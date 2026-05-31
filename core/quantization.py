"""KV cache quantization hooks with configurable granularity.

Three quantization modes:
- per_tensor: single scale for entire tensor (Pipeline A default, section 5/6)
- per_token: one scale per token position (Pipeline B default, section 4)
- per_channel: one scale per channel dimension
- per_group: one scale per group of G channels (e.g., group-64)

Two formulas:
- symmetric: scale = max(|x|) / qmax, zero_point = 0
- asymmetric: scale = (max-min) / (qmax-qmin), zero_point computed
"""

from dataclasses import dataclass
from typing import List, Optional
from contextlib import contextmanager

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class QuantConfig:
    """Configuration for KV cache quantization."""
    bits: int = 4
    symmetric: bool = True
    granularity: str = "per_tensor"  # per_tensor | per_token | per_channel | per_group
    group_size: int = 64
    layers: Optional[List[int]] = None  # None=all layers, list=specific layers
    channels: Optional[List[int]] = None  # None=all channels, list=specific channels
    quantize_keys: bool = True
    quantize_values: bool = True


# Convenience presets matching paper configurations
PRESET_SECTION4 = QuantConfig(symmetric=False, granularity="per_token")
PRESET_SECTION5 = QuantConfig(symmetric=True, granularity="per_tensor")
PRESET_GROUP64 = QuantConfig(symmetric=True, granularity="per_group", group_size=64)


def quantize_tensor(
    x: Tensor,
    bits: int,
    symmetric: bool = True,
    granularity: str = "per_tensor",
    group_size: int = 64,
    channels: Optional[List[int]] = None,
) -> Tensor:
    """Quantize-dequantize a single tensor.

    Supports all granularity modes and symmetric/asymmetric formulas.
    Optionally quantizes only specific channels (for channel ablation).
    """
    if bits >= 16:
        return x

    # Channel-selective: only quantize specified channels
    if channels is not None:
        result = x.clone()
        ch_idx = torch.tensor(channels, device=x.device)
        ch_slice = x[..., ch_idx]
        ch_slice_q = _quantize_core(ch_slice, bits, symmetric, granularity, group_size)
        result[..., ch_idx] = ch_slice_q
        return result

    return _quantize_core(x, bits, symmetric, granularity, group_size)


def _quantize_core(
    x: Tensor,
    bits: int,
    symmetric: bool,
    granularity: str,
    group_size: int,
) -> Tensor:
    """Core quantize-dequantize implementation."""
    if granularity == "per_group":
        return _quantize_per_group(x, bits, symmetric, group_size)

    # Compute qmin/qmax
    if symmetric:
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
    else:
        qmin = 0
        qmax = 2 ** bits - 1

    # Compute scale and zero_point based on granularity
    if granularity == "per_tensor":
        if symmetric:
            max_val = x.abs().max()
            if max_val < 1e-10:
                return x
            scale = max_val / qmax
            x_q = torch.clamp(torch.round(x / scale), qmin, qmax)
            return (x_q * scale).to(x.dtype)
        else:
            x_min, x_max = x.min(), x.max()
            scale = (x_max - x_min) / (qmax - qmin)
            scale = torch.clamp(scale, min=1e-8)
            zero_point = torch.clamp(torch.round(qmin - x_min / scale), qmin, qmax)
            x_q = torch.clamp(torch.round(x / scale + zero_point), qmin, qmax)
            return ((x_q - zero_point) * scale).to(x.dtype)

    elif granularity == "per_token":
        # Per-token: one scale per token position (dim=-1 reduction)
        if symmetric:
            abs_max = x.abs().amax(dim=-1, keepdim=True)
            abs_max = torch.clamp(abs_max, min=1e-8)
            scale = abs_max / qmax
            x_q = torch.clamp(torch.round(x / scale), qmin, qmax)
            return (x_q * scale).to(x.dtype)
        else:
            x_min = x.amin(dim=-1, keepdim=True)
            x_max = x.amax(dim=-1, keepdim=True)
            scale = (x_max - x_min) / (qmax - qmin)
            scale = torch.clamp(scale, min=1e-8)
            zero_point = torch.clamp(torch.round(qmin - x_min / scale), qmin, qmax)
            x_q = torch.clamp(torch.round(x / scale + zero_point), qmin, qmax)
            return ((x_q - zero_point) * scale).to(x.dtype)

    elif granularity == "per_channel":
        # Per-channel: one scale per channel (last dim), reduce over all other dims
        if symmetric:
            abs_max = x.abs().amax(dim=tuple(range(x.ndim - 1)), keepdim=True)
            abs_max = torch.clamp(abs_max, min=1e-8)
            scale = abs_max / qmax
            x_q = torch.clamp(torch.round(x / scale), qmin, qmax)
            return (x_q * scale).to(x.dtype)
        else:
            x_min = x.amin(dim=tuple(range(x.ndim - 1)), keepdim=True)
            x_max = x.amax(dim=tuple(range(x.ndim - 1)), keepdim=True)
            scale = (x_max - x_min) / (qmax - qmin)
            scale = torch.clamp(scale, min=1e-8)
            zero_point = torch.clamp(torch.round(qmin - x_min / scale), qmin, qmax)
            x_q = torch.clamp(torch.round(x / scale + zero_point), qmin, qmax)
            return ((x_q - zero_point) * scale).to(x.dtype)

    else:
        raise ValueError(f"Unknown granularity: {granularity}")


def _quantize_per_group(
    x: Tensor,
    bits: int,
    symmetric: bool,
    group_size: int,
) -> Tensor:
    """Per-group quantization: groups of channels share a scale.

    Vectorized: reshapes last dim into (n_groups, group_size), computes
    scales per-group in parallel, then reshapes back. No Python loop.
    """
    if symmetric:
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
    else:
        qmin = 0
        qmax = 2 ** bits - 1

    n_channels = x.shape[-1]

    # Pad last dim to multiple of group_size if needed
    remainder = n_channels % group_size
    if remainder != 0:
        pad_size = group_size - remainder
        x_padded = F.pad(x, (0, pad_size), value=0.0)
    else:
        pad_size = 0
        x_padded = x

    # Reshape: (..., n_groups, group_size)
    orig_shape = x_padded.shape
    grouped = x_padded.reshape(*orig_shape[:-1], -1, group_size)

    if symmetric:
        abs_max = grouped.abs().amax(dim=-1, keepdim=True)
        abs_max = torch.clamp(abs_max, min=1e-8)
        scale = abs_max / qmax
        g_q = torch.clamp(torch.round(grouped / scale), qmin, qmax)
        result = (g_q * scale).to(x.dtype)
    else:
        g_min = grouped.amin(dim=-1, keepdim=True)
        g_max = grouped.amax(dim=-1, keepdim=True)
        scale = (g_max - g_min) / (qmax - qmin)
        scale = torch.clamp(scale, min=1e-8)
        zero_point = torch.clamp(torch.round(qmin - g_min / scale), qmin, qmax)
        g_q = torch.clamp(torch.round(grouped / scale + zero_point), qmin, qmax)
        result = ((g_q - zero_point) * scale).to(x.dtype)

    # Reshape back and trim padding
    result = result.reshape(orig_shape)
    if pad_size > 0:
        result = result[..., :n_channels]

    return result


class KVQuantizer:
    """Attaches quantize-dequantize hooks to model KV projections.

    Supports:
    - Standard models (k_proj/v_proj): LLaMA, Mistral, Qwen, DeepSeek, Yi, Gemma-2
    - Fused QKV models (qkv_proj): Phi-3.5 (splits at hidden_size boundaries)
    - Layer-selective: quantize only specific layers
    - Channel-selective: quantize only specific channels

    Usage:
        config = QuantConfig(bits=3, symmetric=True, granularity="per_tensor")
        quantizer = KVQuantizer(model, config, model_info)
        quantizer.attach()
        # ... run inference ...
        quantizer.detach()

    Or as context manager:
        with KVQuantizer(model, config, model_info) as q:
            # ... run inference ...
            mse = q.get_mean_mse()
    """

    def __init__(self, model: nn.Module, config: QuantConfig, model_info: dict):
        self.model = model
        self.config = config
        self.model_info = model_info
        self.handles: list = []
        self._mse_values: list = []
        self._per_layer_mse: dict = {}  # layer_idx -> list of MSE values

        # Determine number of layers
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            self.n_layers = len(model.model.layers)
        else:
            self.n_layers = model_info.get("n_layers", 32)

        # Determine which layers to quantize
        if config.layers is not None:
            self._target_layers = set(config.layers)
        else:
            self._target_layers = set(range(self.n_layers))

    def _make_hook(self, layer_idx: int, proj_type: str):
        """Create a quantization hook for a k_proj or v_proj module."""
        config = self.config

        def hook(module, input, output):
            if layer_idx not in self._target_layers:
                return output

            # Skip if this projection type shouldn't be quantized
            if proj_type == "k" and not config.quantize_keys:
                return output
            if proj_type == "v" and not config.quantize_values:
                return output

            # Store original for MSE computation
            original = output.detach()

            # Quantize
            quantized = quantize_tensor(
                output,
                bits=config.bits,
                symmetric=config.symmetric,
                granularity=config.granularity,
                group_size=config.group_size,
                channels=config.channels,
            )

            # Track MSE (global and per-layer)
            mse = F.mse_loss(quantized.detach(), original).item()
            self._mse_values.append(mse)
            self._per_layer_mse.setdefault(layer_idx, []).append(mse)

            return quantized

        return hook

    def _make_fused_qkv_hook(self, layer_idx: int):
        """Create a hook for fused qkv_proj (Phi-3.5).

        The output is [batch, seq, 3*hidden_size] where:
        - Q = output[..., :hs]
        - K = output[..., hs:2*hs]
        - V = output[..., 2*hs:3*hs]
        """
        config = self.config
        hs = self.model_info["hidden_size"]

        def hook(module, input, output):
            if layer_idx not in self._target_layers:
                return output

            result = output.clone()
            original = output.detach()

            # Quantize K portion
            if config.quantize_keys:
                k_slice = output[..., hs:2*hs]
                k_q = quantize_tensor(
                    k_slice,
                    bits=config.bits,
                    symmetric=config.symmetric,
                    granularity=config.granularity,
                    group_size=config.group_size,
                    channels=config.channels,
                )
                result[..., hs:2*hs] = k_q

            # Quantize V portion
            if config.quantize_values:
                v_slice = output[..., 2*hs:3*hs]
                v_q = quantize_tensor(
                    v_slice,
                    bits=config.bits,
                    symmetric=config.symmetric,
                    granularity=config.granularity,
                    group_size=config.group_size,
                    channels=config.channels,
                )
                result[..., 2*hs:3*hs] = v_q

            # Track MSE (on K+V portions only)
            kv_orig = torch.cat([original[..., hs:2*hs], original[..., 2*hs:3*hs]], dim=-1)
            kv_quant = torch.cat([result[..., hs:2*hs].detach(), result[..., 2*hs:3*hs].detach()], dim=-1)
            mse = F.mse_loss(kv_quant, kv_orig).item()
            self._mse_values.append(mse)

            return result

        return hook

    def attach(self) -> "KVQuantizer":
        """Install hooks on the model."""
        self.detach()
        self._mse_values.clear()

        fused_qkv = self.model_info.get("fused_qkv", False)

        if fused_qkv:
            self._attach_fused()
        else:
            self._attach_standard()

        return self

    def _attach_standard(self):
        """Attach hooks to k_proj and v_proj modules."""
        for name, module in self.model.named_modules():
            for layer_idx in range(self.n_layers):
                if f"layers.{layer_idx}." not in name:
                    continue
                if "k_proj" in name and name.endswith("k_proj"):
                    handle = module.register_forward_hook(
                        self._make_hook(layer_idx, "k")
                    )
                    self.handles.append(handle)
                elif "v_proj" in name and name.endswith("v_proj"):
                    handle = module.register_forward_hook(
                        self._make_hook(layer_idx, "v")
                    )
                    self.handles.append(handle)

    def _attach_fused(self):
        """Attach hooks to qkv_proj modules (Phi-3.5)."""
        for name, module in self.model.named_modules():
            for layer_idx in range(self.n_layers):
                if f"layers.{layer_idx}." not in name:
                    continue
                if "qkv_proj" in name and name.endswith("qkv_proj"):
                    handle = module.register_forward_hook(
                        self._make_fused_qkv_hook(layer_idx)
                    )
                    self.handles.append(handle)

    def detach(self):
        """Remove all hooks."""
        for h in self.handles:
            h.remove()
        self.handles = []

    def get_mean_mse(self) -> float:
        """Get mean KV MSE from all tracked quantization operations."""
        if not self._mse_values:
            return 0.0
        return sum(self._mse_values) / len(self._mse_values)

    def get_per_layer_stats(self) -> dict:
        """Get per-layer MSE statistics.

        Returns dict: {layer_idx: {"min": float, "max": float, "mean": float, "std": float, "count": int}}
        """
        stats = {}
        for layer_idx, mse_list in self._per_layer_mse.items():
            t = torch.tensor(mse_list)
            stats[layer_idx] = {
                "min": t.min().item(),
                "max": t.max().item(),
                "mean": t.mean().item(),
                "std": t.std().item() if len(mse_list) > 1 else 0.0,
                "count": len(mse_list),
            }
        return stats

    def reset_stats(self):
        """Reset MSE tracking."""
        self._mse_values.clear()
        self._per_layer_mse.clear()

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, *args):
        self.detach()
        self.model = None


class FlexQuantizer:
    """Applies different quantization configs to different layers.

    For experiments that need per-layer granularity control (e.g., FP16 protection
    for layer 0, per-channel for layer 1, per-tensor for the rest).

    Usage:
        layer_configs = {
            0: QuantConfig(bits=16),  # FP16 protection
            1: QuantConfig(bits=3, granularity="per_channel"),
        }
        default = QuantConfig(bits=3, granularity="per_tensor")
        with FlexQuantizer(model, layer_configs, default, model_info) as fq:
            # ... run inference ...
    """

    def __init__(
        self,
        model: nn.Module,
        layer_configs: dict,
        default_config: Optional[QuantConfig] = None,
        model_info: Optional[dict] = None,
    ):
        self.model = model
        self.layer_configs = layer_configs
        self.default_config = default_config
        self.model_info = model_info or {}
        self.handles: list = []
        self._mse_values: list = []

        if hasattr(model, "model") and hasattr(model.model, "layers"):
            self.n_layers = len(model.model.layers)
        else:
            self.n_layers = self.model_info.get("n_layers", 32)

    def _get_config(self, layer_idx: int) -> Optional[QuantConfig]:
        """Get the config for a specific layer."""
        if layer_idx in self.layer_configs:
            return self.layer_configs[layer_idx]
        return self.default_config

    def _make_hook(self, layer_idx: int, proj_type: str):
        def hook(module, input, output):
            config = self._get_config(layer_idx)
            if config is None:
                return output
            if config.bits >= 16:
                return output
            if proj_type == "k" and not config.quantize_keys:
                return output
            if proj_type == "v" and not config.quantize_values:
                return output

            original = output.detach()
            quantized = quantize_tensor(
                output,
                bits=config.bits,
                symmetric=config.symmetric,
                granularity=config.granularity,
                group_size=config.group_size,
                channels=config.channels,
            )
            mse = F.mse_loss(quantized.detach(), original).item()
            self._mse_values.append(mse)
            return quantized

        return hook

    def _make_fused_qkv_hook(self, layer_idx: int):
        """Create a hook for fused qkv_proj (Phi-3.5) with per-layer configs."""
        hs = self.model_info["hidden_size"]

        def hook(module, input, output):
            config = self._get_config(layer_idx)
            if config is None:
                return output
            if config.bits >= 16:
                return output

            result = output.clone()
            original = output.detach()

            if config.quantize_keys:
                k_slice = output[..., hs:2*hs]
                k_q = quantize_tensor(
                    k_slice, bits=config.bits, symmetric=config.symmetric,
                    granularity=config.granularity, group_size=config.group_size,
                    channels=config.channels,
                )
                result[..., hs:2*hs] = k_q

            if config.quantize_values:
                v_slice = output[..., 2*hs:3*hs]
                v_q = quantize_tensor(
                    v_slice, bits=config.bits, symmetric=config.symmetric,
                    granularity=config.granularity, group_size=config.group_size,
                    channels=config.channels,
                )
                result[..., 2*hs:3*hs] = v_q

            kv_orig = torch.cat([original[..., hs:2*hs], original[..., 2*hs:3*hs]], dim=-1)
            kv_quant = torch.cat([result[..., hs:2*hs].detach(), result[..., 2*hs:3*hs].detach()], dim=-1)
            mse = F.mse_loss(kv_quant, kv_orig).item()
            self._mse_values.append(mse)

            return result

        return hook

    def attach(self) -> "FlexQuantizer":
        self.detach()
        self._mse_values.clear()

        fused_qkv = self.model_info.get("fused_qkv", False)
        if fused_qkv:
            self._attach_fused()
        else:
            self._attach_standard()
        return self

    def _attach_standard(self):
        for name, module in self.model.named_modules():
            for layer_idx in range(self.n_layers):
                if f"layers.{layer_idx}." not in name:
                    continue
                config = self._get_config(layer_idx)
                if config is None:
                    continue
                if "k_proj" in name and name.endswith("k_proj"):
                    handle = module.register_forward_hook(
                        self._make_hook(layer_idx, "k")
                    )
                    self.handles.append(handle)
                elif "v_proj" in name and name.endswith("v_proj"):
                    handle = module.register_forward_hook(
                        self._make_hook(layer_idx, "v")
                    )
                    self.handles.append(handle)

    def _attach_fused(self):
        for name, module in self.model.named_modules():
            for layer_idx in range(self.n_layers):
                if f"layers.{layer_idx}." not in name:
                    continue
                config = self._get_config(layer_idx)
                if config is None:
                    continue
                if "qkv_proj" in name and name.endswith("qkv_proj"):
                    handle = module.register_forward_hook(
                        self._make_fused_qkv_hook(layer_idx)
                    )
                    self.handles.append(handle)

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def get_mean_mse(self) -> float:
        if not self._mse_values:
            return 0.0
        return sum(self._mse_values) / len(self._mse_values)

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, *args):
        self.detach()
        self.model = None


class KVQuantizerDual(KVQuantizer):
    """KV quantizer with separate configs for keys and values (e.g., KIVI).

    KIVI uses per-channel asymmetric for keys and per-group asymmetric for
    values. This class dispatches to different QuantConfigs based on whether
    the hook is on a key or value projection.

    Usage:
        key_cfg = QuantConfig(bits=2, symmetric=False, granularity="per_channel")
        val_cfg = QuantConfig(bits=2, symmetric=False, granularity="per_group", group_size=32)
        with KVQuantizerDual(model, key_cfg, val_cfg, model_info) as q:
            # ... run inference ...
    """

    def __init__(self, model: nn.Module, key_config: QuantConfig,
                 value_config: QuantConfig, model_info: dict):
        # Use key_config as the base for layer targeting and attach logic
        super().__init__(model, key_config, model_info)
        self.key_config = key_config
        self.value_config = value_config

    def _make_hook(self, layer_idx: int, proj_type: str):
        """Create a hook that uses key_config or value_config based on proj_type."""
        key_config = self.key_config
        value_config = self.value_config

        def hook(module, input, output):
            if layer_idx not in self._target_layers:
                return output

            config = key_config if proj_type == "k" else value_config

            if proj_type == "k" and not config.quantize_keys:
                return output
            if proj_type == "v" and not config.quantize_values:
                return output

            original = output.detach()

            # Per-channel quantization is degenerate for single-token tensors
            # (each channel has only one value → scale=0 → garbage).
            # Fall back to per-token, matching KIVI behavior where a single
            # token's key quantization uses one scale for the whole token.
            granularity = config.granularity
            if granularity == "per_channel" and output.shape[-2] == 1:
                granularity = "per_token"

            quantized = quantize_tensor(
                output,
                bits=config.bits,
                symmetric=config.symmetric,
                granularity=granularity,
                group_size=config.group_size,
                channels=config.channels,
            )

            mse = F.mse_loss(quantized.detach(), original).item()
            self._mse_values.append(mse)
            self._per_layer_mse.setdefault(layer_idx, []).append(mse)

            return quantized

        return hook

    def _make_fused_qkv_hook(self, layer_idx: int):
        """Fused qkv_proj hook (Phi-3.5) with separate K/V configs."""
        key_config = self.key_config
        value_config = self.value_config
        hs = self.model_info["hidden_size"]

        def hook(module, input, output):
            if layer_idx not in self._target_layers:
                return output

            result = output.clone()
            original = output.detach()

            if key_config.quantize_keys:
                k_slice = output[..., hs:2*hs]
                k_gran = key_config.granularity
                if k_gran == "per_channel" and k_slice.shape[-2] == 1:
                    k_gran = "per_token"
                k_q = quantize_tensor(
                    k_slice, bits=key_config.bits, symmetric=key_config.symmetric,
                    granularity=k_gran, group_size=key_config.group_size,
                    channels=key_config.channels,
                )
                result[..., hs:2*hs] = k_q

            if value_config.quantize_values:
                v_slice = output[..., 2*hs:3*hs]
                v_q = quantize_tensor(
                    v_slice, bits=value_config.bits, symmetric=value_config.symmetric,
                    granularity=value_config.granularity, group_size=value_config.group_size,
                    channels=value_config.channels,
                )
                result[..., 2*hs:3*hs] = v_q

            kv_orig = torch.cat([original[..., hs:2*hs], original[..., 2*hs:3*hs]], dim=-1)
            kv_quant = torch.cat([result[..., hs:2*hs].detach(), result[..., 2*hs:3*hs].detach()], dim=-1)
            mse = F.mse_loss(kv_quant, kv_orig).item()
            self._mse_values.append(mse)
            self._per_layer_mse.setdefault(layer_idx, []).append(mse)

            return result

        return hook


# KIVI preset: per-channel asymmetric keys, per-group asymmetric values (Liu et al., ICML 2024)
PRESET_KIVI_KEY = QuantConfig(symmetric=False, granularity="per_channel")
PRESET_KIVI_VALUE = QuantConfig(symmetric=False, granularity="per_group", group_size=32)


@contextmanager
def quantize_kv(model, config: QuantConfig, model_info: dict):
    """Context manager for quick KV quantization."""
    q = KVQuantizer(model, config, model_info)
    q.attach()
    try:
        yield q
    finally:
        q.detach()


@contextmanager
def quantize_kv_dual(model, key_config: QuantConfig, value_config: QuantConfig,
                     model_info: dict):
    """Context manager for dual-config KV quantization (e.g., KIVI)."""
    q = KVQuantizerDual(model, key_config, value_config, model_info)
    q.attach()
    try:
        yield q
    finally:
        q.detach()


@contextmanager
def flex_quantize(model, layer_configs, default_config=None, model_info=None):
    """Context manager for FlexQuantizer."""
    fq = FlexQuantizer(model, layer_configs, default_config, model_info)
    fq.attach()
    try:
        yield fq
    finally:
        fq.detach()

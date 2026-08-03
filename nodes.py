"""ComfyUI nodes for MiniMax H3 NestedTensor latent inspection and upscaling."""

from __future__ import annotations

from .utils import UPSCALE_METHODS, inspect_latent, upscale_nested_latent


class MiniMaxH3LatentInspector:
    """Pass-through LATENT node that prints a full NestedTensor structure map."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "inspect"
    CATEGORY = "latent/minimax_h3"
    DESCRIPTION = (
        "Debug MiniMax H3 / NestedTensor LATENT structure. "
        "Prints type, keys, NestedTensor members, shapes, and probed attributes to the console."
    )

    def inspect(self, samples):
        report = inspect_latent(samples)
        print(report)
        return (samples,)


class MiniMaxH3LatentUpscale:
    """Spatially upscale MiniMax H3 video latents inside a NestedTensor AV pair."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "scale_by": ("FLOAT", {"default": 1.5, "min": 0.01, "max": 8.0, "step": 0.01}),
                "method": (list(UPSCALE_METHODS), {"default": "bilinear"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "upscale"
    CATEGORY = "latent/minimax_h3"
    DESCRIPTION = (
        "Upscale MiniMax H3 latent video spatial dims (H, W) via torch.nn.functional.interpolate. "
        "Audio NestedTensor member is passed through unchanged. "
        "Rebuilds samples with comfy.nested_tensor.NestedTensor."
    )

    def upscale(self, samples, scale_by, method):
        return (upscale_nested_latent(samples, scale_by, method),)

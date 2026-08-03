"""MiniMaxH3_LatentUpscaler — NestedTensor upscale + CONST re-noise for MiniMax H3."""

from .nodes import MiniMaxH3LatentUpscaleCombined

NODE_CLASS_MAPPINGS = {
    "MiniMaxH3LatentUpscaleCombined": MiniMaxH3LatentUpscaleCombined,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3LatentUpscaleCombined": "MiniMax H3 Latent Upscale Combined",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

"""MiniMaxH3_LatentUpscaler — NestedTensor-aware latent inspect / upscale for MiniMax H3."""

from .nodes import MiniMaxH3LatentInspector, MiniMaxH3LatentUpscale

NODE_CLASS_MAPPINGS = {
    "MiniMaxH3LatentInspector": MiniMaxH3LatentInspector,
    "MiniMaxH3LatentUpscale": MiniMaxH3LatentUpscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3LatentInspector": "MiniMax H3 Latent Inspector",
    "MiniMaxH3LatentUpscale": "MiniMax H3 Latent Upscale",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

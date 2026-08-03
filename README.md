# MiniMaxH3_LatentUpscaler

ComfyUI custom nodes that make **latent spatial upscaling** work with MiniMax H3 between samplers.

This is **not** a learned AI upscaler. Stock `LatentUpscaleBy` crashes on MiniMax H3 because `common_upscale` calls `.reshape` on a ComfyUI `NestedTensor`, which is not a `torch.Tensor`.

## Discovered MiniMax H3 latent structure

Source of truth in this install:

- Class: `comfy.nested_tensor.NestedTensor` in `ComfyUI/comfy/nested_tensor.py`
- Creation: `ComfyUI/comfy_extras/nodes_minimax_h3.py` (`_empty_av_latent`)

This is **ComfyUI’s** NestedTensor, **not** `torch.nested`.

### Constructor

```python
NestedTensor(tensors)  # stores list(tensors); sets is_nested = True
```

Repo usage:

```python
{"samples": comfy.nested_tensor.NestedTensor((video, audio))}
```

### What it stores

| Piece | Detail |
|--------|--------|
| `tensors` | `list` of plain `torch.Tensor` |
| `is_nested` | always `True` |
| `unbind()` | returns `self.tensors` |
| `shape` / `dtype` / `device` | taken from **`tensors[0]` only** |
| mask / padding fields | **none** (no `.mask`, `.payload`, etc.) |

### MiniMax H3 pair layout

```
LATENT dict
  └── samples: NestedTensor
        ├── tensors[0]  video  [B, 24, T, H/16, W/16]
        └── tensors[1]  audio  [B, 32, 2, T_audio]
```

- Video `T` comes from MiniMax’s frame grid (`video_latent_t` / 17k+5 at 24 fps).
- Audio time is `round(duration * 40)` (`AUDIO_LATENT_FPS = 40`).
- Optional `noise_mask` may also be a NestedTensor pair (same video/audio indexing), as used by LTX AV concat helpers.

### Why stock upscale crashes

`LatentUpscaleBy` → `comfy.utils.common_upscale(samples["samples"], ...)` → `samples.reshape(...)`.

`NestedTensor` has no `.reshape`, hence:

`AttributeError: 'NestedTensor' object has no attribute 'reshape'`

## Nodes

### MiniMax H3 Latent Inspector

- **Input / output:** `LATENT` (pass-through)
- Prints type, keys, NestedTensor members, shapes, dtypes, devices, public attrs, and probed fields (`.tensor`, `.values`, `.data`, `.mask`, `.storage`, `.payload`) to the console.

### MiniMax H3 Latent Upscale

- **Inputs:** `LATENT`, `scale_by`, `method` (`nearest` | `bilinear` | `bicubic`)
- **Output:** `LATENT`
- Extracts NestedTensor members → spatially interpolates **video** (`H`, `W`) with `torch.nn.functional.interpolate` → rebuilds with `comfy.nested_tensor.NestedTensor(...)`.
- **Audio is left unchanged.**
- If `noise_mask` is NestedTensor, the video mask is scaled the same way; the audio mask is passed through.
- Other LATENT dict keys are preserved via a shallow copy.

## Helpers (`utils.py`)

- `extract_tensor(samples)` → `(list[Tensor], was_nested)`
- `wrap_tensor(tensors, *, was_nested)` → NestedTensor via the official constructor, or a plain tensor
- `upscale_nested_latent(latent, scale_by, method)` → full LATENT dict transform

## Install

Package path:

`ComfyUI/custom_nodes/MiniMaxH3_LatentUpscaler/`

Restart ComfyUI (or reload custom nodes). Search the node menu under `latent/minimax_h3`.

## Typical use

1. Sample at base latent resolution.
2. Insert **MiniMax H3 Latent Upscale** (`scale_by`, method).
3. Continue sampling / decode with the upscaled NestedTensor latent.

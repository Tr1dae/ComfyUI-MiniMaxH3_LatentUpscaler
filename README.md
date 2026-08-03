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

### MiniMax H3 Latent Upscale Combined

- **Inputs:** `LATENT`, `scale_by`, `method`, `MODEL`, `NOISE`, `SIGMAS`
- **Output:** `LATENT` (upscaled + re-noised)
- Same spatial upscale as above, then NestedTensor-safe AddNoise (stock `AddNoise` crashes on NestedTensor at `torch.count_nonzero`).
- Prefer this for split-sampler / hires-style workflows.

### MiniMax H3 Add Noise

- NestedTensor-only counterpart of stock AddNoise (member-wise `process_latent_in` → `noise_scaling` → `process_latent_out`).

## Helpers (`utils.py`)

- `extract_tensor(samples)` → `(list[Tensor], was_nested)`
- `wrap_tensor(tensors, *, was_nested)` → NestedTensor via the official constructor, or a plain tensor
- `upscale_nested_latent(latent, scale_by, method)` → full LATENT dict transform
- `add_noise_nested_latent(model, noise, sigmas, latent)` → NestedTensor-safe AddNoise
- `upscale_and_add_noise(...)` → upscale then re-noise

## Install

Package path:

`ComfyUI/custom_nodes/MiniMaxH3_LatentUpscaler/`

Restart ComfyUI (or reload custom nodes). Search the node menu under `latent/minimax_h3`.

## Typical use (split samplers / hires-style)

MiniMax H3 uses CONST flow sampling. Correct start-of-pass mixing is:

`x = σ · noise + (1 − σ) · clean_latent`

Do **not** upscale the live noisy `output` and resume with DisableNoise — that interpolates residual noise and causes large colored dots.

Do **not** upscale `denoised_output` and resume with DisableNoise only — you feed nearly-clean latents into a low-σ tail, so pass 2 barely moves and looks soft.

### Recommended wiring

1. **SamplerCustomAdvanced #1** with high σ half (`SplitSigmas` high output).
2. Take **`denoised_output`** (predicted clean / x0) — do **not** use `output` (noisy; upscaling it → colored dots).
3. **MiniMax H3 Latent Upscale Combined**:
   - `scale_by` / `method`
   - `model` = same MiniMax model
   - `noise` = **RandomNoise**
   - `sigmas` = the **low** half from SplitSigmas (same schedule as sampler #2)
4. **SamplerCustomAdvanced #2**:
   - `noise` = **DisableNoise**
   - `sigmas` = same low half
   - `latent_image` = combined node output

**Why soft results happened before:** MiniMax is CONST/flow. DisableNoise starts as `(1−σ)·latent`. Sampler `output` is already `inverse_noise_scaling`'d so this reconstitutes correctly. Stock-style AddNoise omitted that inverse step, so pass 2 dampened the mix again and looked soft. Combined now applies `inverse_noise_scaling(sigmas[0])` after mixing.

**Note:** Stock ComfyUI `AddNoise` also fails on NestedTensor (`count_nonzero` TypeError). Use this pack’s Combined / Add Noise nodes.

### Why the two SamplerCustomAdvanced sockets behave differently

| Socket | What it is | After upscale + EmptyNoise resume |
|--------|------------|-----------------------------------|
| `output` | Latent still on the noise trajectory (`x` at end of pass 1) | Spatially warped noise → chromatic dots |
| `denoised_output` | Model’s clean estimate (`x0`) | Soft / under-denoised unless you **AddNoise** at pass-2 σ |

Upscaling is only a spatial resize. **AddNoise** restores a valid noisy latent at the new resolution for the second schedule.

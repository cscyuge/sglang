# Wan2.2-Remix Comfy Workflow Static Alignment

SGLang does not execute ComfyUI workflow graphs. The `*-comfy-v1` workflow
presets encode the subset of ComfyUI node semantics that affects Wan2.2-Remix
generation and execute them through SGLang-native pipeline stages.

The static alignment tool extracts these fields from a ComfyUI UI workflow JSON:

- generation task (`t2v` or `i2v`)
- width, height, frame count, and fps
- KSamplerAdvanced steps, CFG, sampler, scheduler, start/end step ranges
- ModelSamplingSD3 flow shift
- high-noise and low-noise transformer ranges

It intentionally ignores prompt text, file names, preview metadata, node
positions, and other ComfyUI UI-only fields.

## Commands

```bash
python -m sglang.multimodal_gen.tools.wan_comfy_workflow \
  --workflow /path/to/Wan2.2-Remix-comfy-t2v-workflow.json \
  --preset wan2.2-remix/nsfw-t2v-comfy-v1
```

```bash
python -m sglang.multimodal_gen.tools.wan_comfy_workflow \
  --workflow /path/to/Wan2.2-Remix-comfy-i2v-workflow.json \
  --preset wan2.2-remix/nsfw-i2v-comfy-v1
```

When the workflow and preset match, the command prints the extracted semantics
and a final match message. A mismatch exits non-zero and lists the differing
fields.

## Expected Wan2.2-Remix Values

| Field | T2V Comfy workflow | I2V Comfy workflow |
|-------|--------------------|--------------------|
| Preset | `wan2.2-remix/nsfw-t2v-comfy-v1` | `wan2.2-remix/nsfw-i2v-comfy-v1` |
| Task | `t2v` | `i2v` |
| Resolution | `1280x720` | `1280x720` |
| Frames | `81` | `33` |
| FPS | `16` | `16` |
| Steps | `12` | `12` |
| Sampler | `euler` | `euler` |
| Schedule | `simple` | `simple` |
| Flow shift | `5.0` | `8.0` |
| High-noise range | `[0, 10)` on `transformer` | `[0, 10)` on `transformer` |
| Low-noise range | `[10, end)` on `transformer_2` | `[10, end)` on `transformer_2` |
| CFG | `1.0` / `1.0` | `1.0` / `1.0` |

This alignment verifies that the SGLang presets track the important execution
parameters in the provided ComfyUI workflows. It does not prove pixel-level
equivalence with ComfyUI, because ComfyUI is not running in this environment.

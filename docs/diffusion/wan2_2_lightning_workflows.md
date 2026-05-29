# Wan2.2 Lightning Workflows

This page documents the SGLang workflow preset for
`wan2.2_LIGHTNING-EDITION_long-video_FP8GGUF.json`.

SGLang does not execute the ComfyUI JSON graph directly. The executable preset
maps the workflow's active I2V first segment into SGLang's `/v1/workflows`
service. The source JSON also contains long-video continuation subgraphs, but
those subgraphs are disabled in the workflow and are listed as unsupported.

## Presets

| Preset | Status | Meaning |
|--------|--------|---------|
| `wan2.2-lightning/nsfw-i2v-comfy-v1` | ready | Active I2V first segment from the Comfy workflow |
| `wan2.2-lightning/nsfw-long-video-comfy-v1` | unsupported | Full multi-segment long-video chaining |

The ready preset uses:

- `832x480`, `81` frames, `16` fps
- `4` denoising steps
- Euler sampler with simple schedule
- high-noise transformer for steps `[0, 2)` with flow shift `5`
- low-noise transformer for steps `[2, end)` with flow shift `8`
- CFG values `guidance_scale=1.0` and `guidance_scale_2=1.0`
- the Comfy workflow's negative prompt as the default negative prompt

The Comfy first segment has separate `ModelSamplingSD3` shifts for high and low
models (`5` and `8`). SGLang maps those into per-expert request-local schedulers
for this preset. This aligns the first-segment sampler shifts, but does not make
the service a general ComfyUI graph executor.

## Serve

Serve the converted Diffusers-format model tree:

```bash
sglang serve \
  --model-type diffusion \
  --model-path /path/to/Wan2p2_I2V_A14B_NSFWSVICamera_Q8Dequant_fp16_SGLang \
  --model-id Wan2.2-I2V-A14B-Diffusers \
  --num-gpus 4 \
  --ulysses-degree 2 \
  --enable-cfg-parallel \
  --dit-layerwise-offload true \
  --text-encoder-cpu-offload \
  --pin-cpu-memory \
  --host 0.0.0.0 \
  --port 30020 \
  --output-path /path/to/outputs \
  --input-save-path /path/to/inputs
```

Tune GPU count and offload settings for the target machine. The preset itself is
independent of the launch script.

## List

```bash
curl --noproxy '*' -sS \
  "http://localhost:30020/v1/workflows?model_family=wan2.2-lightning"
```

## Run I2V

`input_reference` can be a file path visible to the server, a remote image URL,
or a data image.

```bash
curl --noproxy '*' -sS "http://localhost:30020/v1/workflows/runs" \
  -H "Content-Type: application/json" \
  -d '{
    "workflow": "wan2.2-lightning/nsfw-i2v-comfy-v1",
    "model": "Wan2.2-I2V-A14B-Diffusers",
    "input": {
      "prompt": "cinematic camera motion, natural motion, high quality",
      "input_reference": "/path/to/input.png"
    },
    "parameters": {
      "seed": 514625469125877
    },
    "output": {
      "response_format": "url"
    }
  }'
```

The request may override `width`, `height`, `num_frames`, `fps`,
`num_inference_steps`, `guidance_scale`, `guidance_scale_2`, and `seed`. If
`input.negative_prompt` is omitted, the preset uses the negative prompt from the
Comfy workflow.

Poll and download the run like other workflow jobs:

```bash
curl --noproxy '*' -sS "http://localhost:30020/v1/workflows/runs/<RUN_ID>"
curl --noproxy '*' -sS -L \
  "http://localhost:30020/v1/workflows/runs/<RUN_ID>/content" \
  -o output.mp4
```

## Current Limits

The ready preset does not implement arbitrary ComfyUI node execution. In
particular, it does not execute the disabled `PainterLongVideo` continuation
subgraphs, RIFE interpolation, upscaling nodes, or custom Comfy-only latent
handoff logic. Use it when the desired product behavior is one SGLang request
with image plus text input and one video output.

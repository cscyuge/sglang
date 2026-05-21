# Wan2.2-Remix Client Guide

This page is for callers of a running SGLang Wan2.2-Remix service. It does not
cover model conversion, server launch, or ComfyUI setup.

## Service Model

Wan2.2-Remix is exposed through the SGLang workflow API:

- `POST /v1/workflows/runs`: create a video generation run
- `GET /v1/workflows/runs/{run_id}`: poll run status
- `GET /v1/workflows/runs/{run_id}/content`: download the completed video

Callers do not send or execute ComfyUI workflow JSON. Use these presets:

| Task | Preset | Typical Port |
|------|--------|--------------|
| Text to video | `wan2.2-remix/nsfw-t2v-comfy-v1` | `30010` |
| Image to video | `wan2.2-remix/nsfw-i2v-comfy-v1` | `30020` |

The `*-comfy-v1` presets match the generation semantics extracted from the
Wan2.2-Remix ComfyUI workflows, including sampler, schedule, flow shift, step
count, CFG values, high/low transformer split, default frame count, fps, and
resolution.

## Text To Video

```bash
curl --noproxy '*' -sS "http://<T2V_HOST>:30010/v1/workflows/runs" \
  -H "Content-Type: application/json" \
  -d '{
    "workflow": "wan2.2-remix/nsfw-t2v-comfy-v1",
    "model": "Wan2.2-T2V-A14B-Diffusers",
    "input": {
      "prompt": "a calm cinematic shot of clouds moving over a mountain lake",
      "negative_prompt": "low quality, blurry, artifacts"
    },
    "parameters": {
      "width": 1280,
      "height": 720,
      "num_frames": 81,
      "fps": 16,
      "num_inference_steps": 12,
      "seed": 20260519,
      "guidance_scale": 1.0,
      "guidance_scale_2": 1.0
    },
    "output": {
      "response_format": "url"
    }
  }'
```

## Image To Video

`input_reference` can be a server-visible file path, remote image URL, or data
image. The image must be accessible to the SGLang server process.

```bash
curl --noproxy '*' -sS "http://<I2V_HOST>:30020/v1/workflows/runs" \
  -H "Content-Type: application/json" \
  -d '{
    "workflow": "wan2.2-remix/nsfw-i2v-comfy-v1",
    "model": "Wan2.2-I2V-A14B-Diffusers",
    "input": {
      "prompt": "animate the image with slow natural camera motion",
      "negative_prompt": "low quality, blurry, artifacts",
      "input_reference": "/path/to/input.png"
    },
    "parameters": {
      "width": 1280,
      "height": 720,
      "num_frames": 33,
      "fps": 16,
      "num_inference_steps": 12,
      "seed": 20260520,
      "guidance_scale": 1.0,
      "guidance_scale_2": 1.0
    },
    "output": {
      "response_format": "url"
    }
  }'
```

For I2V, `width` and `height` define the target output area. The final video
keeps the input image aspect ratio and is rounded to the model grid. Read
`effective_parameters.output_width` and `effective_parameters.output_height` for
the actual output size.

## Poll And Download

Create returns immediately with `status: queued`:

```json
{
  "id": "b0295a7a-856e-4ead-b307-7edaf9f28597",
  "status": "queued",
  "output": {
    "url": null,
    "file_path": "/server/output/path/b0295a7a-856e-4ead-b307-7edaf9f28597.mp4"
  },
  "effective_parameters": {
    "width": 480,
    "height": 832,
    "output_width": 464,
    "output_height": 832
  }
}
```

Poll until `status` is `completed`:

```bash
curl --noproxy '*' -sS "http://<HOST>:<PORT>/v1/workflows/runs/<RUN_ID>"
```

Then download:

```bash
curl --noproxy '*' -L "http://<HOST>:<PORT>/v1/workflows/runs/<RUN_ID>/content" \
  -o output.mp4
```

If cloud storage is configured, `output.url` may point directly to uploaded
content. Otherwise it is a relative URL served by the same SGLang process.

## Parameter Notes

- Keep `guidance_scale` and `guidance_scale_2` at `1.0` for Comfy-derived
  Wan2.2-Remix behavior unless you intentionally want different CFG behavior.
- Use a fixed `seed` for repeatable runs.
- Lower `num_frames`, `width`, `height`, or `num_inference_steps` for smoke
  tests.
- Do not send ComfyUI workflow JSON; choose the preset name instead.
- Use the T2V service for T2V presets and the I2V service for I2V presets.

## Common Errors

- `Unknown workflow preset`: the `workflow` name is wrong or the server is not
  running a build with Wan2.2-Remix presets.
- `requires task`: the request was sent to the wrong service, for example I2V
  preset to the T2V server.
- `input_reference is required`: I2V request did not include an image source.
- Localhost requests route through a proxy: add `--noproxy '*'` for curl or
  disable proxy handling in the client.

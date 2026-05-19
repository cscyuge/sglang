import json
import pathlib
import tempfile
import unittest

import torch
from safetensors.torch import load_file, save_file

from sglang.multimodal_gen.tools.wan_comfy_repack import (
    COMFYUI_TRANSFORMER_PREFIX,
    convert_comfyui_transformer,
    preserve_workflow_files,
    repack_comfyui_wan,
)
from sglang.multimodal_gen.tools.wan_repack import convert_transformer_key


class TestWanComfyRepack(unittest.TestCase):
    def test_convert_transformer_key_strips_comfyui_prefix_and_renames(self):
        key = f"{COMFYUI_TRANSFORMER_PREFIX}" "blocks.0.self_attn.q.weight"

        self.assertEqual(
            convert_transformer_key(key, source_prefix=COMFYUI_TRANSFORMER_PREFIX),
            "blocks.0.attn1.to_q.weight",
        )

    def test_convert_transformer_key_swaps_norm2_norm3_names(self):
        self.assertEqual(
            convert_transformer_key("blocks.0.norm2.weight"),
            "blocks.0.norm3.weight",
        )
        self.assertEqual(
            convert_transformer_key("blocks.0.norm3.weight"),
            "blocks.0.norm2.weight",
        )

    def test_convert_comfyui_transformer_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            source_path = tmp_path / "source.safetensors"
            config_path = tmp_path / "config.json"
            output_dir = tmp_path / "transformer"

            save_file(
                {
                    f"{COMFYUI_TRANSFORMER_PREFIX}patch_embedding.weight": torch.zeros(
                        2, 16, 1, 2, 2
                    ),
                    f"{COMFYUI_TRANSFORMER_PREFIX}blocks.0.self_attn.q.weight": torch.ones(
                        2, 2
                    ),
                    f"{COMFYUI_TRANSFORMER_PREFIX}blocks.0.norm2.weight": torch.ones(2),
                },
                source_path,
            )
            config_path.write_text(json.dumps({"in_channels": 16}))

            convert_comfyui_transformer(
                model_type="Wan2.2-T2V-A14B",
                source_path=source_path,
                output_dir=output_dir,
                transformer_config_path=config_path,
                max_shard_size_gb=1e-6,
            )

            shard_paths = sorted(
                output_dir.glob("diffusion_pytorch_model-*.safetensors")
            )
            self.assertGreaterEqual(len(shard_paths), 1)
            converted = {}
            for shard_path in shard_paths:
                converted.update(load_file(shard_path))

            self.assertIn("patch_embedding.weight", converted)
            self.assertIn("blocks.0.attn1.to_q.weight", converted)
            self.assertIn("blocks.0.norm3.weight", converted)
            self.assertNotIn(
                f"{COMFYUI_TRANSFORMER_PREFIX}blocks.0.self_attn.q.weight",
                converted,
            )
            self.assertTrue((output_dir / "config.json").is_file())

            index = json.loads(
                (
                    output_dir / "diffusion_pytorch_model.safetensors.index.json"
                ).read_text()
            )
            self.assertEqual(set(index["weight_map"]), set(converted))
            self.assertIn(
                index["weight_map"]["blocks.0.attn1.to_q.weight"],
                {path.name for path in shard_paths},
            )

    def test_convert_comfyui_transformer_rejects_wrong_base_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            source_path = tmp_path / "source.safetensors"
            config_path = tmp_path / "config.json"
            output_dir = tmp_path / "transformer"

            save_file(
                {
                    f"{COMFYUI_TRANSFORMER_PREFIX}patch_embedding.weight": torch.zeros(
                        2, 36, 1, 2, 2
                    ),
                },
                source_path,
            )
            config_path.write_text(json.dumps({"in_channels": 16}))

            with self.assertRaisesRegex(ValueError, "input channels"):
                convert_comfyui_transformer(
                    model_type="Wan2.2-T2V-A14B",
                    source_path=source_path,
                    output_dir=output_dir,
                    transformer_config_path=config_path,
                )

    def test_repack_comfyui_wan_builds_cascade_model_tree(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            base_path = tmp_path / "base"
            output_path = tmp_path / "out"
            high_path = tmp_path / "high.safetensors"
            low_path = tmp_path / "low.safetensors"

            (base_path / "transformer").mkdir(parents=True)
            (base_path / "transformer_2").mkdir()
            (base_path / "scheduler").mkdir()
            (base_path / "text_encoder").mkdir()
            (base_path / "model_index.json").write_text("{}")
            (base_path / "scheduler" / "scheduler_config.json").write_text("{}")
            (base_path / "text_encoder" / "config.json").write_text("{}")
            (base_path / "transformer" / "config.json").write_text(
                json.dumps({"in_channels": 16})
            )
            (base_path / "transformer_2" / "config.json").write_text(
                json.dumps({"in_channels": 16})
            )

            for path, fill_value in ((high_path, 1.0), (low_path, 2.0)):
                save_file(
                    {
                        f"{COMFYUI_TRANSFORMER_PREFIX}patch_embedding.weight": torch.full(
                            (2, 16, 1, 2, 2), fill_value
                        ),
                        f"{COMFYUI_TRANSFORMER_PREFIX}head.head.weight": torch.full(
                            (2, 2), fill_value
                        ),
                    },
                    path,
                )

            repack_comfyui_wan(
                model_type="Wan2.2-T2V-A14B",
                original_model_path=base_path,
                output_path=output_path,
                high_path=high_path,
                low_path=low_path,
            )

            self.assertTrue((output_path / "model_index.json").is_file())
            self.assertTrue((output_path / "scheduler").is_dir())
            self.assertTrue((output_path / "text_encoder").is_dir())

            high = load_file(next((output_path / "transformer").glob("*.safetensors")))
            low = load_file(next((output_path / "transformer_2").glob("*.safetensors")))
            self.assertIn("proj_out.weight", high)
            self.assertIn("proj_out.weight", low)

    def test_preserve_workflow_files_copies_workflow_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            workflow_path = tmp_path / "source_workflow"
            output_path = tmp_path / "out"
            workflow_path.mkdir()
            output_path.mkdir()
            (workflow_path / "example.json").write_text(
                json.dumps({"nodes": [], "links": []})
            )

            preserve_workflow_files(workflow_path, output_path)

            self.assertEqual(
                json.loads((output_path / "workflow" / "example.json").read_text()),
                {"nodes": [], "links": []},
            )


if __name__ == "__main__":
    unittest.main()

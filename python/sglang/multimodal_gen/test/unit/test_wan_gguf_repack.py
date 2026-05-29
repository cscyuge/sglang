import json
import pathlib
import tempfile
import unittest

import gguf
import numpy as np
import torch
from safetensors.torch import load_file

from sglang.multimodal_gen.tools.wan_gguf_repack import (
    convert_gguf_transformer,
    repack_gguf_wan,
)


def _write_gguf(path: pathlib.Path, tensors: dict[str, np.ndarray]) -> None:
    writer = gguf.GGUFWriter(path, "wan")
    for name, tensor in tensors.items():
        writer.add_tensor(name, tensor)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


class TestWanGGUFRepack(unittest.TestCase):
    def test_convert_gguf_transformer_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            source_path = tmp_path / "source.gguf"
            config_path = tmp_path / "config.json"
            output_dir = tmp_path / "transformer"

            _write_gguf(
                source_path,
                {
                    "patch_embedding.weight": np.zeros(
                        (2, 16, 1, 2, 2), dtype=np.float32
                    ),
                    "blocks.0.self_attn.q.weight": np.ones(
                        (2, 2), dtype=np.float16
                    ),
                    "blocks.0.norm2.weight": np.ones(2, dtype=np.float16),
                },
            )
            config_path.write_text(json.dumps({"in_channels": 16}))

            stats = convert_gguf_transformer(
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

            self.assertEqual(stats.tensor_count, 3)
            self.assertIn("patch_embedding.weight", converted)
            self.assertIn("blocks.0.attn1.to_q.weight", converted)
            self.assertIn("blocks.0.norm3.weight", converted)
            self.assertNotIn("blocks.0.self_attn.q.weight", converted)
            self.assertEqual(
                converted["blocks.0.attn1.to_q.weight"].dtype, torch.float16
            )
            self.assertTrue((output_dir / "config.json").is_file())
            self.assertTrue((output_dir / "sglang_gguf_conversion.json").is_file())

            index = json.loads(
                (
                    output_dir / "diffusion_pytorch_model.safetensors.index.json"
                ).read_text()
            )
            self.assertEqual(set(index["weight_map"]), set(converted))

    def test_convert_gguf_transformer_rejects_wrong_base_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            source_path = tmp_path / "source.gguf"
            config_path = tmp_path / "config.json"
            output_dir = tmp_path / "transformer"

            _write_gguf(
                source_path,
                {
                    "patch_embedding.weight": np.zeros(
                        (2, 36, 1, 2, 2), dtype=np.float32
                    ),
                },
            )
            config_path.write_text(json.dumps({"in_channels": 16}))

            with self.assertRaisesRegex(ValueError, "input channels"):
                convert_gguf_transformer(
                    model_type="Wan2.2-T2V-A14B",
                    source_path=source_path,
                    output_dir=output_dir,
                    transformer_config_path=config_path,
                )

    def test_repack_gguf_wan_builds_cascade_model_tree(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            base_path = tmp_path / "base"
            output_path = tmp_path / "out"
            high_path = tmp_path / "high.gguf"
            low_path = tmp_path / "low.gguf"

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
                _write_gguf(
                    path,
                    {
                        "patch_embedding.weight": np.full(
                            (2, 16, 1, 2, 2), fill_value, dtype=np.float32
                        ),
                        "head.head.weight": np.full(
                            (2, 2), fill_value, dtype=np.float16
                        ),
                    },
                )

            repack_gguf_wan(
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
            self.assertTrue(torch.equal(high["proj_out.weight"], torch.ones(2, 2)))
            self.assertTrue(
                torch.equal(low["proj_out.weight"], torch.full((2, 2), 2.0))
            )


if __name__ == "__main__":
    unittest.main()

import os
import tempfile
import unittest

import torch
from torch import nn

from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.multimodal_gen.runtime.utils.stream_r1_checkpoint import (
    clean_stream_r1_state_dict_keys,
    load_stream_r1_generator_checkpoint,
    resolve_stream_r1_checkpoint_path,
    select_stream_r1_state_dict,
)


class TinyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, 2))
        self.mapped = nn.Parameter(torch.zeros(1))


class TestStreamR1CheckpointLoader(unittest.TestCase):
    def _checkpoint_path(self, payload):
        tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        tmp.close()
        torch.save(payload, tmp.name)
        self.addCleanup(lambda: os.path.exists(tmp.name) and os.remove(tmp.name))
        return tmp.name

    def test_selects_ema_when_requested(self):
        checkpoint = {
            "generator_ema": {"weight": torch.ones(1)},
            "generator": {"weight": torch.zeros(1)},
        }

        state_dict, source_key = select_stream_r1_state_dict(checkpoint, use_ema=True)

        self.assertEqual(source_key, "generator_ema")
        torch.testing.assert_close(state_dict["weight"], torch.ones(1))

    def test_can_skip_ema(self):
        checkpoint = {
            "generator_ema": {"weight": torch.ones(1)},
            "generator": {"weight": torch.zeros(1)},
        }

        state_dict, source_key = select_stream_r1_state_dict(checkpoint, use_ema=False)

        self.assertEqual(source_key, "generator")
        torch.testing.assert_close(state_dict["weight"], torch.zeros(1))

    def test_cleans_wrapper_prefixes_and_skips_metadata(self):
        cleaned, skipped = clean_stream_r1_state_dict_keys(
            {
                "_fsdp_wrapped_module._orig_mod.module.weight": torch.ones(2, 2),
                "model.mapped": torch.ones(1),
                "step": 800,
            }
        )

        self.assertEqual(set(cleaned), {"weight", "mapped"})
        self.assertEqual(skipped, ("step",))

    def test_loads_checkpoint_with_name_mapping(self):
        module = TinyModule()
        checkpoint_path = self._checkpoint_path(
            {
                "generator": {
                    "_checkpoint_wrapped_module.official.weight": torch.ones(1),
                    "_checkpoint_wrapped_module.weight": torch.full((2, 2), 2.0),
                }
            }
        )
        mapping = get_param_names_mapping({r"^official\.weight$": "mapped"})

        info = load_stream_r1_generator_checkpoint(
            module,
            checkpoint_path,
            use_ema=False,
            param_names_mapping=mapping,
        )

        self.assertEqual(info.source_key, "generator")
        self.assertEqual(info.num_tensors, 2)
        self.assertFalse(info.unexpected_keys)
        torch.testing.assert_close(module.mapped, torch.ones(1))
        torch.testing.assert_close(module.weight, torch.full((2, 2), 2.0))

    def test_rejects_checkpoint_with_no_matching_keys(self):
        module = TinyModule()
        checkpoint_path = self._checkpoint_path(
            {"generator": {"not_a_model_key": torch.ones(1)}}
        )

        with self.assertRaisesRegex(RuntimeError, "did not match"):
            load_stream_r1_generator_checkpoint(
                module,
                checkpoint_path,
                use_ema=False,
            )

    def test_resolves_checkpoint_directory_by_basename(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = os.path.join(tmp_dir, f"{os.path.basename(tmp_dir)}.pt")
            torch.save({"generator": {"weight": torch.ones(2, 2)}}, checkpoint_path)

            self.assertEqual(
                resolve_stream_r1_checkpoint_path(tmp_dir), checkpoint_path
            )


if __name__ == "__main__":
    unittest.main()

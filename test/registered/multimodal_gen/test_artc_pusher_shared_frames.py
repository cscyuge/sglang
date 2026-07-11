import os

import numpy as np

from sglang.multimodal_gen.runtime.utils import artc_pusher


def _descriptor(tmp_path, values):
    req_dir = tmp_path / "req"
    req_dir.mkdir()
    path = req_dir / "frames.rgb24"
    values.tofile(path)
    return {
        "__sglang_shared_frames__": True,
        "path": str(path),
        "cleanup_dir": str(req_dir),
        "shape": values.shape,
        "dtype": str(values.dtype),
    }


def test_open_shared_frames_transfers_and_unlinks_ownership(tmp_path):
    expected = np.arange(24, dtype=np.uint8).reshape(2, 2, 2, 3)
    descriptor = _descriptor(tmp_path, expected)

    frames = artc_pusher._open_shared_frames(descriptor)

    assert isinstance(frames, np.memmap)
    assert np.array_equal(frames, expected)
    assert not os.path.exists(descriptor["path"])
    assert not os.path.exists(descriptor["cleanup_dir"])
    assert np.array_equal(frames, expected)


def test_release_shared_frames_cleans_dropped_queue_item(tmp_path):
    expected = np.zeros((2, 2, 2, 3), dtype=np.uint8)
    descriptor = _descriptor(tmp_path, expected)

    artc_pusher._release_chunk_item(("chunk", descriptor, None, {}))

    assert not os.path.exists(descriptor["path"])
    assert not os.path.exists(descriptor["cleanup_dir"])


def test_regular_arrays_remain_unchanged():
    frames = np.zeros((3, 2, 2, 3), dtype=np.uint8)

    assert artc_pusher._open_shared_frames(frames) is frames
    assert artc_pusher._shared_frame_count(frames) == 3
    artc_pusher._release_shared_frames(frames)


def test_tracked_shared_frames_are_released_on_stop_cleanup(tmp_path):
    expected = np.zeros((1, 2, 2, 3), dtype=np.uint8)
    descriptor = _descriptor(tmp_path, expected)
    pusher = artc_pusher.ArtcPusher.__new__(artc_pusher.ArtcPusher)
    pusher._shared_frame_descriptors = []

    pusher._track_shared_frames(descriptor)
    pusher._release_tracked_shared_frames()

    assert pusher._shared_frame_descriptors == []
    assert not os.path.exists(descriptor["path"])
    assert not os.path.exists(descriptor["cleanup_dir"])

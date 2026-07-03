import threading

import pytest

import sglang.multimodal_gen.runtime.managers.forward_context as fc_mod
from sglang.multimodal_gen.runtime.managers.forward_context import (
    ForwardContext,
    get_forward_context,
    set_forward_context,
)


def test_set_forward_context_is_thread_local():
    old_fallback_context = fc_mod._forward_context
    fc_mod._forward_context = None
    worker_entered = threading.Event()
    release_worker = threading.Event()
    worker_errors = []

    def worker():
        try:
            with set_forward_context(current_timestep=2, attn_metadata=None):
                worker_entered.set()
                assert get_forward_context().current_timestep == 2
                release_worker.wait(timeout=2)
            with pytest.raises(AssertionError):
                get_forward_context()
        except BaseException as exc:
            worker_errors.append(exc)

    try:
        with set_forward_context(current_timestep=1, attn_metadata=None):
            thread = threading.Thread(target=worker)
            thread.start()
            assert worker_entered.wait(timeout=2)
            assert get_forward_context().current_timestep == 1
            release_worker.set()
            thread.join(timeout=2)
            assert not thread.is_alive()
            assert get_forward_context().current_timestep == 1

        assert worker_errors == []
        with pytest.raises(AssertionError):
            get_forward_context()
    finally:
        release_worker.set()
        fc_mod._forward_context = old_fallback_context


def test_forward_context_direct_global_fallback_remains_compatible():
    old_fallback_context = fc_mod._forward_context
    fc_mod._forward_context = ForwardContext(current_timestep=7, attn_metadata=None)

    try:
        assert get_forward_context().current_timestep == 7
        with set_forward_context(current_timestep=8, attn_metadata=None):
            assert get_forward_context().current_timestep == 8
        assert get_forward_context().current_timestep == 7
    finally:
        fc_mod._forward_context = old_fallback_context

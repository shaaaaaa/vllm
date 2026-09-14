# SPDX-License-Identifier: Apache-2.0
"""Resident-boundary admission and fallback without device initialization."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


def manager(alignment, bundle):
    path = (
        Path(__file__).resolve().parents[2]
        / "vllm/v1/core/single_type_kv_cache_manager.py"
    )
    cls = next(
        n
        for n in ast.parse(path.read_text(encoding="utf8")).body
        if isinstance(n, ast.ClassDef) and n.name == "DSALatentManager"
    )
    cls.bases = [ast.Name(id="object", ctx=ast.Load())]
    names = {
        "_compact_required_indices",
        "_blocks_per_bundle",
        "_round_up_to_bundle",
        "_round_down_to_bundle",
    }
    cls.body = [n for n in cls.body if getattr(n, "name", None) in names]
    ns = dict(cdiv=lambda a, b: (a + b - 1) // b)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
            str(path),
            "exec",
        ),
        ns,
    )
    obj = ns[cls.name]()
    obj.block_size, obj.scratch_blocks = 4, 2
    obj.block_pool = NS(blocks_per_bundle=bundle)
    obj.checkpoint_tail_alignment = alignment
    return obj


@pytest.mark.parametrize("bundle", [1, 3])
@pytest.mark.parametrize("prefix", [16, 17, 23, 31, 32, 33])
def test_bootstrap_and_decode_keep_whole_boundary(bundle, prefix):
    obj = manager(16, bundle)
    for bootstrap in (True, False):
        indices = obj._compact_required_indices(
            prefix, prefix + 5, prefix + 1, force_compact=bootstrap
        )
        needed = set(range((prefix // 16 * 16) // 4, (prefix + 5 + 3) // 4))
        assert needed <= set(indices)
        assert len(indices) == len(set(indices))
        assert len(indices) % bundle == 0


def test_disabled_policy_keeps_only_the_original_tail_block():
    obj = manager(0, 1)
    assert obj._compact_required_indices(23, 27, 24) == [0, 1, 5, 6]
    assert manager(16, 1)._compact_required_indices(23, 27, 24) == [0, 1, 4, 5, 6]


def test_saved_full_chunks_stay_released_and_fallback_rebuilds_dense_prefix():
    obj = manager(16, 1)
    assert obj._compact_required_indices(23, 43, 42, released_blocks=8) == [
        0,
        1,
        8,
        9,
        10,
    ]
    assert obj._compact_required_indices(23, 20, 20) == list(range(5))

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.2 shared-indexer construction semantics for DeepseekV2MLAAttention.

Checks the checkpoint-aware indexer rules ported for GLM-5.2:
- "shared" backbone layers (indexer_types) construct no Indexer, set
  skip_topk, and still reach the shared top-k buffer;
- "full" producers and MTP/nextn layers always keep a full split
  wk/weights_proj Indexer;
- GLM-5.1 configs (no indexer_types, optionally runtime index cache via
  index_topk_freq) keep an Indexer on every layer;
- the loader-side weight filter only drops '.indexer.' weights whose
  layer prefix has no instantiated indexer module.
"""

import types

import pytest
import torch

from vllm.config import CacheConfig, DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLAAttention

pytestmark = pytest.mark.cpu_test


@pytest.fixture()
def default_vllm_config():
    """CPU VllmConfig (the shared fixture relies on platform auto-detection)."""
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        yield


NUM_LAYERS = 78
# GLM-5.2 backbone producers: 0, 1, 2, 6, 10, ..., 74 (21 producers)
PRODUCER_LAYERS = {0, 1, 2} | {6 + 4 * i for i in range(18)}
INDEXER_TYPES_78 = [
    "full" if i in PRODUCER_LAYERS else "shared" for i in range(NUM_LAYERS)
]


def _hf_config(**extra):
    cfg = dict(
        num_hidden_layers=NUM_LAYERS,
        index_topk=2048,
        index_n_heads=64,
        index_head_dim=128,
        qk_rope_head_dim=64,
        indexer_types=None,
        index_topk_freq=None,
        index_skip_topk_offset=None,
        index_topk_pattern=None,
        indexer_rope_interleave=True,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_type": "default"},
    )
    cfg.update(extra)
    return types.SimpleNamespace(**cfg)


def _vllm_config(hf_config):
    return types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            hf_config=hf_config, max_model_len=8192, dtype="bfloat16"
        ),
        scheduler_config=types.SimpleNamespace(max_num_batched_tokens=4096),
        quant_config=None,
        cache_config=CacheConfig("auto"),
    )


@pytest.fixture()
def _stub_attn_backend(monkeypatch):
    """No sparse MLA backend exists on CPU; stub the backend selection."""
    import vllm.model_executor.layers.attention.mla_attention as mla_attn_mod

    class _StubImpl:
        def __init__(self, **kwargs):
            pass

    class _StubBackend:
        @staticmethod
        def get_name():
            return "stub_mla_sparse"

        @staticmethod
        def get_impl_cls():
            return _StubImpl

    monkeypatch.setattr(
        mla_attn_mod, "get_attn_backend", lambda *a, **kw: _StubBackend()
    )


@pytest.fixture()
def _single_tp(monkeypatch):
    """Construction needs rank metadata; these tests execute no collectives."""
    import vllm.distributed.parallel_state as ps

    monkeypatch.setattr(ps, "_TP", types.SimpleNamespace(world_size=1, rank_in_group=0))


def _build(default_vllm_config, _single_tp, _stub_attn_backend, prefix, cfg, buffer):
    return DeepseekV2MLAAttention(
        _vllm_config(cfg),
        cfg,
        hidden_size=512,
        num_heads=8,
        qk_nope_head_dim=64,
        qk_rope_head_dim=64,
        v_head_dim=64,
        q_lora_rank=256,
        kv_lora_rank=512,
        max_position_embeddings=8192,
        cache_config=CacheConfig("auto"),
        quant_config=None,
        prefix=prefix,
        topk_indices_buffer=buffer,
    )


def test_glm52_shared_layer_omits_indexer(
    default_vllm_config, _single_tp, _stub_attn_backend
):
    buffer = torch.empty(64, 2048, dtype=torch.int32)
    cfg = _hf_config(indexer_types=INDEXER_TYPES_78)
    attn = _build(
        default_vllm_config,
        _single_tp,
        _stub_attn_backend,
        "model.layers.3.self_attn",
        cfg,
        buffer,
    )
    assert attn.indexer is None
    assert attn.indexer_rope_emb is None
    assert attn.skip_topk is True
    # the shared buffer stays reachable without a local Indexer
    assert attn.mla_attn.topk_indices_buffer is buffer
    assert attn.mla_attn.topk_tokens == 2048
    assert attn.mla_attn.skip_topk is True
    # no indexer weights on a shared-consumer layer
    assert not [n for n, _ in attn.named_parameters() if n.startswith("indexer.")]


def test_glm52_producer_layer_keeps_full_indexer(
    default_vllm_config, _single_tp, _stub_attn_backend
):
    buffer = torch.empty(64, 2048, dtype=torch.int32)
    cfg = _hf_config(indexer_types=INDEXER_TYPES_78)
    attn = _build(
        default_vllm_config,
        _single_tp,
        _stub_attn_backend,
        "model.layers.6.self_attn",
        cfg,
        buffer,
    )
    assert attn.indexer is not None
    assert attn.skip_topk is False
    # split wk/weights_proj layout preserved
    assert hasattr(attn.indexer, "wk")
    assert hasattr(attn.indexer, "weights_proj")
    assert attn.indexer.topk_indices_buffer is buffer
    assert [n for n, _ in attn.named_parameters() if n.startswith("indexer.")]


def test_glm52_mtp_layer_always_keeps_indexer(
    default_vllm_config, _single_tp, _stub_attn_backend
):
    buffer = torch.empty(64, 2048, dtype=torch.int32)
    cfg = _hf_config(indexer_types=INDEXER_TYPES_78)
    attn = _build(
        default_vllm_config,
        _single_tp,
        _stub_attn_backend,
        "model.layers.78.self_attn",
        cfg,
        buffer,
    )
    assert attn.indexer is not None
    assert attn.skip_topk is False


def test_glm51_keeps_indexer_on_every_layer(
    default_vllm_config, _single_tp, _stub_attn_backend
):
    buffer = torch.empty(64, 2048, dtype=torch.int32)
    cfg = _hf_config()
    attn = _build(
        default_vllm_config,
        _single_tp,
        _stub_attn_backend,
        "model.layers.5.self_attn",
        cfg,
        buffer,
    )
    assert attn.indexer is not None
    assert attn.skip_topk is False


def test_glm51_runtime_override_keeps_production_compute_policy(
    default_vllm_config, _single_tp, _stub_attn_backend
):
    buffer = torch.empty(64, 2048, dtype=torch.int32)
    # An old optional override must not implicitly enable a new compute policy.
    cfg = _hf_config(index_topk_freq=4, index_skip_topk_offset=2)
    attn = _build(
        default_vllm_config,
        _single_tp,
        _stub_attn_backend,
        "model.layers.3.self_attn",
        cfg,
        buffer,
    )
    assert attn.indexer is not None
    assert attn.skip_topk is False
    assert attn.mla_attn.skip_topk is False


def test_glm52_invalid_indexer_types_fail_closed(
    default_vllm_config, _single_tp, _stub_attn_backend
):
    buffer = torch.empty(64, 2048, dtype=torch.int32)
    with pytest.raises(ValueError, match="num_hidden_layers"):
        _build(
            default_vllm_config,
            _single_tp,
            _stub_attn_backend,
            "model.layers.3.self_attn",
            _hf_config(indexer_types=INDEXER_TYPES_78[:77]),
            buffer,
        )
    with pytest.raises(ValueError, match="full.*shared"):
        _build(
            default_vllm_config,
            _single_tp,
            _stub_attn_backend,
            "model.layers.3.self_attn",
            _hf_config(indexer_types=["weird"] * NUM_LAYERS),
            buffer,
        )
    with pytest.raises(ValueError, match="start with a 'full'"):
        _build(
            default_vllm_config,
            _single_tp,
            _stub_attn_backend,
            "model.layers.3.self_attn",
            _hf_config(indexer_types=["shared"] + INDEXER_TYPES_78[1:]),
            buffer,
        )
    bad_pattern = ["F"] * NUM_LAYERS
    bad_pattern[3] = "S"
    with pytest.raises(ValueError, match="disagree"):
        _build(
            default_vllm_config,
            _single_tp,
            _stub_attn_backend,
            "model.layers.0.self_attn",
            _hf_config(indexer_types=INDEXER_TYPES_78, index_topk_pattern=bad_pattern),
            buffer,
        )


def test_loader_filter_drops_only_omitted_indexer_weights(
    default_vllm_config, _single_tp, _stub_attn_backend
):
    buffer = torch.empty(64, 2048, dtype=torch.int32)
    cfg = _hf_config(indexer_types=INDEXER_TYPES_78)
    producer = _build(
        default_vllm_config,
        _single_tp,
        _stub_attn_backend,
        "model.layers.6.self_attn",
        cfg,
        buffer,
    )
    shared = _build(
        default_vllm_config,
        _single_tp,
        _stub_attn_backend,
        "model.layers.3.self_attn",
        cfg,
        buffer,
    )
    # name parameters as the full model would (the loader filter matches
    # '.indexer.' inside the full module path)
    params_dict = {
        f"model.layers.6.self_attn.{n}": v for n, v in producer.named_parameters()
    } | {f"model.layers.3.self_attn.{n}": v for n, v in shared.named_parameters()}
    present = {n.rsplit(".indexer.", 1)[0] for n in params_dict if ".indexer." in n}
    assert "model.layers.6.self_attn" in present
    assert "model.layers.3.self_attn" not in present
    checkpoint_names = [
        "model.layers.3.self_attn.indexer.wk.weight",  # omitted layer -> drop
        "model.layers.6.self_attn.indexer.wk.weight",  # producer -> keep
        "model.layers.6.self_attn.o_proj.weight",  # not an indexer weight
    ]
    dropped = [
        n
        for n in checkpoint_names
        if ".indexer." in n and n.rsplit(".indexer.", 1)[0] not in present
    ]
    assert dropped == ["model.layers.3.self_attn.indexer.wk.weight"]

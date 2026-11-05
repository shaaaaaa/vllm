# SPDX-License-Identifier: Apache-2.0
"""Manual end-to-end smoke test for GLM5.1 DSA shared KV pool.

Run this from the vLLM repository root on an Ascend NPU machine with the
patched vLLM and vLLM-Ascend branches installed or importable.

Example:
    GLM51_MODEL=/models/GLM-5-w4a8 \
    python run_dsa_shared_pool_glm51.py --tp 16

If vLLM-Ascend is checked out next to vLLM instead of installed, this script
will automatically prepend ../vllm-ascend to sys.path. You can also set:
    VLLM_ASCEND_REPO=/workspace/lmy/vllm-ascend
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


DEFAULT_MODEL = "/workspace/models/GLM-5.1-w4a8"
DEFAULT_DATASET_PATH = "/workspace/dataset/custom_32_context_64k.jsonl"
DEFAULT_DATASET_NAME = "custom"
DEFAULT_ENDPOINT = "/v1/completions"

DSA_ENV_DEFAULTS = {
    "VLLM_USE_V1": "1",
    "VLLM_LOG_STATS_INTERVAL": "1",
    "VLLM_ASCEND_DSA_UNBUNDLE": "1",
    "VLLM_ASCEND_DSA_TWO_GROUPS": "1",
    "VLLM_ASCEND_DSA_SHARED_POOL": "1",
    "VLLM_ASCEND_DSA_SHRINK_LATENT": "2",
    # Prefix caching is not supported by DSA two-group/shared-pool mode.
    "VLLM_ENABLE_PREFIX_CACHING": "0",
}

LMCACHE_ENV_DEFAULTS = {
    "LMCACHE_MAX_LOCAL_CPU_SIZE": "50",
    "LMCACHE_ENABLE_SPARSE_ATTENTION": "true",
    "LMCACHE_SAVE_UNFULL_CHUNK": "true",
    "LMCACHE_CHUNK_SIZE": "256",
    "LMCACHE_USE_LAYERWISE": "true",
}

DEFAULT_HF_OVERRIDES = '{"num_hidden_layers": 8}'
DEFAULT_KV_TRANSFER_CONFIG = (
    '{"kv_connector":"LMCacheAscendConnectorV1Dynamic",'
    '"kv_role":"kv_both",'
    '"kv_connector_module_path":"lmcache_ascend.integration.vllm.'
    'lmcache_ascend_connector_v1"}'
)


@dataclass
class RunConfig:
    model: str | None
    tp: int
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    num_prompts: int
    dataset_name: str
    dataset_path: str | None
    endpoint: str
    apply_chat_template: bool
    prompt_tokens: int
    max_tokens: int
    ignore_eos: bool
    temperature: float
    gpu_memory_utilization: float
    dtype: str
    quantization: str | None
    load_format: str
    hf_overrides: dict[str, Any]
    kv_transfer_config: dict[str, Any] | None
    enforce_eager: bool
    backend_device: str | None
    simulate_cpu: bool
    simulate_index_topk: int
    correctness_baseline: bool


def maybe_add_vllm_ascend_repo() -> None:
    configured = os.getenv("VLLM_ASCEND_REPO")
    candidates = []
    if configured:
        candidates.append(Path(configured))
    candidates.append(Path(__file__).resolve().parent.parent / "vllm-ascend")

    for candidate in candidates:
        if (candidate / "vllm_ascend").is_dir():
            sys.path.insert(0, str(candidate))
            return


def parse_json_dict(raw: str | None, name: str) -> dict[str, Any]:
    if raw is None or raw == "":
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError(f"{name} must decode to a JSON object.")
    return value


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description="Run GLM5.1 once with DSA shared KV pool enabled."
    )
    parser.add_argument(
        "--model",
        default=os.getenv("GLM51_MODEL", DEFAULT_MODEL),
        help="GLM5.1 model path or HF id. Can also be set by GLM51_MODEL.",
    )
    parser.add_argument("--tp", type=int, default=int(os.getenv("TP_SIZE", "1")))
    parser.add_argument(
        "--max-model-len", "--max_model_len", type=int, default=70000
    )
    parser.add_argument("--max-num-seqs", "--max_num_seqs", type=int, default=64)
    parser.add_argument(
        "--max-num-batched-tokens",
        "--max_num_batched_tokens",
        type=int,
        default=16384,
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=8,
        help="Number of prompts submitted by this offline smoke test.",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="Benchmark dataset name. Only custom JSONL is consumed directly.",
    )
    parser.add_argument(
        "--dataset-path",
        default=os.getenv("VLLM_BENCH_DATASET_PATH", DEFAULT_DATASET_PATH),
        help="Custom JSONL dataset path used by the benchmark command.",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="Benchmark endpoint being mirrored; informational for this offline run.",
    )
    parser.add_argument(
        "--skip-chat-template",
        action="store_true",
        help="Match vllm bench serve --skip-chat-template when needed.",
    )
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=65536,
        help="Fallback synthetic prompt length when --dataset-path is unavailable.",
    )
    parser.add_argument("--max-tokens", "--output-len", type=int, default=1000)
    parser.add_argument(
        "--ignore-eos",
        dest="ignore_eos",
        action="store_true",
        default=True,
        help="Match vllm bench serve --ignore-eos. Enabled by default.",
    )
    parser.add_argument(
        "--no-ignore-eos",
        dest="ignore_eos",
        action="store_false",
        help="Allow EOS to stop generation.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--gpu-memory-utilization",
        "--gpu_memory_utilization",
        type=float,
        default=0.75,
    )
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--quantization", default=os.getenv("VLLM_QUANTIZATION"))
    parser.add_argument(
        "--load-format",
        "--load_format",
        default=os.getenv("VLLM_LOAD_FORMAT", "dummy"),
        help="Default is dummy so the smoke test reaches KV/cache paths on one NPU.",
    )
    parser.add_argument(
        "--hf-overrides",
        "--hf_overrides",
        default=os.getenv("VLLM_HF_OVERRIDES", DEFAULT_HF_OVERRIDES),
        help="JSON object passed to vLLM hf_overrides.",
    )
    parser.add_argument(
        "--kv-transfer-config",
        "--kv_transfer_config",
        default=os.getenv("VLLM_KV_TRANSFER_CONFIG", DEFAULT_KV_TRANSFER_CONFIG),
        help="JSON object passed to vLLM kv_transfer_config.",
    )
    parser.add_argument(
        "--no-kv-transfer-config",
        action="store_true",
        help="Disable the default LMCache kv_transfer_config.",
    )
    parser.add_argument(
        "--no-enforce-eager",
        action="store_true",
        help="Allow graph mode. Eager is the default for bring-up stability.",
    )
    parser.add_argument(
        "--backend-device",
        choices=("cpu", "npu"),
        default=os.getenv("VLLM_ASCEND_DSA_OFFLOAD_BACKEND_DEVICE"),
        help="Optional reference offload backend device. Unset by default.",
    )
    parser.add_argument(
        "--simulate-cpu",
        action="store_true",
        help="Run a CPU-only shared-pool simulation instead of loading GLM5.1.",
    )
    parser.add_argument(
        "--simulate-index-topk",
        type=int,
        default=2048,
        help="index_topk used by --simulate-cpu when --model is not provided.",
    )
    parser.add_argument(
        "--correctness-baseline",
        action="store_true",
        help=(
            "Run a no-LMCache semantic baseline: real weights by default, no "
            "hf layer truncation, no kv_transfer_config, and no latent shrink."
        ),
    )
    args = parser.parse_args()
    if not args.model and not args.simulate_cpu:
        parser.error("missing --model or GLM51_MODEL")

    argv = sys.argv[1:]

    def _arg_was_set(*names: str) -> bool:
        return any(arg == name or arg.startswith(f"{name}=") for arg in argv for name in names)

    load_format = args.load_format
    hf_overrides_raw = args.hf_overrides
    if args.correctness_baseline:
        if not _arg_was_set("--load-format", "--load_format") and "VLLM_LOAD_FORMAT" not in os.environ:
            load_format = "auto"
        if not _arg_was_set("--hf-overrides", "--hf_overrides") and "VLLM_HF_OVERRIDES" not in os.environ:
            hf_overrides_raw = "{}"

    hf_overrides = parse_json_dict(hf_overrides_raw, "--hf-overrides")
    kv_transfer_config = None
    if not args.no_kv_transfer_config and not args.correctness_baseline:
        kv_transfer_config = parse_json_dict(
            args.kv_transfer_config, "--kv-transfer-config"
        )
    return RunConfig(
        model=args.model,
        tp=args.tp,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        num_prompts=args.num_prompts,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        endpoint=args.endpoint,
        apply_chat_template=not args.skip_chat_template,
        prompt_tokens=args.prompt_tokens,
        max_tokens=args.max_tokens,
        ignore_eos=args.ignore_eos,
        temperature=args.temperature,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        quantization=args.quantization,
        load_format=load_format,
        hf_overrides=hf_overrides,
        kv_transfer_config=kv_transfer_config,
        enforce_eager=not args.no_enforce_eager,
        backend_device=args.backend_device,
        simulate_cpu=args.simulate_cpu,
        simulate_index_topk=args.simulate_index_topk,
        correctness_baseline=args.correctness_baseline,
    )


def apply_env(config: RunConfig) -> None:
    for key, value in DSA_ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)
    if config.correctness_baseline:
        # A no-LMCache correctness baseline must keep full latent resident.
        # Otherwise shrink/remap makes the output depend on LMCache being present.
        os.environ["VLLM_ASCEND_DSA_SHRINK_LATENT"] = "0"
    else:
        for key, value in LMCACHE_ENV_DEFAULTS.items():
            os.environ.setdefault(key, value)
    if config.backend_device is not None:
        os.environ["VLLM_ASCEND_DSA_OFFLOAD_BACKEND_DEVICE"] = config.backend_device


def require_npu() -> None:
    import torch

    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("torch_npu is required for this GLM5.1 NPU test.") from exc
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("No available Ascend NPU was found.")


def get_index_topk(model: str) -> int:
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    text_config = getattr(hf_config, "text_config", hf_config)
    index_topk = getattr(text_config, "index_topk", None)
    if index_topk is None:
        raise RuntimeError(
            f"{model!r} does not expose index_topk; this is not a DSA sparse model."
        )
    return int(index_topk)


def get_simulated_index_topk(config: RunConfig) -> int:
    if config.model:
        try:
            return get_index_topk(config.model)
        except Exception as exc:
            print(
                "[DSA-SHARED-SIM] could not read model config; "
                f"falling back to --simulate-index-topk={config.simulate_index_topk}: {exc}"
            )
    return config.simulate_index_topk


def load_custom_dataset_prompts(
    model: str,
    dataset_path: str | None,
    batch_size: int,
    apply_chat_template: bool,
) -> list[str] | None:
    if not dataset_path:
        return None
    path = Path(dataset_path)
    if not path.is_file():
        return None

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if len(prompts) >= batch_size:
                break
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            prompt = item.get("prompt")
            if not isinstance(prompt, str):
                raise RuntimeError(
                    f"Custom dataset row in {dataset_path} does not contain "
                    "a string 'prompt' field."
                )
            if apply_chat_template:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
            prompts.append(prompt)
    if len(prompts) < batch_size:
        raise RuntimeError(
            f"Dataset {dataset_path} only has {len(prompts)} usable prompts, "
            f"but --num-prompts={batch_size}."
        )
    return prompts


def make_synthetic_prompts(model: str, target_tokens: int, batch_size: int) -> list[str]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    seed = (
        "This is a deterministic DSA shared pool smoke-test context. "
        "The shared KV cache pool manages latent and indexer pages by bundle. "
        "After prefill, extra latent pages are released, while decode uses the "
        "indexer top-k selection to gather only the needed latent tokens. "
    )
    prompt = seed
    # Build a deterministic prompt long enough to trigger the post-prefill shrink.
    while len(tokenizer.encode(prompt, add_special_tokens=False)) < target_tokens:
        prompt += "\n" + seed
    prompt += "\nQuestion: What does this smoke test verify?"
    return [prompt for _ in range(batch_size)]


def make_prompts(config: RunConfig) -> list[str]:
    assert config.model is not None
    if config.dataset_name == "custom":
        prompts = load_custom_dataset_prompts(
            config.model,
            config.dataset_path,
            config.num_prompts,
            config.apply_chat_template,
        )
        if prompts is not None:
            return prompts
        print(
            "[DSA-SHARED-E2E] dataset not found; using synthetic prompts: "
            f"{config.dataset_path}"
        )
    elif config.dataset_path:
        raise RuntimeError(
            f"Unsupported --dataset-name={config.dataset_name!r}; this script "
            "only reads custom JSONL directly."
        )
    return make_synthetic_prompts(
        config.model, config.prompt_tokens, config.num_prompts
    )


def run_generation(config: RunConfig) -> None:
    require_npu()
    assert config.model is not None
    index_topk = get_index_topk(config.model)
    if config.num_prompts > config.max_num_seqs:
        raise RuntimeError(
            f"--num-prompts={config.num_prompts} must be <= "
            f"--max-num-seqs={config.max_num_seqs}."
        )
    shrink_enabled = os.environ.get("VLLM_ASCEND_DSA_SHRINK_LATENT", "0") != "0"
    if shrink_enabled and config.prompt_tokens <= index_topk:
        raise RuntimeError(
            f"--prompt-tokens={config.prompt_tokens} must be > index_topk={index_topk} "
            "to exercise DSA latent shrink."
        )
    if shrink_enabled and config.max_model_len <= config.prompt_tokens + config.max_tokens:
        raise RuntimeError(
            "--max-model-len must exceed prompt_tokens + max_tokens "
            f"({config.prompt_tokens + config.max_tokens})."
        )

    from vllm import LLM, SamplingParams

    prompts = make_prompts(config)
    sampling = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        ignore_eos=config.ignore_eos,
    )
    llm_kwargs = {
        "model": config.model,
        "trust_remote_code": True,
        "tensor_parallel_size": config.tp,
        "max_model_len": config.max_model_len,
        "max_num_seqs": config.max_num_seqs,
        "max_num_batched_tokens": config.max_num_batched_tokens,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "dtype": config.dtype,
        "load_format": config.load_format,
        "hf_overrides": config.hf_overrides,
        "enforce_eager": config.enforce_eager,
        "enable_prefix_caching": False,
    }
    if config.quantization:
        llm_kwargs["quantization"] = config.quantization
    if config.kv_transfer_config:
        llm_kwargs["kv_transfer_config"] = config.kv_transfer_config

    print("[DSA-SHARED-E2E] config:")
    print(f"  model={config.model}")
    print(f"  correctness_baseline={config.correctness_baseline}")
    print(f"  tp={config.tp}")
    print(f"  load_format={config.load_format}")
    print(f"  hf_overrides={config.hf_overrides}")
    print(f"  kv_transfer_config={config.kv_transfer_config}")
    print(f"  max_model_len={config.max_model_len}")
    print(f"  max_num_seqs={config.max_num_seqs}")
    print(f"  max_num_batched_tokens={config.max_num_batched_tokens}")
    print(f"  num_prompts={config.num_prompts}")
    print(f"  dataset_name={config.dataset_name}")
    print(f"  dataset_path={config.dataset_path}")
    print(f"  endpoint={config.endpoint}")
    print(f"  index_topk={index_topk}")
    print(f"  fallback_prompt_tokens={config.prompt_tokens}")
    print(f"  max_tokens={config.max_tokens}")
    print(f"  ignore_eos={config.ignore_eos}")
    print(f"  backend_device={config.backend_device}")
    for key in sorted(DSA_ENV_DEFAULTS):
        print(f"  {key}={os.environ.get(key)}")
    for key in sorted(LMCACHE_ENV_DEFAULTS):
        print(f"  {key}={os.environ.get(key)}")
    if config.load_format == "dummy" or "num_hidden_layers" in config.hf_overrides:
        print(
            "[DSA-SHARED-E2E] WARNING: this is not a semantic correctness run "
            "(dummy weights and/or truncated num_hidden_layers are enabled)."
        )
    if config.correctness_baseline:
        print(
            "[DSA-SHARED-E2E] correctness baseline: LMCache disabled, "
            "latent shrink disabled, using resident KV path."
        )

    llm = LLM(**llm_kwargs)
    outputs = llm.generate(prompts, sampling, use_tqdm=False)

    if len(outputs) != len(prompts):
        raise RuntimeError(f"Expected {len(prompts)} outputs, got {len(outputs)}.")
    for i, output in enumerate(outputs):
        if not output.outputs:
            raise RuntimeError(f"Request {i} returned no completion.")
        completion = output.outputs[0]
        token_ids = getattr(completion, "token_ids", None) or []
        text = completion.text
        print(f"[DSA-SHARED-E2E] request={i} tokens={len(token_ids)} text={text!r}")
        if len(token_ids) == 0 and not text:
            raise RuntimeError(f"Request {i} generated neither token ids nor text.")

    print("[DSA-SHARED-E2E] PASS")


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _align_up(value: int, alignment: int) -> int:
    return _ceil_div(value, alignment) * alignment


def run_cpu_simulation(config: RunConfig) -> None:
    import torch

    from vllm.v1.core.dsa_shared_pool import (
        DSASharedBlockLayout,
        DSASharedBlockOwner,
        DSASharedBundleAllocator,
        dsa_scratch_blocks_for_topk,
    )

    index_topk = get_simulated_index_topk(config)
    block_size = 128
    dtype_bytes = 2
    latent_dim = 576
    indexer_dim = 128
    latent_page = block_size * latent_dim * dtype_bytes
    indexer_page = block_size * indexer_dim * dtype_bytes
    # Use a larger prompt for CPU simulation so the shrink/free/reuse path moves
    # multiple bundles, not just one tail bundle.
    sim_prompt_tokens = max(config.prompt_tokens, index_topk + 16 * block_size)
    sim_batch = max(config.num_prompts, 4)
    prompt_blocks = _ceil_div(sim_prompt_tokens, block_size)
    scratch_blocks = dsa_scratch_blocks_for_topk(index_topk, block_size)
    latent_prefill_bundles = _ceil_div(prompt_blocks, 2)
    indexer_prefill_bundles = _ceil_div(prompt_blocks, 9)
    capacity_bundles = latent_prefill_bundles + indexer_prefill_bundles

    layout = DSASharedBlockLayout(
        latent_page_size_bytes=latent_page,
        indexer_page_size_bytes=indexer_page,
        capacity_bundles=capacity_bundles,
    )
    latent_prefill_bundles = _ceil_div(prompt_blocks, layout.latent_blocks_per_bundle)
    indexer_prefill_bundles = _ceil_div(prompt_blocks, layout.indexer_blocks_per_bundle)
    keep_latent_blocks = _align_up(
        scratch_blocks, layout.latent_blocks_per_bundle
    )
    keep_latent_bundles = _ceil_div(
        keep_latent_blocks, layout.latent_blocks_per_bundle
    )

    allocator = DSASharedBundleAllocator(layout)
    latent_bundles = allocator.allocate(
        DSASharedBlockOwner.LATENT, latent_prefill_bundles
    )
    indexer_bundles = allocator.allocate(
        DSASharedBlockOwner.INDEXER, indexer_prefill_bundles
    )
    freed_latent_bundles = latent_bundles[keep_latent_bundles:]
    pinned_latent_bundles = set(latent_bundles[:keep_latent_bundles])
    allocator.free(DSASharedBlockOwner.LATENT, freed_latent_bundles)
    reused_by_indexer = allocator.allocate(
        DSASharedBlockOwner.INDEXER, min(4, len(freed_latent_bundles))
    )

    latent_block_table = [
        block_id
        for bundle_id in latent_bundles[:keep_latent_bundles]
        for block_id in layout.block_ids_for_bundle(
            DSASharedBlockOwner.LATENT, bundle_id
        )
    ]
    pinned_latent_blocks = {
        block_id
        for bundle_id in pinned_latent_bundles
        for block_id in layout.block_ids_for_bundle(
            DSASharedBlockOwner.LATENT, bundle_id
        )
    }
    indexer_block_table = [
        block_id
        for bundle_id in indexer_bundles + reused_by_indexer
        for block_id in layout.block_ids_for_bundle(
            DSASharedBlockOwner.INDEXER, bundle_id
        )
    ]

    print("[DSA-SHARED-SIM] config:")
    print(f"  model={config.model or '<none; fake config>'}")
    print(f"  index_topk={index_topk}")
    print(f"  prompt_tokens={sim_prompt_tokens}")
    print(f"  batch={sim_batch}")
    print(f"  prompt_blocks={prompt_blocks}")
    print(f"  scratch_blocks={scratch_blocks}")
    print(f"  keep_latent_blocks={keep_latent_blocks}")
    print(f"  latent_blocks_per_bundle={layout.latent_blocks_per_bundle}")
    print(f"  indexer_blocks_per_bundle={layout.indexer_blocks_per_bundle}")
    print(f"  latent_prefill_bundles={latent_bundles}")
    print(f"  indexer_prefill_bundles={indexer_bundles}")
    print(f"  freed_latent_bundles={freed_latent_bundles}")
    print(f"  reused_by_indexer={reused_by_indexer}")
    print(f"  pinned_latent_bundles={tuple(sorted(pinned_latent_bundles))}")
    print(f"  latent_block_table_prefix={latent_block_table[:12]}")
    print(f"  indexer_block_table_prefix={indexer_block_table[:18]}")

    if not freed_latent_bundles:
        raise RuntimeError("simulation did not free any latent bundle")
    if reused_by_indexer != freed_latent_bundles[: len(reused_by_indexer)]:
        raise RuntimeError("indexer did not reuse the freed latent bundles first")
    if not set(latent_block_table).issubset(pinned_latent_blocks):
        raise RuntimeError("latent block table contains an unpinned/freed latent block")

    try:
        from vllm_ascend.worker.dsa_shared_pool import reshape_dsa_shared_pool_raw

        raw = torch.empty(
            layout.slot_count * layout.bundle_page_size_bytes,
            dtype=torch.int8,
        )
        k_nope, k_pe = reshape_dsa_shared_pool_raw(
            raw,
            torch.float16,
            block_size,
            1,
            512,
            64,
            indexer_dim,
            is_indexer=False,
        )
        (indexer,) = reshape_dsa_shared_pool_raw(
            raw,
            torch.float16,
            block_size,
            1,
            512,
            64,
            indexer_dim,
            is_indexer=True,
        )
        if k_nope.untyped_storage().data_ptr() != raw.untyped_storage().data_ptr():
            raise RuntimeError("k_nope does not alias the raw shared slab")
        if k_pe.untyped_storage().data_ptr() != raw.untyped_storage().data_ptr():
            raise RuntimeError("k_pe does not alias the raw shared slab")
        if indexer.untyped_storage().data_ptr() != raw.untyped_storage().data_ptr():
            raise RuntimeError("indexer does not alias the raw shared slab")
        print(f"  k_nope_shape={tuple(k_nope.shape)}")
        print(f"  k_pe_shape={tuple(k_pe.shape)}")
        print(f"  indexer_shape={tuple(indexer.shape)}")
    except ImportError as exc:
        print(f"[DSA-SHARED-SIM] skip vllm-ascend view check: {exc}")

    run_cpu_lmcache_gather_simulation(
        batch_size=sim_batch,
        prompt_tokens=sim_prompt_tokens,
        index_topk=index_topk,
        block_size=block_size,
    )

    print("[DSA-SHARED-SIM] PASS")


class _TrackingInMemoryBackend:
    def __init__(self, device: str = "cpu") -> None:
        from vllm_ascend.distributed.kv_transfer.sparse_offload.offload_backend import (
            InMemoryLatentOffloadBackend,
        )

        self.impl = InMemoryLatentOffloadBackend(device=device)
        self.saved: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}
        self.load_calls: list[tuple[str, torch.Tensor, list[int], list[str]]] = []
        self._req_ids: list[str] = []

    def register_load_buffer(self, load_buffer: torch.Tensor) -> None:
        self.impl.register_load_buffer(load_buffer)

    def save_layer(
        self,
        layer_name: str,
        req_id: str,
        token_positions: torch.Tensor,
        latent: torch.Tensor,
    ) -> None:
        self.saved[(req_id, layer_name)] = (
            token_positions.detach().cpu().clone(),
            latent.detach().cpu().clone(),
        )
        self.impl.save_layer(layer_name, req_id, token_positions, latent)

    def wait_for_layer_load(
        self,
        layer_name: str,
        selected_tokens: torch.Tensor,
        token_start_index: list[int],
    ) -> None:
        self.load_calls.append(
            (
                layer_name,
                selected_tokens.detach().cpu().clone(),
                list(token_start_index),
                list(self._req_ids),
            )
        )
        self.impl.wait_for_layer_load(layer_name, selected_tokens, token_start_index)

    def set_load_req_ids(self, req_ids: list[str]) -> None:
        self._req_ids = list(req_ids)
        self.impl.set_load_req_ids(req_ids)

    def free_request(self, req_id: str) -> None:
        self.impl.free_request(req_id)


def _make_latent_rows(
    req_idx: int,
    positions: torch.Tensor,
    width: int,
    *,
    salt: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    pos = positions.to(torch.float32).unsqueeze(1)
    cols = torch.arange(width, dtype=torch.float32).unsqueeze(0)
    # Keep values small enough to be represented exactly after fp16 conversion.
    values = (req_idx + 1) * 17 + (pos % 97) + (cols % 13) + salt
    return values.to(dtype)


def run_cpu_lmcache_gather_simulation(
    *,
    batch_size: int,
    prompt_tokens: int,
    index_topk: int,
    block_size: int,
) -> None:
    from vllm_ascend.distributed.kv_transfer.sparse_offload.decode_latent_pool import (
        GrowingDecodeLatentPool,
    )
    from vllm_ascend.distributed.kv_transfer.sparse_offload.offload_manager import (
        SparseLatentOffloadManager,
        SparseOffloadConfig,
        build_gather_plan,
        resolve_scratch_gather,
    )
    from vllm_ascend.distributed.kv_transfer.sparse_offload.paged_latent_pool import (
        PagedLatentPool,
    )

    dtype = torch.float16
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    layer_name = "model.layers.0.self_attn.attn"
    req_ids = [f"req-{idx}" for idx in range(batch_size)]
    prompt_lens = torch.tensor(
        [prompt_tokens + idx * block_size for idx in range(batch_size)],
        dtype=torch.long,
    )
    max_prompt = int(prompt_lens.max().item())
    decode_count = min(64, max(1, index_topk // 16))
    prefill_count = index_topk - decode_count
    topk_rows = []
    backend = _TrackingInMemoryBackend(device="cpu")
    config = SparseOffloadConfig(
        num_layers=1,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_size=block_size,
        max_num_seqs=batch_size,
        topk_tokens=index_topk,
        dtype=dtype,
        device=torch.device("cpu"),
        pool_num_blocks=batch_size * (_ceil_div(max_prompt + decode_count, block_size) + 2),
    )
    scratch_knope = torch.zeros(
        (
            config.scratch_num_blocks,
            block_size,
            1,
            kv_lora_rank,
        ),
        dtype=dtype,
    )
    scratch_kpe = torch.zeros(
        (
            config.scratch_num_blocks,
            block_size,
            1,
            qk_rope_head_dim,
        ),
        dtype=dtype,
    )
    load_buffer = torch.zeros(
        (batch_size * index_topk, config.latent_dim),
        dtype=dtype,
    )
    manager = SparseLatentOffloadManager(
        config=config,
        backend=backend,
        layer_names=[layer_name],
        scratch_knope=scratch_knope,
        scratch_kpe=scratch_kpe,
        load_buffer=load_buffer,
        decode_pool=GrowingDecodeLatentPool(
            num_layers=1,
            block_size=block_size,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            dtype=dtype,
            device="cpu",
        ),
        paged_latent_pool=PagedLatentPool(
            num_layers=1,
            num_blocks=config.pool_num_blocks,
            block_size=block_size,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            dtype=dtype,
            device="cpu",
        ),
    )

    expected_by_req: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for req_idx, req_id in enumerate(req_ids):
        plen = int(prompt_lens[req_idx].item())
        prompt_pos = torch.arange(plen, dtype=torch.long)
        k_nope = _make_latent_rows(
            req_idx, prompt_pos, kv_lora_rank, salt=0, dtype=dtype
        )
        k_pe = _make_latent_rows(
            req_idx, prompt_pos, qk_rope_head_dim, salt=3, dtype=dtype
        )
        manager.store_prefill_layer(req_id, layer_name, prompt_pos, k_nope, k_pe)

        decode_pos = torch.arange(plen, plen + decode_count, dtype=torch.long)
        decode_k_nope = _make_latent_rows(
            req_idx, decode_pos, kv_lora_rank, salt=7, dtype=dtype
        )
        decode_k_pe = _make_latent_rows(
            req_idx, decode_pos, qk_rope_head_dim, salt=11, dtype=dtype
        )
        manager._paged_latent_pool.store(
            req_id,
            0,
            decode_pos,
            decode_k_nope,
            decode_k_pe,
        )

        step = max(1, plen // prefill_count)
        prefill_topk = (torch.arange(prefill_count, dtype=torch.long) * step) % plen
        topk_row = torch.cat([prefill_topk, decode_pos])
        topk_rows.append(topk_row)
        expected_by_req[req_id] = (
            torch.cat([k_nope.index_select(0, prefill_topk), decode_k_nope]),
            torch.cat([k_pe.index_select(0, prefill_topk), decode_k_pe]),
            topk_row,
        )

    topk_indices = torch.stack(topk_rows)
    plan = build_gather_plan(
        topk_indices,
        prompt_lens,
        block_size,
        config.scratch_blocks_per_req,
    )
    is_pref = plan.prefill_positions != -1
    expected_flat_selected = plan.prefill_positions[is_pref].cpu()
    expected_starts = (
        torch.cat(
            [
                torch.zeros(1, dtype=torch.long),
                is_pref.sum(dim=1).cumsum(0)[:-1].cpu(),
            ]
        )
        .to(torch.long)
        .tolist()
    )
    (
        scratch_knope_out,
        scratch_kpe_out,
        sparse_indices,
        scratch_block_table,
        seq_lens_kv,
    ) = manager.gather_decode_layer(layer_name, req_ids, plan)

    if len(backend.load_calls) != 1:
        raise RuntimeError(f"expected one LMCache load call, got {len(backend.load_calls)}")
    load_layer, selected_tokens, token_start_index, load_req_ids = backend.load_calls[0]
    if load_layer != layer_name:
        raise RuntimeError("LMCache load used the wrong layer")
    if load_req_ids != req_ids:
        raise RuntimeError("LMCache load used the wrong request ids")
    if token_start_index != expected_starts:
        raise RuntimeError(
            f"LMCache token_start_index mismatch: {token_start_index} vs {expected_starts}"
        )
    if not torch.equal(selected_tokens, expected_flat_selected):
        raise RuntimeError("LMCache selected_tokens did not match prefill topk positions")

    expected_prefill_latent = []
    starts = token_start_index + [int(expected_flat_selected.numel())]
    for req_idx, req_id in enumerate(req_ids):
        positions, latent = backend.saved[(req_id, layer_name)]
        if not torch.equal(positions, torch.arange(int(prompt_lens[req_idx]))):
            raise RuntimeError(f"saved LMCache positions are wrong for {req_id}")
        lo, hi = starts[req_idx], starts[req_idx + 1]
        expected_prefill_latent.append(
            latent.index_select(0, selected_tokens[lo:hi].to(torch.long))
        )
    expected_prefill_latent_tensor = torch.cat(expected_prefill_latent)
    loaded = manager._load_buffer[: expected_prefill_latent_tensor.shape[0]].cpu()
    if not torch.equal(loaded, expected_prefill_latent_tensor):
        raise RuntimeError("LMCache-loaded data in load_buffer does not match saved latent")

    # This mirrors the KV rows npu_sparse_flash_attention will read from
    # (scratch_knope/scratch_kpe, sparse_indices, scratch_block_table).
    resolved = resolve_scratch_gather(
        scratch_knope_out,
        scratch_kpe_out,
        sparse_indices,
        scratch_block_table,
        block_size,
        seq_lens_kv,
    )
    scratch_blocks_per_req = config.scratch_blocks_per_req
    for req_idx, req_id in enumerate(req_ids):
        expected_knope, expected_kpe, expected_topk = expected_by_req[req_id]
        actual_knope, actual_kpe = resolved[req_idx]
        if not torch.equal(actual_knope.cpu(), expected_knope):
            raise RuntimeError(f"kernel-resolved k_nope data mismatch for {req_id}")
        if not torch.equal(actual_kpe.cpu(), expected_kpe):
            raise RuntimeError(f"kernel-resolved k_pe data mismatch for {req_id}")
        if not torch.equal(topk_indices[req_idx], expected_topk):
            raise RuntimeError(f"topk row mutated for {req_id}")

        valid_local = sparse_indices[req_idx, : int(seq_lens_kv[req_idx])].to(torch.long)
        phys = (
            scratch_block_table[req_idx][valid_local // block_size].to(torch.long)
            * block_size
            + valid_local % block_size
        )
        lo = req_idx * scratch_blocks_per_req * block_size
        hi = (req_idx + 1) * scratch_blocks_per_req * block_size
        if int(phys.min()) < lo or int(phys.max()) >= hi:
            raise RuntimeError(
                f"sparse attention block table for {req_id} points outside its pinned latent scratch region"
            )

    print("[DSA-SHARED-SIM] lmcache/control-plane data check:")
    print(f"  topk_shape={tuple(topk_indices.shape)}")
    print(f"  prefill_selected={int(expected_flat_selected.numel())}")
    print(f"  decode_selected={batch_size * decode_count}")
    print(f"  scratch_block_table_shape={tuple(scratch_block_table.shape)}")
    print("  lmcache_selected_tokens_match=True")
    print("  lmcache_loaded_data_match=True")
    print("  sparse_attention_block_table_points_to_pinned_latent=True")
    print("  sparse_attention_resolved_data_match=True")


def main() -> int:
    maybe_add_vllm_ascend_repo()
    config = parse_args()
    apply_env(config)
    try:
        if config.simulate_cpu:
            run_cpu_simulation(config)
        else:
            run_generation(config)
    except Exception as exc:
        print(f"[DSA-SHARED-E2E] FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

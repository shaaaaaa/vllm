# SPDX-License-Identifier: Apache-2.0
"""Run the actual admission/output methods without model/device bootstrap."""
import ast
from collections import defaultdict
import enum
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock
import time

import msgspec
import pytest

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'vllm/v1/core/sched/scheduler.py'
ASCEND = ROOT.parent / 'vllm-ascend/vllm_ascend/core'


def _class(path, name):
    return next(n for n in ast.parse(path.read_text(encoding='utf-8')).body
                if isinstance(n, ast.ClassDef) and n.name == name)


def _runtime():
    source = ast.parse((ROOT / 'vllm/v1/engine/__init__.py').read_text(encoding='utf-8'))
    schema = ast.parse('from __future__ import annotations')
    schema.body.extend(n for n in source.body if isinstance(n, ast.ClassDef)
                       and n.name in ('FinishReason', 'EngineCoreOutput', 'EngineCoreOutputs'))
    scope = dict(msgspec=msgspec, enum=enum, time=time,
                 FINISH_REASON_STRINGS=('stop', 'length', 'abort', 'error', 'repetition'))
    exec(compile(schema, '<actual engine output schemas>', 'exec'), scope)

    class Status(enum.IntEnum):
        WAITING = 0
        RUNNING = 1
        WAITING_FOR_STREAMING_REQ = 2
        WAITING_FOR_REMOTE_KVS = 3
        FINISHED_ERROR = 5
        @staticmethod
        def is_finished(value): return value >= 5
    scope.update(logger=Mock(), RequestStatus=Status,
                 remove_all=lambda values, removed: [v for v in values if v not in removed])
    names = {'_reject_impossible_initial_dsa_request', '_drain_dsa_admission_errors',
             '_update_after_schedule', 'has_finished_requests', 'finish_requests'}
    cls = _class(BASE, 'Scheduler')
    cls.bases = []
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    module = ast.parse('from __future__ import annotations')
    module.body.append(cls)
    exec(compile(module, str(BASE), 'exec'), scope)
    return scope


class Queue(list):
    def remove_requests(self, requests):
        self[:] = [r for r in self if r not in requests]
    def pop_request(self): return self.pop(0)
    def prepend_request(self, request): self.insert(0, request)


def _scheduler(capacity=100, length=1000):
    scope = _runtime()
    obj = scope['Scheduler']()
    obj._dsa_dense_request_capacity = capacity
    request = NS(request_id='long', num_tokens=length, num_prompt_tokens=length,
        num_preemptions=0, num_computed_tokens=0, dsa_compact_allocated=False,
        status=scope['RequestStatus'].WAITING, client_index=7,
        bootstrap_sample_pending=False, bootstrap_final_hidden=None)
    request.is_finished = lambda: scope['RequestStatus'].is_finished(request.status)
    request.get_finished_reason = lambda: scope['FinishReason'].ERROR
    obj.requests = {'long': request}
    obj.waiting, obj.skipped_waiting, obj.running = Queue([request]), Queue(), []
    obj.finished_req_ids = set()
    obj.finished_recving_kv_req_ids, obj.failed_recving_kv_req_ids = set(), set()
    releases = []
    def free(req, delay_free_blocks=False):
        releases.append(req.request_id)
        obj.requests.pop(req.request_id)
        obj.finished_req_ids.add(req.request_id)
    obj._free_request = free
    return obj, request, releases, scope


@pytest.mark.parametrize('path,name', [(BASE, 'Scheduler'),
    (ASCEND / 'recompute_scheduler.py', 'RecomputeScheduler'),
    (ASCEND / 'scheduler_dynamic_batch.py', 'SchedulerDynamicBatch')])
@pytest.mark.parametrize('lookup', [None, 0, 50, 999])
def test_actual_waiting_lookup_guards_miss_but_defers_unknown_and_admits_full_hit(path, name, lookup):
    obj, request, releases, _ = _scheduler()
    obj.connector = NS(get_num_new_matched_tokens=lambda *args: (lookup, lookup == 999),
                       supports_dsa_compact_external_load=True)
    obj.kv_cache_manager = NS(get_computed_blocks=lambda req: (None, 0))
    cls = _class(path, name)
    schedule = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'schedule')
    body = next(n.body for n in ast.walk(schedule) if isinstance(n, (ast.For, ast.While))
                and any(isinstance(s, ast.If) and isinstance(s.test, ast.Call)
                        and isinstance(s.test.func, ast.Attribute)
                        and s.test.func.attr == '_reject_impossible_initial_dsa_request' for s in n.body))
    first = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'num_external_computed_tokens' for t in n.targets))
    last = next(i for i, n in enumerate(body) if isinstance(n, ast.If)
                and isinstance(n.test, ast.Call) and isinstance(n.test.func, ast.Attribute)
                and n.test.func.attr == '_reject_impossible_initial_dsa_request')
    loop = ast.parse('for request in requests: pass').body[0]
    loop.body = body[first:last + 1] + ast.parse('admitted.append(request.request_id)').body
    scope = dict(self=obj, requests=[request], admitted=[], request_queue=obj.waiting,
        step_skipped_waiting=Queue(), skipped_waiting_requests=Queue(),
        SERVING_PERF_ENABLED=False, bootstrap_only_step=False, bootstrap_sample_req_ids=[],
        scheduled_req_id='', num_scheduled_tokens={}, request_id='long', logger=Mock())
    exec(compile(ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[])), str(path), 'exec'), scope)
    if lookup is None:
        assert not releases and not scope['admitted'] and not getattr(obj, '_dsa_admission_errors', None)
    elif lookup == 999:
        assert not releases and scope['admitted'] == ['long']
    else:
        assert releases == ['long'] and not scope['admitted'] and not obj.requests
        assert obj.has_finished_requests()


@pytest.mark.parametrize('change', [dict(num_preemptions=1), dict(num_computed_tokens=1),
    dict(dsa_compact_allocated=True), dict(num_tokens=1001)])
def test_initial_only_guard_preserves_existing_recovery_paths(change):
    obj, request, releases, _ = _scheduler()
    request.__dict__.update(change)
    assert not obj._reject_impossible_initial_dsa_request(request, compact_external_load=False)
    assert not releases


@pytest.mark.parametrize('capacity,length', [(None, 1000), (100, 100)])
def test_unmarked_or_dense_fit_keeps_temporary_pressure_path(capacity, length):
    obj, request, releases, _ = _scheduler(capacity, length)
    assert not obj._reject_impossible_initial_dsa_request(request, compact_external_load=False)
    assert not releases


@pytest.mark.parametrize('output_path,output_class', [(BASE, 'Scheduler'),
    (ASCEND / 'recompute_scheduler.py', 'RecomputeScheduler')])
def test_rejection_only_step_emits_terminal_error_on_exact_async_output(output_path, output_class):
    obj, request, releases, schema = _scheduler()
    earlier = NS(num_scheduled_tokens={})
    obj._update_after_schedule(earlier)
    assert obj._reject_impossible_initial_dsa_request(request, compact_external_load=False)
    obj.finished_req_ids.clear()  # The pending terminal payload alone keeps the scheduler active.
    assert obj.has_finished_requests()
    rejected_step = NS(num_scheduled_tokens={})
    obj._update_after_schedule(rejected_step)
    outputs = defaultdict(list)
    obj._drain_dsa_admission_errors(earlier, outputs)
    assert not outputs  # An older in-flight batch must not steal this error.
    obj._drain_dsa_admission_errors(rejected_step, outputs)
    result = outputs[7][0]
    assert result.request_id == 'long' and result.finished and result.new_token_ids == []
    assert str(result.finish_reason) == 'error' and releases == ['long']
    assert msgspec.msgpack.encode(result)  # Real wire schema, not a finished-ID-only notification.
    obj._drain_dsa_admission_errors(rejected_step, outputs)
    assert len(outputs[7]) == 1
    method = next(n for n in _class(output_path, output_class).body
                  if isinstance(n, ast.FunctionDef) and n.name == 'update_from_output')
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == '_drain_dsa_admission_errors' for n in ast.walk(method))
    output_assignment = next(n for n in method.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'engine_core_outputs' for t in n.targets))
    scope = dict(outputs=outputs, EngineCoreOutputs=schema['EngineCoreOutputs'])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[output_assignment], type_ignores=[])), str(output_path), 'exec'), scope)
    assert scope['engine_core_outputs'][7].outputs == [result]


def test_bootstrap_full_hit_bypass_uses_real_compact_allocation():
    import importlib.util
    path = Path(__file__).with_name('test_compact_boundary_admission.py')
    spec = importlib.util.spec_from_file_location('compact_allocation_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kwargs, _ = module.schedule_allocation(1000, dsa_compact_external_load=False,
        bootstrap_full_hit=True, load_kv_async=False, num_external_computed_tokens=1000)
    assert kwargs['dsa_compact_external_load'] is True


def test_skipped_waiting_head_is_removed_by_existing_finish_api():
    obj, request, releases, _ = _scheduler()
    obj.skipped_waiting.append(obj.waiting.pop())
    assert obj._reject_impossible_initial_dsa_request(request, compact_external_load=False)
    assert not obj.skipped_waiting and releases == ['long']


@pytest.mark.parametrize('hit', [0, 500, 999, 1000])
def test_real_lmcache_initial_compact_predicate_requires_full_prefix(hit):
    path = ROOT.parent / 'LMCache-NPU/lmcache/integration/vllm/vllm_v1_adapter.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'dsa_cold_compact_load' for t in node.targets)]
    expression = next(node for node in assignments if isinstance(node.value, ast.BoolOp))
    resident = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'resident_cold_load' for t in node.targets))
    scope = dict(self=NS(supports_dsa_cold_compact_load=lambda: True,
                        config=NS(pd_role='receiver'),
                        _block_size=128, _dsa_scratch_capacity=128, _dsa_kv_policy_threshold=128),
        num_computed_tokens=0, full_request_hit=hit == 1000, full_resumed_query_hit=False,
        need_to_allocate=hit - int(hit == 1000), num_external_hit_tokens=hit,
        compact_remap_frontier=hit - int(hit == 1000), resumed=False, query_scope='all_tokens',
        request=NS(num_preemptions=0, num_tokens=1000), request_prompt_tokens=1000,
        cdiv=lambda value, divisor: (value + divisor - 1) // divisor)
    exec(compile(ast.Module(body=[resident, expression], type_ignores=[]), str(path), 'exec'), scope)
    assert scope['dsa_cold_compact_load'] is (hit == 1000)


@pytest.mark.parametrize('queued', [False, True])
def test_actual_enginecore_empty_step_delivers_rejection_with_async_older_batch(queued):
    from collections import deque
    from concurrent.futures import Future
    from contextlib import nullcontext
    from typing import cast
    obj, request, _, schema = _scheduler()
    path = ROOT / 'vllm/v1/engine/core.py'
    methods = [n for n in _class(path, 'EngineCore').body if isinstance(n, ast.FunctionDef)
               and n.name in ('step', 'step_with_batch_queue')]
    module = ast.parse('from __future__ import annotations')
    module.body.extend(methods)
    scope = dict(SERVING_PERF_ENABLED=False, cast=cast, Future=Future, ModelRunnerOutput=object)
    exec(compile(module, str(path), 'exec'), scope)
    older = NS(num_scheduled_tokens={}, total_num_scheduled_tokens=0)
    obj._update_after_schedule(older)
    def schedule():
        assert obj._reject_impossible_initial_dsa_request(request, compact_external_load=False)
        result = NS(num_scheduled_tokens={}, total_num_scheduled_tokens=0)
        obj._update_after_schedule(result)
        return result
    def update(output, model_result):
        outputs = defaultdict(list)
        obj._drain_dsa_admission_errors(output, outputs)
        return {client: schema['EngineCoreOutputs'](outputs=values) for client, values in outputs.items()}
    obj.schedule = schedule
    obj.has_requests = lambda: bool(obj.requests) or obj.has_finished_requests()
    obj.get_grammar_bitmask = lambda output: None
    obj.update_from_output = update
    def result_future(*args, **kwargs):
        future = Future()
        future.set_result(object())
        return future
    old_future = result_future()
    engine = NS(scheduler=obj, model_executor=NS(execute_model=result_future),
        log_error_detail=lambda output: nullcontext(), log_iteration_details=lambda output: nullcontext(),
        _process_aborts_queue=lambda: None, is_ec_consumer=False, is_pooling_model=False,
        batch_queue=deque([(old_future, older, old_future)]) if queued else None,
        batch_queue_size=2)
    if queued:
        outputs, executed = scope['step_with_batch_queue'](engine)
        assert outputs == {} and executed is False
        outputs, executed = scope['step_with_batch_queue'](engine)
    else:
        outputs, executed = scope['step'](engine)
    assert executed is False
    assert outputs[7].outputs[0].request_id == 'long'
    assert str(outputs[7].outputs[0].finish_reason) == 'error'

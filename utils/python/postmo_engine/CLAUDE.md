# CLAUDE.md

## Current Status

This workspace is not yet a real-device validated end-to-end Qwen3.5 Engine.
`perf/`, Backend, core contracts, Qwen35 Process/Sampler/Module/Engine offline
implementations now have unit-test coverage, but there is no real-device or
Qwen3.5 output alignment validation. Tensor Transfer, contract tests, and smoke
tests are not complete. See `docs/IMPLEMENTATION_STATUS.md` before treating
design text as implemented behavior.

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workspace

This directory is a **standalone working copy** of `postmo_engine`. It is not the original tree under `imodelzoo`. Do not modify `/hmdd/imodelzoo` (or any other source checkout) from this workspace.

Imports assume the package name `postmo_engine`. Run tools with the **parent** of this directory on `PYTHONPATH`:

```bash
export PYTHONPATH=/hmdd/lyz_codes/codes_tmp
```

There is no `pyproject.toml`, `setup.py`, or `requirements.txt` in this copy. Runtime/tests that are present need Python 3.10+ style typing, `pytest`, `numpy`, and `loguru`. YAML export (`perf.dumper`) additionally needs `PyYAML`. `TcimBackend` imports `tcim_lite` only when constructed without an injected `runtime`.

## Commands

```bash
# All unit tests (from this directory)
PYTHONPATH=/hmdd/lyz_codes/codes_tmp python3 -m pytest tests/unit -q

# One file
PYTHONPATH=/hmdd/lyz_codes/codes_tmp python3 -m pytest tests/unit/test_backend_base.py -q

# One test
PYTHONPATH=/hmdd/lyz_codes/codes_tmp python3 -m pytest \
  tests/unit/test_backend_base.py::test_base_class_owns_all_four_perf_boundaries -q
```

There is no project lint/format/build command. Unit tests are offline: `TcimBackend` tests inject a fake TCIM runtime; they do not need a device.

The current unit suite covers the implemented Perf and Backend layers. It does not prove an end-to-end model path or real TCIM behavior.

## What exists vs what is planned

Implemented today:

- `backend/` — `PostMoBackend` contract + `TcimBackend`
- `perf/` — Houmo-style free-path tracker, report formatter, and PostMo YAML dumper
- `tests/unit/` — Perf and Backend offline unit tests

Not implemented yet (described in `FIRST_VERSION_PLAN.md`): `core/`, `sampling/`, `process/`, `module/`, `engine/`, tensor-transfer modules (`backend/tensor.py`, `backend/tcim_tensor.py`), and `tests/contract` / `tests/smoke`. Do not create empty VLM/ASR/TTS/LALM layers or a product `FakeBackend`.

Authoritative docs:

- `DESIGN.md` — target layering and capability matrix
- `DESIGN_REVISION.md` — semantic corrections before implementation
- `FIRST_VERSION_PLAN.md` — phased v1 plan and suggested tree
- `PERF_PLAN.md` — perf scope and timing boundaries
- `docs/postmo_engine设计doc.md` + `docs/*.mmd` — earlier product-level design (mentions HMM/HMONNX/ONNX). **v1 does not implement those backends.**
- `docs/QWEN35_TEXT_PLAN.md` — current Qwen3.5 Text-only implementation plan and file naming

## Architecture

Target call chain (not implemented yet beyond the Backend endpoint):

```text
PostMoEngine  ->  Processor  ->  PostMoModule  ->  PostMoBackend  ->  TCIM Lite
```

Layer rules that later code must keep:

- **Engine**: e2e orchestration, decode outer loop, stop reasons, streaming. Must not see Prefill chunks, TCIM models, or tensors.
- **Processor**: preprocess / construct model inputs / sample / parse outputs. Must not own real cache length.
- **Module**: chunked prefill inner loop, single-step decode, device state. **Only Module may write `context_length`.**
- **Backend**: load / bind / run+sync / read. Must not know Token, EOS, sampling, Prefill/Decode, or `context_length`.

v1 product scope: Qwen3.5 text-only, TCIM Lite, single device, batch 1, one request, session reset per request, greedy sampling, sync streaming text. Unsupported features (vision, MTP, prefix cache, continuous batching, PagedAttention, PP/TP/DP/EP, multi-LoRA, multi-turn, random sampling) must surface in the capability matrix with a layer-specific reason — never silently drop or downgrade.

Effective capability is the intersection of PostMo model, Backend, Engine, and Deployment capabilities. Engine access states are `AVAILABLE` / `PLANNED` / `BLOCKED`; `PLANNED` is not callable.

## Backend

`PostMoBackend` is a template method. Public methods own validation, opaque `ModelHandle` checks, and the four timed operations. Subclasses implement only the `_prepare_*` / `_load_*` / `_set_*` / `_run_and_sync` / `_copy_*` / `_normalize_*` hooks. Runtime model handles retain `model_category` and `model_role` for scope paths.

Timed vs untimed (must stay this way):

| Public API | Measured scope | Outside the timer |
|---|---|---|
| `load_model` | `_load_prepared_model` | path/option/dummy-input prep |
| `set_host_input` | `_set_prepared_host_input` | host validate/reshape/contiguous |
| `run` | `_run_and_sync` (TCIM `run` **and** `sync`) | — |
| `get_output` | `_copy_output_to_host` | host numpy/copy/normalize |

`ModelHandle` is bound to the creating backend via a private token; another backend cannot unwrap it. `load_model` accepts explicit `model_category` and `model_role` strings rather than a typed registry identity.

`TcimBackend` is the only official backend. It lazy-imports `tcim_lite.runtime` unless a `runtime=` double is injected (tests do this). Host outputs are copied to a new NumPy array so later `run()` cannot overwrite caller data.

## Perf

Perf currently has one free-path collector and a report model:

- `PerfTracker` — Houmo-compatible free-form scope tracker; disabled trackers do not validate or record paths
- `PerfReport` / `ScopeStats` — aggregate arbitrary scope paths and derived metrics

Backend runtime scopes are `llm.<role>.load_model`, `set_input`, `run`, and `get_output`. Device bind/D2D/ROI/clone/zero and CPU pre/postprocess are not auto-timed. Business-level timings use arbitrary valid paths such as `engine.request.e2e`.

Engine-level timings (`engine.e2e`, TTFT, …) can use `PerfTracker.scope|start|end` and are exported by the Dumper as custom timings; no Engine currently produces them. Perf reports retain arbitrary scopes and may derive metrics such as TTFT/TPOT/tokens-per-sec according to the Houmo rules.

`dump_yaml` / `dumps_yaml` write schema_version 1, `time_unit: ms`, via PyYAML; file write is atomic (`os.replace` / `os.link`).

`PerfTracker` is an independent copy of the Houmo tracker API. `create(enabled=False)` is a no-op, while enabled trackers support `scope`, `start`, `end`, `set_metrics`, `summary`, `reset`, and arbitrary valid custom paths. PostMo-specific `dumps_yaml` / `dump_yaml` export only Runtime Operation records, custom timings, and whitelisted `ttft_ms` / `e2e_ms` derived values.

## License

Apache-2.0 (HOUMO AI). Keep the existing SPDX header on new Python files.

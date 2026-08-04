# Copyright (c) 2026 HOUMO AI
#
# File: test_artifact_cache_store.py
# Description:
#  Unit tests for artifact manifests, fingerprints, file scanning, and atomic
#    cache publication and recovery.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests extracted from the former model-flow contract suite: test_artifact_cache_store.py."""

import pytest
from pathlib import (
    Path,
)
from tests.models_tests.model_workflow.artifact_cache_store import (
    ArtifactCache,
    ArtifactManifest,
    ArtifactRequirement,
    ArtifactType,
    AtomicArtifactWriter,
    CacheStatus,
    calculate_config_fingerprint,
)
from tests.models_tests.model_workflow.artifact_file_scanner import (
    build_required_file_roles,
    find_nonempty_artifact_files,
    find_nonempty_hmm_files,
    prune_compiler_intermediates,
)
from tests.models_tests.model_workflow.flow_contracts import (
    ArtifactValidationError,
    ModelFamily,
    ModelFlow,
)

pytestmark = pytest.mark.unit


def test_shared_artifact_file_enumeration_and_roles(tmp_path: Path) -> None:
    (tmp_path / "model.bin").write_bytes(b"model")
    (tmp_path / "empty.bin").write_bytes(b"")
    (tmp_path / "lock.lock").write_text("lock\n", encoding="utf-8")
    (tmp_path / "artifact_manifest.json").write_text("{}\n", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "decode.hmm").write_bytes(b"decode")
    (nested / "prefill.hmms").write_bytes(b"prefill")

    artifact_files = find_nonempty_artifact_files(tmp_path)
    assert {path.relative_to(tmp_path) for path in artifact_files} == {
        Path("model.bin"),
        Path("nested/decode.hmm"),
        Path("nested/prefill.hmms"),
    }
    hmm_files = find_nonempty_hmm_files(tmp_path)
    assert build_required_file_roles(tmp_path, hmm_files, prefix="hmm") == {
        "hmm_0": "nested/decode.hmm",
        "hmm_1": "nested/prefill.hmms",
    }


def test_artifact_manifest_matches_downloaded_hmm_for_demo(tmp_path: Path) -> None:
    (tmp_path / "prefill.hmm").write_bytes(b"prefill")
    (tmp_path / "decode.hmm").write_bytes(b"decode")
    fingerprint = calculate_config_fingerprint(
        {"model": "qwen3", "backend": "xh2", "case": "ctx32k"}
    )
    manifest = ArtifactManifest(
        schema_version=1,
        fingerprint_version=1,
        artifact_type=ArtifactType.COMPILED_MODEL.value,
        model_name="qwen3",
        model_family=ModelFamily.LLM.value,
        backend="xh2",
        case_id="ctx32k",
        producer_flow=ModelFlow.GET_MODEL.value,
        source_type="download",
        config_fingerprint=fingerprint,
        required_files={
            "prefill_hmm": "prefill.hmm",
            "decode_hmm": "decode.hmm",
        },
    )
    cache = ArtifactCache()
    cache.write_manifest(tmp_path, manifest)
    inspection = cache.inspect(
        tmp_path,
        ArtifactRequirement(
            artifact_type=ArtifactType.COMPILED_MODEL,
            model_name="qwen3",
            backend="xh2",
            case_id="ctx32k",
            required_roles=("prefill_hmm", "decode_hmm"),
        ),
        expected_fingerprint=fingerprint,
    )
    assert inspection.status == CacheStatus.VALID
    assert inspection.manifest is not None
    assert inspection.manifest.producer_flow == ModelFlow.GET_MODEL.value


def test_artifact_fingerprint_mismatch_is_stale(tmp_path: Path) -> None:
    (tmp_path / "model.hmm").write_bytes(b"hmm")
    cache = ArtifactCache()
    cache.write_manifest(
        tmp_path,
        ArtifactManifest(
            schema_version=1,
            fingerprint_version=1,
            artifact_type=ArtifactType.COMPILED_MODEL.value,
            model_name="resnet50",
            model_family=ModelFamily.CV.value,
            backend="xh2",
            case_id="default",
            producer_flow=ModelFlow.COMPILE.value,
            source_type="local_compile",
            config_fingerprint="sha256:old",
            required_files={"hmm": "model.hmm"},
        ),
    )
    inspection = cache.inspect(
        tmp_path,
        ArtifactRequirement(
            ArtifactType.COMPILED_MODEL,
            "resnet50",
            "xh2",
            "default",
            ("hmm",),
        ),
        expected_fingerprint="sha256:new",
    )
    assert inspection.status == CacheStatus.STALE


def test_manifest_required_files_are_always_validated(tmp_path: Path) -> None:
    cache = ArtifactCache()
    cache.write_manifest(
        tmp_path,
        ArtifactManifest(
            schema_version=1,
            fingerprint_version=1,
            artifact_type=ArtifactType.COMPILED_MODEL.value,
            model_name="demo",
            model_family=ModelFamily.CV.value,
            backend="xh2",
            case_id="default",
            producer_flow=ModelFlow.COMPILE.value,
            source_type="local_compile",
            config_fingerprint="sha256:test",
            required_files={"hmm": "missing.hmm"},
        ),
    )
    inspection = cache.inspect(
        tmp_path,
        ArtifactRequirement(ArtifactType.COMPILED_MODEL, "demo", "xh2", "default"),
    )
    assert inspection.status == CacheStatus.CORRUPTED


def test_atomic_artifact_writer_preserves_old_result_until_commit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "results"
    destination = root / "hmm_xh2"
    destination.mkdir(parents=True)
    (destination / "old.hmm").write_bytes(b"old")

    aborted = AtomicArtifactWriter(destination, root=root, token="aborted")
    with aborted as staging:
        (staging / "new.hmm").write_bytes(b"new")
    assert (destination / "old.hmm").read_bytes() == b"old"
    assert not aborted.staging.exists()
    assert not aborted.staging_marker.exists()

    committed = AtomicArtifactWriter(destination, root=root, token="committed")
    with committed as staging:
        (staging / "new.hmm").write_bytes(b"new")
        committed.commit()
    assert not (destination / "old.hmm").exists()
    assert (destination / "new.hmm").read_bytes() == b"new"
    assert not committed.backup.exists()


def test_atomic_artifact_writer_rejects_destination_outside_cache(
    tmp_path: Path,
) -> None:
    writer = AtomicArtifactWriter(
        tmp_path / "outside", root=tmp_path / "results", token="unsafe"
    )
    with pytest.raises(ArtifactValidationError, match="inside its owned cache root"):
        writer.__enter__()


def test_atomic_artifact_writer_recovers_backup_across_run_tokens(
    tmp_path: Path,
) -> None:
    root = tmp_path / "results"
    destination = root / "hmm_xh2"
    destination.mkdir(parents=True)
    (destination / "old.hmm").write_bytes(b"old")

    interrupted = AtomicArtifactWriter(destination, root=root, token="old-run")
    interrupted.backup_marker.write_text("owned\n", encoding="utf-8")
    destination.replace(interrupted.backup)

    resumed = AtomicArtifactWriter(destination, root=root, token="new-run")
    with resumed as staging:
        assert (destination / "old.hmm").read_bytes() == b"old"
        assert not resumed.backup.exists()
        assert not resumed.backup_marker.exists()
        (staging / "new.hmm").write_bytes(b"new")
        resumed.commit()

    assert (destination / "new.hmm").read_bytes() == b"new"
    assert not (destination / "old.hmm").exists()


def test_prune_compiler_intermediates_keeps_only_diagnostic_sources(
    tmp_path: Path,
) -> None:
    tcim = tmp_path / "xh2" / "tcim" / "resnet50"
    tcim.mkdir(parents=True)
    kept = {
        "model.cpp": b"source",
        "graph.json": b"{}",
    }
    # ".cpp" appears inside these names without making them C++ sources, so a
    # substring match would wrongly keep them.
    dropped = {
        "model.cpp.o": b"object" * 16,
        "model.cpp.hdpl.lib.a": b"archive" * 16,
        "model.cpp.0.asm.hu": b"asm" * 16,
        "weights.bin": b"weights" * 16,
    }
    for name, payload in {**kept, **dropped}.items():
        (tcim / name).write_bytes(payload)
    empty_leaf = tcim / "scratch"
    empty_leaf.mkdir()
    (empty_leaf / "temp.bin").write_bytes(b"temp")
    outside = tmp_path / "xh2" / "model.hmm"
    outside.write_bytes(b"artifact")

    released = prune_compiler_intermediates(tmp_path)

    assert released == sum(len(payload) for payload in dropped.values()) + 4
    for name in kept:
        assert (tcim / name).exists()
    for name in dropped:
        assert not (tcim / name).exists()
    assert not empty_leaf.exists()
    assert outside.read_bytes() == b"artifact"


def test_prune_compiler_intermediates_ignores_missing_directories(
    tmp_path: Path,
) -> None:
    assert prune_compiler_intermediates(tmp_path / "absent") == 0
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "model.hmm").write_bytes(b"artifact")
    assert prune_compiler_intermediates(plain) == 0
    assert (plain / "model.hmm").exists()

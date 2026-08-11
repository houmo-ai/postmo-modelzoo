# Copyright (c) 2025 HOUMO AI
#
# File: cache_path_resolver.py
# Description:
#  Model Cache Placeholder and Case Path Resolution.
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

"""Resolve symbolic model/result cache paths and artifact case references."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .parameter_matrix import ParameterCase


__all__ = [
    "MODEL_CACHE_ROOT",
    "RESULT_CACHE_ROOT",
    "cache_case_reference",
    "cache_root_directory",
    "get_model_case_artifact_id",
    "replace_case_output_dir",
    "resolve_cached_path",
    "resolve_case_paths",
]


MODEL_CACHE_ROOT = "cached_models"
RESULT_CACHE_ROOT = "cached_results"


def resolve_case_paths(
    case: ParameterCase,
    model_cache_dir: Path,
    result_cache_dir: Path,
) -> ParameterCase:
    """Resolve cached model and result placeholders in a parameter case."""
    values = {
        key: resolve_cached_path(
            value,
            model_cache_dir=model_cache_dir,
            result_cache_dir=result_cache_dir,
        )
        for key, value in case.values.items()
    }
    return ParameterCase(case.index, values)


def resolve_cached_path(
    value: Any,
    *,
    model_cache_dir: Path,
    result_cache_dir: Path,
) -> Any:
    """Resolve cache placeholders in one configuration value.

    Placeholders are matched as whole path segments only, so a segment that
    merely *contains* the placeholder (e.g. ``old_cached_models``) is left
    untouched. Both bare placeholders (``cached_models``) and placeholders
    followed by a sub-path (``cached_models/CoPaw-9B``) are resolved.
    """
    if not isinstance(value, str):
        return value
    # Normalize separators so segment splitting is consistent on all platforms;
    # the result is reconstructed with the original separator style below.
    normalized = value.replace("\\", "/")
    segments = normalized.split("/")
    resolved_segments = [
        _resolve_cache_segment(segment, model_cache_dir, result_cache_dir)
        for segment in segments
    ]
    # If no segment changed, return the original value unchanged.
    if resolved_segments == segments:
        return value
    # Preserve the original separator when a single resolved segment now
    # contains the other separator style (e.g. a Windows model_cache_dir).
    separator = "\\" if "\\" in value and "/" not in value else "/"
    return separator.join(resolved_segments)


def _resolve_cache_segment(
    segment: str,
    model_cache_dir: Path,
    result_cache_dir: Path,
) -> str:
    """Resolve a single path segment if it is exactly a cache placeholder."""
    if segment == MODEL_CACHE_ROOT:
        return str(model_cache_dir)
    if segment == RESULT_CACHE_ROOT:
        return str(result_cache_dir)
    return segment


def cache_case_reference(value: str) -> tuple[str, str] | None:
    """Return the cache root and case directory a configuration path refers to.

    ``cached_results/hmm_xh2_2k/hmquant/quant_embedding.pt`` refers to case
    ``hmm_xh2_2k`` under the result cache.  Values without a cache placeholder,
    or without a directory below it, have no case reference.
    """
    parts = Path(value.replace("\\", "/")).parts
    for root in (MODEL_CACHE_ROOT, RESULT_CACHE_ROOT):
        if root not in parts:
            continue
        index = parts.index(root)
        if index + 1 < len(parts):
            return root, parts[index + 1]
    return None


def cache_root_directory(
    root: str, *, model_cache_dir: Path, result_cache_dir: Path
) -> Path:
    """Resolve a cache placeholder root to its concrete directory."""
    return model_cache_dir if root == MODEL_CACHE_ROOT else result_cache_dir


def get_model_case_artifact_id(case: ParameterCase) -> str | None:
    """Return the cache case id produced by a get-model case.

    A configured output may point below the case directory, for example
    ``cached_results/hmm_xh2_256k/xh2``.  The stable id is still the first path
    segment below the recognized cache root.  Paths outside the cache roots
    retain the legacy final-directory fallback.
    """
    file_type = str(case.values.get("type", ""))
    keys = (
        ("download_dir", "model_dir")
        if file_type == "raw"
        else (
            "extract_dir",
            "quant_model_dir",
            "build_model_dir",
            "model_dir",
            "download_dir",
        )
    )
    for key in keys:
        value = case.values.get(key)
        if isinstance(value, str) and value:
            reference = cache_case_reference(value)
            if reference is not None:
                return reference[1]
            return Path(value).name
    return None


def replace_case_output_dir(
    case: ParameterCase, directory: Path
) -> ParameterCase:
    """Return a parameter case with its output-directory option replaced."""
    values = dict(case.values)
    for key in ("out-dir", "output_dir", "out_dir", "output_path"):
        if key in values:
            values[key] = str(directory)
            return ParameterCase(case.index, values)
    return case

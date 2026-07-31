# Copyright (c) 2025 HOUMO AI
#
# File: parameter_matrix.py
# Description:
#  Parameter Matrix Validation and Command-Line Rendering.
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

"""Convert column-oriented JSON parameters into executable command cases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .flow_contracts import ConfigError

__all__ = ["ParameterCase", "ParameterMatrix", "render_case_options"]


OMITTED_VALUES = (None, "default")


@dataclass(frozen=True)
class ParameterCase:
    """One indexed command parameter combination."""

    index: int
    values: Mapping[str, Any]


@dataclass(frozen=True)
class ParameterMatrix:
    """Validated collection of parameter cases created from JSON columns."""

    cases: tuple[ParameterCase, ...]

    @classmethod
    def from_columns(
        cls,
        columns: Mapping[str, Sequence[Any]],
        *,
        ignored_keys: Iterable[str] = (),
        location: str = "parameter matrix",
    ) -> "ParameterMatrix":
        """Build a validated parameter matrix from column-oriented values."""
        ignored = set(ignored_keys)
        active = {
            key: value
            for key, value in columns.items()
            if key not in ignored and isinstance(value, Sequence) and not isinstance(value, str)
        }
        if not active:
            return cls(())
        lengths = {key: len(value) for key, value in active.items()}
        if len(set(lengths.values())) != 1:
            raise ConfigError(
                f"Parameter columns must have equal lengths at {location}",
                details={"lengths": lengths},
            )
        count = next(iter(lengths.values()))
        return cls(
            tuple(
                ParameterCase(index, {key: values[index] for key, values in active.items()}) for index in range(count)
            )
        )


def render_case_options(
    case: ParameterCase,
    *,
    positional_keys: Iterable[str] = (),
    skipped_keys: Iterable[str] = (),
    skipped_values: Mapping[str, set[Any]] | None = None,
    option_prefix: str = "--",
) -> tuple[str, ...]:
    """Render one parameter case as normalized command-line options."""
    positional = set(positional_keys)
    skipped = set(skipped_keys)
    skipped_values = skipped_values or {}
    argv: list[str] = []
    for key, value in case.values.items():
        argv.extend(
            _render_option(
                key,
                value,
                positional=key in positional,
                skipped=key in skipped or value in OMITTED_VALUES,
                skipped_value=value in skipped_values.get(key, set()),
                option_prefix=option_prefix,
            )
        )
    return tuple(argv)


def _render_option(
    key: str,
    value: Any,
    *,
    positional: bool,
    skipped: bool,
    skipped_value: bool,
    option_prefix: str,
) -> tuple[str, ...]:
    """Render one key/value pair, returning an empty tuple when omitted."""
    if skipped or skipped_value:
        return ()
    if isinstance(value, bool):
        if not value:
            return ()
        return (str(value) if positional else f"{option_prefix}{key}",)
    if positional:
        return (str(value),)
    return (f"{option_prefix}{key}", str(value))

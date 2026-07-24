# iModelzoo Coding Style Guide

This document defines repository-wide coding and day-to-day engineering conventions for iModelzoo.

## Scope and precedence

- Explicit task instructions take precedence over this guide.
- More-specific repository instructions and configuration files closer to a changed file take precedence over repository-wide defaults.
- Preserve established conventions in the touched component when they do not conflict with higher-priority instructions.
- Machine-enforced formatting and linting are defined by the applicable checked-in configuration files, such as `.clang-format`, `.pre-commit-config.yaml`, `pyproject.toml`, and `setup.cfg`.
- When this guide and an applicable formatter configuration disagree on mechanical formatting, follow the formatter configuration and report the inconsistency.
- If applicable instructions conflict materially, report the conflict instead of silently choosing one.

## General principles

- Prefer **small, focused diffs**. Avoid drive-by refactors and unrelated formatting.
- Preserve existing naming and style in the touched area unless a change is explicitly requested.
- Prefer explicit, readable code over cleverness.
- Keep functions focused; separate decisions from mechanics when practical.
- Avoid deep nesting and use early returns where they improve clarity.
- Do not add dependencies or change public API/ABI without explicit approval.
- Do not leave debug logging, tracing, synchronization, or instrumentation in performance-critical paths unless explicitly requested.
- Do not run repository-wide formatting or cleanup when only a limited set of files is in scope.

## Protected and generated code

- Do not modify vendored or third-party source trees unless explicitly requested.
- Known vendored paths include `apis/common/eigen3/`, `apis/common/yaml-cpp/`, `apis/common/nlohmann/`, `apis/common/hpp/spdlog/`, and `tools/common/spdlog/`.
- Do not assume every file under `apis/common/` or `tools/common/` is vendored. First-party shared components, such as `tools/common/houmo-llm-engine/`, may be modified when they are explicitly in scope.
- Do not edit generated or build artifacts directly, including `build/`, `builds/`, `*.inc`, and `*.gen`.
- When regeneration is required, modify the true source, such as a TableGen `.td` file or generator, and document the regeneration command.

## Languages and tooling

- Primary implementation languages are C++ and Python.
- Common supporting formats include Shell, YAML, and CMake.
- Use the nearest applicable formatter, linter, build configuration, and component documentation.
- Do not install tools or dependencies merely to run validation unless explicitly approved.

## C++ style

### Readability and control flow

- Always use braces for `if`, `for`, `while`, and `do`, even when a formatter permits an unbraced single-line statement.
- Make ownership, lifetime, and error-handling behavior explicit.
- Prefer existing project abstractions over introducing parallel helpers for the same purpose.

### Naming

- Follow naming conventions in the surrounding component.
- Avoid vague names such as `tmp` or `temp` for persistent source files; use names that describe the file's role.

### Includes and dependencies

- Include what is used and keep includes minimal.
- Preserve the include ordering produced by the applicable formatter.
- Do not add third-party dependencies without explicit approval.
- Keep declarations and definitions synchronized when changing headers or public interfaces.

## Python style

### Readability and control flow

- Use clear, conventional Python control flow.
- Avoid deeply nested branches; prefer early returns where appropriate.
- Do not compress non-trivial control flow into one-line statements.
- Keep CLI parsing, configuration resolution, and execution logic separated when practical.

### Naming

- Follow naming conventions in the surrounding component.
- Avoid vague names such as `tmp` or `temp` for persistent source files; use names that describe the file's role.
- Add or preserve type annotations when they are already part of the touched component's style; do not perform unrelated typing rewrites.

### Imports and dependencies

- Import only what is used; keep imports minimal and explicit.
- Follow the repository's configured import ordering.
- Avoid wildcard imports unless required by an established local pattern.
- Do not reorder or rewrite unrelated imports outside the touched scope.
- Do not add third-party dependencies without explicit approval.

## Shell style

- Preserve the script's existing shell and platform compatibility.
- Quote variable expansions unless intentional word splitting or globbing is required.
- Preserve existing CLI arguments, environment-variable semantics, and exit behavior unless a change is explicitly requested.
- Prefer actionable error messages and clear error handling.
- Run `shellcheck` on touched scripts when it is already available and applicable.

## Configuration and build files

- Preserve local formatting and key ordering in YAML, CMake, and other configuration files unless normalization is explicitly requested.
- Avoid unrelated reformatting of configuration and build files.
- When changing a configuration field, update directly coupled consumers, examples, tests, and documentation that are in scope.
- Keep platform-specific paths and behavior separated where the existing structure distinguishes Linux, Windows, Android, or other targets.

## Compatibility and performance

- Preserve existing CLI flags, defaults, output formats, model artifact names, and public API/ABI unless the task explicitly requires a change.
- When a compatibility-affecting change is requested, update directly related tests, examples, and documentation.
- Do not silently change model paths, batch sizes, sequence lengths, device counts, or performance-reporting semantics.
- Preserve benchmark methodology, warm-up behavior, timing boundaries, and reported metrics unless changing them is part of the task.

## Validation

- Validate changes in proportion to their scope and risk.
- Prefer the narrowest relevant checks for the files and components changed.
- Run formatters and linters only on touched files unless broader validation is explicitly requested.
- For Python changes, run an appropriate syntax, import, or focused pytest check when the environment supports it.
- For C/C++ changes, use the nearest component-specific formatting, build, or test command when available.
- Preserve existing tests and add or update focused tests when behavior changes.
- Do not access the network, install dependencies, or alter the environment merely to run validation unless explicitly approved.
- If a relevant check cannot be run, state what was not run and why.

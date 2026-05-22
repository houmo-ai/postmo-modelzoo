# iModelzoo Coding Style Guide

This document is the source of truth for coding style and day-to-day engineering conventions in this repo.

## General principles
- Prefer **small, focused diffs**. Avoid drive-by refactors and unrelated formatting.
- Preserve existing naming/style in the touched area unless explicitly requested.
- Do not edit vendored deps under `apis/common/` and `tools/common/`.
- Do not edit generated/build artifacts directly (e.g. `build/`, `builds/`, `*.inc`, `*.gen`). If regeneration is needed, modify the true sources (TableGen `.td`, generators, etc.) and document how to regenerate.
- Avoid introducing debug-only changes in performance-critical code unless explicitly asked.

## Languages & toolchain
- Primary: C++ (project standard) + Python.

## C++ style (general)
### Readability
- Prefer explicit, readable code over cleverness.
- Keep functions small; separate “decision” from “mechanics” when possible.
- Avoid deep nesting; return early on error paths.

### Braces and control flow
- Always use braces for `if/for/while/do`, even for single-line bodies.

### Naming
- Follow existing conventions in the surrounding code.
- Avoid vague names like `tmp`, `temp` in file names; prefer descriptive names.

### Includes & dependencies
- Include what you use; keep includes minimal.
- Do not add new third-party deps without explicit approval.

## Python style (general)
### Readability
- Prefer explicit, readable code over cleverness.
- Keep functions small; separate “decision” from “mechanics” when possible.
- Avoid deep nesting; return early on error paths.

### Braces and control flow
- Always use braces for `if/for/while/do`, even for single-line bodies.

### Naming
- Follow existing conventions in the surrounding code.
- Avoid vague names like `tmp`, `temp` in file names; prefer descriptive names.

### Includes & dependencies
- Include what you use; keep includes minimal.
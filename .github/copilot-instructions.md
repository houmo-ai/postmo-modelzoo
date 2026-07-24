# Copilot Instructions for iModelzoo

## Language

- Default prose language: Chinese.
- Do NOT translate code, paths, commands, identifiers, or tool output.

## Project

iModelzoo: a model and example repository for Houmo's AI platform, which showcases model conversion, model quantization (via the Houmo Quantization Tool), model compilation (via the Python TCIM interface), model deployment and inference (via the TCIM interface in Python or C++), evaluation, API usage, as well as tools for quantization and performance testing.
Primary languages: C++ and Python.

## Non-negotiables

- Make small, focused diffs. No drive-by refactors or unrelated formatting.
- Preserve existing naming/style unless explicitly asked.
- Do not modify vendored deps under `apis/common/` and `tools/common/`.
- Do not edit build outputs or generated artifacts (`build/`, `builds/`, `*.inc`, `*.gen`, etc.).
- Do not add dependencies or change public API/ABI without explicit approval.

## Hard safety bans (unless I explicitly approve)

- No git remote/history changes: `git pull`, `git fetch`, `git push`, `git reset`, `git rebase`
- No network/install/escalation: `curl`, `wget`, `apt`, `pip`, `brew`, `sudo`, `docker`

## References (read-only)

- Repository layout: `.github/guidance/repo-layout.md`
- Coding style: `.github/guidance/coding-style.md`
- Review: `.github/guidance/review-guidelines.md`

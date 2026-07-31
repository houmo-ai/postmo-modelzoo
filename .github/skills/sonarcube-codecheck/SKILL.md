---
name: sonarcube-codecheck
description: SonarQube code check for a git repo's latest commit, producing a Markdown report. Use when the user asks to "scan a commit", "run codecheck", "run a SonarQube scan", "code quality check", "sonar-scanner", or "扫描最新提交/代码检查/质量门禁". Adapted from the CI codecheck.sh / imodelzoo_codecheck.sh scripts.
---

# sonarcube-codecheck

Runs a SonarQube analysis on the source files changed in a repository's latest
commit (HEAD), waits for the Compute Engine task, then writes a Markdown report
to `<output-dir>/<full-commit-id>.md`.

This mirrors the CI scripts (`codecheck.sh` / `imodelzoo_codecheck.sh` /
`logging.sh`) but scans the commit's changed files instead of a CI diff file,
and generates a report instead of just gating.

## Default workflow (when invoked without specific instructions)

When the user asks to "scan my latest commit / run codecheck / 扫描最新提交",
run this skill's script against the repository the SKILL.md lives in, scanning
`HEAD`, and write the report into `<repo>/.sonarcube/<commit-id>.md`.

Steps:

1. Ensure `SONARQUBE_URL`, `SONARQUBE_TOKEN`, and `SONARQUBE_PROJECT_KEY` are set
   in the environment (and optionally `SONARQUBE_BRANCH`). If a required one is
   missing, ask the user for it. `SONARQUBE_PROJECT_KEY` defaults to
   `imodelzoo_codecheck` for this repo unless the user says otherwise.
2. Run the scan, pointing `--output-dir` at the repo's `.sonarcube/` directory
   (the script creates it if needed). The report file name is the full commit id.

```bash
# from the repository root (the repo containing this .github/skills dir)
export SONARQUBE_URL="http://<host>:9000"
export SONARQUBE_TOKEN="squ_xxxxxxxx"
export SONARQUBE_PROJECT_KEY="imodelzoo_codecheck"
export SONARQUBE_BRANCH="develop"     # optional

bash .github/skills/sonarcube-codecheck/scripts/codecheck.sh \
  --repo "$(git rev-parse --show-toplevel)" \
  --output-dir "$(git rev-parse --show-toplevel)/.sonarcube"
```

The report is written to `<repo>/.sonarcube/<full-commit-id>.md`, and the script
prints that path on success. Report the path and a one-line summary (quality
gate + issue count) back to the user.

The SonarQube **projectKey is built per-commit**:
`${SONARQUBE_PROJECT_KEY}_<full-commit-id>`. Each commit therefore gets its own
project in SonarQube, matching the CI convention (e.g.
`imodel_zoo_xh2_imodelzoo_codecheck_1009`).

## Prerequisites (on the machine running the scan — the scanner host)

These MUST be on the scanner host, not the SonarQube server. The script runs a
preflight check and, for anything missing, prints an OS-specific install command
(it auto-detects apt / dnf / yum / apk / brew / pacman) instead of failing
silently.

Environment variables (read automatically; the script exits with an example
export line for any required one that is unset):

- `SONARQUBE_URL` (required) — e.g. `http://10.64.35.181:9000`
- `SONARQUBE_TOKEN` (required) — a valid user token, `squ_...`
- `SONARQUBE_PROJECT_KEY` (required) — the **base** key; the final projectKey is
  `${SONARQUBE_PROJECT_KEY}_<commit-id>`
- `SONARQUBE_BRANCH` (optional) — passed as `sonar.branch.name`; if unset it is
  derived from git

Tools:

- Required — script hard-fails with an install hint if any are missing:
  `sonar-scanner`, `curl`, `python3`
- Recommended — script warns + prints an install hint if missing:
  **`shellcheck`** — REQUIRED for `.sh` files to be analyzed. The SonarQube
  ShellCheck sensor invokes the local `shellcheck` binary. If it is missing,
  shell issues are silently skipped (the scan still "succeeds" but
  under-reports), and the generated report notes that shell analysis was
  skipped. Verify with `command -v shellcheck`.

## Usage

```bash
export SONARQUBE_URL="http://<host>:9000"
export SONARQUBE_TOKEN="squ_xxxxxxxx"
export SONARQUBE_PROJECT_KEY="my_project"      # final key = my_project_<commit>
export SONARQUBE_BRANCH="develop"              # optional

bash scripts/codecheck.sh \
  --repo /path/to/repo \
  --output-dir /path/to/output      # optional, default: repo's parent dir
```

Options (each arg overrides its corresponding env var):

- `--repo <path>` repository to scan (required)
- `--project-key <key>` base project key (overrides `SONARQUBE_PROJECT_KEY`)
- `--branch <name>` branch name (overrides `SONARQUBE_BRANCH`)
- `--output-dir <path>` where to write `<commit>.md` (default: parent of repo)
- `--commit <ref>` commit to scan (default: `HEAD`)
- `--scope changed|all` scan only changed source files, or the whole tree (default: `changed`)
- `--quality-gate <name>` select this gate for the project before scanning
- `--project-version <v>` sonar.projectVersion (default: `1.0`)
- `--python-version <v>` sonar.python.version (default: `3.8`)
- `--extra "<args>"` extra raw args appended to sonar-scanner

## What the script does

1. Reads `SONARQUBE_URL`/`SONARQUBE_TOKEN`/`SONARQUBE_PROJECT_KEY`/
   `SONARQUBE_BRANCH` from the environment (args override), validates env vars,
   tools, token (`api/authentication/validate`), and server status
   (`api/system/status` must be `UP`).
2. Adds the repo to git `safe.directory`, reads commit metadata and branch
   (uses `SONARQUBE_BRANCH`/`--branch` if provided, else derives from git).
3. Builds the final projectKey `${SONARQUBE_PROJECT_KEY}_<commit-id>`.
4. For `--scope changed`: lists files changed in the commit and filters out
   images/data/binaries (`.jpg .png .mat .bin .pt .onnx .so ...`), passing the
   rest via `-Dsonar.inclusions`. If no source files changed, it exits early.
5. Optionally selects a quality gate.
6. Runs `sonar-scanner` (with `sonar.branch.name` when a branch is set),
   extracts the CE task id from the log.
7. Polls `api/ce/task?id=...` until `SUCCESS`. On a transient `FAILED` (common on
   a project's first analysis) it surfaces the server error and retries the scan
   once.
8. Pulls branch-scoped quality gate, measures, and issues, then renders a
   Markdown report to `<output-dir>/<full-commit-id>.md` and prints its path.

## Determining the base projectKey

The base key comes from `SONARQUBE_PROJECT_KEY` (or `--project-key`). If the user
works in an IDE, check the VS Code SonarLint binding first:
`<repo>/.vscode/settings.json` for `sonarlint.connectedMode.project.projectKey`,
or a `.sonarlint/` directory. If there is no binding and no env var, ask the user
(in CI the base comes from the job's `jobInfo`).

## Troubleshooting

- **`Not authorized` / `{"valid":false}`**: the token is invalid or expired.
  Generate a new one: SonarQube > My Account > Security > Generate Tokens, then
  re-export `SONARQUBE_TOKEN`.
- **Plugin download `HTTP 500` / server `RESTARTING`**: the server is mid-restart
  or half-started. Check `api/system/health` (want `GREEN`) and
  `api/system/status` (want `UP`). If it is stuck in `RESTARTING` with an empty
  CE queue and health `RED`, the CE/web/Elasticsearch process failed to come up —
  that needs host-level intervention (check `logs/es.log`, `logs/ce.log`,
  `vm.max_map_count`, disk space, DB connectivity), not an API fix.
- **Shell issues missing from the report**: `shellcheck` is not installed on the
  scanner host. Install it and re-scan.
- **Results differ from another SonarQube server**: results depend on the
  server-side Quality Profiles. Two servers with different Python/Shell profiles
  produce different issue sets even for the same code. Compare the bound profiles
  via `api/qualityprofiles/search?project=<key>` and align them
  (`api/qualityprofiles/backup` to export, then import) to reproduce.
- **`jq` not available**: the script uses `python3` for JSON parsing, so `jq` is
  not required.

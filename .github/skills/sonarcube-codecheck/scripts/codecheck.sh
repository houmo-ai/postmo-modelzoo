#!/bin/bash
# codecheck.sh - SonarQube scan of a repo's latest commit + report generation.
#
# Adapted from the CI codecheck scripts (codecheck.sh / imodelzoo_codecheck.sh /
# logging.sh). Instead of a CI diff file it scans the source files changed in the
# latest commit (HEAD), waits for the Compute Engine task, then writes a Markdown
# report to <output-dir>/<full-commit-id>.md.
#
# Requirements on the machine running this script (the scanner host):
#   - sonar-scanner in PATH
#   - curl, python3 in PATH
#   - shellcheck in PATH  (REQUIRED for shell files to be analyzed; if missing,
#     the ShellCheck sensor is silently skipped and .sh issues will NOT appear)
#
# Environment variables (read automatically):
#   - SONARQUBE_URL          SonarQube server URL (required)
#   - SONARQUBE_TOKEN        user token, squ_... (required)
#   - SONARQUBE_PROJECT_KEY  base project key; final key = <base>_<commit_id> (required)
#   - SONARQUBE_BRANCH       branch name passed as sonar.branch.name (optional;
#                            derived from git if unset)
#
# Usage:
#   codecheck.sh --repo <path> [options]
#
# The final SonarQube projectKey is: ${SONARQUBE_PROJECT_KEY}_<full-commit-id>
#
# Options (args override the corresponding env var):
#   --repo <path>          Path to the git repository (required)
#   --project-key <key>    Base project key (overrides SONARQUBE_PROJECT_KEY)
#   --branch <name>        Branch name (overrides SONARQUBE_BRANCH)
#   --output-dir <path>    Where to write <commit>.md (default: repo parent dir)
#   --commit <ref>         Commit to scan (default: HEAD)
#   --scope changed|all    Scan only files changed in the commit, or the whole
#                          tree (default: changed)
#   --quality-gate <name>  Select this quality gate for the project before scan
#   --project-version <v>  sonar.projectVersion (default: 1.0)
#   --python-version <v>   sonar.python.version (default: 3.8)
#   --extra "<args>"       Extra raw args appended to sonar-scanner
#
# Exit codes: 0 success (report written); non-zero on hard failure.

set -o pipefail

# ---------- logging ----------
log_info() { echo -e "\033[32m[INFO] $(date '+%Y-%m-%d %H:%M:%S') $*\033[0m"; }
log_warn() { echo -e "\033[33m[WARN] $(date '+%Y-%m-%d %H:%M:%S') $*\033[0m"; }
log_err()  { echo -e "\033[31m[ERROR] $(date '+%Y-%m-%d %H:%M:%S') $*\033[0m" >&2; }

# ---------- defaults ----------
repo=""
project_key_base=""   # override for $SONARQUBE_PROJECT_KEY (base key)
branch=""             # override for $SONARQUBE_BRANCH
output_dir=""
commit="HEAD"
scope="changed"
quality_gate=""
project_version="1.0"
python_version="3.8"
extra_args=""

# ---------- arg parsing ----------
while (( $# )); do
  case "$1" in
    --repo)            repo="$2"; shift 2;;
    --project-key)     project_key_base="$2"; shift 2;;   # overrides SONARQUBE_PROJECT_KEY
    --branch)          branch="$2"; shift 2;;             # overrides SONARQUBE_BRANCH
    --output-dir)      output_dir="$2"; shift 2;;
    --commit)          commit="$2"; shift 2;;
    --scope)           scope="$2"; shift 2;;
    --quality-gate)    quality_gate="$2"; shift 2;;
    --project-version) project_version="$2"; shift 2;;
    --python-version)  python_version="$2"; shift 2;;
    --extra)           extra_args="$2"; shift 2;;
    *) log_err "unknown arg: $1"; exit 2;;
  esac
done

# ---------- resolve from environment (args override env) ----------
# SONARQUBE_URL / SONARQUBE_TOKEN / SONARQUBE_BRANCH / SONARQUBE_PROJECT_KEY
[[ -z "$project_key_base" ]] && project_key_base="$SONARQUBE_PROJECT_KEY"
[[ -z "$branch" ]]           && branch="$SONARQUBE_BRANCH"

# ---------- helpers: package manager detection + install hints ----------
detect_pkg_mgr() {
  if command -v apt-get >/dev/null 2>&1;  then echo "apt";    return; fi
  if command -v dnf     >/dev/null 2>&1;  then echo "dnf";    return; fi
  if command -v yum     >/dev/null 2>&1;  then echo "yum";    return; fi
  if command -v apk     >/dev/null 2>&1;  then echo "apk";    return; fi
  if command -v brew    >/dev/null 2>&1;  then echo "brew";   return; fi
  if command -v pacman  >/dev/null 2>&1;  then echo "pacman"; return; fi
  echo "unknown"
}
PKG_MGR="$(detect_pkg_mgr)"

# install_hint <tool> -> prints a suggested install command for the detected OS.
install_hint() {
  local tool="$1"
  case "$tool" in
    shellcheck)
      case "$PKG_MGR" in
        apt)    echo "  sudo apt-get update && sudo apt-get install -y shellcheck";;
        dnf)    echo "  sudo dnf install -y ShellCheck";;
        yum)    echo "  sudo yum install -y ShellCheck  # (needs EPEL)";;
        apk)    echo "  sudo apk add shellcheck";;
        brew)   echo "  brew install shellcheck";;
        pacman) echo "  sudo pacman -S shellcheck";;
        *)      echo "  Download a static binary: https://github.com/koalaman/shellcheck/releases (extract 'shellcheck' into a PATH dir)";;
      esac;;
    curl)
      case "$PKG_MGR" in
        apt)    echo "  sudo apt-get install -y curl";;
        dnf)    echo "  sudo dnf install -y curl";;
        yum)    echo "  sudo yum install -y curl";;
        apk)    echo "  sudo apk add curl";;
        brew)   echo "  brew install curl";;
        pacman) echo "  sudo pacman -S curl";;
        *)      echo "  Install 'curl' via your OS package manager.";;
      esac;;
    python3)
      case "$PKG_MGR" in
        apt)    echo "  sudo apt-get install -y python3";;
        dnf)    echo "  sudo dnf install -y python3";;
        yum)    echo "  sudo yum install -y python3";;
        apk)    echo "  sudo apk add python3";;
        brew)   echo "  brew install python";;
        pacman) echo "  sudo pacman -S python";;
        *)      echo "  Install 'python3' via your OS package manager.";;
      esac;;
    sonar-scanner)
      echo "  Download SonarScanner CLI: https://docs.sonarsource.com/sonarqube/latest/analyzing-source-code/scanners/sonarscanner/"
      echo "  Extract it and add its bin/ directory to PATH.";;
    *)
      echo "  Install '$tool' via your OS package manager.";;
  esac
}

# ---------- validation ----------
[[ -z "$repo" ]]   && { log_err "--repo is required"; exit 2; }
[[ ! -d "$repo" ]] && { log_err "repo not found: $repo"; exit 2; }

# env vars
missing_env=0
if [[ -z "$SONARQUBE_URL" ]]; then
  log_err "SONARQUBE_URL env is not set. Set it to your SonarQube server, e.g.:"
  echo "  export SONARQUBE_URL=\"http://<host>:9000\""
  missing_env=1
fi
if [[ -z "$SONARQUBE_TOKEN" ]]; then
  log_err "SONARQUBE_TOKEN env is not set. Generate a token in SonarQube > My Account > Security, then:"
  echo "  export SONARQUBE_TOKEN=\"squ_xxxxxxxx\""
  missing_env=1
fi
if [[ -z "$project_key_base" ]]; then
  log_err "Project key base is not set. Provide it via SONARQUBE_PROJECT_KEY env or --project-key, e.g.:"
  echo "  export SONARQUBE_PROJECT_KEY=\"my_project\""
  missing_env=1
fi
[[ $missing_env -ne 0 ]] && exit 2

url="$SONARQUBE_URL"
token="$SONARQUBE_TOKEN"
# SONARQUBE_BRANCH is optional; if unset it is derived from git later.

# required tools: fail with per-OS install instructions
missing_req=0
for tool in sonar-scanner curl python3; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    log_err "Required tool '$tool' not found in PATH. Install it:"
    install_hint "$tool"
    missing_req=1
  fi
done
[[ $missing_req -ne 0 ]] && { log_err "Aborting: install the missing required tool(s) above and re-run."; exit 2; }

repo="$(cd "$repo" && pwd)"
[[ -z "$output_dir" ]] && output_dir="$(dirname "$repo")"
mkdir -p "$output_dir"

# optional but strongly recommended: shellcheck (needed for .sh analysis)
shellcheck_ok=1
if ! command -v shellcheck >/dev/null 2>&1; then
  shellcheck_ok=0
  log_warn "shellcheck not found in PATH. Shell (.sh) files will NOT be analyzed by SonarQube's ShellCheck sensor (issues will be under-reported)."
  log_warn "To enable shell analysis, install it on THIS machine (the scanner host):"
  install_hint "shellcheck"
fi

# ---------- token / server checks ----------
valid=$(curl -s -u "$token:" "$url/api/authentication/validate" | python3 -c "import sys,json;print(json.load(sys.stdin).get('valid'))" 2>/dev/null)
if [[ "$valid" != "True" ]]; then
  log_err "SONARQUBE_TOKEN is not valid (authentication/validate returned: $valid). Generate a new token in SonarQube > My Account > Security."
  exit 1
fi

status=$(curl -s -u "$token:" "$url/api/system/status" | python3 -c "import sys,json;print(json.load(sys.stdin).get('status'))" 2>/dev/null)
if [[ "$status" != "UP" ]]; then
  log_err "SonarQube server is not UP (status=$status). Wait for it to finish (re)starting, then retry."
  exit 1
fi

# ---------- git info ----------
git config --global --add safe.directory "$repo" 2>/dev/null
pushd "$repo" >/dev/null || { log_err "cannot enter repo"; exit 1; }
  commit_full=$(git rev-parse "$commit") || { log_err "bad commit: $commit"; popd >/dev/null || true; exit 1; }
  commit_author=$(git log -1 --pretty=%an "$commit_full")
  commit_date=$(git log -1 --pretty=%ad "$commit_full")
  commit_subject=$(git log -1 --pretty=%s "$commit_full")
  # branch: prefer SONARQUBE_BRANCH env / --branch; otherwise derive from git.
  if [[ -z "$branch" ]]; then
    branch=$(git log -10 --oneline --decorate "$commit_full" 2>/dev/null | grep -o "origin/[^),]*" | head -1 | cut -d'/' -f2)
    [[ -z "$branch" ]] && branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
  fi

  # Build inclusion list of changed SOURCE files (skip images/data/binaries).
  inclusions=""
  if [[ "$scope" == "changed" ]]; then
    mapfile -t changed < <(git diff-tree --no-commit-id --name-only -r "$commit_full")
    src=()
    for f in "${changed[@]}"; do
      case "$f" in
        *.jpg|*.jpeg|*.png|*.gif|*.bmp|*.JPEG|*.JPG|*.mat|*.bin|*.pt|*.pth|*.onnx|*.so|*.a|*.o) continue;;
        *) [[ -f "$repo/$f" ]] && src+=("$f");;
      esac
    done
    if [[ ${#src[@]} -eq 0 ]]; then
      log_warn "No source files changed in $commit_full (only data/binary/deleted files). Nothing to analyze."
      popd >/dev/null || true
      exit 0
    fi
    inclusions=$(IFS=,; echo "${src[*]}")
    log_info "Changed source files to scan: $inclusions"
  fi
popd >/dev/null || true

# ---------- construct projectKey = <SONARQUBE_PROJECT_KEY>_<commit_id> ----------
project_key="${project_key_base}_${commit_full}"
log_info "projectKey: $project_key (base=$project_key_base, commit=$commit_full)"
[[ -n "$branch" ]] && log_info "branch: $branch"

# ---------- optional: select quality gate ----------
if [[ -n "$quality_gate" ]]; then
  log_info "Selecting quality gate '$quality_gate' for project '$project_key'"
  curl -s -u "$token:" -X POST "$url/api/qualitygates/select" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "projectKey=${project_key}&gateName=${quality_gate}" >/dev/null
fi

# ---------- run scanner ----------
scan_log=$(mktemp /tmp/codecheck_scan.XXXXXX.log)
log_info "Running sonar-scanner (projectKey=$project_key, scope=$scope)"
pushd "$repo" >/dev/null || { log_err "cannot enter repo"; exit 1; }
  # shellcheck disable=SC2086
  sonar-scanner \
    -Dsonar.projectKey="$project_key" \
    -Dsonar.sources=. \
    ${inclusions:+-Dsonar.inclusions="$inclusions"} \
    ${branch:+-Dsonar.branch.name="$branch"} \
    -Dsonar.exclusions="**.java,**3rdparty**" \
    -Dsonar.scm.disabled=true \
    -Dsonar.sourceEncoding=UTF-8 \
    -Dsonar.python.version="$python_version" \
    -Dsonar.projectVersion="$project_version" \
    -Dsonar.host.url="$url" \
    -Dsonar.token="$token" \
    $extra_args 2>&1 | tee "$scan_log"
  scan_rc=${PIPESTATUS[0]}
popd >/dev/null || true

if [[ $scan_rc -ne 0 ]]; then
  log_err "sonar-scanner failed (rc=$scan_rc). See $scan_log"
  exit 1
fi

task_id=$(grep -oP 'api/ce/task\?id=\K[^ ]+' "$scan_log" | head -1)
if [[ -z "$task_id" ]]; then
  log_err "Could not find CE task id in scanner output."
  exit 1
fi
log_info "CE task id: $task_id"

# ---------- wait for CE task ----------
wait_ce() {
  local tid="$1" status="PENDING" attempt=0 max_attempts=90 resp
  while [[ ( "$status" == "IN_PROGRESS" || "$status" == "PENDING" ) && $attempt -lt $max_attempts ]]; do
    resp=$(curl -s -u "$token:" "$url/api/ce/task?id=$tid")
    status=$(echo "$resp" | python3 -c "import sys,json;print(json.load(sys.stdin)['task']['status'])" 2>/dev/null)
    if [[ "$status" == "IN_PROGRESS" || "$status" == "PENDING" ]]; then
      attempt=$((attempt+1))
      log_info "CE task $status ... (try $attempt/$max_attempts), waiting 10s"
      sleep 10
    fi
  done
  CE_STATUS="$status"
  CE_ERROR=$(echo "$resp" | python3 -c "import sys,json;print(json.load(sys.stdin)['task'].get('errorMessage',''))" 2>/dev/null)
}

wait_ce "$task_id"
if [[ "$CE_STATUS" != "SUCCESS" ]]; then
  log_err "CE task did not succeed (status=$CE_STATUS). ${CE_ERROR:+errorMessage: $CE_ERROR}"
  # A brand-new project+branch sometimes fails transiently on the first
  # analysis. Retry the scan once before giving up.
  log_warn "Retrying the scan once (transient CE failures happen on a project's first analysis)."
  pushd "$repo" >/dev/null || { log_err "cannot enter repo"; exit 1; }
    # shellcheck disable=SC2086
    sonar-scanner \
      -Dsonar.projectKey="$project_key" \
      -Dsonar.sources=. \
      ${inclusions:+-Dsonar.inclusions="$inclusions"} \
      ${branch:+-Dsonar.branch.name="$branch"} \
      -Dsonar.exclusions="**.java,**3rdparty**" \
      -Dsonar.scm.disabled=true \
      -Dsonar.sourceEncoding=UTF-8 \
      -Dsonar.python.version="$python_version" \
      -Dsonar.projectVersion="$project_version" \
      -Dsonar.host.url="$url" \
      -Dsonar.token="$token" \
      $extra_args 2>&1 | tee "$scan_log"
    scan_rc=${PIPESTATUS[0]}
  popd >/dev/null || true
  [[ $scan_rc -ne 0 ]] && { log_err "retry scan failed (rc=$scan_rc)."; exit 1; }
  task_id=$(grep -oP 'api/ce/task\?id=\K[^ ]+' "$scan_log" | head -1)
  log_info "retry CE task id: $task_id"
  wait_ce "$task_id"
  if [[ "$CE_STATUS" != "SUCCESS" ]]; then
    log_err "CE task failed again (status=$CE_STATUS). ${CE_ERROR:+errorMessage: $CE_ERROR}"
    exit 1
  fi
fi
log_info "CE task SUCCESS"

# ---------- fetch results (branch-scoped when a branch was used) ----------
bq=""; [[ -n "$branch" ]] && bq="&branch=$branch"
qg_json=$(curl -s -u "$token:" "$url/api/qualitygates/project_status?projectKey=$project_key$bq")
measures_json=$(curl -s -u "$token:" "$url/api/measures/component?component=$project_key$bq&metricKeys=bugs,vulnerabilities,code_smells,security_hotspots,coverage,duplicated_lines_density,ncloc,reliability_rating,security_rating,sqale_rating,sqale_index")
issues_json=$(curl -s -u "$token:" "$url/api/issues/search?componentKeys=$project_key$bq&resolved=false&ps=500")

report="$output_dir/$commit_full.md"

# ---------- render report via python3 ----------
export CC_QG="$qg_json" CC_MEASURES="$measures_json" CC_ISSUES="$issues_json"
export CC_URL="$url" CC_PK="$project_key" CC_REPO="$repo" CC_TASK="$task_id"
export CC_COMMIT="$commit_full" CC_BRANCH="$branch" CC_AUTHOR="$commit_author"
export CC_DATE="$commit_date" CC_SUBJECT="$commit_subject" CC_INCL="$inclusions"
export CC_REPORT="$report" CC_SHELLCHECK="$shellcheck_ok"

python3 - <<'PY'
import os, json

def j(env):
    try: return json.loads(os.environ.get(env, "") or "{}")
    except Exception: return {}

qg = j("CC_QG"); measures = j("CC_MEASURES"); issues = j("CC_ISSUES")
url=os.environ["CC_URL"]; pk=os.environ["CC_PK"]

RATING={"1.0":"A","2.0":"B","3.0":"C","4.0":"D","5.0":"E"}
m={x["metric"]:x.get("value","") for x in measures.get("component",{}).get("measures",[])}
gate=qg.get("projectStatus",{}).get("status","UNKNOWN")

lines=[]
lines.append("# SonarQube Codecheck 报告\n")
lines.append("## 提交信息\n")
lines.append("| 项目 | 内容 |")
lines.append("| --- | --- |")
lines.append(f"| 仓库 | {os.environ['CC_REPO']} |")
lines.append(f"| Commit | `{os.environ['CC_COMMIT']}` |")
lines.append(f"| 分支 | {os.environ.get('CC_BRANCH','')} |")
lines.append(f"| 作者 | {os.environ.get('CC_AUTHOR','')} |")
lines.append(f"| 时间 | {os.environ.get('CC_DATE','')} |")
lines.append(f"| 说明 | {os.environ.get('CC_SUBJECT','')} |")
lines.append("")
br=os.environ.get("CC_BRANCH","")
dash=f"{url}/dashboard?id={pk}" + (f"&branch={br}" if br else "")
lines.append(f"- SonarQube: {url}")
lines.append(f"- Project Key: `{pk}`")
lines.append(f"- Dashboard: {dash}")
lines.append(f"- 分析任务 ID: `{os.environ['CC_TASK']}`")
lines.append("")

lines.append("## Quality Gate\n")
lines.append(f"**状态: {gate}**\n")
conds=qg.get("projectStatus",{}).get("conditions",[])
if conds:
    lines.append("| 指标 | 状态 | 比较 | 阈值 | 实际值 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for c in conds:
        lines.append(f"| {c.get('metricKey')} | {c.get('status')} | {c.get('comparator')} | {c.get('errorThreshold')} | {c.get('actualValue')} |")
    lines.append("")

lines.append("## 度量指标\n")
lines.append("| 指标 | 值 |")
lines.append("| --- | --- |")
lines.append(f"| Bugs | {m.get('bugs','-')} (Reliability: {RATING.get(m.get('reliability_rating',''),'-')}) |")
lines.append(f"| Vulnerabilities | {m.get('vulnerabilities','-')} (Security: {RATING.get(m.get('security_rating',''),'-')}) |")
lines.append(f"| Security Hotspots | {m.get('security_hotspots','-')} |")
lines.append(f"| Code Smells | {m.get('code_smells','-')} (Maintainability: {RATING.get(m.get('sqale_rating',''),'-')}) |")
lines.append(f"| 技术债 (sqale_index) | {m.get('sqale_index','-')} min |")
lines.append(f"| 重复率 | {m.get('duplicated_lines_density','-')}% |")
lines.append(f"| 覆盖率 | {m.get('coverage','-')}% |")
lines.append(f"| 有效代码行 (ncloc) | {m.get('ncloc','-')} |")
lines.append("")

lst=issues.get("issues",[])
lines.append(f"## Issues (共 {issues.get('total',len(lst))})\n")
if lst:
    lines.append("| 严重度 | 类型 | 文件 | 行 | 说明 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for i in sorted(lst,key=lambda x:(x.get("component",""),x.get("line",0))):
        comp=i.get("component","").split(":")[-1]
        lines.append(f"| {i.get('severity')} | {i.get('type')} | {comp} | {i.get('line','-')} | {i.get('message','').replace('|','\\|')} |")
    lines.append("")
else:
    lines.append("未发现任何 issue。\n")

incl=os.environ.get("CC_INCL","")
if incl:
    lines.append("## 扫描范围\n")
    lines.append("本次仅分析最新提交中改动的源码文件（数据/图片/二进制文件已排除）：\n")
    for f in incl.split(","):
        lines.append(f"- `{f}`")
    lines.append("")

notes=[]
if os.environ.get("CC_SHELLCHECK","1") != "1":
    notes.append("**shellcheck 未安装**：扫描环境缺少 `shellcheck`，`.sh` 文件的 Shell 静态检查被跳过，Shell 相关问题未纳入本报告。在扫描机安装 shellcheck 后重跑可补全。")
if m.get("coverage","") in ("0.0",""):
    notes.append("**覆盖率数据缺失/为 0**：本次未提供覆盖率报告 (jacoco/cobertura)。")
if notes:
    lines.append("## 说明与注意事项\n")
    for i,n in enumerate(notes,1):
        lines.append(f"{i}. {n}")
    lines.append("")

with open(os.environ["CC_REPORT"],"w") as f:
    f.write("\n".join(lines)+"\n")
print("REPORT_WRITTEN:"+os.environ["CC_REPORT"])
PY

rc=$?
rm -f "$scan_log"
if [[ $rc -ne 0 ]]; then
  log_err "report generation failed"
  exit 1
fi
log_info "Report written: $report"
echo "$report"

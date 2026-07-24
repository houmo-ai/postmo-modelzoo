---
name: gerrit-review-publish
description: '通过 SSH 将 code review 评论（含 inline comments）发布到内网 Gerrit。Use when posting review results to Gerrit, publishing inline comments on specific code lines, or when asked to "发 review 到 gerrit"、"post review"、"publish comments"。覆盖内网访问限制、SSH 连接、JSON 构造、行号定位、inline comment 发布全流程。'
---

# Gerrit Review 发布

本 skill 将 code review 结果通过 Gerrit SSH API 发布到内网 Gerrit，重点解决内网无法直连、shell 转义陷阱、inline comment 发布等问题。

**Announce at start:** "我正在使用 gerrit-review-publish skill 来发布 review 到 Gerrit。"

## When to Use This Skill

适用场景：

- 需要把 code review 结果发布到 Gerrit change 上
- 需要在 Gerrit 上发布行内评论（inline comments）锚定到具体文件和行号
- 需要通过 SSH 与内网 Gerrit 交互（查询 change、fetch diff、post review）

不适用：

- 做 code review 本身（应使用 `imodelzoo-code-review` 及其子 skill）
- 修改 Gerrit 配置或权限
- 与 GitHub PR 交互（应使用 `gh` CLI）

## 前置条件

1. 已配置 SSH key 可访问 Gerrit（`ssh -p 29418 gerrit.houmo.ai gerrit version` 能成功）
2. git remote 已指向 Gerrit（`origin` 为 `ssh://<user>@gerrit.houmo.ai:29418/toolchain/imodelzoo`）
3. 已完成 code review，准备好评论内容

## Step-by-Step 工作流

### Step 1: 获取 Change 信息

内网 Gerrit 无法通过 HTTP/HTTPS 直接访问（WebFetch 被阻断），必须通过 SSH API。

```bash
# 查询 change 基本信息（commit message、files、current revision）
ssh -p 29418 gerrit.houmo.ai gerrit query <CHANGE_NUMBER> \
  --format=JSON --current-patch-set --files --commit-message
```

从返回的 JSON 中提取：

- `currentPatchSet.revision`：当前 patch set 的 commit SHA
- `currentPatchSet.ref`：fetch 用的 ref（如 `refs/changes/24/38724/1`）
- `currentPatchSet.files`：变更文件列表
- `branch`：目标分支

### Step 2: Fetch 并查看 Diff

```bash
# fetch change 到本地
git fetch origin <REF>   # 如 refs/changes/24/38724/1

# 查看完整 diff
git diff <PARENT_SHA>..<REVISION_SHA>
```

**行号定位关键**：inline comment 的 `line` 字段必须是 **patched 文件中的行号**（即 revision SHA 对应的文件），不是 diff 中的 +/- 行号。用以下方式确认：

```bash
# 查看 patched 文件中目标行的内容，确认行号
git show <REVISION_SHA>:<FILE_PATH> | sed -n '<START>,<END>p'

# 或用 grep 定位具体行
git show <REVISION_SHA>:<FILE_PATH> | grep -n "PATTERN"
```

### Step 3: 构造 Review JSON

**必须使用 Python 构造 JSON**，不要用 shell heredoc 或变量拼接——这是核心防坑点（详见下方"Shell 转义陷阱"）。

```python
import json

review = {
    "message": "整体 cover message（摘要级别，不需要重复 inline comment 内容）",
    "labels": {"Code-Review": 0},   # 0=neutral, +1=LGTM, -1=需要修改
    "comments": {
        "<FILE_PATH>": [           # 相对于仓库根目录的路径
            {
                "line": <LINE_NUMBER>,  # patched 文件中的绝对行号
                "message": "行内评论内容..."
            },
            # 可在同一文件添加多条评论
        ],
        # 可在多个文件添加评论
    }
}

print(json.dumps(review))
```

#### 关键字段说明

| 字段                       | 类型   | 说明                                                    |
| -------------------------- | ------ | ------------------------------------------------------- |
| `message`                  | string | Cover message，显示在 change 页面顶部，作为整体评审摘要 |
| `labels`                   | object | 评分标签，`Code-Review` 取值 -2..+2                     |
| `comments`                 | object | 行内评论，key 为文件路径，value 为评论数组              |
| `comments[path][].line`    | int    | **patched 文件中的行号**，不是 diff hunk 行号           |
| `comments[path][].message` | string | 评论内容，支持 `\n` 换行                                |

#### 行号注意事项

- `line` 是 **patched 文件中的绝对行号**，不是 diff 中的偏移
- 必须基于 `git show <REVISION>:<file>` 确认行号，不能凭 diff 猜测
- 如果评论针对新增文件的第 N 行，line 就是 N
- 如果要评论的是未修改的行（上下文行），line 也是该行在 patched 文件中的行号

### Step 4: 发布 Review

```bash
python3 -c "
import json
# ... 构造 review dict ...
print(json.dumps(review))
" | ssh -p 29418 gerrit.houmo.ai gerrit review <REVISION_SHA> --json -l Code-Review=0
```

**关键点**：

- 使用 `--json` 标志从 stdin 读取 JSON
- 同时用 `-l Code-Review=0` 传入评分（JSON 中的 `labels` 也会生效，但命令行参数作为 fallback）
- `<REVISION_SHA>` 是 Step 1 获取的 commit SHA

### Step 5: 验证发布结果

```bash
ssh -p 29418 gerrit.houmo.ai gerrit query <CHANGE_NUMBER> \
  --format=JSON --comments --current-patch-set
```

从返回的 `comments` 数组中确认：

- 最新评论的 `message` 包含你的 cover message
- 最新评论显示 "(N comments)"，N = 你的 inline comments 数量
- 每个 comment 的 `file` 和 `line` 与预期一致

## Shell 转义陷阱（已踩过的坑）

### 问题 1: `--message` 参数含空行导致参数分裂

```bash
# 错误：空行会被 shell 解释为参数分隔符
ssh ... gerrit review SHA --message "$(cat <<'EOF'
第一段

第二段
EOF
)" --label Code-Review=0
# 报错：fatal: "第二段" is not a valid patch set
```

**原因**：Gerrit SSH API 的 `--message` 参数不能包含空行，空行后的内容会被解释为位置参数。

### 问题 2: 中文/特殊字符在 shell 中的转义

```bash
# 错误：heredoc + 中文 + 引号嵌套极易出错
ssh ... gerrit review SHA --message "P1-1: 建议 int() 转换..."
```

**原因**：多层 shell 引用（bash -> ssh -> gerrit）下，引号、反斜杠、中文的转义极难保证一致。

### 解决方案：Python + JSON stdin

```bash
# 正确：Python 构造 JSON，通过 --json 从 stdin 传入
python3 -c "
import json
review = {'message': '...', 'labels': {...}, 'comments': {...}}
print(json.dumps(review))
" | ssh -p 29418 gerrit.houmo.ai gerrit review SHA --json -l Code-Review=0
```

**为什么这样做**：

1. Python `json.dumps` 处理所有转义（换行、引号、Unicode），无需手动处理
2. `--json` 从 stdin 读取，不经过 shell 参数解析，避免空行分裂问题
3. JSON 结构化，inline comments 可以自然表达为嵌套 dict
4. Python heredoc 内可以自由使用引号和中文

## 与 imodelzoo-code-review 的集成

本 skill 不负责 code review 本身，只负责发布。典型工作流：

1. 使用 `imodelzoo-code-review`（及子 skill）完成评审，产出结构化 review 报告
2. 从报告中提取 P0/P1/P2 条目，映射为 inline comments
3. 使用本 skill 发布到 Gerrit

映射规则：

- 每个 P0/P1/P2 条目 → 一条 inline comment，锚定到对应文件和行号
- Cover message → 整体摘要（分类、总评、是否阻塞）
- 需要评分时，通过 `labels` 字段设置 `Code-Review` 值

## Troubleshooting

| 问题                                    | 原因                                     | 解决方案                                            |
| --------------------------------------- | ---------------------------------------- | --------------------------------------------------- |
| `fatal: "XXX" is not a valid patch set` | `--message` 含空行或特殊字符导致参数分裂 | 改用 `--json` + stdin                               |
| WebFetch 无法访问 gerrit.houmo.ai       | 内网 Gerrit 不对外暴露                   | 使用 SSH API 代替                                   |
| `gerrit query` 返回空                   | change number 未加引号或权限不足         | 确认 SSH key 配置；用 `gerrit query "38724"`        |
| inline comment 未出现在对应行           | `line` 字段用了 diff 行号而非文件行号    | 用 `git show REV:file \| grep -n` 确认              |
| JSON 解析失败                           | Python heredoc 中引号嵌套错误            | 确保 Python 代码内部字符串用单引号，JSON 值用双引号 |
| 评论发到了 cover message 但没有 inline  | `comments` 字段缺失或路径不正确          | 确认文件路径相对于仓库根目录；确认 `--json` flag    |

## References

- Gerrit SSH API: `ssh -p 29418 gerrit.houmo.ai gerrit review --help`
- Gerrit Query API: `ssh -p 29418 gerrit.houmo.ai gerrit query --help`
- 行内评论 JSON 格式详见：`references/gerrit-review-json-schema.md`

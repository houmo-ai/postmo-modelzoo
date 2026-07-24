# Gerrit Review JSON Schema

通过 `--json` flag 传入 stdin 的 review 输入的完整 JSON 结构。

## Top-level 对象

```json
{
  "message": "string (optional)",
  "labels": { "Label-Name": int },
  "comments": { "file/path": [ ... ] },
  "tag": "string (optional)",
  "notify": "NONE | OWNER | OWNER_REVIEWERS | ALL"
}
```

## 字段说明

### message

- 类型：`string`
- 可选
- Cover message，显示在 change 页面的评论列表顶部
- 建议内容：分类 + 摘要 + 总评结论（如"无 P0 阻塞，建议处理 P1-1 后合入"）
- 不需要重复 inline comment 的内容，后者直接锚定在代码行上

### labels

- 类型：`object`（key=label name, value=整数值）
- 可选
- iModelzoo 常用标签：

| Label         | 值  | 含义                               |
|---------------|----|----------------------------------|
| `Code-Review` | +2 | 批准合入（不要使用)                       |
| `Code-Review` | +1 | LGTM 但非最终批准（review 没有发现任何需修改的地方) |
| `Code-Review` | 0  | 中立（仅评论）                          |
| `Code-Review` | -1 | 需要修改后再审（有P0级别的问题）                |
| `Code-Review` | -2 | 阻塞合入（不要使用)                       |
| `Verified`    | +1 | 验证通过（不要使用)                       |
| `Verified`    | -1 | 验证失败（不要使用)                       |

### comments

- 类型：`object`（key=文件路径, value=评论数组）
- 可选
- 文件路径**相对于仓库根目录**，如 `models/llm/qwen3.5/test.sh`
- 每条评论的结构：

```json
{
  "line": 1711,
  "message": "评论内容，支持 \\n 换行"
}
```

#### line 字段

- 类型：`int`
- **patched 文件中的绝对行号**（即 revision commit 对应的文件版本）
- 不是 diff 中的 `@@ -a,b +c,d @@` 偏移量
- 不是旧文件的行号
- 确认方式：`git show <REVISION>:<file> | grep -n "PATTERN"`

#### message 字段

- 类型：`string`
- 支持 `\n` 换行
- Python 中写多行字符串时，`json.dumps` 会自动转义为 `\n`

### tag

- 类型：`string`
- 可选
- 为此次 review 打标签，在 Gerrit UI 中显示

### notify

- 类型：`string` 枚举
- 可选
- 邮件通知范围：`NONE`、`OWNER`、`OWNER_REVIEWERS`、`ALL`

## 命令行发布

```bash
python3 -c "
import json
review = { ... }  # 构造上述结构
print(json.dumps(review))
" | ssh -p 29418 gerrit.houmo.ai gerrit review <REVISION_SHA> --json -l Code-Review=0
```

注意：
- `--json` 表示从 stdin 读取 JSON
- `-l Code-Review=0` 作为 fallback 评分（当 JSON 中未设置 labels 时生效）
- 两者同时存在时，JSON 中的 `labels` 优先

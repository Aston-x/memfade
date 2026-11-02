# memfade · AstrBot 插件

给 AstrBot 用的长期记忆：**会记得，也会忘得恰到好处。**

两个时机：

* `on_llm_request`——把召回的记忆追加到 system prompt，模型这一轮就能看见；
* `on_llm_response`——这一轮说完之后，把新内容抽取成记忆。

## 装法

把 `adapters/astrbot/` 整个目录放进 AstrBot 的插件目录，启用即可。

## 指令

| 指令 | 作用 |
|---|---|
| `记忆 状态` | 这个会话记住了多少东西 + 最近记下的内容 |
| `记忆 搜 <关键词>` | 检索（必须真命中） |
| `记忆 整理` | 立刻跑一轮整理 |
| `记忆 置顶 <内容>` | 让它永不淡化 |
| `记住 <内容>` | 显式记住 |
| `忘掉 <内容>` | 忘掉一条 |

## 配置

| 键 | 默认 | 说明 |
|---|---|---|
| `db_path` | `memfade.db` | SQLite 记忆库路径 |
| `top_k` | 6 | 每轮最多召回多少条 |
| `max_inject_tokens` | 600 | 注入 token 预算 |
| `echo_max` | 2 | 已淡出的旧事最多注入几条 |
| `extract_enabled` | true | 是否抽取记忆 |
| `extract_every_n` | 3 | 每隔多少轮抽取一次 |
| `min_importance` | 0.35 | 低于此不入库 |
| `use_embedding` | true | 是否用向量检索 |

## 注意

**作用域**：有群号就是群聊、按群隔离；私聊按发送者隔离。

抽取**不阻塞**：它在 `on_llm_response` 之后跑，且丢后台任务，
LLM 慢也不挡住这一轮回复。

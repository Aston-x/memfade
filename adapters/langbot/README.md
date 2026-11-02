# memfade · LangBot 插件

给 LangBot 用的长期记忆：**会记得，也会忘得恰到好处。**

两个组件：

* `EventListener`——在 `PromptPreProcessing` 时把召回的记忆注入
  `default_prompt`，并在这一轮结束后抽取新记忆；
* `Command`——指令：查状态、检索、整理、置顶、记住、忘掉。

## 装法

把 `adapters/langbot/` 整个目录作为 LangBot 插件安装（`manifest.yaml` 里有
`spec.components`，两个组件都在里面）。

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

见 `manifest.yaml` 的 `spec.config`：数据库路径、召回条数、注入预算、
抽取节奏、显著性门槛、LLM 与嵌入接口。

## 注意

⚠ **组件清单不能少**。LangBot 靠 `manifest.yaml` 的 `spec.components`
发现组件——少了那一段，插件能装上、能启用，但**什么都不做**
（事件监听器与指令都进不去运行时）。

**作用域**用会话名（`session.session_name`），群聊与私聊不会串。

抽取**不阻塞**：丢后台任务，LLM 慢也不挡住回复。

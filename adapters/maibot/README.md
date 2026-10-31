# 记忆淡化引擎 · MaiBot 插件

给麦麦（许杏玖）用的长期记忆插件。**会记得，也会忘得恰到好处。**

## 装法

把整个目录复制到 MaiBot 的 `plugins/` 下（目录名别叫 `memfade`，见下方注意事项），
然后在插件管理里启用。

```powershell
cd G:\GitHub\memfade
python scripts\build_plugin.py --deploy 'G:\AI\许杏玖-MaiBot\MaiBot\plugins'
```

**不需要配 API key**：抽取用的大模型与向量模型，直接复用宿主
`model_config.toml` 里已经配好的 `memory` 与 `embedding` 任务。

## 它做什么

每条消息进来时：

1. 把「关于对方」的稳定事实（名字、偏好、约定）常驻注入；
2. 检索这轮相关的记忆，按淡化档位分两层注入：
   - **事实层**——现在仍然成立的知识；
   - **回声层**——已经开始淡出的旧事，带「可能已经变了」标记，且限量；
3. 每 3 轮从对话里抽一次新记忆；
4. 定期做一轮整理：重算淡化档位、压缩旧记忆成一行摘要、合并重复、仲裁矛盾。

## 命令

| 命令 | 作用 |
|---|---|
| `/记忆` | 查看当前会话记住了多少东西 |
| `/记忆 搜 <关键词>` | 检索 |
| `/记忆 整理` | 立刻跑一轮整理 |
| `/记住 <内容>` | 显式记住（显著性 0.95） |
| `/忘掉 <片段>` | 忘掉一条 |

## WebUI

插件管理里有两个页面：**记忆引擎**（总览 + 淡化档位分布 + 最近记忆）
和 **记忆检索**。

## 配置

| 键 | 默认 | 说明 |
|---|---|---|
| `plugin.enabled` | false | 总开关 |
| `memory.inject_enabled` | true | 是否注入 |
| `memory.max_inject_tokens` | 600 | 注入 token 预算 |
| `memory.top_k` | 8 | 每轮召回条数 |
| `memory.echo_max` | 3 | 回声层最多几条 |
| `memory.profile_items` | 6 | 画像最多几条 |
| `memory.extract_every_n` | 3 | 每 N 轮抽取一次 |
| `memory.min_importance` | 0.35 | 低于此不入库 |
| `memory.cross_session` | false | 是否允许跨会话共享 |
| `memory.llm_task_name` | memory | 宿主模型任务名 |
| `fade.recall_alpha` | 0.35 | 被想起时的强度回升幅度 |
| `fade.archive_after_days` | 0 | 归档后多少天物理删（0=永不） |
| `fade.maintain_interval_minutes` | 60 | 整理间隔 |

## ⚠ 两个坑

**插件目录名不能叫 `memfade`。**
宿主加载插件时会把 `plugins/` 塞进 `sys.path`，目录同名会把自己遮蔽掉，
报 `cannot import name 'MemoryDraft' from 'memfade'`。
本插件用的目录名是 `memfade-plugin`。

**改插件代码不需要重启 MaiBot**，宿主有文件监听会自动重载。
但新增插件目录时会触发一次 Runner 重启。

## 记忆存哪

`data/plugins/memfade.memory/memfade.db`（SQLite 单文件）。
作用域键是 MaiBot 的 `session_id`——**一个群一份、私聊按人一份**。

## 原理

见仓库的 `docs/DECAY.md`（淡化算法）与 `docs/ARCHITECTURE.md`（架构）。



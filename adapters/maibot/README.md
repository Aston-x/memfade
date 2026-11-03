# memfade · MaiBot 长期记忆插件

给聊天机器人用的长期记忆：**会记得，也会忘得恰到好处。**

记忆分两层——**事实层**放现在还成立的知识，**回声层**放已经开始淡出的旧事
（带「可能已经变了」标记，且限量）。越重要、越常被想起的记忆留得越久；
一次性事件靠自然退场淘汰，不会「总提旧事」。

## 装法

把整个目录复制到 MaiBot 的 `plugins/` 下，在插件管理里启用。

```powershell
cd <memfade 仓库>
python scripts\build_plugin.py --deploy '<MaiBot 的 plugins 目录>'
```

**不需要配 API key**：抽取用的大模型与向量模型直接复用宿主
`model_config.toml` 里已经配好的 `memory` 与 `embedding` 任务。

> ⚠ **插件目录名不能叫 `memfade`。**
> 宿主加载插件时会把 `plugins/` 塞进 `sys.path`，目录同名会把自己遮蔽掉，
> 报 `cannot import name 'MemoryDraft' from 'memfade'`。
> 本插件用的目录名是 `memfade-plugin`。

## 它做什么

每条消息进来时：

1. 把「关于对方」的稳定事实常驻注入（**画像块**）；
2. 检索这轮相关的记忆，按淡化档位分两层注入（**事实层 / 回声层**）；
3. 每 N 轮从对话里抽一次新记忆；
4. 定期整理：重算淡化档位、压缩旧记忆成一行摘要、合并重复、仲裁矛盾。

记忆库只记**关于对话里那个人**的事：机器人的自述、群公告、转发内容都不入库
（详见仓库 `docs/DECAY.md` 第 11 节）。

## 命令

| 命令 | 作用 |
|---|---|
| `/记忆` | 当前会话记住了多少东西 + 最近记下的内容 |
| `/记忆 搜 <关键词>` | 检索（必须真命中，不掺无关记忆） |
| `/记忆 整理` | 立刻跑一轮整理 |
| `/记忆 置顶 <内容>` | 让它永不淡化（`取消置顶` 反向） |
| `/记忆 修正 <老内容> => <新内容>` | 改掉记错的那条（抽取难免出错，得有补救口） |
| `/记忆 接管` | 看长期记忆归谁管（有没有别的记忆系统还在跑） |
| `/记住 <内容>` | 显式记住 |
| `/忘掉 <片段>` | 忘掉一条 |

## 模型能用的工具

| 工具 | 作用 |
|---|---|
| `remember_something` | 让模型自己写一条记忆（用户明确要求记住时） |
| `recall_memories` | 让模型主动检索长期记忆（「我以前说过什么」这类问题） |

## WebUI

插件管理里有两个页面：**记忆引擎**（总览 + 淡化档位分布 + 最近记忆 + 体检）
和 **记忆检索**。

「最近写入的记忆」表里带**编号**与**依据原话**两列——
依据原话就是这条记忆在对话里的出处（用户自己说的那句）。
看到记错了，在「改掉记错的那条」卡片里填编号与新内容就能改；
也能直接删掉那条。

## 配置

| 键 | 默认 | 说明 |
|---|---|---|
| `plugin.enabled` | false | 总开关 |
| `memory.inject_enabled` | true | 是否注入 |
| `memory.max_inject_tokens` | 600 | 注入 token 预算 |
| `memory.top_k` | 8 | 每轮召回条数 |
| `memory.echo_max` | 3 | 回声层最多几条 |
| `memory.profile_items` | 6 | 画像最多几条 |
| `memory.extract_enabled` | true | 是否抽取 |
| `memory.extract_every_n` | 3 | 每 N 轮抽取一次 |
| `memory.flush_idle_minutes` | 10 | 安静多久后把攒下的消息抽掉 |
| `memory.min_importance` | 0.45 | 稳定事实的入库门槛（事件类另用更低的门槛） |
| `memory.store_evidence` | true | 给每条记忆存「依据原话」，写入前的闸门靠它核对 |
| `memory.cross_session` | false | 是否允许跨会话共享 |
| `memory.self_names` | `[]` | 机器人自己的名字（留空则自动读宿主 `bot_config` 的 `nickname`） |
| `memory.focus_subjects` | `[]` | 只记这些主语的事（留空=不限制） |
| `memory.subject_hint` | `""` | 抽取提示词里的人物提示，形如 `（小明 / 小红）` |
| `memory.require_person_subject` | true | 主语必须是人，公共知识不入库 |
| `memory.strip_quotes` | true | 抽取前剥掉被引用的原文与合并转发 |
| `memory.llm_task_name` | memory | 宿主模型任务名 |
| `memory.db_filename` | memfade.db | 记忆库文件名 |
| `fade.recall_alpha` | 0.35 | 被想起时的强度回升幅度 |
| `fade.archive_after_days` | 0 | 归档后多少天物理删（0=永不） |
| `fade.maintain_interval_minutes` | 60 | 整理间隔 |

**零配置可用**：`self_names` 留空时自动读宿主配置；
`focus_subjects` 留空时记「这次对话里出现过的人」；
`subject_hint` 留空时抽取器用对话里出现的名字。

## 记忆存哪

`data/plugins/memfade.memory/memfade.db`（SQLite 单文件，含全文索引与向量）。

作用域键是宿主的 `session_id`——**一个群一份、私聊按人一份**。
所有读取路径共用同一条作用域过滤：没有「找不到就退回全局」的兜底，
私聊的记忆不会漏进群里。

## 自带的运维工具

插件目录下的 `memfade_tools/`：

| 工具 | 作用 |
|---|---|
| `backup_memory_stack.py` | 备份整栈记忆（插件库 + 宿主旧记忆 + 各插件状态 + 可读 JSON） |
| `retire_legacy_memory.py` | 让宿主旧记忆系统退场（memfade 独占长期记忆） |
| `clean_library.py` | 体检式清理：按规则/名单删改，重复条目合并 |
| `audit_db.py` | 审一份库：作用域分布、索引与正文是否一致、疑似测试数据 |
| `verify_against_chat.py` | 拿记忆回聊天记录里核对，判断有没有依据 |

```powershell
python memfade_tools\backup_memory_stack.py --host-root '<MaiBot 根目录>'
python memfade_tools\retire_legacy_memory.py --host-root '<MaiBot 根目录>' --dry-run
```

## 接管长期记忆

宿主内置的长期记忆系统（A_Memorix）如果还开着，会和本插件**并行写、并行读**，
而旧系统的段落**没有淡化机制**——写进去就赖着不走。

插件启动时会检查并给出提示，`/记忆 接管` 也能随时查看现状。
退场两步：

1. `config/bot_config.toml` 里把下面几项改成 `false`：
   `a_memorix.integration.enable_memory_query_tool`、
   `person_fact_writeback_enabled`、`chat_summary_writeback_enabled`、
   `enable_person_profile_injection`、`heuristic_memory_recall_enabled`，
   以及 `a_memorix.plugin.enabled`；
2. 跑 `memfade_tools/retire_legacy_memory.py` 清掉旧数据（会自动备份）。

## 迁移旧记忆库

已经有别的记忆系统？导出成 JSON，再导入：

```powershell
python -m memfade_cli --db <新库> import --file <导出的.json>
```

或用面板的「导入」动作。导入保留种类、显著性、三元组与**淡化状态**
（强度与锚点）——不保留的话，搬过来的记忆会「重新变年轻」。

## 卸载

停用插件即可，**记忆数据不会丢**（在 `data/plugins/memfade.memory/`）。
要彻底清掉就删那个目录；迁移前先跑一次 `backup_memory_stack.py`。

## 已知坑

**插件目录名不能叫 `memfade`**（见上）。

**改插件代码不需要重启 MaiBot**，宿主有文件监听会自动重载；
但新增插件目录时会触发一次 Runner 重启。

**诊断会改库**：注入会强化被注入的记忆（强度回升、锚点重置），
而「被想起次数」还参与间隔重复成长——诊断一次等于给记忆续命。
所以体检/自检要么用库副本，要么显式 `reinforce=False`。

## 原理

仓库的 `docs/DECAY.md`（淡化算法）与 `docs/ARCHITECTURE.md`（架构）。

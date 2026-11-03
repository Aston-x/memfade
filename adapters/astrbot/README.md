# memfade · AstrBot 插件

给 AstrBot 用的长期记忆：**会记得，也会忘得恰到好处。**

两个时机：

* `on_llm_request`——把召回的记忆追加到 system prompt，模型这一轮就能看见；
* `on_llm_response`——这一轮说完之后，把新内容抽取成记忆。

---

## 装法

把本目录整个放进 AstrBot 的插件目录：

```
<astrbot>/data/plugins/astrbot_plugin_memfade/
```

**目录名不能改**，它同时决定了三件事：

| 由目录名决定 | 路径 |
|---|---|
| 配置文件 | `data/config/astrbot_plugin_memfade_config.json` |
| 数据目录 | `data/plugin_data/astrbot_plugin_memfade/`（记忆库在这里） |
| 导入名 | `data.plugins.astrabot_plugin_memfade` |

改名字＝用户的配置和记忆全部失联。

> 目录名必须是合法 Python 标识符（**不能带横杠**）。AstrBot 以
> `data.plugins.<目录名>` 导入，带横杠会直接导入失败。

---

## 作用域模型：三层

AstrBot 没有「群内按人区分的会话」这个概念——`umo` 在群聊里对全群成员是同一个值。
所以隔离由适配器自己做：

| 层 | 作用域键 | 语义 |
|---|---|---|
| 群级公共 | `astrbot:group:<群号>` | 全群共享 |
| 群内人级私密 | `astrbot:group_user:<群号>#<用户号>` | 只属于该成员 |
| 私聊 | `astrbot:private:<用户号>` | 1:1，只有本人可见 |

**读取**：群聊读「群级公共 + 当前说话人的人级私密」两层；私聊只读自己那一层。
私聊永远不会读、写任何 `group_*` 作用域——宁可少记，也不能把私聊内容漏进群里。

**写入**：按记忆的主语路由——

* 主语命中本群成员 → 写**人级私密层**
* 主语等于群别名（默认「本群」）→ 写**群级公共层**
* 其它（陌生昵称 / 空 / 泛称）→ **丢弃**

第三条是丢弃而不是退到群层：退群层等于把某个人的私密事实公开。

召回时两层各自召回 → 按记忆 ID 去重 → 按分数合并 → **只组装一次**。
（不能两次组装再拼文本：会出两个 `[记忆]` 头、token 预算翻倍、MMR 各做各的导致
两层近重复内容同时出现。）

---

## 指令

| 指令 | 作用 |
|---|---|
| `记忆 状态` | 各层记住了多少 + 最近记下的 + 抽取引擎诊断 |
| `记忆 搜 <关键词>` | 检索（必须真命中，两层都搜） |
| `记忆 整理` | 立刻跑一轮整理 |
| `记忆 置顶 <内容>` | 让它永不淡化 |
| `记忆 修正 老内容 => 新内容` | 改掉记错的那条 |
| `记住 <内容>` | 显式记住（记给你自己） |
| `群记 <内容>` | 记住一件对全群都成立的事 |
| `忘掉 <内容>` | 忘掉一条记忆 |

`记忆 状态` 末尾有诊断信息：

```
私聊层：记住了 3 条（活跃 3 / 淡出 0 / 摘要 0 / 归档 0）
抽取引擎：LLM（成功 12 / 降级 0 / 报错 0）
向量检索：已启用
```

「抽取引擎：规则兜底（没接 LLM）」说明宿主没配 provider，抽取靠规则——
能记但记得糙。向量检索同理，宿主没配 embedding provider 就自动降级为关键词检索。

---

## 配置

配置项共 50 项，完整列表见 `_conf_schema.json`，AstrBot 面板会自动渲染。

最常调的几个：

| 键 | 默认 | 说明 |
|---|---|---|
| `use_host_llm` | true | 用宿主已配的模型做抽取，不用客户再填 API key |
| `llm_provider_id` | 空 | 指定用哪个 provider。留空＝当前会话正在用的那个 |
| `use_host_embedding` | true | 用宿主的 embedding provider |
| `min_importance` | 0.45 | 入库门槛。调低会让噪声进库 |
| `top_k` | 8 | 每轮最多召回几条 |
| `max_inject_tokens` | 600 | 注入 token 预算 |
| `recall_alpha` | 0.35 | 淡化曲线：被想起时残留多少学习强度 |
| `archive_after_days` | 0 | 归档后多少天物理删除。0＝永不删除 |
| `maintain_interval_minutes` | 60 | 后台整理间隔。**关掉的话摘要压缩、合并、淘汰都不会发生** |
| `cross_scope` | false | 是否允许「永远别忘」跨会话共享 |
| `group_subject` | 本群 | 群级公共层用的主语词 |
| `flush_idle_minutes` | 10 | 安静多久后冲刷抽取缓冲 |

> ⚠ **`cross_scope` 默认是关的，别随便开。** `cross_scope=1` 的记忆对**所有**
> 作用域可见——多人群聊里 A 说「永远别忘我的生日」会漏给 B、C 和别的群。

---

## 常见故障

| 现象 | 原因 |
|---|---|
| 面板里看不到日志 | 正常——插件用 `astrbot.api.logger`，日志在 WebUI 的日志面板，不在终端 |
| `记忆 状态` 显示「没接 LLM」 | 宿主没配 provider，或 provider 类型不是 Chat_Completion |
| 一直「向量检索：未启用」 | 宿主没配 embedding provider，会自动降级为关键词检索 |
| 配了不生效 | 配置文件名 = 目录名 + `_config.json`，改过目录名就找不到 |
| 抽不出记忆 | LLM 报错时看日志里的 `[memfade] LLM 抽取失败`，会退化成规则抽取 |
| 记忆不区分人 | 确认装的是 0.4.0 以上（0.3.x 只有群级一层） |

---

## 运维工具

`memfade_tools/` 是随插件发货的独立 CLI（在自己进程里跑，不依赖宿主）：

```bash
python memfade_tools/audit_db.py --db <记忆库路径>     # 体检
python memfade_tools/acceptance.py                     # 跑一遍完整闭环自检
python memfade_tools/clean_library.py --db <路径>      # 清理污染条目
python memfade_tools/edit_memory.py --db <路径> --patch <清单.json>  # 批量修
```

（`backup_memory_stack.py` 和 `retire_legacy_memory.py` 写死了 MaiBot 的目录结构，
在 AstrBot 下没有意义，所以不发。）

---

## 注意

* **抽取不阻塞**：它在 `on_llm_response` 之后跑，且丢后台任务，LLM 慢也不挡住这一轮回复。
* **导入用相对导入**：插件以 `data.plugins.<目录名>` 导入，核心副本跟其它插件物理隔离。
  顶层 `import memfade` 绝不能出现——`sys.modules` 是进程级全局缓存，谁先 import
  谁就污染全进程，而且不同副本里的 `MemoryKind` 枚举虽然 `==` 相等但 `is` 不相等，
  核心里十几处身份比较会静默走错分支。

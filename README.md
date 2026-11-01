# memfade · 记忆淡化引擎

给聊天机器人用的**可移植长期记忆插件**。

> 别的记忆库负责记住「发生过什么」，
> memfade 还管「这件事**还该不该被想起来**」。

```
核心 4032 行 · 零第三方依赖 · 单文件 SQLite · 191 项测试
适配器 2128 行 · 覆盖 MaiBot / LangBot / AstrBot / NoneBot2 / Koishi + HTTP 侧车
```

## 为什么需要它

聊天机器人的记忆，长期只有两个极端：只用上下文窗口，隔一天就失忆；
或者全塞进向量库，噪声越来越大、检索越来越慢，还说不清「为什么想起这件事」。

更具体的三个真实症状：

1. **记忆里重复语段太多**——同一件事换个标点、换个语序存成好几条；
2. **总提起已经过去的小事**——一次性事件和稳定事实被同等对待，每次聊天都翻旧账；
3. **私聊内容漏进群里**——同一个机器人在不同会话间共享了不该共享的记忆。

memfade 就是照着这三条来设计的。

## 五个核心机制

### 1. 双轴模型：把「重要」和「持久」拆开

大多数记忆库把这两件事压进同一个字段，于是重要事实会跟着时间一起掉分。
memfade 坚持两条独立的轴：

* **`importance`（显著性）**——这条内容本身有多重要。只在写入时判定，之后永不随时间变化。
  它决定检索权重，以及这条记忆能活多久。
* **`strength`（强度）**——这条记忆现在还剩多少。随「距上次被想起的时间」衰减，
  每次被想起就回升。它回答的是「此刻还想不想得起来」。

### 2. 淡化梯度

淡化不是二元的「留 / 删」，是一条有档位的坡道：

| 档位 | 含义 | 还能被召回吗 | 注入时 |
|---|---|---|---|
| `ACTIVE` | 正常状态 | 是 | 作为「事实层」，当作现在仍然成立的知识 |
| `FADING` | 开始淡出 | 是 | 作为「回声层」，带「可能已经变了」的标记，且限量 |
| `GISTED` | 已压成一行摘要 | 是（极低权重） | 只带摘要，细节已不可考 |
| `ARCHIVED` | 归档 | 否 | 完全不进上下文，但原文仍留在库里可审计 |

旧事不会消失，但也不会占据日常对话。这是「总提旧事」的解药。

### 3. 幂等的艾宾浩斯衰减

```
stability = base_stability(种类)
            x (1 + importance_gain x importance)      重要 -> 活得久
            x min(ceiling, growth ^ recall_count)      复习 -> 更稳（间隔重复）

effective_strength(now) = strength x exp(-Δdays / stability)
```

关键工程细节：**始终按「锚点时刻」重算，而不是每天乘一次系数。**
维护任务跑一次、跑两次、隔一天再跑，结果完全一致。

实际保留率（重要性 0.5，实机面板里的同一张表）：

| 种类 | 7 天后 | 30 天后 | 90 天后 | 1 年后 |
|---|---|---|---|---|
| 事件 | 56% | 8% | 0.1% | 0% |
| 事实 | 96% | 85% | 61% | 13% |
| 喜好 | 94% | 76% | 44% | 4% |
| 关系 | 90% | 65% | 28% | 0.5% |

一次性事件几周就退场，稳定事实能记上半年——这就是「记得住但不忘太多」。

⚠ **「保留率」和「档位」是两件事**：保留率随时间平滑下降，档位只在跨过阈值时才变。
而且稳定性带间隔重复的成长系数（`growth ^ recall_count`，上限 12 倍）——
一条被反复提到 4 次的事实，稳定性会从 270 天涨到 2834 天。
这是设计意图（常用的记忆就该更牢固），但读表时要记得：**表里是「不再被提起时」的数字。**

### 4. 失败即闭的会话隔离

每条记忆带一个作用域键（`qq:group:1127155154` / `qq:private:1738157278`）。
检索时只有一条过滤条件：

```sql
scope_key = 当前会话  OR  cross_scope = 1
```

这里**没有**「找不到就退回全局」的兜底。宁可这条消息没有记忆可用，
也不能把别的会话的内容漏进来。跨会话共享必须显式开启。

### 5. 分层注入

召回结果按淡化档位分两层进 prompt：

```
[记忆]
- 许杏玖喜欢辣的食物
- 许杏玖住在上海

[记忆·旧事]
- （可能已经变了，仅供参考）上周三一起去吃了炸鸡
```

事实层给「现在仍然成立」的知识，回声层只在被直接问起时才带出来。

### 6. 画像常驻

「这个人是谁」（名字、偏好、约定、忌讳）**不等被问到才想起**——
它常驻注入一个精简小块。检索靠「这次问了什么」，
画像是「这个人是谁」，两回事。

```
[系统·关于对方]
- 许杏玖喜欢辣的食物
- 他讨厌别人迟到
```

画像与记忆块**不重复**：画像里的条目 id 会在组装记忆块时被排除。
画像只收**短事实**（≤120 字），几百字的百科段落归记忆块；
注入时单条记忆截到 120 字、画像单条截到 60 字。

### 7. 三条检索通道

| 通道 | 干什么 | 什么时候用 |
|---|---|---|
| 全文（BM25） | 中文二元组索引，「炸鸡」对得上「今天吃了炸鸡」 | 主力 |
| 单字兜底 | 内容里含这个字就算命中 | 全文一无所获、且查询短到像关键词时 |
| 向量 | 语义相似 | 宿主配了嵌入模型时 |

单字通道是**独立**的，不混进 BM25 索引。原因很实在：
同一个 FTS 表里混着单字，长文档涨的 token 比短文档多，
BM25 的长度归一化会被带偏——「重要的长约定」会被「炸鸡」这种短条目反超。
（有测试专门守着这条，见 `tests/test_retrieve.py`。）

而显式检索（「/搜」「/忘掉」、WebUI 搜索框）会打开「必须真的命中」这道门槛：
不把「最近写入」但跟关键词无关的记忆一并列出来。
「/忘掉」尤其重要——忘掉错的那条是不可逆的。

---

## 已实现的防护

| 风险 | 处理 |
|---|---|
| 私聊内容漏进群 | 所有读取路径共用一条作用域过滤，没有全局兜底 |
| 同一件事存了好几份 | 归一化哈希 + Dice 系数双重判定，合并时保留证据链 |
| 旧事反复被提起 | 一次性事件靠淡化梯度自然退场；旧事进「回声层」且限量 |
| 矛盾的事实并存 | 新事实取代旧事实，旧事实保留为证据链但不召回 |
| 记忆库无限增长 | 单会话上限 + 按档位淘汰最弱的，置顶的不动 |
| 提示词失效 | 显著性按种类在**代码层**夹紧，不只靠提示词约束 |
| 模型超时/限流 | 抽取失败退回规则抽取，不影响淡化与隔离 |
| 整理任务重复执行 | 衰减按锚点重算，天然幂等 |
| 单字查询查不到 | 独立的单字通道（不污染 BM25 索引） |
| 显式检索捞到无关记忆 | 搜索/忘掉路径强制「必须命中」 |
| 长记忆占满注入预算 | 写入前按标点拆成「一件事一条」，拆不动就原样保留 |


## 架构

核心包零框架依赖、零第三方依赖；每个适配器只负责三件事：
把宿主消息翻译成统一格式、把记忆翻译回宿主上下文、把宿主的 LLM 接到核心协议上。

```
memfade/
  types.py         Memory / Scope / MemoryKind / MemoryStatus
  decay.py         淡化引擎：稳定性、保留率、事件、档位求解
  retrieve.py      混合检索 + 加权打分 + 分层组装
  store.py         SQLite + FTS5 二元组索引 + 向量 BLOB
  consolidate.py   离线整理：衰减重算 / gist / 合并 / 矛盾仲裁
  extract.py       抽取（LLM 为主，规则兜底）
  dedup.py         近似重复判定与合并
  textutil.py      中文二元组、Dice 相似度
  llm.py           LLM 协议与标准库 HTTP 实现
  embedding.py     嵌入协议与标准库 HTTP 实现
  engine.py        门面：适配器唯一入口

adapters/
  maibot/          MaiBot 插件（复用宿主 LLM 与嵌入模型）
  nonebot2/        NoneBot2
  langbot/         LangBot
  astrbot/         AstrBot
  koishi/          Koishi
  http/            通用 JSON-over-HTTP 侧车
```

**热路径同步、冷路径异步**：召回与注入是纯同步函数，不含一个 `await`；
只有抽取和摘要（每 N 轮一次）走 async。这样在同步与异步宿主里都能用。

## 快速上手

### MaiBot

把 `adapters/maibot/` 整个目录复制到 MaiBot 的 `plugins/` 目录，
在插件管理里启用即可。**不需要额外配 API key**——
抽取用的大模型与向量模型直接复用宿主已经在 MaiBot 里配好的那些。

```bash
python scripts/build_plugin.py --dest G:/AI/许杏玖-MaiBot/MaiBot/plugins
```

命令：

* `/记忆`——查看当前会话记住了多少东西；
* `/记忆 搜 <关键词>`——检索；
* `/记忆 整理`——立刻跑一轮整理；
* `/记住 <内容>`——显式记住；
* `/忘掉 <片段>`——忘掉一条。

### 各适配器的验证程度

不吹牛，逐个说清楚每个适配器被验证到哪一步：

| 适配器 | 验证方式 | 覆盖 |
|---|---|---|
| MaiBot | **实机部署在许杏玖上**，并用宿主自己的反序列化器验证注入的 item | 全链路实机 |
| LangBot | 装了真的 `langbot-plugin` SDK，5 项测试跑在真 SDK 上（找不到就退到本地研究目录里的 SDK 副本） | API 与注入行为 |
| AstrBot | 装了真的 AstrBot SDK 源码，用 ast 静态核对 API 面 + 桩件测逻辑 | API 面与作用域逻辑 |
| Koishi | 装了真的 koishi@4.18，`tsc` 类型检查 + 行为测试（会起真的 Python 侧车） | 类型与跨语言契约 |
| NoneBot2 | 装了真的 nonebot2，4 项测试跑在真 SDK 上 | 作用域与公开 API |

```bash
pip install langbot-plugin            # LangBot 适配器测试需要
pip install nonebot2                  # NoneBot2 适配器测试需要
npm --prefix adapters/koishi test     # Koishi 类型检查 + 行为测试
```

两个 Python 环境各跑一遍，**每个适配器的真 SDK 测试都能跑到**：

```powershell
# MaiBot 的 venv（带 maibot_sdk）
& 'G:\AI\许杏玖-MaiBot\MaiBot\.venv\Scripts\python.exe' -m unittest discover -s tests -t .
# 系统 Python（带 langbot-plugin / nonebot2）
python -m unittest discover -s tests -t .
```

### 直接用核心包

```python
from memfade import MemoryEngine, MemoryDraft, MemoryKind, Scope

engine = MemoryEngine(db_path="memfade.db")
scope = Scope(platform="qq", chat_type="private", chat_id="1738157278")

engine.remember(
    MemoryDraft(content="许杏玖喜欢辣的食物", kind=MemoryKind.SEMANTIC, importance=0.8),
    scope,
)

ctx = engine.build_context("今天吃什么", scope)
print(ctx.text)
```

### 命令行工具

```bash
python scripts/memfade_cli.py --db memfade.db health              # 记忆库体检
python scripts/memfade_cli.py --db memfade.db search --query 显卡  # 检索并解释召回
python scripts/memfade_cli.py --db memfade.db export --out m.json  # 导出（可移植、可审计）
python scripts/memfade_cli.py --db memfade.db fade                # 跑一轮整理
```

体检会指出：记忆太长（注入时会被截断）、一条塞了多件事、缺三元组
（矛盾检测抓不到）、还有多少近似重复。

### 从旧的记忆库迁移

已经有一套记忆系统（比如 A_Memorix）？可以把它的事实**洗干净**再带过来：

```bash
python scripts/import_a_memorix.py \
    --source "<旧库>/metadata.db" \
    --target "<memfade>/memfade.db" \
    --maibot-account 2760135586 --maibot-user 1738157278 \
    --host-root "<MaiBot 根目录>"
```

流程是：读旧库 → 用宿主 LLM 规范化成「一件事一条 + 三元组」→ memfade 去重入库。
旧库里「称对方为宝宝」有 3 种说法、「二十块加价与铁子无关」有 4 种说法，
迁移后会被合并。实测 41 条旧事实 → 入库 21 条。

作用域用 MaiBot 的 `session_id` 公式算，**chat_type 必须是 `session`**——
写成 `group`/`private` 机器人就读不到。

## 借鉴来源

这个项目的思路不是凭空来的，下面这些项目提供了关键模块的参考：

| 来源 | 借鉴了什么 |
|---|---|
| [congling](https://github.com/LUOSHUI319/astrbot_plugin_concling)（AstrBot） | 分级遗忘曲线、稳定性随复习次数增长、**按锚点重算的幂等衰减** |
| [MemoryBank](https://github.com/zhongwanjun/MemoryBank-SiliconFriend) | 艾宾浩斯曲线 `exp(-t/τ)`、召回即强化的思想 |
| [Engram](https://github.com/foramoment/engram-ai-memory) | 幂等衰减与剪枝、RRF 混合检索、访问提权 |
| [memory-brain](https://github.com/peterpoal134/memory-brain)（AstrBot） | core + 多适配器的架构、**中文二元组 FTS5**、混合检索打分 |
| [A-MEM](https://github.com/WujiangXu/AgenticMemory) | 记忆的动态索引与链接 |
| [memorylens-bench](https://github.com/Neal006/memorylens-bench) | Recall@T / Temporal Drift / Contradiction 三个评测指标 |
| [feiyu-memory](https://github.com/Deepsleep-one/astrbot-plugin-feiyu-memory) | 范围隔离与证据链记忆 |
| [MaiBot A_Memorix](https://github.com/MaiM-with-u/maibot) | 残留学习式强化、双轴生命周期的工程化写法 |

memfade 在这些基础上的差异化：显著性/持久度双轴分离、
四档淡化梯度、事实层/回声层双层注入、失败即闭的会话隔离。

## 开发

```bash
python -m unittest discover -s tests -t .     # 全部测试（标准库 unittest，无第三方依赖）
python scripts/build_plugin.py --zip          # 打包各宿主适配器
python scripts/bench_real_data.py --db <MaiBot.db>   # 在真实聊天数据上量去重/淡化/隔离
```

## 文档

* [架构](docs/ARCHITECTURE.md)——为什么是 core + adapters，热路径/冷路径怎么切
* [淡化算法](docs/DECAY.md)——每一个设计选择的理由，尤其是「为什么不那么做」
* [GitHub 调研](docs/RESEARCH.md)——查了哪些项目、借鉴了什么、什么没照搬

## 实机验证

用 MaiBot 的真实消息库（535 条）跑一遍：

```
① 去重率     : 17.5%（83 条被判重复，含实际踩过的「显卡跑不动两个模型」）
② 淡化       : 30 天后只剩 3 条稳定事实在事实层，其余退场
③ 跨会话泄漏 : 0 条
```

**实机（许杏玖线上）**：

* 提示词快照里能直接找到 memfade 构造的 item——真实发给模型的上下文里
  确实带着记忆块（群会话 6/8、私聊 3/5 的快照命中，其余是检索没命中的预期情况）；
* 线上记忆库：私聊 25 条 + 群 3 条，其中 21 条是从 A_Memorix 清洗后迁过来的；
  其中已有 1 条被矛盾检测自动取代（旧事实退场、新事实上位）；
* 记忆质量：中位长度 60→18 字、超长率 85%→21%、缺三元组 85%→28%；
* 问「她喜欢吃什么」的注入量 482→218 字，无关条目不再进 prompt。

**记忆可解释**：面板里能直接看「为什么想起这条」——
相关性 / 重要性 / 强度 / 新鲜度 / 频次五个分量逐条列出。
实机问「显卡能同时跑几个模型」，显卡那条 rel=1.00 排第一，
语音+SD 那条 rel=0.63 第二，不相关的 rel=0.00 沉底。

### 消融实验：淡化到底带来了什么

`scripts/bench_decay.py` 拿 [memorylens-bench](https://github.com/Neal006/memorylens-bench)
的同款指标，在一段脚本化对话上对比「只记不遗忘的 naive 基线」与 memfade：

```
指标                             naive       memfade
Recall@T                         1.0           1.0     该记住的记住了吗
老话题召回                        1.0           1.0     问起旧事还答得上吗
旧值残留（事实层）                   1.0           0.0   ✓ 旧值被当成仍然成立
新旧并存（事实层）                   1.0           0.0   ✓ 新旧同时被当成事实
平均注入条数                      5.72          5.72
平均上下文字数                     90.2          98.2
```

**召回不掉、旧值归零**——这就是「忘得对」而不是「忘得多」。
上下文字数略大，是因为画像常驻，换来稳定事实从不缺席。

漂移指标按「每条旧值自己的更新时间」统计：
「住在杭州」在脚本中点被取代，「喜欢吃辣」要到 3/4 处才被取代——
统一按中点算的话，会在「吃辣」还没变的那几轮里就把它算成漂移。
指标算错比没有指标更糟。

第一版跑这个实验时还揪出两个潜伏 bug：FTS5 的隐式 AND（多词查询几乎查不到），
以及已归档的记忆**永远召不回来**（哪怕对方明确问起）。两个都已修复。

## License

MIT



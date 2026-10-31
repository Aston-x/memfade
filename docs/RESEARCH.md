# GitHub 调研与借鉴记录

做这个插件之前，先把 GitHub 上现成的方案摸了一遍。
本文记录**查了什么、借鉴了什么、以及为什么没有全盘照搬**。

## 检索范围

围绕「QQ 机器人 + 长期记忆 + 遗忘/淡化」三条线检索，
覆盖 Python 与 TypeScript 两大生态。

### 市面上的 QQ bot 记忆插件

| 项目 | 星 | 生态 | 定位 |
|---|---|---|---|
| [kawayiYokami/astrbot_plugin_angel_memory](https://github.com/kawayiYokami/astrbot_plugin_angel_memory) | 200 | AstrBot | 让 AI 不仅能记住，还能主动思考、自主进化 |
| [zhanzhao2/astrbot-plugin-persistent-memory](https://github.com/zhanzhao2/astrbot-plugin-persistent-memory) | 87 | AstrBot | 记忆捕获、召回注入、LanceDB 持久化、混合检索与重排 |
| [menglimi/astrbot_plugin_memory_companion](https://github.com/menglimi/astrbot_plugin_memory_companion) | 77 | AstrBot | 更拟人化的记忆调度与记忆编排 |
| [peterpoal134/memory-brain](https://github.com/peterpoal134/memory-brain) | 49 | AstrBot | 五类记忆、混合检索、遗忘治理、自主生活循环 |
| [leafliber/astrbot_plugin_iris_memory](https://github.com/leafliber/astrbot_plugin_iris_memory) | 38 | AstrBot | 通用记忆插件 |
| [Klein-Luckary/GroupMemoryMini](https://github.com/Klein-Luckary/GroupMemoryMini) | 5 | LangBot | 简易记忆插件 |
| [TIGillusion/simple_memory](https://github.com/TIGillusion/simple_memory) | 3 | MaiBot | 海马体记忆机制简化版：概念存储、检索、遗忘、整合 |
| [Elmeir/maibot-plugin-memory-enhance](https://github.com/Elmeir/maibot-plugin-memory-enhance) | 0 | MaiBot | 跨聊天流、事实与日记的增强检索 |
| [TanTan-yyds/maibot-short-term-memory](https://github.com/TanTan-yyds/maibot-short-term-memory) | 0 | MaiBot | 全局短期记忆，注入 Planner/Replyer |
| [luoxinmob/nonebot-plugin-memory](https://github.com/lanxinmob/nonebot-plugin-memory) | 4 | NoneBot2 | 用户画像 + Redis 会话历史 |
| [langbot-app/langbot-longterm-memory](https://github.com/langbot-app/langbot-longterm-memory) | 0 | LangBot | 官方长期记忆插件（L1 画像 + L2 情景） |
| [Deepsleep-one/astrbot-plugin-feiyu-memory](https://github.com/Deepsleep-one/astrbot-plugin-feiyu-memory) | 1 | AstrBot | 主人优先的身份核验、范围隔离、证据链记忆 |
| [LUOSHUI319/astrbot_plugin_congling](https://github.com/LUOSHUI319/astrbot_plugin_congling) | 8 | AstrBot | 分级会淡忘的长期记忆 + 四阶段睡眠 + 日记 |

### 通用记忆研究 / 引擎

| 项目 | 借鉴点 |
|---|---|
| [zhongwanjun/MemoryBank-SiliconFriend](https://github.com/zhongwanjun/MemoryBank-SiliconFriend) | 艾宾浩斯曲线 `exp(-t/(5·S))`；召回即强化（`S += 1`，重置时钟） |
| [foramoment/engram-ai-memory](https://github.com/foramoment/engram-ai-memory) | 幂等衰减与剪枝；RRF 混合检索；访问提权；三层记忆 |
| [WujiangXu/AgenticMemory](https://github.com/WujiangXu/AgenticMemory)（A-MEM） | 记忆的动态索引与链接（Zettelkasten 式） |
| [Neal006/memorylens-bench](https://github.com/Neal006/memorylens-bench) | Recall@T / Temporal Drift / Contradiction 三个评测指标；衰减基准方法论 |
| [getzep/graphiti](https://github.com/getzep/graphiti) | 时间知识图谱的边衰减 |
| [mem0ai/mem0](https://github.com/mem0ai/mem0) | 记忆抽取与合并的流水线 |

## 逐项借鉴

### congling 的 `memory/forgetting.py` —— 最有价值的一份

它的曲线是：

```python
stability = S_BASE[level] * min(CEILING, S_GROWTH[level] ** hits)
retention = exp(-days / stability)
```

并且**按「距上次被想起多少天」重算保留率，而不是每天乘一次**。
这条工程细节 memfade 原样采纳（`decay.py`），并把它推广到
所有记忆种类，而不只是 0~5 级。

它还有两处 memfade 学了：

* **稳定性的复习增长**（`S_GROWTH ** hits`）——间隔重复；
* **有界化打分**（`retrieve.py` 里的 `(0.55 + 0.45 × decay)`）——
  衰减进入总分时只能是折扣，不能是清零。

它的 `maintenance.py` 里还有一条幂等保护（按业务日跳过重复衰减），
memfade 用了更彻底的做法——从锚点重算，天然幂等，不需要额外的去重标记。

### memory-brain 的 `memory-core` —— 架构范式

它把核心放在 `memory-core/memory_core/`，AstrBot 只是壳，
还额外支持 DSH 与 QQBridge。**core + 多适配器**这个结构 memfade 直接采纳。

它还有一个关键技巧：**FTS5 存字符二元组**。
SQLite 自带的 unicode61 分词器会把一整句中文当成一个词，
「炸鸡」搜不到「今天吃了炸鸡」。memfade 的 `textutil.py` 用了同样的办法，
并且额外保留英文/数字整词（bigram 对拉丁字母的召回质量明显不如整词）。

它的检索打分是「语义 + BM25 + 重要性 + 时间 + 图谱」的加权和，
memfade 沿用了这个思路，但把「时间」拆成了 strength 与 recency 两个信号。

### MemoryBank 的 `forgetting_curve`

```python
def forgetting_curve(t, S):
    return math.exp(-t / 5 * S)
```

它还提供了一条 memfade 直接采用的机制：**被召回的记忆，
`memory_strength += 1` 且 `last_recall_date` 重置**。
memfade 把它改造成残留学习（`new = eff + α(1-eff)`），
并加了冷却期防止正反馈。

它用随机采样决定丢不丢（`random.random() > retention`），
memfade 改成确定性的四档梯度——随机删除不可复现，也没法审计。

### Engram 的 `consolidation.js`

* `strength *= decayRate ^ daysSinceLastConsolidation`——幂等衰减；
* 访问强化 `strength = MIN(strength * 1.1, 1.0)`；
* 近重复合并（余弦 0.92）；
* `permanent` 标签豁免衰减——对应 memfade 的 `pinned`。

它的 FoA 打分是 `relevance × importance × strength × recency`，
memfade 改成加权和（理由见 `docs/DECAY.md` 第 6 节）。

### simple_memory 的激活扩散

它用关键词在记忆图上做 BFS 扩散，激活值按边的强度递减。
memfade 采纳了「关联」这个概念，但砍掉了图结构：
只与首条命中共享 2 个以上关键词的候选获得小幅加成。
收益（捞出旁支记忆）保留，成本（每次查询建图遍历）降到几乎为零。

### memorylens-bench 的评测指标

它定义了一个记忆衰减基准，三个指标：

* **Recall@T**——在第 T 轮，检索到的上下文里有没有当前正确的事实；
* **Temporal Drift**——事实更新之后，上下文里还残留多少旧值；
* **Contradiction**——上下文里同时出现新旧两个互相矛盾的版本。

memfade 的 `scripts/bench_real_data.py` 用了同样的思路，
在真实消息上量三个行为：去重率、淡化曲线、跨会话泄漏。

## 没有照搬的地方

### 显著性衰减（memory-brain）

memory-brain 的遗忘是 `importance *= 0.99`，衰减直接作用在 importance 上。
问题是：**importance 同时承担了「多重要」和「多新鲜」两个含义**，
衰减之后一条重要事实的检索权重也跟着掉。

memfade 把它拆成两条轴（见 `docs/DECAY.md` 第 1 节），
衰减只作用在 strength 上。

### 硬删除（Engram / memory-brain 的合并）

两者在剪枝时都会物理删除或归档掉整条记忆。
memfade 的默认策略是**归档不删**：原文留在库里可审计，
只有显式配置 `archive_after_days` 才会物理清理。

### 随机遗忘（MemoryBank）

`random.random() > retention` 决定丢不丢——同样的数据跑两次结果不同。
memfade 的档位求解是纯函数，给定同样的输入永远得到同样的输出。

### 每种语言重写一遍（通用做法）

Koishi 是 TypeScript。常见的做法是给每个语言各写一份记忆插件。
memfade 让核心只跑一份，Koishi 通过 HTTP 侧车调用——
淡化算法只有一份实现，不会走样。

## memfade 的四个差异化

把上面的调研收拢成四点，是 memfade 与现有方案的主要区别：

1. **显著性 / 持久度双轴分离**
   —— 重要事实不会因为时间而掉分，一次性事件不会因为「刚写入」而反复被提起。

2. **四档淡化梯度（含 gist）**
   —— 旧事压缩成一行摘要，而不是突然消失或一直占着位置。

3. **事实层 / 回声层双层注入**
   —— 「现在仍然成立」和「可能已经变了」在 prompt 里是分开表述的。

4. **失败即闭的会话隔离**
   —— 所有读取路径共用一条作用域过滤，没有全局兜底。

## 实机验证（实验对象：许杏玖）

用 MaiBot 的真实消息库（535 条）跑 `scripts/bench_real_data.py`：

```
① 去重：同样的话存了几份？
  原始候选     : 475
  实际入库     : 392
  被判定重复   : 83
  去重率       : 17.5%
    · 合并掉：不行啊，我显卡跑不动同时两个模型
    · 合并掉：干嘛，戳得我痒痒的。

② 淡化：时间推移后，还剩多少记忆留在事实层？
  假设所有记忆都是「30 天前」写的：
    +  0 天 → 事实层   3 条 / 回声层 277 条 / 已退场 112 条
    +  7 天 → 事实层   3 条 / 回声层   0 条 / 已退场 389 条

③ 隔离：私聊的记忆会不会漏进群？
  群作用域     : 3 个
  私聊作用域   : 3 个
  跨会话泄漏   : 0 条（必须为 0）
```

「显卡跑不动两个模型」这条正是实际踩过的重复——归档里记着它被存过两份。



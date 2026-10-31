# 架构

## 为什么是 core + adapters

市面上的 QQ 机器人框架，编程语言和插件模型都不同：

| 框架 | 语言 | 插件模型 | 注入 prompt 的方式 |
|---|---|---|---|
| MaiBot | Python | SDK 装饰器 + 命名 Hook | `ctx.maisaka.context.append` |
| LangBot | Python | `BasePlugin` + `EventListener` | `event.default_prompt.append` |
| AstrBot | Python | `Star` + `@filter` | `req.system_prompt` |
| NoneBot2 | Python | `PluginMetadata` + Matcher | 无统一钩子，提供助手 API |
| Koishi | TypeScript | `apply(ctx, config)` | 无统一钩子，走 HTTP 侧车 |

如果给每个框架各写一份记忆系统，淡化算法一定会走样——
五种实现五个 bug，而且每加一个框架就多一份维护负担。

所以 memfade 的结构是：核心 `memfade/` 不知道任何框架的存在，
只依赖标准库；每个适配器只负责「翻译」：
把宿主消息翻译成统一格式、把记忆翻译回宿主上下文、
把宿主的 LLM 接到核心协议上。

Koishi 走 HTTP 侧车，是因为它是 TypeScript 生态——
与其用 TS 重写一遍淡化算法（必然走样），不如让核心只跑一份。

## 模块职责

| 模块 | 职责 | 行数 |
|---|---|---|
| `types.py` | `Memory` / `Scope` / `MemoryKind` / `MemoryStatus` | 147 |
| `decay.py` | 稳定性、保留率、生命周期事件、档位求解 | 331 |
| `retrieve.py` | 候选生成、加权打分、关联加成、分层组装 | 228 |
| `store.py` | SQLite、FTS5 二元组索引、向量 BLOB、作用域过滤 | 493 |
| `consolidate.py` | 衰减重算、gist 压缩、合并、矛盾仲裁、清理 | 175 |
| `extract.py` | LLM 抽取 + 规则兜底、敏感信息过滤 | 240 |
| `dedup.py` | 精确/近似重复判定与合并 | 87 |
| `textutil.py` | 中文二元组、Dice 相似度、token 估算 | 123 |
| `llm.py` / `embedding.py` | 协议定义 + 标准库 HTTP 实现 | 192 |
| `config.py` | 配置对象（dict/TOML/JSON 通吃） | 157 |
| `engine.py` | 门面：适配器唯一入口 | 262 |

## 热路径 / 冷路径

**热路径**（每条消息都跑）：`build_context()` → 召回 → 打分 → 组装。
纯同步，不含一个 `await`，不碰网络。实测在几千条记忆的库上是毫秒级。

**冷路径**（每 N 轮 / 定时跑）：抽取、摘要、整理。这些要调 LLM，走 async。

这个划分让核心在**同步宿主**（脚本、CLI）和**异步宿主**（几乎所有 bot 框架）
里都能直接用——同步宿主调热路径，异步宿主额外 await 冷路径即可。

## 存储

单文件 SQLite：

```sql
memories(id, scope_key, kind, content, summary, importance, strength,
         anchor_at, status, created_at, updated_at, last_recalled_at,
         recall_count, pinned, cross_scope, subject, predicate, object,
         source, superseded_by, content_hash, fts_tokens, meta, embedding)
```

* `fts_tokens` 是预计算的字符二元组串，FTS5 触发器保证索引与主表同步；
* `embedding` 是 float32 BLOB，numpy 在就用矩阵运算，不在就纯 Python；
* `meta` 是 JSON，装合并历史、来源等附加信息。

## 作用域过滤只有一条

```sql
WHERE (scope_key = ? OR cross_scope = 1)
```

这条出现在**所有**读取路径上：FTS 检索、向量检索、近期召回、整理、
去重候选。没有第二条「找不到就退回全局」的路径。

跨会话共享（`cross_scope = 1`）必须显式开启——只有抽取器认定
「用户明确要求记住」时才会置位。

## 一次调用的完整链路

```
入站消息
   │
   ├─► 解析作用域（平台 + 会话类型 + 会话 ID）
   │      群 1127155154  →  qq:group:1127155154
   │      私聊 1738157278 →  qq:private:1738157278
   │
   ├─► 热路径：召回
   │      FTS5 二元组检索  ─┐
   │      向量检索         ─┼─► 候选集合
   │      近期写入         ─┘
   │
   ├─► 打分
   │      rel / importance / strength / recency / freq
   │      + 与首条命中共享关键词的关联加成
   │
   ├─► 分层
   │      ACTIVE      → 事实层
   │      FADING/GIST → 回声层（限量 + 过时标记）
   │      ARCHIVED    → 丢弃
   │
   ├─► 预算裁剪 → 渲染成文本
   │
   └─► 注入宿主上下文 + 强化被注入的记忆

（冷路径，每 N 轮）
   入站消息 → 抽取候选 → 去重 → 落库
   （定时）
   全库 → 衰减重算 → gist 压缩 → 合并重复 → 矛盾仲裁
```

## 注入时机：为什么不是「收到消息就注入」

宿主把「注入上下文」这件事放在**模型请求之前**，而不是「消息到达时」。
原因有二：

1. 消息到达时，会话可能还没建立，注入会失败；
2. 只有在请求构造阶段才能**直接改写真正发给模型的 Context Items**——
   写进聊天历史的话，会被上下文的裁剪逻辑顺手丢掉。

以 MaiBot 为例，钩子选在 `maisaka.planner.before_request`
（规划器）与 `maisaka.replyer.before_model_request`（回复器）：
两次模型请求各注入各的，互不干扰，而且都有现成的 `session_id` 可当作用域键。




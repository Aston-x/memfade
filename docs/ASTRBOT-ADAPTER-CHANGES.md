# AstrBot 适配器改造说明（外部改动交接）

> 2026-11-03 · 改动人：WorkBuddy（甲方交付侧）
> 目的：把 memfade 接进一个要卖给终端用户的 AstrBot 机器人产品。
> 本文说明**改了什么、为什么改、怎么验证的**，以及**哪些决定留给维护者**。

---

## 零、一句话总结

原来的 AstrBot 适配器**装上等于白装**：LLM 没接、embedder 没接、群聊不区分说话人。
这次把这三个 P0 修掉，并按产品需求把作用域改成**群级公共 + 人级私密**双层。

核心引擎 `memfade/` 只改了 2 行（见第五节），其余全在 `adapters/astrbot/`。

---

## 一、改了什么

### 1.1 `adapters/astrbot/main.py`（重写，346 → 约 760 行）

| 改动 | 为什么 |
|---|---|
| **接上宿主的 LLM**（新增 `clients.py`） | 原来 `MemoryEngine(mem_cfg)` 没传 `llm=` → `extract.py:428` 的 LLM 抽取分支永不进入，`extract.py:543` 的 canonize 直接原样返回。**结果是只能靠「记住」手动存，对话里的话记不下来** |
| **接上宿主的 embedder** | 原来没传 `embedder=` → `engine.py:786` 静默降级成纯关键词检索，但 WebUI 上开关显示「已启用」。现在改成「真的有 embedder 才开向量通道」 |
| **作用域改三层** | 原来 `_scope_of()` 群聊只按群号建作用域 → 群里 A 说的话会注入给 B |
| **写入按主语路由** | 原来抽取结果整批写进同一个 scope → 即使召回时区分了人，写入侧不改照样串层 |
| **召回改成双层合并** | 两层各自召回 → 按记忆 ID 去重 → 按分数合并 → **只组装一次**（不能两次组装再拼文本） |
| **补 `fade` / `maintenance` 配置节** | 原来只传了 storage/extraction/retrieval，核心认识的 7 个顶层节里**这两个整段没传** → `archive_after_days` 在 schema 里存在却从没生效 |
| **加后台维护任务** | 没有它，核心一半能力是哑的：gist 压缩、近似重复合并、矛盾取代、按上限淘汰全都不发生，淡化只会单向衰减 |
| **加空闲冲刷** | 群里一天说 2 句话时永远攒不到 3 条 → 记忆永远是空的 |
| **`logger` 不再用标准 logging** | `logging.getLogger("memfade")` 覆盖掉了 `astrbot.api.logger` → **WebUI 面板看不到任何插件日志** |
| **改相对导入** | 见第三节 |
| **`@register` 参数顺序修正** | 原来是 `(name, desc, version, repo)`，AstrBot 的签名是 `(name, author, desc, version, repo)` → 面板显示成 `Plugin memfade_memory (memfade) by 记忆淡化引擎…: 0.1.0` |
| **`db_path` 绝对化** | 原来默认 `"memfade.db"` 相对路径 → 落到进程 CWD。改用 `StarTools.get_data_dir()` |
| **`min_importance` 0.35 → 0.45** | 核心与 MaiBot 都是 0.45。0.35 会让「今天只有一节课」这类噪声进库 |
| **`promote_cross_scope` 默认关** | 核心默认是 True，但 `cross_scope=1` 的行对**所有**作用域可见 → 多用户部署下 A 的私密记忆会漏给 B 和别的群 |
| **`self.config` 自己兜** | `Star` 基类**不保存** config，且 config 可能是 None（宿主没给配置时只传 context） |
| **新增 `群记` 命令** | 写群级公共层。`记住` 走人级私密层 |
| **新增 `记忆 状态` 诊断** | 显示「抽取引擎：LLM / 规则兜底」和「向量检索：启用 / 未启用」 |

### 1.2 `adapters/astrbot/scopes.py`（新增）

作用域键设计与路由规则。

```
群级公共   astrbot:group:3000000003
人级私密 astrbot:group_user:3000000003#u9987
私聊       astrbot:private:u9987
```

**为什么用 `#` 而不是 `:` 分隔 group/user**：`Scope.key()` 是三段 `platform:chat_type:chat_id`，
`from_key` 用 `split(":", 2)`，第三段整体是 chat_id（可以含冒号）。但
`engine.import_memories` 在 scope 为空时手写 `raw_scope.split(":")[-1]` 拆键，
用 `#` 能让那条路径也恰好拆对。

**写入路由**（按 `MemoryDraft.subject`）：

1. 主语命中本群成员昵称 → 人级私密层
2. 主语等于群别名（配置项 `group_subject`，默认「本群」）→ 群级公共层
3. 其它 → **丢弃**

第 3 条是丢弃而不是退群层：退群层等于把某个人的私密事实公开。

### 1.3 `adapters/astrbot/clients.py`（新增）

`AstrBotLLMClient` / `AstrBotEmbeddingClient` 两个薄适配，复用宿主已配好的 provider——
客户不用再填一份 API key。

跨版本处理：`get_using_provider_async`（4.29.4）→ `get_using_provider`（4.14.6）→
退回首个 provider，全用 `getattr` 探测。`EmbeddingProvider` 没有从
`astrabot.api.provider` 公开导出，所以用鸭子类型调 `get_embeddings()`，不 import 那个类。

### 1.4 `adapters/astrbot/_conf_schema.json`（9 项 → 50 项）

新增 LLM/embedding 接入项、淡化参数、维护参数、作用域相关项。

### 1.5 `adapters/astrbot/metadata.yaml`

`name` → `astrbot_plugin_memfade`（与目录名、`@register` 名、`get_data_dir()` 入参**四处一致**），
`version` → `0.4.0`（对齐核心 `__version__`），删掉 `display_name`
（4.14.6 认、4.29.4 不认，删掉两边行为一致）。

### 1.6 `adapters/astrbot/memfade_tools/`

| 文件 | 改动 |
|---|---|
| `edit_memory.py` | `_scope_of()` 原来把 platform 写死成 `"qq"` → 改用 `Scope.from_key()`。否则在 AstrBot 下「改记忆」会改到错误的作用域上去 |
| `verify_against_chat.py` | 原来直接 `from src.services import llm_service`（MaiBot 内部模块）→ **非 MaiBot 宿主下必崩**。改成两条路：给了 `--host-root` 且能 import 就用它，否则用核心自带的 `HttpLLMClient`（新增 `--llm-base-url/--llm-model/--llm-api-key`） |
| `backup_memory_stack.py` / `retire_legacy_memory.py` | 写死了 MaiBot 的目录结构和插件 ID → **从 AstrBot 包里剔除**（AstrBot 的备份 = `data/plugin_data/<name>/` + `data/config/<name>_config.json`，MaiBot 那套对它无意义） |

### 1.7 `scripts/build_plugin.py`

- `DEPLOYABLE` 从 `{maibot: path}` 改成按适配器的 spec：`{name: {source, dir_name, preserve_in_dir, tools}}`
- astrbot 的 `dir_name` = `astrbot_plugin_memfade`，`preserve_in_dir` = `[]`
  （**AstrBot 的配置和数据都在插件目录外，天然保住用户配置和记忆**，不像 MaiBot 的 config.toml 在插件目录内需要单独保留）
- zip 根目录名用 `dir_name`（原来用 `target.name` → 会产出 `astrbot/`，而配置文件名和数据目录都跟着目录名走，名字对不上等于用户配置失联）
- 运维工具清单按适配器裁剪

### 1.8 核心（`memfade/engine.py`，2 行）

```python
# 原来
target_scope = Scope(
    platform="qq",
    chat_type="session",
    chat_id=raw_scope.split(":")[-1] if ":" in raw_scope else raw_scope,
    cross_scope=bool(item.get("cross_scope", False)),
)
# 改成
target_scope = Scope.from_key(raw_scope)
```

手写拆键把 platform 写死成 `"qq"` → AstrBot 导出再导入后作用域整个错位。
`Scope.from_key` 用 `split(":", 2)`，语义正确，`#` 分隔也安全。

**改了核心必须跑 `python scripts/build_plugin.py` 同步 4 份副本**，`tests/test_vendored_sync.py` 会守住这条。

---

## 二、测试

### 新增

| 文件 | 项数 | 内容 |
|---|---|---|
| `tests/test_astrbot_scope.py` | 16 | **双层隔离证明**：A 的记忆用 B 的 scope 召回为空、用 A 的命中；群级对全员可见；私聊不漏进群；`cross_scope` 行合并后去重；`from_key` 往返；三层 key 两两不等；未知主语丢弃不泄漏 |
| `tests/test_astrbot_config.py` | 13 | schema 键与代码读写的键**双向对齐**（防「配了不生效」）；四处名称一致；版本对齐核心；`archive_after_days` 真的生效；无 embedder 时向量通道关闭 |

### 改动

`tests/test_astrbot_adapter.py`（10 项，仍全过）：桩件补 `StarTools` 和假 provider；
`_scope_of` 相关断言按新 API 改写；静态核对扩展（`text_chat` / `get_embeddings` /
`get_all_embedding_providers` / `StarTools.get_data_dir` / `register_command_group`）；
版本容忍（`get_using_provider_async` **或** `get_using_provider`）。

### 结果

```
python -m unittest discover -s tests -t .
Ran 301 tests in 15.2s
OK (skipped=52)
```

基线是 272 项（skipped=52 是各适配器 SDK 相关，本机没装 maibot_sdk / langbot-plugin / nonebot2）。
新增 29 项。

---

## 三、`import memfade` 的遮蔽问题（请重点看这条）

**AstrBot 不把 `data/plugins/` 加进 `sys.path`**，插件以 `data.plugins.<目录名>`
这个 PEP 420 命名空间包导入 → 两个插件各带一份 `memfade/` 本来就互不遮蔽。

**真正的风险是适配器自己**：原来的 `_bootstrap_import()` 把插件目录插到 `sys.path[0]`，
让顶层 `memfade` 变得可导入。而 `sys.modules` 是**进程级全局缓存**：

```
插件 A 加载 → import memfade → sys.modules["memfade"] = A 的副本
插件 B 加载 → import memfade → 拿到【A 的副本】，不是 B 的
pip 装的 memfade → import memfade → 也拿到 A 的副本
```

后果：① 版本漂移；② **跨副本枚举身份分裂**——`MemoryKind` 是 `str, Enum`，
两副本之间 `==` 和 `hash` 相等但 `is` **不相等**，而核心里有 13 处
`is MemoryKind.X` / `is MemoryStatus.X` 的身份比较（`engine.py:329`、
`guards.py:322`、`decay.py:277-304`、`retrieve.py:161/328`、`consolidate.py:133/143`），
会静默走错分支。

**解法**（已实施）：生产路径走**包内相对导入**，全局 `sys.modules["memfade"]`
根本不会被创建；兜底路径（脱离包上下文）用 `sys.path.append` 插到**末尾**，
并校验拿到的 `memfade.__file__` 确实在本插件目录下——解析到别处就**宁可明确报错
也不静默用错版本**。

**建议维护者也检查其它适配器**：如果它们也用 `_bootstrap_import()` 插 `sys.path[0]`，
同样的隐患存在。

---

## 四、留给维护者决定的三件事

### 4.1 群级公共层的自动抽取目前不可靠

核心的抽取提示词有一条硬规则（第 8 条）：**主语必须是「人」**。
这跟「让 LLM 输出 `subject=本群`」直接冲突——`require_person_subject=True` 时
群别名会被核心直接丢弃。

本次的做法是：群路径用**独立的 extractor 实例**
（`dataclasses.replace` 换掉 `require_person_subject=False` + `subject_hint` 注入成员名单），
让群别名能活到适配器层，再由适配器用白名单二次过滤。

**但现实是群级公共层主要靠 `群记` 命令长起来，自动抽取贡献有限。**
三个选项：

1. 保持现状（群层靠显式指令）
2. 改核心提示词，允许「集体主语」——**要同步 4 份副本**
3. 加一个配置项 `allow_group_subject`，让核心在抽取后过滤而不是在提示词里禁止

**我们暂时选了 1**（改核心影响面大，且核心的铁律是「宁可少记」）。

### 4.2 `recall_alpha` 默认值要不要调

核心默认 `0.35`（`decay.py:182`）：被想起时新强度 = eff + α(1-eff)。
MaiBot 用的是 0.35，适配器也跟了 0.35。

这个值直接决定「总提旧事」的严重程度。**如果实盘反馈她还是老提旧事**，
优先调这个（调低 = 强化更弱 = 更快淡出），而不是去调 `min_importance`。

### 4.3 AstrBot 要不要进 `build_plugin.py` 的 `DEPLOYABLE`

已经进了（`dir_name=astrbot_plugin_memfade`）。但 AstrBot 的安装路径是
`data/plugins/<目录名>/`，而不是 MaiBot 的 `plugins/<目录名>/`，
`--deploy` 的目标目录要由使用者给绝对路径。

**要不要给 `--deploy` 加个 `--host astrbot|maibot` 的快捷方式？** 现在是
`--deploy <plugins目录>`，用户得自己知道该指哪。

---

## 五、部署形态（四处名称必须一致）

| 位置 | 值 |
|---|---|
| 部署目录名 | `astrbot_plugin_memfade` |
| `metadata.yaml` → `name` | `astrbot_plugin_memfade` |
| `@register(...)` 第 1 参 | `astrbot_plugin_memfade` |
| `StarTools.get_data_dir(...)` | `astrbot_plugin_memfade` |
| 由此得到配置文件 | `data/config/astrbot_plugin_memfade_config.json` |
| 由此得到数据目录 | `data/plugin_data/astrbot_plugin_memfade/` |

**目录名必须是合法 Python 标识符**（不能带 `-`）。
MaiBot 的 `memfade-plugin` 在 AstrBot 下会直接导入失败。

---

## 六、实测记录（真实 AstrBot 4.29.4）

装到 `data/plugins/astrbot_plugin_memfade/` 后：

```
Plugin astrbot_plugin_memfade (0.4.0) by memfade: 会记得，也会忘得恰到好处。
艾宾浩斯衰减、四档淡化梯度、分层作用域隔离。
```

对话实测：

```
记住 我叫Alice，我住在杭州
→ 记住了：我叫Alice，我住在杭州

记忆 状态
→ 私聊层：记住了 2 条（活跃 2 / 淡出 0 / 摘要 0 / 归档 0）
  · 我叫Alice，我住在杭州
  · 我叫Alice
  抽取引擎：LLM（成功 0 / 降级 0 / 报错 0）
  向量检索：未启用（宿主没配 embedding provider）
```

**没能端到端验证到的**：LLM 抽取的完整链路。测试时 DeepSeek 一直返回 503
（服务商繁忙，非本插件问题），而 `on_llm_response` 在 LLM 失败时不会触发
（没有响应就没有东西可抽），所以抽取任务根本没跑起来。
**这一段需要有可用模型的机器上复验。**

---

## 七、还没做的

1. `memfade_tools/backup_memory_stack.py` / `retire_legacy_memory.py` 在 AstrBot 侧
   没有替代品——AstrBot 的备份需求（`data/plugin_data/<name>/` + `data/config/<name>_config.json`）
   目前没有对应脚本
2. AstrBot 侧缺「模型主动翻记忆」的能力（MaiBot 有 `recall_memories` 工具，
   NoneBot2 有公开 API，README 的对照表里 AstrBot 一栏是「—」）
3. 群聊多人群场景的并发抽取没压测过（`ExtractionConfig` 是引擎级共享对象，
   群路径用了独立 extractor 实例，但同一群的多消息并发仍待验证）

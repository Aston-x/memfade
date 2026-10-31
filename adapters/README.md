# 适配器

核心 `memfade/` 不知道任何框架的存在。每个适配器只做三件事：

1. 把宿主的消息翻译成统一格式（`Scope` + 文本）；
2. 把召回的记忆翻译回宿主的上下文；
3. 把宿主的 LLM / 嵌入能力接到核心协议上。

所有适配器共用同一份核心，所以**淡化算法只有一份实现**，
不会像「每个框架各写一份」那样慢慢走样。

| 目录 | 宿主 | 注入方式 | 抽取时机 |
|---|---|---|---|
| `maibot/` | MaiBot | `ctx.maisaka.context.append` | `chat.receive.after_process` |
| `langbot/` | LangBot | `event.default_prompt.append` | `NormalMessageResponded` |
| `astrbot/` | AstrBot | `req.system_prompt` | `on_llm_response` |
| `nonebot2/` | NoneBot2 | 无统一钩子，提供助手 API | 旁听消息 |
| `koishi/` | Koishi | 无统一钩子，走 HTTP 侧车 | 中间件 |
| `http/` | 任何能发 HTTP 的 bot | 返回现成文本，调用方自己注入 | 调用方决定 |

## 通用约定

### 作用域

每个宿主都要能给出「这次对话属于哪个会话」，格式：

```python
Scope(platform="qq", chat_type="group", chat_id="1127155154")
```

**默认会话内隔离**。跨会话共享必须显式开启（`cross_scope=True`），
只有「用户明确要求记住」时才应该开。

### 注入前缀

注入的文本要带系统前缀，否则模型会把这段当成「用户刚说的话」：

```
[系统·长期记忆] 以下是你自己长期记住的内容，属于背景信息，
不需要主动提起；只有对方问到或话题正好相关时才自然带出来：
```

### LLM

没有 LLM 也能跑：抽取会退化成规则匹配（姓名、偏好、承诺、称呼），
淡化、去重、隔离这些核心能力不受影响。

## 各框架的装法

### MaiBot

```powershell
python scripts\build_plugin.py --deploy '<MaiBot>/plugins'
```

详细说明见 [`maibot/README.md`](maibot/README.md)。

### LangBot

把 `langbot/` 整个目录放进 LangBot 的插件目录，在插件市场里启用。
抽取用的 LLM 在插件配置里填 `llm_base_url` / `llm_api_key` / `llm_model`
（OpenAI 兼容接口）；留空则只用规则抽取。

### AstrBot

把 `astrbot/` 放进 AstrBot 的插件目录，`metadata.yaml` 里已声明名字与版本。
配置项见 `astrbot/_conf_schema.json`。

### NoneBot2

```bash
pip install -e adapters/nonebot2
```

NoneBot2 没有统一的 prompt 注入钩子，所以本插件提供助手 API：

```python
from nonebot_plugin_memfade import build_context, remember, recall

text = await build_context(event)      # 可直接拼进 prompt 的记忆文本
await remember(event, "他叫许杏玖")      # 显式写入一条记忆
```

同时也把记忆文本挂到 `event.set_extra("memfade_context", ...)`，
想自动接管的插件直接读这个 extra。

### Koishi

Koishi 是 TypeScript 生态，所以它走 HTTP 侧车——核心只跑一份，
不在 TS 里重写淡化算法。

```powershell
python adapters\http\server.py --db memfade.db --port 8787 --token <令牌>
```

```bash
cd adapters/koishi && npm install && npm run build
```

然后把 `lib/` 装进 Koishi，在配置里填侧车地址与令牌。

### HTTP 侧车

任何能发 HTTP 的 bot 都能用。接口：

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查 |
| POST | `/remember` | 写入记忆 |
| POST | `/recall` | 召回记忆 |
| POST | `/context` | 召回 + 分层组装（返回可直接注入的文本） |
| POST | `/consolidate` | 跑一轮整理 |
| GET | `/stats` | 统计 |

鉴权：请求头 `X-MemFade-Token`，与服务端 `--token` 一致才放行。
默认只监听 `127.0.0.1`。


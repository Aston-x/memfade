# memfade · Koishi 插件

给 Koishi 用的长期记忆：**会记得，也会忘得恰到好处。**

插件本身很薄：记忆的存取与抽取都在 **Python 侧车**里（`adapters/http/server.py`），
Koishi 这边只负责取回上下文、转发抽取请求。

## 装法

```bash
# 1) 起侧车
python ../http/server.py --db memfade.db --port 8787 --token <令牌> \
    --llm-url <OpenAI 兼容地址> --llm-model <模型名>

# 2) 装插件
npm install koishi-plugin-memfade
```

## 指令

| 指令 | 作用 |
|---|---|
| `记忆` | 这个会话记住了多少东西 + 最近记下的内容 |
| `记忆.搜 <关键词>` | 检索（必须真命中） |
| `记忆.整理` | 立刻跑一轮整理 |
| `记忆.置顶 <内容>` | 让它永不淡化 |
| `记住 <内容>` | 显式记住 |
| `忘掉 <内容>` | 忘掉一条 |

## 配置

| 键 | 默认 | 说明 |
|---|---|---|
| `endpoint` | `http://127.0.0.1:8787` | 侧车地址 |
| `token` | 空 | 侧车令牌 |
| `topK` | 6 | 每轮最多召回多少条 |
| `extractEnabled` | true | 是否抽取记忆 |
| `extractEveryN` | 3 | 每隔多少条消息抽取一次 |

## 注意

⚠ **绝不原样写入**：抽取走侧车的 `/extract`，让模型把整句话提炼成
「关于这个人的事实」。侧车没配 LLM 时用规则抽取兜底——
**无论哪条路，都不把整句聊天记录当记忆存进去**
（那是聊天记录备份，不是记忆，也是记忆污染最直接的来源）。

侧车接口：`/remember` `/extract` `/recall` `/context` `/list` `/stats`
`/consolidate` `/forget` `/pin`。

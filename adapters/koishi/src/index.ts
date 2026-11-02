/**
 * memfade · Koishi 适配器
 *
 * Koishi 是 Node/TypeScript 生态，而 memfade 的核心是 Python。
 * 这里不去重写一遍淡化算法（那必然走样），而是让核心只跑一份，
 * Koishi 通过 HTTP 侧车（adapters/http/server.py）调用它。
 *
 * 职责：
 * - 中间件旁听消息，按节奏送去抽取；
 * - 每轮请求前取回相关记忆，挂到 session.memfade_context；
 * - 暴露 ctx.memfade 助手 API（recall / remember / context）。
 *
 * 配合 LLM 插件：把 session.memfade_context 拼进 system prompt 即可。
 */

import { Context, Schema } from 'koishi'

export interface MemFadeHit {
  id: number
  content: string
  summary: string
  kind: string
  status: string
  tier: string
  score: number
  relevance: number
  strength: number
  importance: number
  recall_count: number
  scope_key: string
}

export interface MemFadeContext {
  text: string
  token_estimate: number
  facts: MemFadeHit[]
  echoes: MemFadeHit[]
}

export const Config = Schema.object({
  endpoint: Schema.string()
    .default('http://127.0.0.1:8787')
    .description('memfade HTTP 侧车地址'),
  token: Schema.string()
    .default('')
    .description('侧车鉴权令牌（对应服务端的 --token）'),
  extractEnabled: Schema.boolean()
    .default(true)
    .description('是否从对话中抽取记忆'),
  extractEveryN: Schema.number()
    .default(3)
    .description('每隔多少条消息抽取一次'),
  topK: Schema.number()
    .default(6)
    .description('每轮最多召回多少条记忆'),
})

/**
 * 配置**数据**的类型。
 *
 * `Schema.object()` 返回的是 `Schema<输入, 输出>`，
 * 宿主传给 apply 的是「输出」那一侧——解析后的配置数据。
 * 从 schema 直接推导，schema 改了类型跟着改，不会漂移。
 */
export type MemFadeConfig = typeof Config extends Schema<any, infer T> ? T : never

/**
 * 记忆服务。
 *
 * 刻意**不**继承 Koishi 的 `Service`：那套生命周期（cordis 的注册表、
 * inject 依赖注入）在纯插件场景里是多余的负担。插件把它挂在 `ctx.memfade`
 * 上，其它插件直接 `ctx.memfade.context(session, text)` 就行。
 */
export class MemoryService {
  private buffer = new Map<string, string[]>()
  private counter = new Map<string, number>()

  constructor(
    public ctx: Context,
    public config: MemFadeConfig,
  ) {
  }

  /** 会话作用域键：群聊按群号，私聊按用户号。 */
  scopeOf(session: any) {
    const channel = session?.channel?.id ?? session?.guild?.id ?? ''
    const isGroup = Boolean(session?.channel?.type === 'group' || session?.guild?.id)
    return {
      platform: 'koishi',
      chat_type: isGroup ? 'group' : 'private',
      chat_id: String(isGroup ? channel : (session?.author?.id ?? session?.userId ?? 'unknown')),
    }
  }

  private async request(path: string, body: unknown) {
    const headers: Record<string, string> = { 'Content-Type': 'application/json' }
    if (this.config.token) headers['X-MemFade-Token'] = this.config.token
    const response = await fetch(`${this.config.endpoint.replace(/\/$/, '')}${path}`, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
    })
    if (!response.ok) throw new Error(`memfade sidecar ${response.status}`)
    return (await response.json()) as any
  }

  /** 取回这一轮相关的记忆。 */
  async context(session: any, query: string): Promise<MemFadeContext> {
    const result = await this.request('/context', {
      scope: this.scopeOf(session),
      query,
      top_k: this.config.topK,
    })
    return result as MemFadeContext
  }

  /** 显式写入一条记忆。 */
  async remember(session: any, content: string, kind = 'semantic', importance = 0.95) {
    const result = await this.request('/remember', {
      scope: this.scopeOf(session),
      content,
      kind,
      importance,
    })
    return result
  }

  /** 召回记忆（不注入）。 */
  async recall(session: any, query: string): Promise<MemFadeHit[]> {
    const result = await this.request('/recall', {
      scope: this.scopeOf(session),
      query,
      top_k: this.config.topK,
    })
    return (result.hits ?? []) as MemFadeHit[]
  }

  /** 送去抽取。 */
  private async collect(session: any, text: string) {
    const scope = this.scopeOf(session)
    const key = `${scope.chat_type}:${scope.chat_id}`
    const list = this.buffer.get(key) ?? []
    list.push(text)
    this.buffer.set(key, list)
    const count = (this.counter.get(key) ?? 0) + 1
    this.counter.set(key, count)
    if (count % Math.max(1, this.config.extractEveryN) !== 0) return
    this.buffer.set(key, [])
    // 有 LLM 就走抽取：把整句话提炼成「关于这个人的事实」。
    // 原样写入等于把聊天记录塞进记忆库——那是备份，不是记忆。
    const speaker = String(session?.author?.name ?? session?.author?.nickname ?? '').trim()
    try {
      const result = await this.request('/extract', {
        scope,
        messages: list.map((content) => ({ role: 'user', content, speaker })),
      })
      if (result?.llm) return
    } catch (error) {
      // 老侧车没有 /extract，或者没配 LLM——退回原样写入。
    }
    await this.request('/remember', {
      scope,
      drafts: list.map((content) => ({ content, importance: 0.5 })),
    })
  }

  async start() {
    this.ctx.middleware(async (session, next) => {
      const text = String(session.text ?? '').trim()
      if (text) {
        try {
          const memory = await this.context(session, text)
          if (memory.text) {
            session.memfade_context =
              '[系统·长期记忆] 以下是你自己长期记住的内容，属于背景信息，' +
              '不需要主动提起；只有对方问到或话题正好相关时才自然带出来：\n\n' +
              memory.text
          }
        } catch (error) {
          this.ctx.logger('memfade').warn(`召回失败: ${(error as Error).message}`)
        }
        if (this.config.extractEnabled) {
          void this.collect(session, text).catch((error) => {
            this.ctx.logger('memfade').warn(`抽取失败: ${(error as Error).message}`)
          })
        }
      }
      return next()
    })
  }
}

declare module 'koishi' {
  interface Context {
    memfade: MemoryService
  }
}

declare module 'koishi' {
  interface Session {
    memfade_context?: string
  }
}

export const apply = (ctx: Context, config: MemFadeConfig) => {
  ctx.memfade = new MemoryService(ctx, config)
}




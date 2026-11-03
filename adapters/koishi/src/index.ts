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

  /** 这个作用域的统计。 */
  async stats(session: any) {
    const result = await this.request('/stats', { scope: this.scopeOf(session) })
    return result.stats ?? {}
  }

  /** 最近记下的内容——**要看得见**才知道记忆库干不干净。 */
  async list(session: any, limit = 8): Promise<any[]> {
    const result = await this.request('/list', { scope: this.scopeOf(session), limit })
    return (result.memories ?? []) as any[]
  }

  /** 跑一轮整理（重算淡化档位、合并重复、仲裁矛盾）。 */
  async maintain(session: any) {
    const result = await this.request('/consolidate', { scope: this.scopeOf(session) })
    return result.report ?? {}
  }

  /** 置顶一条记忆（此后永不淡化）。 */
  async pin(session: any, content: string, pinned = true) {
    const hits = await this.recall(session, content)
    if (!hits.length) return null
    const target = hits[0]
    await this.request('/pin', { scope: this.scopeOf(session), id: target.id, pinned })
    return target
  }

  /** 忘掉一条记忆。 */
  async forget(session: any, content: string) {
    const result = await this.request('/forget', { scope: this.scopeOf(session), content })
    return result
  }

  /** 改掉记错的那条记忆（老内容 => 新内容）。 */
  async repair(session: any, old: string, next: string) {
    const result = await this.request('/repair', {
      scope: this.scopeOf(session),
      old,
      new: next,
    })
    return result
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
    // ⚠ **绝不原样写入**：那等于把聊天记录塞进记忆库——那是备份，不是记忆，
    // 也是记忆污染最直接的来源（主人引用机器人时，整段原话会进库）。
    // 侧车没配 LLM 时会用规则抽取兜底；真写不进去就什么都不写。
    const speaker = String(session?.author?.name ?? session?.author?.nickname ?? '').trim()
    try {
      const result = await this.request('/extract', {
        scope,
        messages: list.map((content) => ({ role: 'user', content, speaker })),
      })
    } catch (error) {
      // 老侧车没有 /extract 这条路——那就不写。
      // 宁可少记，也不要把整句聊天记录当记忆存进去。
      this.ctx.logger('memfade').warn(`抽取失败：${(error as Error).message}`)
    }
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
  registerCommands(ctx, config)
}

/**
 * 指令：查状态、检索、整理、置顶、修正、记住、忘掉。
 *
 * 只给数字（「记住了 12 条」）没人能判断记忆库干不干净——
 * 得**看得见具体内容**才知道有没有记错东西。
 */
function registerCommands(ctx: Context, config: MemFadeConfig) {
  const service = ctx.memfade

  ctx.command('记忆', '查看这个会话记住了多少东西').action(async ({ session }) => {
    const [stats, recent] = await Promise.all([
      service.stats(session).catch(() => ({})),
      service.list(session, 8).catch(() => []),
    ])
    const byStatus = stats.by_status ?? {}
    const lines = [
      `这个会话目前记住了 ${stats.total ?? 0} 条：` +
        `活跃 ${byStatus.active ?? 0} 条，淡出中 ${byStatus.fading ?? 0} 条，` +
        `已成摘要 ${byStatus.gisted ?? 0} 条，已归档 ${byStatus.archived ?? 0} 条。`,
    ]
    if (recent.length) {
      lines.push('最近记下的：')
      for (const memory of recent) {
        const marker = memory.status === 'active' ? '' : '（已淡出）'
        lines.push(`· ${memory.content}${marker}`)
      }
    }
    return lines.join('\n')
  })

  ctx.command('记忆.搜 <query>').action(async ({ session }, query: string) => {
    const hits = await service.recall(session, query)
    if (!hits.length) return '没找到相关的记忆。'
    const lines = hits.map((hit: any) => {
      const marker = hit.tier === 'echo' ? '（可能已经变了）' : ''
      return `· ${hit.content}${marker}`
    })
    return `记得这些：\n${lines.join('\n')}`
  })

  ctx.command('记忆.整理').action(async ({ session }) => {
    const report = await service.maintain(session)
    return (
      `整理完成：扫描 ${report.scanned ?? 0} 条，淡出 ${report.decayed ?? 0} 条，` +
      `合并 ${report.merged ?? 0} 条，取代 ${report.superseded ?? 0} 条。`
    )
  })

  ctx.command('记忆.置顶 <content>').action(async ({ session }, content: string) => {
    const target = await service.pin(session, content)
    if (!target) return '没找到相关的记忆。'
    return `置顶了：${target.content}`
  })

  ctx.command('记忆.修正 <query>').action(async ({ session }, query: string) => {
    const raw = String(query ?? '').trim()
    const separator = ['=>', '→', '改成', '改为'].find((token) => raw.includes(token))
    if (!separator) return '要改哪条？照这个格式发：记忆.修正 老内容 => 新内容'
    const index = raw.indexOf(separator)
    const old = raw.slice(0, index).trim()
    const next = raw.slice(index + separator.length).trim()
    if (!old || !next) return '要改哪条？照这个格式发：记忆.修正 老内容 => 新内容'
    const result = await service.repair(session, old, next)
    if (!result?.ok) return '没找到要改的那条记忆。'
    return `改好了：${result.previous ?? ''}\n现在记的是：${result.content ?? ''}`
  })

  ctx.command('记住 <content>').action(async ({ session }, content: string) => {
    const result = await service.remember(session, content, 'semantic', 0.9)
    return result?.written ? '记住了。' : '没记下来（内容不重要或与已有记忆重复）。'
  })

  ctx.command('忘掉 <content>').action(async ({ session }, content: string) => {
    const result = await service.forget(session, content)
    return result?.ok ? `忘掉了：${result.content ?? ''}` : '没找到要忘掉的记忆。'
  })
}




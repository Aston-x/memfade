/**
 * Koishi 插件的行为测试。
 *
 * 两层验证：
 * 1. 插件自身的逻辑（作用域解析、往 session 上挂记忆文本）；
 * 2. **与 Python 侧车的真实契约**——起一个真的侧车进程，
 *    用插件的 fetch 打过去，看返回结构对不对。
 *
 * 跑法：
 *   node --import ./test/register-stub.mjs --test test/plugin.test.mjs
 */

import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { after, before, describe, it } from 'node:test'
import { setTimeout as delay } from 'node:timers/promises'

import { Config, MemoryService, apply } from '../lib/index.js'

const PORT = 18787
const ENDPOINT = `http://127.0.0.1:${PORT}`
const DB_PATH = join(tmpdir(), 'memfade-koishi-test.db')

describe('memfade koishi 插件', () => {
  let sidecar = null

  before(async () => {
    // 每轮用全新库：残留记忆会和新内容构成近似重复，
    // 引擎（正确地）复用旧记忆，断言就会失真。
    for (const suffix of ['', '-wal', '-shm']) {
      try {
        rmSync(DB_PATH + suffix)
      } catch {
        // 不存在就算了
      }
    }
    sidecar = spawn(
      process.env.PYTHON || 'python',
      [
        '../http/server.py',
        '--db',
        DB_PATH,
        '--port',
        String(PORT),
      ],
      { stdio: 'ignore' },
    )
    for (let i = 0; i < 50; i++) {
      try {
        const resp = await fetch(`${ENDPOINT}/health`)
        if (resp.ok) break
      } catch {
        await delay(100)
      }
    }
  })

  after(() => {
    sidecar?.kill()
  })

  it('配置声明了侧车地址与令牌', () => {
    assert.ok(Config.dict.endpoint)
    assert.ok(Config.dict.token)
  })

  it('群会话按群号隔离，私聊按用户号', () => {
    const service = new MemoryService({ logger: () => undefined }, {})
    const group = service.scopeOf({
      channel: { id: '1127155154', type: 'group' },
      author: { id: 'u1' },
    })
    assert.equal(group.chat_type, 'group')
    assert.equal(group.chat_id, '1127155154')

    const priv = service.scopeOf({ author: { id: 'u1' } })
    assert.equal(priv.chat_type, 'private')
    assert.equal(priv.chat_id, 'u1')
  })

  it('能真的把记忆写进侧车并读回来', async () => {
    const service = new MemoryService({ logger: () => undefined }, {
      endpoint: ENDPOINT,
      token: '',
      extractEnabled: true,
      extractEveryN: 3,
      topK: 6,
    })
    const session = { channel: { id: 'koishi-group', type: 'group' }, author: { id: 'u1' }, text: '你好' }
    // 侧车库是跨轮次复用的，内容要唯一，否则第二次跑会被去重挡掉
    const content = `许杏玖喜欢辣的东西 #${Date.now()}`
    const written = await service.remember(session, content, 'semantic', 0.9)
    assert.equal(written.ok, true)
    assert.ok(written.written >= 1)

    const ctx = await service.context(session, '喜欢吃什么')
    assert.ok(ctx.text.includes(content), '侧车返回的上下文应含刚写入的记忆')
  })

  it('apply 会注册服务', () => {
    const commands = []
    const ctx = {
      logger: () => undefined,
      // 假 Koishi 的 ctx：记录注册了哪些指令
      command: (def, desc) => {
        const entry = { def, desc, action: null }
        commands.push(entry)
        return {
          action: (fn) => {
            entry.action = fn
            return entry
          },
        }
      },
    }
    apply(ctx, {
      endpoint: ENDPOINT,
      token: '',
      extractEnabled: true,
      extractEveryN: 3,
      topK: 6,
    })
    assert.ok(ctx.memfade, '应该把服务挂到 ctx.memfade 上')
    assert.equal(ctx.memfade.config.topK, 6)
    const names = commands.map((c) => c.def)
    for (const expected of ['记忆', '记忆.搜 <query>', '记忆.整理', '记忆.置顶 <content>', '记住 <content>', '忘掉 <content>']) {
      assert.ok(names.includes(expected), `缺少指令：${expected}`)
      assert.ok(commands.find((c) => c.def === expected).action, `指令没有动作：${expected}`)
    }
  })

  it('记忆 状态 会列出记下的内容', async () => {
    const service = new MemoryService({ logger: () => undefined }, {
      endpoint: ENDPOINT,
      token: '',
      extractEnabled: true,
      extractEveryN: 3,
      topK: 6,
    })
    const session = { channel: { id: 'koishi-group', type: 'group' }, author: { id: 'u1' } }
    const content = `列出内容用的记忆 #${Date.now()}`
    await service.remember(session, content, 'semantic', 0.9)
    const stats = await service.stats(session)
    const recent = await service.list(session, 8)
    assert.ok(stats.total >= 1, '侧车没返回统计')
    assert.ok(
      recent.some((m) => m.content === content),
      '侧车没列出刚写入的记忆',
    )
  })

  it('侧车没配 LLM 时，抽取退回原样写入而不是报错', async () => {
    const service = new MemoryService({ logger: () => undefined }, {
      endpoint: ENDPOINT,
      token: '',
      extractEnabled: true,
      extractEveryN: 1,
      topK: 6,
    })
    const session = { channel: { id: 'koishi-group', type: 'group' }, author: { id: 'u1' } }
    const content = `没有 LLM 时的原样写入 #${Date.now()}`
    // 走的是 middleware 之外的 collect：直接调它，
    // 因为 middleware 需要完整的 Koishi 运行时。
    await service.collect?.(session, content)
    const ctx = await service.context(session, '原样写入')
    assert.ok(ctx.text.includes(content), '没配 LLM 时应该退回 /remember 原样写入')
  })
})


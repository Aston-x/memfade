/**
 * ESM loader hook：把 'koishi' 解析到桩件。
 *
 * 用 node --import ./test/register-stub.mjs 启动测试。
 */

import { register } from 'node:module'
import { pathToFileURL } from 'node:url'

register('./stub-hooks.mjs', import.meta.url)


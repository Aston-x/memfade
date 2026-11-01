/**
 * loader hook 实现：把 'koishi' 指到 koishi-stub.mjs。
 */

const STUB = new URL('./koishi-stub.mjs', import.meta.url).href

export async function resolve(specifier, context, nextResolve) {
  if (specifier === 'koishi') {
    return { url: STUB, shortCircuit: true }
  }
  return nextResolve(specifier, context)
}


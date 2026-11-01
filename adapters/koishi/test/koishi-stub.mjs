/**
 * 最小 koishi 桩件。
 *
 * 真正的 koishi 包在裸 Node 下 import 就会炸（它的 loader 需要自己的运行时），
 * 所以测试里用这个桩件顶替。桩件只实现插件真正用到的东西：
 * Schema（字符串/数字/布尔）、Service（带 ctx.register）、Context。
 */

class SchemaClass {
  constructor(type, def) {
    this.type = type
    this.defaultValue = def
  }

  default(value) {
    const next = new SchemaClass(this.type, value)
    return next
  }

  description(value) {
    const next = new SchemaClass(this.type, this.defaultValue)
    next.desc = value
    return next
  }

  static string() {
    return new SchemaClass('string')
  }

  static number() {
    return new SchemaClass('number')
  }

  static boolean() {
    return new SchemaClass('boolean')
  }

  static object(dict) {
    const schema = new SchemaClass('object')
    schema.dict = dict
    return schema
  }
}

export const Schema = SchemaClass

export class Service {
  constructor(ctx, name) {
    this.ctx = ctx
    this.name = name
  }

  static [Symbol.species]() {
    return this
  }
}

export class Context {
  constructor() {
    this.services = new Map()
    this.middlewares = []
    this.logger = () => undefined
  }

  register(name, value) {
    this.services.set(name, value)
    return value
  }

  middleware(handler) {
    this.middlewares.push(handler)
  }
}

export const service = Service


/**
 * Parsing and formatting of jaxtyping-style shape descriptors.
 *
 * A shape is an array of {@link Token}s, each of which is either:
 *   - a literal integer  (`number`)        e.g. `128`
 *   - a named axis       (`string` "B")    identifiers, e.g. `"B"`, `"C_in"`
 *   - the rest token     (`string` "...")  matches zero or more arbitrary axes
 *
 * Shapes from the manifest arrive as arrays of strings - we coerce numeric
 * strings into numbers here so unification can do strict equality checks.
 */

/**
 * A single dimension of a shape: a concrete size (`number`), a named axis
 * variable (`string`), or the variadic rest marker (`"..."`).
 */
export type Token = number | string

/** An ordered list of shape dimensions, e.g. `["B", "C", 128]`. */
export type Shape = Token[]

/**
 * Substitution map produced by the unifier: maps an axis variable to either a
 * concrete size or another axis variable (a chain the resolver walks to its
 * terminal value).
 */
export type Substitution = Map<string, string | number>

const REST = '...'

export const REST_TOKEN: string = REST

/** Normalize a manifest shape (array of strings/numbers) into {@link Token}s. */
export function normalize(shape: unknown): Shape {
  if (!Array.isArray(shape)) return []
  return shape.map((tok: unknown): Token => {
    if (typeof tok === 'number') return tok
    const s = String(tok).trim()
    if (s === REST) return REST
    if (/^-?\d+$/.test(s)) return Number(s)
    return s
  })
}

/** True if a token is a literal integer. */
export function isLiteral(tok: Token): tok is number {
  return typeof tok === 'number'
}

/** True if a token is a named axis variable (not a literal, not the rest marker). */
export function isVariable(tok: Token): tok is string {
  return typeof tok === 'string' && tok !== REST
}

/** True if a token is the variadic rest marker (`"..."`). */
export function isRest(tok: Token): boolean {
  return tok === REST
}

/** True if the shape contains a rest token (allows an arbitrary middle). */
export function hasRest(shape: Shape): boolean {
  return shape.some(isRest)
}

/** Stringify a shape token list to its display form, e.g. `"B C 128 H W"`. */
export function formatShape(shape: Shape | null | undefined): string {
  if (!shape || shape.length === 0) return '·' // scalar
  return shape.map((t) => (isRest(t) ? '...' : String(t))).join(' ')
}

/**
 * Rename axis variables to be unique to this node instance.
 * For example, two `ConvBlock`s both use axis "B"; we suffix per-node:
 *
 *   freshen(["B","C_in","H","W"], "node12") -> ["B#node12", "C_in#node12", ...]
 *
 * Literals and "..." pass through unchanged. The unifier resolves "B#node12"
 * against "B#node17" when an edge is added.
 */
export function freshen(shape: Shape, nodeId: string): Shape {
  const tag = `#${nodeId}`
  return shape.map((t) => (isVariable(t) ? `${t}${tag}` : t))
}

/** Pretty-print a (possibly-substituted) shape using axis-base names. */
export function prettyShape(shape: Shape | null | undefined, sub: Substitution): string {
  if (!shape || shape.length === 0) return '·'
  return shape
    .map((t) => {
      if (isRest(t)) return '...'
      if (isLiteral(t)) return String(t)
      const resolved = resolve(t, sub)
      if (typeof resolved === 'number') return String(resolved)
      // Strip the #nodeId tag for readability.
      const base = String(resolved).split('#')[0]
      return base
    })
    .join(' ')
}

/** Walk the substitution chain to its terminal value. */
export function resolve(tok: Token, sub: Substitution): Token {
  let cur: Token = tok
  const seen = new Set<Token>()
  while (sub && typeof cur === 'string' && sub.has(cur) && !seen.has(cur)) {
    seen.add(cur)
    cur = sub.get(cur) as Token
  }
  return cur
}

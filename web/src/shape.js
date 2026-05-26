/**
 * Parsing and formatting of jaxtyping-style shape descriptors.
 *
 * A shape is an array of tokens, each of which is either:
 *   - a literal integer  (Number)         e.g. 128
 *   - a named axis       (String "B")     identifiers, e.g. "B", "C_in"
 *   - the rest token     (String "...")   matches zero or more arbitrary axes
 *
 * Shapes from the manifest arrive as arrays of strings - we coerce numeric
 * strings into Numbers here so unification can do strict equality checks.
 */

const REST = '...'

export const REST_TOKEN = REST

/** Normalize a manifest shape (array of strings) into Tokens. */
export function normalize(shape) {
  if (!Array.isArray(shape)) return []
  return shape.map((tok) => {
    if (typeof tok === 'number') return tok
    const s = String(tok).trim()
    if (s === REST) return REST
    if (/^-?\d+$/.test(s)) return Number(s)
    return s
  })
}

/** True if a token is a literal integer. */
export function isLiteral(tok) {
  return typeof tok === 'number'
}

/** True if a token is a named axis variable. */
export function isVariable(tok) {
  return typeof tok === 'string' && tok !== REST
}

/** True if a token is the variadic rest marker. */
export function isRest(tok) {
  return tok === REST
}

/** True if the shape contains a rest token (allows arbitrary middle). */
export function hasRest(shape) {
  return shape.some(isRest)
}

/** Stringify a shape token list to its display form, e.g. "B C 128 H W". */
export function formatShape(shape) {
  if (!shape || shape.length === 0) return '·'  // scalar
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
export function freshen(shape, nodeId) {
  const tag = `#${nodeId}`
  return shape.map((t) => (isVariable(t) ? `${t}${tag}` : t))
}

/** Pretty-print a (possibly-substituted) shape using axis-base names. */
export function prettyShape(shape, sub) {
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
export function resolve(tok, sub) {
  let cur = tok
  const seen = new Set()
  while (sub && sub.has(cur) && !seen.has(cur)) {
    seen.add(cur)
    cur = sub.get(cur)
  }
  return cur
}

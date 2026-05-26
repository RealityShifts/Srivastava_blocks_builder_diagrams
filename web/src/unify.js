/**
 * Hindley-Milner-style unification on shape token lists.
 *
 * The substitution is a Map from axis-variable -> (literal | other variable).
 * Following the chain via `resolve()` (in shape.js) yields the terminal value.
 *
 * - literal vs literal:    must be equal, otherwise error
 * - variable vs literal:   bind variable -> literal
 * - variable vs variable:  bind one to the other (alias)
 * - "..." (rest):          absorbs zero or more axes on either side
 *
 * Rest handling is conservative: we anchor known-length suffixes/prefixes
 * around a single rest token. Multiple rest tokens are treated as wildcard
 * (no constraints between them).
 */

import { isLiteral, isRest, isVariable, resolve, REST_TOKEN } from './shape.js'

export class UnifyError extends Error {
  constructor(message, details) {
    super(message)
    this.name = 'UnifyError'
    this.details = details
  }
}

/** Walk substitution; returns same type as token if no binding. */
function walk(tok, sub) {
  return resolve(tok, sub)
}

function bind(varName, value, sub) {
  // Avoid trivial cycles.
  if (varName === value) return
  sub.set(varName, value)
}

/** Unify two single tokens in-place against `sub`. Throws UnifyError on conflict. */
export function unifyToken(a, b, sub) {
  a = walk(a, sub)
  b = walk(b, sub)
  if (a === b) return
  if (isRest(a) || isRest(b)) return // rest matches anything as a single axis

  if (isLiteral(a) && isLiteral(b)) {
    if (a !== b) {
      throw new UnifyError(`axis literal mismatch: ${a} vs ${b}`, { a, b })
    }
    return
  }
  if (isVariable(a) && isLiteral(b)) {
    bind(a, b, sub)
    return
  }
  if (isLiteral(a) && isVariable(b)) {
    bind(b, a, sub)
    return
  }
  if (isVariable(a) && isVariable(b)) {
    // Alias newer -> older deterministically so the substitution forms a DAG.
    if (a < b) bind(b, a, sub)
    else bind(a, b, sub)
    return
  }
  throw new UnifyError(`cannot unify ${String(a)} with ${String(b)}`, { a, b })
}

/**
 * Unify two full shape arrays in-place against `sub`.
 * Handles a single `...` token on either side by absorbing the excess.
 * Throws UnifyError on conflict; on success the substitution is updated.
 */
export function unifyShape(a, b, sub) {
  const ai = a.findIndex(isRest)
  const bi = b.findIndex(isRest)

  // No rest tokens - must be same arity.
  if (ai === -1 && bi === -1) {
    if (a.length !== b.length) {
      throw new UnifyError(
        `rank mismatch: ${a.length} vs ${b.length}`,
        { a, b }
      )
    }
    for (let i = 0; i < a.length; i++) unifyToken(a[i], b[i], sub)
    return
  }

  // At least one side has rest - anchor on prefix/suffix around the rest.
  const A = ai !== -1 ? a : expandRest(b, a)
  const B = bi !== -1 ? b : expandRest(a, b)
  if (A === null || B === null) {
    throw new UnifyError(
      `incompatible shapes (rest cannot reconcile)`,
      { a, b }
    )
  }
  // Both sides now contain a single rest token in matching positions.
  const aIdx = A.findIndex(isRest)
  const bIdx = B.findIndex(isRest)
  // Anchor prefix
  const prefixLen = Math.min(aIdx, bIdx)
  for (let i = 0; i < prefixLen; i++) unifyToken(A[i], B[i], sub)
  // Anchor suffix
  const aSuf = A.length - aIdx - 1
  const bSuf = B.length - bIdx - 1
  const suffixLen = Math.min(aSuf, bSuf)
  for (let i = 0; i < suffixLen; i++) {
    unifyToken(A[A.length - 1 - i], B[B.length - 1 - i], sub)
  }
}

/**
 * If `withoutRest` has no rest, but the *other side* has rest, we still need
 * to unify pairwise around the rest. This helper returns `withoutRest`
 * unchanged but verifies that an alignment exists (length must be >= the
 * concrete count of `withRest`).
 */
function expandRest(withRest, withoutRest) {
  const restIdx = withRest.findIndex(isRest)
  if (restIdx === -1) return withoutRest
  const concreteCount = withRest.length - 1
  if (withoutRest.length < concreteCount) return null
  return withoutRest
}

/** Snapshot the substitution into a fresh Map (for trial unifications). */
export function cloneSub(sub) {
  return new Map(sub)
}

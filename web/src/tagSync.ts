/**
 * Helpers for keeping nodes in sync.
 *
 * Two orthogonal identities live on every node:
 *   - **name** drives *parameter* synchronization: nodes that share an explicit
 *     name and block type keep their ctor values in lockstep.
 *   - **tag** drives *weight* sharing in codegen (one `self.<attr>` reused at
 *     several call sites).
 *
 * The functions here answer "do these two nodes belong to the same family?" for
 * each identity, plus the low-level value copy used when mirroring params.
 */

import type { NodeLike } from './types.ts'

/** Normalize a tag into a case-insensitive lookup key (trimmed + lowercased). */
export function nodeTagKey(tag: unknown): string {
  return String(tag ?? '')
    .trim()
    .toLowerCase()
}

/** Effective node name: the editable name, falling back to the block type. */
export function nodeNameValue(node: NodeLike | null | undefined): string {
  const n = String(node?.name ?? '').trim()
  return n || node?.entry?.name || ''
}

/**
 * Case-insensitive key used to group nodes for param synchronization. Only an
 * *explicitly set* name participates: a blank name (which displays as the block
 * type) does not auto-sync, so untouched nodes stay independent as before.
 */
export function nodeNameKey(node: NodeLike | null | undefined): string {
  const explicit = String(node?.name ?? '').trim()
  return explicit ? explicit.toLowerCase() : ''
}

/** Weight-sharing family: same block kind + block type + (non-empty) tag key. */
export function nodesInSameTagFamily(a: NodeLike, b: NodeLike): boolean {
  const key = nodeTagKey(a?.tag)
  if (!key || key !== nodeTagKey(b?.tag)) return false
  if (a.entry?.kind !== b.entry?.kind) return false
  if (a.entry?.name !== b.entry?.name) return false
  return true
}

/**
 * Param-sync family: same block kind + same block type + same (non-empty)
 * name key. Nodes sharing a name keep their ctor values in lockstep.
 */
export function nodesInSameNameFamily(a: NodeLike, b: NodeLike): boolean {
  const key = nodeNameKey(a)
  if (!key || key !== nodeNameKey(b)) return false
  if (a.entry?.kind !== b.entry?.kind) return false
  if (a.entry?.name !== b.entry?.name) return false
  return true
}

/** Copy ctor values from source onto target when they share block identity. */
export function copyNodeValues(source: NodeLike, target: NodeLike): boolean {
  if (!source?.entry || !target?.entry) return false
  if (source.entry.kind !== target.entry.kind) return false
  if (source.entry.name !== target.entry.name) return false
  if (!target.values) target.values = {}
  for (const param of source.entry.ctor ?? []) {
    target.values[param.name] = source.values?.[param.name]
  }
  return true
}

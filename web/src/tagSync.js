/**
 * Helpers for keeping nodes that share a tag in sync (same block type + tag).
 */

export function nodeTagKey(tag) {
  return String(tag ?? '').trim().toLowerCase()
}

/** Effective node name: the editable name, falling back to the block type. */
export function nodeNameValue(node) {
  const n = String(node?.name ?? '').trim()
  return n || node?.entry?.name || ''
}

/**
 * Case-insensitive key used to group nodes for param synchronization. Only an
 * *explicitly set* name participates: a blank name (which displays as the block
 * type) does not auto-sync, so untouched nodes stay independent as before.
 */
export function nodeNameKey(node) {
  const explicit = String(node?.name ?? '').trim()
  return explicit ? explicit.toLowerCase() : ''
}

export function nodesInSameTagFamily(a, b) {
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
export function nodesInSameNameFamily(a, b) {
  const key = nodeNameKey(a)
  if (!key || key !== nodeNameKey(b)) return false
  if (a.entry?.kind !== b.entry?.kind) return false
  if (a.entry?.name !== b.entry?.name) return false
  return true
}

/** Copy ctor values from source onto target when they share block identity. */
export function copyNodeValues(source, target) {
  if (!source?.entry || !target?.entry) return false
  if (source.entry.kind !== target.entry.kind) return false
  if (source.entry.name !== target.entry.name) return false
  if (!target.values) target.values = {}
  for (const param of source.entry.ctor ?? []) {
    target.values[param.name] = source.values?.[param.name]
  }
  return true
}

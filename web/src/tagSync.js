/**
 * Helpers for keeping nodes that share a tag in sync (same block type + tag).
 */

export function nodeTagKey(tag) {
  return String(tag ?? '').trim().toLowerCase()
}

export function nodesInSameTagFamily(a, b) {
  const key = nodeTagKey(a?.tag)
  if (!key || key !== nodeTagKey(b?.tag)) return false
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

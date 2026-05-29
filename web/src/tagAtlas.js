/**
 * Tag atlas: a single source of truth for every synced thing on the canvas.
 *
 * A "member" can be either a regular BlockNode or a group:
 *
 *   - Regular nodes are keyed by their *name* (the editable per-instance name,
 *     falling back to the block type). Members sharing a name-key share the
 *     canonical ctor values + which params are exposed as __param__ ports.
 *   - Groups are keyed by their *tag* (facadeTag). Members sharing a tag-key
 *     share the facade name, description, and boundary signature.
 *
 * The flow is always:
 *   1. Member mutates its own copy via the inspector.
 *   2. `recordValueChange` / `recordGroupMeta` updates the atlas entry.
 *   3. The atlas reports every other member that needs to be brought in sync.
 *
 * No DOM / Rete references live here so the module stays test-friendly.
 */

import { nodeTagKey, nodeNameKey, copyNodeValues } from './tagSync.js'

/** Family used to decide if two members can be merged under one tag. */
function memberFamily(node) {
  if (!node?.entry) return null
  if (node.entry.kind === 'group') return `group:${node.entry.name}`
  if (node.entry.kind === 'module' || node.entry.kind === 'learnable') {
    return `${node.entry.kind}:${node.entry.name}`
  }
  return `${node.entry.kind}:${node.entry.name}`
}

function blankEntry(tag, family) {
  return {
    tag,
    family,
    blockName: '',
    values: {},
    exposedParams: new Set(),
    description: '',
    name: '',
    boundarySignature: null,
    members: new Set(),
  }
}

function exposedParamSet(node) {
  const out = new Set()
  for (const portName of Object.keys(node?.inputs ?? {})) {
    if (portName.startsWith('__param__')) out.add(portName.slice('__param__'.length))
  }
  return out
}

export function makeTagAtlas() {
  return new Map()
}

export function getAtlasEntry(atlas, tag) {
  const key = nodeTagKey(tag)
  if (!key) return null
  return atlas.get(key) ?? null
}

/** Look up the entry a regular node belongs to (keyed by its name). */
export function getNodeAtlasEntry(atlas, node) {
  const key = nodeNameKey(node)
  if (!key) return null
  return atlas.get(key) ?? null
}

/**
 * Register a tagged BlockNode against the atlas. Returns the entry the node
 * now belongs to. When the entry was newly created the node's own values
 * become the canonical seed; when joining an existing entry the caller is
 * expected to apply the canonical values back onto the node.
 */
export function registerNodeMember(atlas, node) {
  const key = nodeNameKey(node)
  if (!key) return null
  const family = memberFamily(node)
  let entry = atlas.get(key)
  if (!entry || entry.family !== family) {
    entry = blankEntry(key, family)
    entry.blockName = node.entry?.name ?? ''
    entry.values = { ...(node.values ?? {}) }
    entry.exposedParams = exposedParamSet(node)
    atlas.set(key, entry)
  }
  entry.members.add(node.id)
  return entry
}

/**
 * Register a group facade against the atlas. The atlas stores both the
 * canonical groupId (any one) and the user-visible name + description so
 * peers can adopt them.
 */
export function registerGroupMember(atlas, group, facadeNode) {
  const key = nodeTagKey(group?.facadeTag ?? facadeNode?.tag)
  if (!key) return null
  const family = `group:${facadeNode?.entry?.name ?? group.name ?? 'Group'}`
  let entry = atlas.get(key)
  if (!entry || entry.family !== family) {
    entry = blankEntry(String(group.facadeTag ?? facadeNode?.tag ?? ''), family)
    entry.blockName = facadeNode?.entry?.name ?? group.name ?? 'Group'
    entry.name = group.name ?? ''
    entry.description = group.description ?? ''
    atlas.set(key, entry)
  }
  entry.members.add(group.id)
  return entry
}

export function unregisterMember(atlas, tag, memberId) {
  const key = nodeTagKey(tag)
  if (!key) return
  const entry = atlas.get(key)
  if (!entry) return
  entry.members.delete(memberId)
  if (entry.members.size === 0) atlas.delete(key)
}

/**
 * Push the atlas's canonical values onto a node and return the list of
 * params that actually changed (so the caller can decide what to refresh).
 */
export function adoptValuesFromAtlas(atlas, node) {
  const entry = getNodeAtlasEntry(atlas, node)
  if (!entry) return []
  const changed = []
  if (!node.values) node.values = {}
  for (const [paramName, value] of Object.entries(entry.values)) {
    if (node.values[paramName] !== value) {
      node.values[paramName] = value
      changed.push(paramName)
    }
  }
  return changed
}

/** Should the exposed-param set on a freshly-joined node be synced too? */
export function adoptExposedParamsFromAtlas(atlas, node) {
  const entry = getNodeAtlasEntry(atlas, node)
  if (!entry) return { toExpose: [], toHide: [] }
  const current = exposedParamSet(node)
  const target = entry.exposedParams
  const toExpose = []
  const toHide = []
  for (const name of target) if (!current.has(name)) toExpose.push(name)
  for (const name of current) if (!target.has(name)) toHide.push(name)
  return { toExpose, toHide }
}

/**
 * Record a value change on a member and return the peer ids that should be
 * updated. The new value becomes canonical for the tag key.
 */
export function recordValueChange(atlas, sourceNode, paramName) {
  const entry = getNodeAtlasEntry(atlas, sourceNode)
  if (!entry) return []
  entry.values[paramName] = sourceNode.values?.[paramName]
  return [...entry.members].filter((id) => id !== sourceNode.id)
}

/** Bulk-record all ctor values from a node (used when seeding from a peer). */
export function recordAllValues(atlas, sourceNode) {
  const entry = getNodeAtlasEntry(atlas, sourceNode)
  if (!entry || !sourceNode?.values) return []
  for (const key of Object.keys(sourceNode.values)) {
    entry.values[key] = sourceNode.values[key]
  }
  return [...entry.members].filter((id) => id !== sourceNode.id)
}

/**
 * Record an exposed-param toggle and return peers that should mirror the
 * structural change.
 */
export function recordExposedParamChange(atlas, sourceNode, paramName, isExposed) {
  const entry = getNodeAtlasEntry(atlas, sourceNode)
  if (!entry) return []
  if (isExposed) entry.exposedParams.add(paramName)
  else entry.exposedParams.delete(paramName)
  return [...entry.members].filter((id) => id !== sourceNode.id)
}

/** Update group meta (name/description) for the tag family. */
export function recordGroupMeta(atlas, sourceGroup, patch) {
  const entry = getAtlasEntry(atlas, sourceGroup?.facadeTag)
  if (!entry) return []
  if (patch.name !== undefined) entry.name = String(patch.name ?? '')
  if (patch.description !== undefined) entry.description = String(patch.description ?? '')
  if (patch.boundarySignature !== undefined) entry.boundarySignature = patch.boundarySignature
  return [...entry.members].filter((gid) => gid !== sourceGroup.id)
}

/** Rebuild the atlas from the current editor + groups snapshot. */
export function rebuildAtlas({ editor, groups, getNode }) {
  const atlas = makeTagAtlas()
  for (const node of editor.getNodes()) {
    if (node?.entry?.kind === 'group') continue // groups go via the groups map
    if (!nodeNameKey(node)) continue
    registerNodeMember(atlas, node)
  }
  if (groups) {
    for (const group of groups.values()) {
      const facade = group.facadeNodeId ? getNode(group.facadeNodeId) : null
      if (!nodeTagKey(group.facadeTag ?? facade?.tag)) continue
      const entry = registerGroupMember(atlas, group, facade)
      if (entry && facade && !entry.boundarySignature) {
        entry.name = group.name ?? entry.name
        entry.description = group.description ?? entry.description
      }
    }
  }
  return atlas
}

/** Convenience: source-of-truth check used by tests. */
export function atlasSummary(atlas) {
  const out = {}
  for (const [key, entry] of atlas) {
    out[key] = {
      tag: entry.tag,
      family: entry.family,
      memberCount: entry.members.size,
      values: { ...entry.values },
      exposedParams: [...entry.exposedParams].sort(),
      name: entry.name,
      description: entry.description,
    }
  }
  return out
}

// Re-export for callers that previously imported from tagSync.
export { nodeTagKey, nodeNameKey, copyNodeValues }

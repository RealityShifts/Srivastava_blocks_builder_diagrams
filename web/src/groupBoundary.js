/**
 * Boundary signatures for tagged group facades. Groups sharing a tag must
 * expose the same receiver/outlet layout (counts, shapes, param ports).
 */

function portSpec(port) {
  return {
    shape: port?.shape ?? ['...'],
    dtype: port?.dtype ?? 'any',
    optional: Boolean(port?.optional),
  }
}

function paramSpec(m) {
  return {
    paramName: m.paramName,
    paramType: m.paramType ?? 'int',
  }
}

/** Extract a tag-syncable signature from a group facade entry. */
export function boundarySignatureFromEntry(entry) {
  const pm = entry?.portMap ?? {}
  return {
    inputs: (entry?.inputs ?? []).map(portSpec),
    outputs: (entry?.outputs ?? []).map(portSpec),
    params: (pm.params ?? []).map(paramSpec),
  }
}

/** Build a signature from a computeBoundary / makeGroupEntry boundary object. */
export function boundarySignatureFromBoundary(boundary) {
  return {
    inputs: (boundary.inputs ?? []).map((b) => ({
      shape: b.shape ?? ['...'],
      dtype: b.dtype ?? 'any',
      optional: Boolean(b.optional),
    })),
    outputs: (boundary.outputs ?? []).map((b) => ({
      shape: b.shape ?? ['...'],
      dtype: b.dtype ?? 'any',
      optional: false,
    })),
    params: (boundary.params ?? []).map(paramSpec),
  }
}

function shapeKey(shape) {
  return JSON.stringify(shape ?? ['...'])
}

function portListsMatch(a, b) {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i++) {
    const x = a[i]
    const y = b[i]
    if (shapeKey(x.shape) !== shapeKey(y.shape)) return false
    if ((x.dtype ?? 'any') !== (y.dtype ?? 'any')) return false
    if (Boolean(x.optional) !== Boolean(y.optional)) return false
  }
  return true
}

function paramListsMatch(a, b) {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i++) {
    if (a[i].paramName !== b[i].paramName) return false
    if ((a[i].paramType ?? 'int') !== (b[i].paramType ?? 'int')) return false
  }
  return true
}

/** True when two boundary signatures describe the same facade interface. */
export function boundarySignaturesMatch(a, b) {
  if (!a || !b) return false
  return (
    portListsMatch(a.inputs ?? [], b.inputs ?? []) &&
    portListsMatch(a.outputs ?? [], b.outputs ?? []) &&
    paramListsMatch(a.params ?? [], b.params ?? [])
  )
}

function pickBinding(localItem, portMapItem) {
  const childNodeId = localItem?.childNodeId ?? portMapItem?.childNodeId ?? null
  const childPort = localItem?.childPort ?? portMapItem?.childPort ?? null
  if (!childNodeId || !childPort) return {}
  return { childNodeId, childPort }
}

function paramBinding(localItem, portMapItems) {
  if (localItem?.childNodeId && localItem?.childPort) {
    return { childNodeId: localItem.childNodeId, childPort: localItem.childPort }
  }
  const byName = (portMapItems ?? []).find((m) => m.paramName === localItem?.paramName)
  if (byName?.childNodeId && byName?.childPort) {
    return { childNodeId: byName.childNodeId, childPort: byName.childPort }
  }
  return {}
}

/**
 * Align a locally computed boundary to a shared tag template. Preserves child
 * bindings by port index (params match by paramName).
 */
export function applySignatureToBoundary(localBoundary, signature, portMap = {}) {
  const local = localBoundary ?? { inputs: [], outputs: [], params: [] }
  const sig = signature ?? boundarySignatureFromBoundary(local)

  const inputs = (sig.inputs ?? []).map((spec, i) => ({
    ...spec,
    ...pickBinding(local.inputs?.[i], portMap.inputs?.[i]),
  }))
  const outputs = (sig.outputs ?? []).map((spec, i) => ({
    ...spec,
    ...pickBinding(local.outputs?.[i], portMap.outputs?.[i]),
  }))
  const params = (sig.params ?? []).map((spec) => {
    const localParam =
      local.params?.find((p) => p.paramName === spec.paramName) ??
      local.params?.[(sig.params ?? []).indexOf(spec)]
    return {
      ...spec,
      ...paramBinding(localParam, portMap.params),
    }
  })

  return { ...local, inputs, outputs, params }
}

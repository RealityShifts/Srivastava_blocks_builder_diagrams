/**
 * Boundary signatures for tagged group facades. Groups sharing a tag must
 * expose the same receiver/outlet layout (counts, shapes, param ports) so the
 * single generated class is correct for every instance.
 */

import type { Shape } from './shape.ts'

/** A child-port binding: which inner node/port a boundary slot maps to. */
export interface ChildBinding {
  childNodeId?: string | null
  childPort?: string | null
}

/** A boundary tensor slot (an input receiver or an output outlet). */
export interface BoundaryPort extends ChildBinding {
  shape: Shape
  dtype?: string
  optional?: boolean
}

/** A boundary param slot (an exposed `__param__` port routed to a child). */
export interface BoundaryParam extends ChildBinding {
  paramName: string
  paramType?: string
}

/** A group's full boundary: ordered inputs, outputs, and param ports. */
export interface Boundary {
  inputs: BoundaryPort[]
  outputs: BoundaryPort[]
  params: BoundaryParam[]
}

/** Comparison shape of a single tensor port, ignoring child bindings. */
interface PortSig {
  shape: Shape
  dtype: string
  optional: boolean
}

/** Comparison shape of a single param port, ignoring child bindings. */
interface ParamSig {
  paramName: string
  paramType: string
}

/** The interface-only fingerprint two tagged facades must agree on. */
export interface BoundarySignature {
  inputs: PortSig[]
  outputs: PortSig[]
  params: ParamSig[]
}

/** A facade entry's portMap, plus the loose inputs/outputs read off the entry. */
interface FacadeEntryLike {
  inputs?: Array<Partial<BoundaryPort>>
  outputs?: Array<Partial<BoundaryPort>>
  portMap?: { params?: Array<Partial<BoundaryParam>> }
}

/** An optional shared portMap used to recover child bindings by index/name. */
interface PortMap {
  inputs?: Array<ChildBinding | undefined>
  outputs?: Array<ChildBinding | undefined>
  params?: Array<BoundaryParam | undefined>
}

function portSpec(port: Partial<BoundaryPort> | undefined): PortSig {
  return {
    shape: port?.shape ?? ['...'],
    dtype: port?.dtype ?? 'any',
    optional: Boolean(port?.optional),
  }
}

function paramSpec(m: Partial<BoundaryParam>): ParamSig {
  return {
    paramName: m.paramName ?? '',
    paramType: m.paramType ?? 'int',
  }
}

/** Extract a tag-syncable signature from a group facade entry. */
export function boundarySignatureFromEntry(entry: FacadeEntryLike | undefined): BoundarySignature {
  const pm = entry?.portMap ?? {}
  return {
    inputs: (entry?.inputs ?? []).map(portSpec),
    outputs: (entry?.outputs ?? []).map(portSpec),
    params: (pm.params ?? []).map(paramSpec),
  }
}

/** Build a signature from a computeBoundary / makeGroupEntry boundary object. */
export function boundarySignatureFromBoundary(boundary: Partial<Boundary>): BoundarySignature {
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

/**
 * Normalize a shape into a stable comparison key. Variable axes carry a
 * `#<nodeId>` suffix from `freshen()` (so two `B` axes on different child
 * instances unify) - those suffixes are not part of the *interface* though,
 * so we strip them here. Without this, two structurally-identical group
 * facades built from different children would always look different.
 */
function shapeKey(shape: Shape | undefined): string {
  const toks = (shape ?? ['...']).map((t) => {
    if (typeof t === 'number') return t
    const s = String(t)
    if (s === '...') return s
    const hash = s.indexOf('#')
    return hash >= 0 ? s.slice(0, hash) : s
  })
  return JSON.stringify(toks)
}

function portListsMatch(a: PortSig[], b: PortSig[]): boolean {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i++) {
    const x = a[i]!
    const y = b[i]!
    if (shapeKey(x.shape) !== shapeKey(y.shape)) return false
    if ((x.dtype ?? 'any') !== (y.dtype ?? 'any')) return false
    if (Boolean(x.optional) !== Boolean(y.optional)) return false
  }
  return true
}

function paramListsMatch(a: ParamSig[], b: ParamSig[]): boolean {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i++) {
    if (a[i]!.paramName !== b[i]!.paramName) return false
    if ((a[i]!.paramType ?? 'int') !== (b[i]!.paramType ?? 'int')) return false
  }
  return true
}

/** True when two boundary signatures describe the same facade interface. */
export function boundarySignaturesMatch(
  a: BoundarySignature | null | undefined,
  b: BoundarySignature | null | undefined
): boolean {
  if (!a || !b) return false
  return (
    portListsMatch(a.inputs ?? [], b.inputs ?? []) &&
    portListsMatch(a.outputs ?? [], b.outputs ?? []) &&
    paramListsMatch(a.params ?? [], b.params ?? [])
  )
}

function pickBinding(
  localItem: ChildBinding | undefined,
  portMapItem: ChildBinding | undefined
): ChildBinding {
  const childNodeId = localItem?.childNodeId ?? portMapItem?.childNodeId ?? null
  const childPort = localItem?.childPort ?? portMapItem?.childPort ?? null
  if (!childNodeId || !childPort) return {}
  return { childNodeId, childPort }
}

function paramBinding(
  localItem: BoundaryParam | undefined,
  portMapItems: Array<BoundaryParam | undefined> | undefined
): ChildBinding {
  if (localItem?.childNodeId && localItem?.childPort) {
    return { childNodeId: localItem.childNodeId, childPort: localItem.childPort }
  }
  const byName = (portMapItems ?? []).find((m) => m?.paramName === localItem?.paramName)
  if (byName?.childNodeId && byName?.childPort) {
    return { childNodeId: byName.childNodeId, childPort: byName.childPort }
  }
  return {}
}

/**
 * Align a locally computed boundary to a shared tag template. Preserves child
 * bindings by port index (params match by paramName).
 */
export function applySignatureToBoundary(
  localBoundary: Boundary | null | undefined,
  signature: BoundarySignature | null | undefined,
  portMap: PortMap = {}
): Boundary {
  const local: Boundary = localBoundary ?? { inputs: [], outputs: [], params: [] }
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

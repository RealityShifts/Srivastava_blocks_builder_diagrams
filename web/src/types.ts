/**
 * Shared domain types for the blocks-builder.
 *
 * These describe the *manifest* (the static catalogue of blocks the user can
 * drop onto the canvas) and the *runtime node* shape that wraps a manifest
 * entry once instantiated in the Rete editor. They are intentionally the single
 * source of truth imported across modules so a change to, say, a ctor-param
 * field ripples through codegen, the validator, and the inspector at once.
 */

import type { Shape, Substitution } from './shape.ts'
import type { BoundarySignature } from './groupBoundary.ts'

/** The category of a block, which decides how it is rendered and code-generated. */
export type NodeKind =
  | 'module' // a real nn.Module that owns weights (e.g. ConvBlock)
  | 'group' // a collapsed subgraph, code-generated as its own class
  | 'input' // a synthetic graph source (forward() argument)
  | 'output' // an explicit graph sink (forward() return value)
  | 'const' // a scalar constant feeding an init-time param port
  | 'constmath' // arithmetic on a wired scalar, feeding an init-time param port
  | 'learnable' // a learnable nn.Parameter / buffer
  | 'rearrange' // einops rearrange op
  | 'reshape' // tensor reshape op
  | 'unbind' // split one tensor along an axis into N outputs (inverse of stack)
  | string // forward-compatible: unknown kinds from future manifests

/** A constructor (init-time) parameter of a block, e.g. `out_ch: int = 64`. */
export interface CtorParam {
  /** Parameter name as it appears in the Python signature. */
  name: string
  /** Declared type hint, e.g. `'int'`, `'float'`, `'bool'`, `'str'`. */
  type: string
  /** Default value, or `null` when the parameter is required with no default. */
  default?: unknown
  /** Whether the parameter must be supplied for valid codegen. */
  required?: boolean
  /** Optional enumerated choices surfaced as a dropdown in the inspector. */
  choices?: string[]
}

/** A tensor port (input or output) on a block. */
export interface Port {
  /** Port name, used as the keyword argument / attribute in generated code. */
  name: string
  /** The port's declared shape as manifest tokens, e.g. `['B', 'C', 'H', 'W']`. */
  shape: Shape
  /** Declared dtype, e.g. `'float'`, `'int'`, `'any'`. */
  dtype?: string
  /** Optional ports may be left unconnected without a validation error. */
  optional?: boolean
  /** Variadic ports accept multiple incoming connections. */
  variadic?: boolean
}

/**
 * A manifest entry: the static description of a block as loaded from the block
 * library (or one of the synthetic built-ins like Input/Output/Constant). Once
 * placed on the canvas it is frozen and referenced by every {@link NodeLike}.
 */
export interface ManifestEntry {
  /** Block type name, e.g. `'ConvBlock'`. Drives imports and class references. */
  name: string
  /** Python module the block is imported from, or a `__builtin__`/`__*__` sentinel. */
  module: string
  /** Framework this block targets, e.g. `'pytorch'`, or `'any'` for built-ins. */
  framework?: string
  /** The block category - see {@link NodeKind}. */
  kind: NodeKind
  /** Constructor parameters exposed in the inspector. */
  ctor: CtorParam[]
  /** Input tensor ports. */
  inputs: Port[]
  /** Output tensor ports. */
  outputs: Port[]
  /** Maps a shape axis to the ctor param that determines it, e.g. `{ C_out: 'out_ch' }`. */
  bindings?: Record<string, string>
  /** Present on group facades: the id of the group this entry represents. */
  groupId?: string
  /** Free-form human-readable description shown in the Info tab. */
  description?: string
  /** Allow forward-compatible extra manifest fields without a cast. */
  [key: string]: unknown
}

/**
 * The minimal node surface the sync/codegen/validation helpers rely on. The
 * real runtime object is a Rete `ClassicPreset.Node` (see {@link BlockNode} in
 * nodes.ts) which structurally satisfies this and adds port plumbing.
 */
export interface NodeLike {
  /** Stable unique id assigned by Rete. */
  id: string
  /** The frozen manifest entry this node instantiates. */
  entry: ManifestEntry
  /** Editable per-instance name; blank means "use the block type". */
  name?: string
  /** Weight-sharing / annotation tag. */
  tag?: string
  /** The owning group's id, or `null`/`undefined` for top-level nodes. */
  groupId?: string | null
  /** Per-instance constructor parameter values, keyed by ctor param name. */
  values?: Record<string, unknown>
  /** Rete input ports keyed by port name (includes exposed `__param__*` ports). */
  inputs?: Record<string, unknown>
  /** Rete output ports keyed by port name. */
  outputs?: Record<string, unknown>
}

/**
 * A group: a subgraph collapsed behind a single *facade* node. The group's
 * **name** drives the generated Python class; its **facadeTag** drives weight
 * sharing between instances of the same group.
 */
export interface Group {
  /** Unique group id (also stored on the facade entry's `groupId`). */
  id: string
  /** User-editable group name -> generated class name. */
  name: string
  /** Free-form description shown in the inspector. */
  description: string
  /** Weight-sharing tag for the group facade (mirrors a node's `tag`). */
  facadeTag: string
  /** Whether the group is currently collapsed (facade visible, children hidden). */
  collapsed: boolean
  /**
   * The outer group this group is nested inside, or `null`/`undefined` when the
   * group is top-level. This is the durable source of truth for containment:
   * the facade NODE also carries it as `node.groupId`, but the facade is
   * destroyed and recreated across collapse/expand cycles, so membership is
   * tracked here on the group descriptor and re-applied to each new facade.
   */
  memberOf?: string | null
  /** Id of the facade node representing this group on the canvas. */
  facadeNodeId: string
  /** Maps facade boundary ports to their inner child ports. */
  portMap: unknown
  /** Last known facade position, restored on collapse. */
  savedPosition?: { x: number; y: number }
  /** Per-child pixel offsets from the facade anchor, for layout restoration. */
  childOffsets?: Record<string, { dx: number; dy: number }>
}

/**
 * One entry in the {@link TagAtlas}: the canonical, shared state for everything
 * that lives under a single key (a name key for nodes, a tag key for groups).
 */
export interface AtlasEntry {
  /** The original (un-normalized) key string that seeded this entry. */
  tag: string
  /** Family discriminator (`"<kind>:<blockName>"`) preventing cross-type merges. */
  family: string | null
  /** Block type name shared by all members. */
  blockName: string
  /** Canonical ctor values mirrored onto every member. */
  values: Record<string, unknown>
  /** Names of ctor params currently exposed as `__param__` ports. */
  exposedParams: Set<string>
  /** Canonical group description (groups only). */
  description: string
  /** Canonical group name (groups only). */
  name: string
  /** Canonical boundary signature (group facades only). */
  boundarySignature: BoundarySignature | null
  /** Ids of every member (node id or group id) sharing this key. */
  members: Set<string>
}

/** The atlas itself: a map from a normalized key to its shared {@link AtlasEntry}. */
export type TagAtlas = Map<string, AtlasEntry>

/** A single edge between two node ports in the graph. */
export interface Connection {
  id: string
  source: string
  sourceOutput: string
  target: string
  targetInput: string
}

/**
 * The slice of the Rete editor the pure modules (codegen, validator, runtime)
 * rely on. The real editor is a `NodeEditor` from Rete; this structural subset
 * keeps those modules decoupled from the plugin wiring.
 */
export interface GraphEditor {
  getNodes(): NodeLike[]
  getConnections(): Connection[]
  getNode?(id: string): NodeLike | null | undefined
  /** Cached substitution from the last validation run (set by main.ts). */
  __lastValidationSub?: Substitution
}

/** Re-export so consumers can name the substitution type without reaching into shape. */
export type { Substitution } from './shape.ts'

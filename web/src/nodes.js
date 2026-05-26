/**
 * Build Rete nodes from manifest entries.
 *
 * Each `BlockNode` is a `ClassicPreset.Node` enriched with:
 *   - `entry`:   the original manifest entry (frozen)
 *   - `values`:  per-instance ctor-param values (mutated by Inspector)
 *   - `axisShape(port, side)`: returns a *freshened* shape, with axis variables
 *     suffixed by the node id so different node instances don't collide.
 *
 * A single global `tensor` Socket is used for all connections; shape
 * validation is performed by the validator on top of the type system.
 */

import { ClassicPreset } from 'rete'
import { normalize, freshen } from './shape.js'

export const tensorSocket = new ClassicPreset.Socket('tensor')

export class BlockNode extends ClassicPreset.Node {
  constructor(entry) {
    super(entry.name)
    this.entry = entry
    this.values = Object.fromEntries(
      entry.ctor.map((p) => [p.name, p.default])
    )
    // Cache normalized shapes (raw, without per-node freshening).
    this._inputShapes = Object.fromEntries(
      entry.inputs.map((p) => [p.name, normalize(p.shape)])
    )
    this._outputShapes = Object.fromEntries(
      entry.outputs.map((p) => [p.name, normalize(p.shape)])
    )

    for (const p of entry.inputs) {
      const input = new ClassicPreset.Input(
        tensorSocket,
        labelFor(p),
        Boolean(p.variadic)
      )
      input.multipleConnections = Boolean(p.variadic)
      // Stash the manifest port spec for later inspection.
      input.portSpec = p
      this.addInput(p.name, input)
    }
    for (const p of entry.outputs) {
      const output = new ClassicPreset.Output(tensorSocket, labelFor(p))
      output.portSpec = p
      this.addOutput(p.name, output)
    }
  }

  /** Freshened shape for the given port, scoped to this node's id. */
  freshenedShape(portName, side /* 'in' | 'out' */) {
    const raw =
      side === 'in' ? this._inputShapes[portName] : this._outputShapes[portName]
    if (!raw) return null
    return freshen(raw, this.id)
  }

  /** Apply ctor-param -> axis bindings to the substitution map. */
  applyParamBindings(sub) {
    const bindings = this.entry.bindings || {}
    for (const [axis, paramName] of Object.entries(bindings)) {
      const value = this.values[paramName]
      if (value === null || value === undefined || value === '') continue
      const v = Number(value)
      if (!Number.isFinite(v) || !Number.isInteger(v)) continue
      const freshAxis = `${axis}#${this.id}`
      // Set the binding directly. The unifier walks substitutions, so a later
      // unifyShape() will see this as the resolved value of freshAxis.
      const existing = sub.get(freshAxis)
      if (existing !== undefined && existing !== v) {
        // Conflict between user-provided ctor value and an already-derived binding.
        // We still write it - the validator surfaces the resulting mismatch.
      }
      sub.set(freshAxis, v)
    }
  }
}

function labelFor(port) {
  const tag = port.variadic ? '[*]' : port.optional ? '?' : ''
  return `${port.name}${tag}`
}

// ---------------------------------------------------------------------------
// Input node - synthetic source node with user-defined shape & dtype.
// ---------------------------------------------------------------------------

/**
 * Built-in palette entry for an Input node. Mixed literals and named axes are
 * fine ("B 3 224 224", "B C H W", "2 3 64 64"). Not tied to any framework.
 */
export const INPUT_ENTRY = {
  name: 'Input',
  module: '__builtin__',
  framework: 'any',
  kind: 'input',
  ctor: [
    { name: 'name', type: 'str', default: 'x', required: false },
    { name: 'shape', type: 'str', default: 'B C H W', required: false },
    {
      name: 'dtype',
      type: 'str',
      default: 'float',
      required: false,
      choices: ['float', 'int', 'bool', 'complex', 'any'],
    },
  ],
  inputs: [],
  outputs: [
    {
      name: 'out',
      shape: ['B', 'C', 'H', 'W'],
      dtype: 'float',
      optional: false,
      variadic: false,
    },
  ],
  bindings: {},
}

/** "B, 3,224 224" -> ["B", "3", "224", "224"] */
export function parseShapeString(s) {
  if (s == null) return []
  return String(s)
    .trim()
    .split(/[\s,]+/)
    .filter((tok) => tok.length > 0)
}

/**
 * Source node whose output shape and dtype come from user-typed values rather
 * than a manifest annotation. Inherits everything else from BlockNode so the
 * inspector / palette / unifier treat it like a normal entry.
 */
export class InputNode extends BlockNode {
  freshenedShape(portName, side) {
    if (side !== 'out') return null
    const tokens = normalize(parseShapeString(this.values.shape))
    // Keep the rete port's dtype tag in sync so the validator's dtype check
    // sees the latest value the user typed.
    const port = this.outputs[portName]
    if (port?.portSpec) port.portSpec.dtype = this.values.dtype || 'any'
    return freshen(tokens, this.id)
  }
}

/** Pick the right node class for a manifest entry. */
export function makeNode(entry) {
  if (entry.kind === 'input') return new InputNode(entry)
  return new BlockNode(entry)
}

/** Group manifest entries by their submodule for the palette UI. */
export function groupByModule(entries) {
  const groups = new Map()
  for (const e of entries) {
    const tail =
      e.module === '__builtin__' ? 'built-in' : e.module.split('.').slice(-1)[0]
    if (!groups.has(tail)) groups.set(tail, [])
    groups.get(tail).push(e)
  }
  for (const list of groups.values())
    list.sort((a, b) => a.name.localeCompare(b.name))
  // Keep "built-in" pinned at the top of the palette.
  return [...groups.entries()].sort(([a], [b]) => {
    if (a === 'built-in') return -1
    if (b === 'built-in') return 1
    return a.localeCompare(b)
  })
}

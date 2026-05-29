# Tree/Node Restructure — Implementation Plan

Status: PLAN (no code written yet). Approach chosen: **plan first, build incrementally,
keeping the app runnable at every step.**

## Goal

Replace the entangled "flat Rete nodes + `state.groups` side-table + facade
recreate/remap" model with a **definition-vs-instance** model:

- A **Tree** is a reusable *class definition* (the main graph, every group, AND
  every leaf block are all Trees).
- A **Node** is an *instance* that references a Tree by `name`.
- Collapse/expand is a **pure derived computation** (`dynamicMainTree`), never a
  mutation of live editor state. This kills the bug class behind the recent
  "collapse / nested-groups / residual-boundary" fixes.

## Decisions locked with the user

1. **Global node dict owns Node objects; Trees hold only id strings.**
   `nodes: Record<NodeId, Node>` is the single store. `Tree.list_of_nodes: NodeId[]`.
   A node belongs to exactly one tree (its owning tree is implicit / indexed once).
2. **Shape inference is per-tree, composed at boundaries.** Each Tree computes its
   own `Signature` (dangling inputs, outputs, constants) in isolation via
   `getInputOutputParamsSignature(tree)`. Facades unify child-tree signatures at
   their boundary. This is what makes a Tree a genuine reusable class.
3. **Incremental build, app stays runnable.** New core lives alongside the old
   model behind conversion shims until each consumer (codegen, render, validator,
   persistence) is ported.

## Core data model (new file: `web/src/tree/model.ts`)

```ts
type NodeId = string          // uuid per instance
type TreeName = string        // unique; also the generated class name

interface Connection {
  from: NodeId
  fromOutput: string
  to: NodeId
  toInput: string
  // unique_id = `${from}::${fromOutput}->${to}::${toInput}`  (FIXED, collision-free)
  id: string
}

interface ParamSpec { value: unknown; dtype: string; shape?: Shape; /* … */ }

interface Tree {
  name: TreeName                         // unique
  list_of_nodes: NodeId[]
  list_of_connections: Connection[]
  // boundary refs: "<localName>" for a dangling port, or "<childId>@<port>" routed inward
  inputs: string[]
  outputs: string[]
  params: Record<string, ParamSpec>
}

interface Node {
  id: NodeId
  name: TreeName                         // which Tree this instantiates (block OR group OR custom)
  tag?: string                           // weight-sharing / class property name
  values?: Record<string, unknown>       // per-instance ctor param overrides
  // inputs/outputs are DERIVED from the referenced Tree's signature, not stored
}

interface Forest {
  nodes: Record<NodeId, Node>            // the single global node store
  trees: Record<TreeName, Tree>          // "main" + every group + every block def
  mainTreeName: TreeName                 // usually "main"
}
```

### Leaf blocks as one-node trees
Every default block from the manifest is registered as a Tree with exactly one
child node whose `name` is the block/operator name. The Tree's name is what Rete
renders and what codegen uses as the class name — **the inner node's name does
not contribute to the rendered name.** This is the "rename a ConvBlock" mechanism.

## Derived view: `buildDynamicMainTree(forest)` (new: `web/src/tree/expand.ts`)

Pure function. Starting from `mainTreeName`, for each node whose referenced tree
is *expanded* (user opened the group), splice the child tree in:

1. Substitute the expanded node with its child tree's nodes.
2. Rewrite every connection that referenced the spliced node's id: a `to`/`from`
   pointing at `node` is replaced by the `childId` taken from the matching
   boundary entry `childId@port`, and the `childId@` prefix is stripped from the
   port name.
3. Append the child tree's remaining internal connections.
4. Recompute each spliced connection's `unique_id` (now collision-free).

`dynamicMainTree` is recomputed on demand; the editor never mutates source trees.

**Implemented** (`web/src/tree/expand.ts`): `buildDynamicMainTree(forest, expanded)`
where `expanded` is the set of node ids the user has opened. Spliced child ids
are namespaced `${instanceId}/${childId}` so two expanded instances of the same
tree never collide; splicing recurses for nested expansion. Tested in
`web/test-tree-expand.ts`.

### Rename + prune (`web/src/tree/edit.ts`)
A node's `name` IS a tree name, so renaming is a definition-level edit:
- `renameNode(forest, nodeId, newName)` — if `newName` already names a tree the
  node SHARES that definition; if it's free, the node's current tree is cloned
  under `newName` (fresh child ids) and the node repointed. This is how one
  instance is renamed without disturbing its siblings.
- `pruneUnusedTrees(forest)` — removes every tree no node references, computed
  to a transitive fixpoint (a tree reachable only through a pruned tree also
  dies), keeping `main` always. Child nodes owned solely by a pruned tree are
  deleted. `renameNode` prunes by default.

## Rendering rule (in render adapter)

While drawing `dynamicMainTree`, for each node look up `name` in `trees`:
- if the tree has exactly **one** child whose name is a known block → render as
  that block (params, inputs, outputs).
- else → render as a generic group block showing params + boundary I/O.

## `getInputOutputParamsSignature(tree: Tree): Signature` (new: `web/src/tree/signature.ts`)

THE keystone. Single source of truth for **three** consumers that must never
disagree (their disagreement is the current bug class):
1. the **rendering boundary** — what ports a collapsed facade shows;
2. the codegen **`forward(...)` signature** of `class <tree.name>`;
3. the codegen **`__init__(...)` signature** of `class <tree.name>`.

Computed **per-tree, in isolation** — it looks only at `tree.list_of_nodes`,
`tree.list_of_connections`, and (for child trees referenced by a node) each
child's *own* recursively-computed signature. It never reaches into the
`dynamicMainTree` or sibling trees. That isolation is what makes a Tree a genuine
reusable class.

```ts
interface SigPort {
  name: string            // boundary port name (the localName, NOT childId@port)
  shape: Shape
  dtype: string
  optional?: boolean
  // provenance — which inner child port this boundary slot routes to:
  childId: NodeId
  childPort: string
}
interface SigParam {
  name: string            // __init__ arg name (sanitized, deduped)
  dtype: string
  default?: unknown        // present ⇒ optional kwarg; absent ⇒ required positional
  shape?: Shape           // for learnable params / buffers
  childId: NodeId         // where the value is injected at init time
  childPort?: string      // the __param__ port it feeds, if exposed-param style
}
interface Signature {
  forwardInputs:  SigPort[]   // → forward() tensor args, in port order
  forwardOutputs: SigPort[]   // → forward() return tuple, in port order
  initParams:     SigParam[]  // → __init__ args
}
```

### Exact derivation rules
Let `incoming(childId, port)` / `outgoing(childId, port)` be membership tests over
`tree.list_of_connections` (only connections whose both ends are in
`list_of_nodes` count — boundary refs in `tree.inputs/outputs` do not).

- **forwardInputs** — every child **tensor input port** with NO incoming
  connection inside this tree. Emit in `(node order, port order)`. Name = the
  child port name, deduped if collision (`x`, `x_2`, …). Carry `{childId, childPort}`.
  - A child that is itself a tree (group/renamed block) contributes its *own*
    `forwardInputs` here, prefixed/routed through the boundary — i.e. recursion,
    not a peek at internals.
- **forwardOutputs** — every child **tensor output port** with NO outgoing
  connection inside this tree (a graph sink). Same ordering + provenance rules.
  - Explicit `output`-kind children also force their wired-in port to be an output
    even if (defensively) something downstream existed.
- **initParams** — union, in stable order, of:
  - `const`-kind children → one init param (the constant's value becomes the default);
  - `learnable`-kind children → an init param carrying `shape` (becomes an
    `nn.Parameter`/buffer attr);
  - exposed `__param__*` ports with no incoming wire → an init param feeding that
    child's ctor;
  - a node's `values` that override a required (no-default) ctor param are baked
    in, NOT surfaced — only *dangling* / *exposed* ones surface.
- **Tag collapsing happens at codegen, not here.** This function reports the raw
  per-instance signature; the shared-`self.<tag>` dedup is applied downstream
  (see Codegen mapping) so the signature stays a faithful structural description.

### Facade composition (the boundary)
A group facade node renders exactly `getInputOutputParamsSignature(childTree)`:
`forwardInputs` → receiver ports, `forwardOutputs` → outlet ports, `initParams` →
`__param__` ports. This *replaces* `computeBoundary` + `portMap` +
`boundarySignature` — the boundary is now derived, never stored, so there is
nothing to keep in sync across collapse/expand.

### Must subsume current behavior (port target)
The new function must reproduce what codegen does today via
`findEntryInputs` (unwired required input ports → forward args, codegen.ts:331) and
the `const`/exposed-param → `__init__` logic. Step 2's unit tests diff
`getInputOutputParamsSignature` output against the current codegen's derived
`entryInputs` / init params on fixtures until they match, before anything is rewired.

## Codegen mapping (rewire `web/src/codegen.ts`)

- Every **Tree** → one Python class named `Tree.name`.
- Every **tag / shared tag** → a class property name in the containing tree, so
  identical tags share weights (one `self.<tag>` reused).
- `__init__` params and `forward` params come straight from
  `getInputOutputParamsSignature(tree)`.
- This replaces `partitionByGroup` / `buildSubgraphView` /
  `mergeGroupFacadeInitParams` / `bubbleChildGroupParams` — the flat-graph
  structure reconstruction is no longer needed because Trees ARE the structure.

## Incremental porting order (each step leaves app runnable)

- **Step 0** — `tree/model.ts` types + `tree/forest.ts` store (no UI wiring).
- **Step 1** — `manifestToTree()` shim: register every manifest block as a 1-node tree.
- **Step 2** — `tree/signature.ts` + unit tests against known blocks/groups.
- **Step 3** — `tree/expand.ts` (`buildDynamicMainTree`) + unit tests.
- **Step 4** — adapter `forest <-> current GraphEditor/state.groups` (two-way), so
  the existing UI keeps working while sourcing truth from the forest.
- **Step 5** — codegen-from-tree behind a flag; diff output vs current codegen on
  fixtures until identical/superset, then flip default.
- **Step 6** — render from `dynamicMainTree`; retire facade recreate/remap +
  `portMap`/`childOffsets`/`boundarySignature` churn.
- **Step 7** — persistence: serialize the forest; migrate old saved graphs via the
  adapter on load. Delete dead group-lifecycle code.

## Open questions to resolve as we build
- Disconnected graphs / array-of-roots: spec says defer. Model already supports it
  (a Tree can have multiple roots); codegen/topo just needs to handle a forest of
  roots when we get there.
- Multiple merge-points from same source: handled naturally (a node can have
  multiple out-connections).
- Tag-collision across families: keep the current `family` discriminator concept
  when mapping tags → shared class properties.

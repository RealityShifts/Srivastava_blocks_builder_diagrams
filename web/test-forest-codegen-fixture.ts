// Regression fixture: a real exported graph (130 nodes, 10 groups incl. nested,
// weight-shared, and Elementwise/Add operators). Asserts forest codegen is
// BYTE-IDENTICAL to the legacy editor codegen. Pure logic - no browser.
//   node test-forest-codegen-fixture.ts
//
// This locks in the fix for: duplicate-class-name groups collapsing into one
// tree, dropped facade boundary edges (in0=undefined), spurious dangling inputs,
// the Elementwise/Add local-variable name, and class emission order.
import { readFileSync } from 'node:fs'
import { generate } from './src/codegen.ts'
import type { NodeLike, Connection as FlatConn } from './src/types.ts'
import {
  INPUT_ENTRY, OUTPUT_ENTRY, CONST_ENTRY, LEARNABLE_TENSOR_ENTRY,
  REARRANGE_ENTRY, RESHAPE_ENTRY, CONCAT_ENTRY, STACK_ENTRY,
  POOL_ENTRY, UPSAMPLE_ENTRY, ELEMENTWISE_ENTRY,
} from './src/nodes.ts'
import { graphDataToForest, forestToGenerateInput } from './src/tree/adapter.ts'

let pass = 0, fail = 0
const check = (name: string, cond: boolean, info?: unknown) => {
  if (cond) { pass++; console.log(`  ok   ${name}`) }
  else { fail++; console.log(`  FAIL ${name}`, info ?? '') }
}

const data = JSON.parse(readFileSync(new URL('./fixtures/big-nested-groups.json', import.meta.url), 'utf8'))

// Reproduce loadManifest's catalogue (builtins win over the fetched manifest).
const manifest = JSON.parse(readFileSync(new URL('./public/manifests/pytorch.json', import.meta.url), 'utf8'))
const builtins = [INPUT_ENTRY, OUTPUT_ENTRY, CONST_ENTRY, LEARNABLE_TENSOR_ENTRY, REARRANGE_ENTRY, RESHAPE_ENTRY, CONCAT_ENTRY, STACK_ENTRY, POOL_ENTRY, UPSAMPLE_ENTRY, ELEMENTWISE_ENTRY]
const bn = new Set(builtins.map((e) => e.name))
const catalogue: Record<string, any> = Object.fromEntries([...builtins, ...manifest.filter((e: any) => !bn.has(e.name))].map((e) => [e.name, e]))

// LEGACY: the serialized nodes are already flat with facade portMaps; feed
// straight to generate() (this is what the live editor path does).
function legacyInput(): { nodes: NodeLike[]; conns: FlatConn[] } {
  const nodes: NodeLike[] = data.nodes.map((spec: any) => {
    let entry = catalogue[spec.name]
    if (spec.kind === 'group') {
      const pm = spec.portMap ?? { inputs: [], outputs: [], params: [] }
      entry = {
        name: spec.name, module: '__group__', kind: 'group', ctor: [],
        inputs: (pm.inputs ?? []).map((m: any) => ({ name: m.facadePort, shape: m.shape ?? ['...'], dtype: 'any' })),
        outputs: (pm.outputs ?? []).map((m: any) => ({ name: m.facadePort, shape: m.shape ?? ['...'], dtype: 'any' })),
        groupId: spec.groupId, portMap: pm,
      }
    }
    return {
      id: spec.id, entry, name: spec.instanceName ?? '', tag: spec.tag ?? '',
      groupId: spec.kind === 'group' ? (spec.memberOf ?? null) : (spec.groupId ?? null),
      values: { ...spec.values },
    }
  })
  const conns: FlatConn[] = data.connections.map((c: any, i: number) => ({ id: `e${i}`, source: c.source, sourceOutput: c.sourceOutput, target: c.target, targetInput: c.targetInput }))
  return { nodes, conns }
}

const { nodes: lNodes, conns: lConns } = legacyInput()
const legacy = generate(lNodes as any, lConns as any, 'pytorch')

const { forest, exposedByNode, facadeMeta, groupClassName, nodeOrder } = graphDataToForest(data)
const { nodes: fNodes, connections: fConns } = forestToGenerateInput(forest, catalogue, exposedByNode, facadeMeta, groupClassName, nodeOrder)
const forestCode = generate(fNodes as any, fConns as any, 'pytorch')

// Sanity: no NEW dangling artifacts the legacy path didn't already have.
const undef = (s: string) => s.split('\n').filter((l) => /=undefined/.test(l)).length
check('forest emits no "=undefined" args', undef(forestCode) === 0, undef(forestCode))

// Headline: byte-identical to the legacy editor codegen.
const same = legacy === forestCode
check('forest codegen byte-identical to legacy on big nested fixture', same)
if (!same) {
  const a = legacy.split('\n'), b = forestCode.split('\n')
  for (let i = 0; i < Math.max(a.length, b.length); i++) {
    if (a[i] !== b[i]) { console.log(`  first diff @ line ${i + 1}:\n    legacy: ${a[i]}\n    forest: ${b[i]}`); break }
  }
}

console.log(`\n${pass} passed, ${fail} failed`)
if (fail > 0) process.exit(1)

# Graph data format

Challenge graphs are stored as **node-link JSON**. The format is intentionally
plain so it can be parsed in any language, and it is directly compatible with
`networkx.node_link_graph` (using the `links` key).

## Top-level structure

```json
{
  "directed": false,
  "multigraph": false,
  "graph": {
    "name": "basic"
  },
  "nodes": [
    {"id": 0, "x": 1.20, "y": 3.40, "z": 0.50}
  ],
  "links": [
    {"source": 0, "target": 1, "cost": 2.34}
  ]
}
```

## Fields

### `graph` (challenge metadata)

| Field | Meaning |
|-------|---------|
| `name` | Human-readable graph id. |

Evaluation settings (`k`, `max_turn_deg`, `drop_prob`, start nodes, phase
thresholds, seed count, and agent count) live in ``[eval]`` in
``exploration_challenge/params.toml`` or on ``run_eval.py`` CLI flags.

### `nodes`

- `id`: unique, stable node identifier (int).
- `x`, `y`, `z`: 3D position in metres (world frame).

### `links` (edges)

- `source`, `target`: node ids. Edges are **undirected**.
- `cost`: traversal cost (defaults to the Euclidean distance between endpoints
  if omitted on load). Flight "time" is proportional to total traversed cost.

## Guarantees

- The graph is **connected**.
- Node ids are unique.
- Edges represent collision-free line-of-sight connections: an edge means you can
  fly straight between two points without hitting anything. Graphs may span
  multiple spatially separated rooms joined by long connector edges across empty
  gaps. Limited-depth observation models a realistic sensor: vision follows
  connectivity, not through walls.

## Loading

Use ``exploration_challenge.graph_io.load_graph(path)`` (also used by
``run_eval.py``). ``load_graph`` retains ``graph.name`` from file metadata.

## Observation depth (`k`) and line-of-sight (`max_turn_deg`)

Observation depth is an evaluation setting: how many graph hops each UAV can
see from its current node. Configure it via ``run_eval.py --k`` or ``[eval].k`` in
``exploration_challenge/params.toml`` (currently ``4``).

Vision also respects a **per-hop turn-angle** line-of-sight rule: a node is only
observed if it can be reached along some path from the UAV where every single
edge-to-edge turn is at most ``max_turn_deg`` degrees. The first hop from the
UAV is always allowed. This prevents vision from flowing around corners: once
a path bends too sharply, nodes beyond that bend are hidden. Configure via
``run_eval.py --max-turn-deg`` or ``[eval].max_turn_deg`` in
``exploration_challenge/params.toml`` (currently ``75``). Set ``max_turn_deg = 180``
to disable the turn gate (pure hop-ball visibility).

## Start node

Configure the start node at evaluation time:

- **Random per seed** (default): leave ``[eval].start`` unset for random distinct
  starts per seed.
  Agent 0 gets a random node per seed; with ``n_agents > 1``, remaining agents get
  distinct random nodes from the same start RNG stream.
- **Fixed**: set ``[eval].start = 42`` in ``exploration_challenge/params.toml`` or
  pass ``run_eval.py --start 42``.

## Phase thresholds

Exploration and surveillance completion thresholds are configured via
``[eval].explore_threshold`` and ``[eval].surveil_threshold`` in
``exploration_challenge/params.toml`` (currently ``0.9`` each).

## Observation types

At each decision point the simulator hands your policy an `Observation` dataclass
(see ``exploration_challenge/observation.py``). Import with:

```python
from exploration_challenge.observation import Observation, ObservedNode, ObservedEdge
```

| Type | Fields |
|------|--------|
| `ObservedNode` | `id`, `x`, `y`, `z` |
| `ObservedEdge` | `u`, `v`, `cost` |
| `Observation` | `agent_id`, `position`, `position_xyz`, `nodes`, `edges`, `visited`, `neighbors()` |

`nodes` and `edges` contain only what is visible in the current depth-`k`,
line-of-sight scan. Build and merge your own map across ticks and infer
unexplored branches from accumulated topology (e.g. degree-1 leaves).

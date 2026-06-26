"""The depth-limited, line-of-sight observation handed to an Explorer policy.

An ``Observation`` is a *restricted copy* of what one UAV can sense from its
current node: every node within ``k`` graph hops that is reachable along a path
whose edge-to-edge turns stay below ``max_turn_deg``, and the edges among those
nodes. Edges to neighbours outside the view are omitted, so a policy cannot tell
from ``edges`` alone whether a visible node has hidden neighbours.

Each scan may independently drop candidate nodes with probability ``drop_prob``.
A dropped node is not visible and expansion does not continue through it, so
nodes beyond a miss are hidden until a later scan detects the intermediate node.

It deliberately holds no reference to the ground-truth graph, so a policy cannot
"see through walls" or peek at undiscovered structure.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObservedNode:
    id: int
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class ObservedEdge:
    u: int
    v: int
    cost: float


@dataclass(frozen=True)
class Observation:
    """What the policy sees at one decision point."""

    agent_id: int
    position: int
    position_xyz: tuple[float, float, float]
    nodes: tuple[ObservedNode, ...]
    edges: tuple[ObservedEdge, ...]
    visited: tuple[int, ...]  # proprioception: every node this drone has physically been at

    def neighbors(self, node_id: int) -> list[int]:
        out: list[int] = []
        for e in self.edges:
            if e.u == node_id:
                out.append(e.v)
            elif e.v == node_id:
                out.append(e.u)
        return out

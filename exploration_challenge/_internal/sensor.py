"""Build depth-limited line-of-sight observations from the ground-truth graph."""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass

import networkx as nx

from ..graph_io import undirected_edge_key
from ..observation import ObservedEdge, ObservedNode, Observation


@dataclass(frozen=True)
class _Geom:
    neighbors: dict[int, list[int]]
    edge_dir: dict[tuple[int, int], tuple[float, float, float]]
    edge_cost: dict[tuple[int, int], float]
    coords: dict[int, tuple[float, float, float]]


def _unit_from_coords(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float, float] | None:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length == 0.0:
        return None
    return (dx / length, dy / length, dz / length)


def _geom(world: nx.Graph) -> _Geom:
    cached = world.graph.get("_geom")
    if cached is not None:
        return cached

    neighbors: dict[int, list[int]] = {}
    coords: dict[int, tuple[float, float, float]] = {}
    for n in world.nodes:
        nd = world.nodes[n]
        coords[n] = (nd["x"], nd["y"], nd["z"])
        neighbors[n] = list(world.neighbors(n))

    edge_dir: dict[tuple[int, int], tuple[float, float, float]] = {}
    edge_cost: dict[tuple[int, int], float] = {}
    for u, v, data in world.edges(data=True):
        cost = float(data["cost"])
        edge_cost[(u, v)] = cost
        edge_cost[(v, u)] = cost
        d_uv = _unit_from_coords(coords[u], coords[v])
        if d_uv is not None:
            edge_dir[(u, v)] = d_uv
            edge_dir[(v, u)] = (-d_uv[0], -d_uv[1], -d_uv[2])

    geom = _Geom(neighbors=neighbors, edge_dir=edge_dir, edge_cost=edge_cost, coords=coords)
    world.graph["_geom"] = geom
    return geom


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _detect(
    node: int,
    position: int,
    drop_prob: float,
    rng: random.Random,
    decided: dict[int, bool],
) -> bool:
    """Return True when ``node`` is detected this scan (``position`` is always kept).

    Memoizes one Bernoulli draw per node so each node is decided at most once per
    scan. ``drop_prob >= 1`` always drops (``random()`` is in ``[0, 1)``).
    """
    if node == position or drop_prob <= 0.0:
        return True
    if node in decided:
        return decided[node]
    detected = rng.random() >= drop_prob
    decided[node] = detected
    return detected


def _check_dropout(drop_prob: float, rng: random.Random | None) -> None:
    if drop_prob > 0.0 and rng is None:
        raise ValueError("drop_prob > 0 requires an rng for reproducible draws")


def bfs_ball(
    world: nx.Graph,
    position: int,
    k: int,
    *,
    drop_prob: float = 0.0,
    rng: random.Random | None = None,
) -> set[int]:
    """Return every node within ``k`` graph hops of ``position`` (inclusive).

    When ``drop_prob > 0``, each newly reached neighbour is independently
    dropped with that probability; a dropped node is not visible and is not
    expanded further. ``rng`` is required when ``drop_prob > 0``.
    """
    _check_dropout(drop_prob, rng)
    geom = _geom(world)
    visible = {position}
    depth = {position: 0}
    queue: deque[int] = deque([position])
    decided: dict[int, bool] = {}
    while queue:
        u = queue.popleft()
        if depth[u] >= k:
            continue
        for v in geom.neighbors[u]:
            if v in depth:
                continue
            if not _detect(v, position, drop_prob, rng, decided):
                continue
            depth[v] = depth[u] + 1
            visible.add(v)
            queue.append(v)
    return visible


def los_ball(
    world: nx.Graph,
    position: int,
    k: int,
    max_turn_deg: float = 60.0,
    *,
    drop_prob: float = 0.0,
    rng: random.Random | None = None,
) -> set[int]:
    """Return nodes visible within ``k`` hops with per-hop turn-angle line-of-sight.

    A node is visible only if it can be reached along some path from ``position``
    where every single edge-to-edge turn is at most ``max_turn_deg``. The first hop
    from ``position`` is always allowed (no incoming edge to measure). Setting
    ``max_turn_deg = 180`` reproduces the geometry-blind ``bfs_ball`` behaviour.

    When ``drop_prob > 0``, each candidate node is independently dropped with
    that probability; a dropped node is not visible and expansion stops at it.
    ``rng`` is required when ``drop_prob > 0``.
    """
    _check_dropout(drop_prob, rng)
    if max_turn_deg >= 180.0:
        return bfs_ball(world, position, k, drop_prob=drop_prob, rng=rng)
    visible = {position}
    if k <= 0:
        return visible

    geom = _geom(world)
    cos_thresh = math.cos(math.radians(max_turn_deg))
    queue: deque[tuple[int, tuple[float, float, float], int]] = deque()
    best_depth: dict[tuple[int, int], int] = {}
    decided: dict[int, bool] = {}

    for nb in geom.neighbors[position]:
        d_out = geom.edge_dir.get((position, nb))
        if d_out is None:
            continue
        if not _detect(nb, position, drop_prob, rng, decided):
            continue
        best_depth[(position, nb)] = 1
        visible.add(nb)
        queue.append((nb, d_out, 1))

    while queue:
        u, d_in, depth = queue.popleft()
        if depth >= k:
            continue
        for w in geom.neighbors[u]:
            d_out = geom.edge_dir.get((u, w))
            if d_out is None:
                continue
            if _dot(d_in, d_out) < cos_thresh:
                continue
            nd = depth + 1
            if best_depth.get((u, w), 10**9) <= nd:
                continue
            if not _detect(w, position, drop_prob, rng, decided):
                continue
            best_depth[(u, w)] = nd
            visible.add(w)
            queue.append((w, d_out, nd))

    return visible


def _visible_subgraph(
    world: nx.Graph,
    visible: set[int],
) -> tuple[dict[int, dict[int, float]], list[ObservedEdge]]:
    """Adjacency and edges among ``visible`` nodes."""
    geom = _geom(world)
    adj: dict[int, dict[int, float]] = {}
    edges: list[ObservedEdge] = []
    seen_edges: set[tuple[int, int]] = set()
    for n in visible:
        nbrs = adj.setdefault(n, {})
        for nb in geom.neighbors[n]:
            if nb in visible:
                nbrs[nb] = geom.edge_cost[(n, nb)]
                key = undirected_edge_key(n, nb)
                if key not in seen_edges:
                    seen_edges.add(key)
                    edges.append(ObservedEdge(key[0], key[1], geom.edge_cost[(n, nb)]))
    return adj, edges


def visible_adj(world: nx.Graph, visible: set[int]) -> dict[int, dict[int, float]]:
    """Edge costs among nodes in ``visible`` (for simulator bookkeeping)."""
    return _visible_subgraph(world, visible)[0]


def build_observation(
    world: nx.Graph,
    position: int,
    k: int,
    agent_id: int = 0,
    visited: set[int] | None = None,
    max_turn_deg: float = 60.0,
    *,
    drop_prob: float = 0.0,
    rng: random.Random | None = None,
    visible: set[int] | None = None,
) -> Observation:
    """Build the depth-``k`` line-of-sight view around ``position``."""
    if visible is None:
        visible = los_ball(
            world,
            position,
            k,
            max_turn_deg,
            drop_prob=drop_prob,
            rng=rng,
        )
    _, edges = _visible_subgraph(world, visible)
    geom = _geom(world)
    nodes = [
        ObservedNode(
            n,
            geom.coords[n][0],
            geom.coords[n][1],
            geom.coords[n][2],
        )
        for n in visible
    ]

    px, py, pz = geom.coords[position]
    return Observation(
        agent_id=agent_id,
        position=position,
        position_xyz=(px, py, pz),
        nodes=tuple(nodes),
        edges=tuple(edges),
        visited=tuple(visited) if visited else (),
    )

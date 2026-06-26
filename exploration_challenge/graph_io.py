"""Load/save challenge graphs in node-link JSON (see docs/graph_format.md).

Also provides small graph helpers used across the package
(``euclidean``, ``node_xyz``, ``undirected_edge_key``).
"""

from __future__ import annotations

import json
import math
from typing import Any

import networkx as nx

# Defaults applied to a graph's metadata when fields are missing.
DEFAULT_GRAPH_ATTRS: dict[str, Any] = {
    "name": "unnamed",
}


def undirected_edge_key(u: int, v: int) -> tuple[int, int]:
    """Canonical key for an undirected edge."""
    return (u, v) if u < v else (v, u)


def euclidean(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def node_xyz(graph: nx.Graph, node: int) -> tuple[float, float, float]:
    d = graph.nodes[node]
    return (d["x"], d["y"], d["z"])


def load_graph(path: str) -> nx.Graph:
    """Read a node-link JSON file into a validated, connected ``nx.Graph``."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    graph = nx.Graph()
    attrs = dict(DEFAULT_GRAPH_ATTRS)
    attrs.update(data.get("graph", {}))
    graph.graph.update(attrs)

    for nd in data["nodes"]:
        graph.add_node(int(nd["id"]), x=float(nd["x"]), y=float(nd["y"]), z=float(nd["z"]))

    for ln in data["links"]:
        u, v = int(ln["source"]), int(ln["target"])
        cost = ln.get("cost")
        if cost is None:
            cost = euclidean(node_xyz(graph, u), node_xyz(graph, v))
        graph.add_edge(u, v, cost=float(cost))

    validate_graph(graph)
    return graph


def node_link_dict(
    graph_attrs: dict[str, Any],
    nodes: list[dict[str, Any]],
    links: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the standard node-link JSON object (see docs/graph_format.md)."""
    return {
        "directed": False,
        "multigraph": False,
        "graph": graph_attrs,
        "nodes": nodes,
        "links": links,
    }


def save_graph(graph: nx.Graph, path: str) -> None:
    """Write an ``nx.Graph`` to node-link JSON."""
    data = node_link_dict(
        dict(graph.graph),
        [
            {"id": n, "x": graph.nodes[n]["x"], "y": graph.nodes[n]["y"], "z": graph.nodes[n]["z"]}
            for n in graph.nodes
        ],
        [
            {"source": u, "target": v, "cost": graph.edges[u, v]["cost"]}
            for u, v in graph.edges
        ],
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def validate_graph(graph: nx.Graph) -> None:
    if graph.number_of_nodes() == 0:
        raise ValueError("graph has no nodes")
    if not nx.is_connected(graph):
        raise ValueError("graph must be connected")
    for n in graph.nodes:
        for key in ("x", "y", "z"):
            if key not in graph.nodes[n]:
                raise ValueError(f"node {n} missing coordinate {key!r}")

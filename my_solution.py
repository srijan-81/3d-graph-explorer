from __future__ import annotations

import heapq
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

from exploration_challenge.observation import Observation

# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def dijkstra(
    graph: Dict[int, Dict[int, float]],
    source: int,
    targets: Optional[Set[int]] = None,
) -> Tuple[Dict[int, float], Dict[int, Optional[int]]]:
    dist: Dict[int, float] = {source: 0.0}
    prev: Dict[int, Optional[int]] = {source: None}
    heap = [(0.0, source)]
    settled: Set[int] = set()
    remaining = set(targets) if targets else None

    while heap:
        d, u = heapq.heappop(heap)
        if u in settled:
            continue
        settled.add(u)
        if remaining is not None:
            remaining.discard(u)
            if not remaining:
                break
        for v, w in graph.get(u, {}).items():
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

    return dist, prev


def reconstruct_path(prev: Dict[int, Optional[int]], target: int) -> List[int]:
    path = []
    cur: Optional[int] = target
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()
    return path if len(path) > 1 else []


def bfs_within_k(
    graph: Dict[int, Dict[int, float]], source: int, k: int
) -> Set[int]:
    visited = {source}
    queue = deque([(source, 0)])
    while queue:
        node, depth = queue.popleft()
        if depth >= k:
            continue
        for nb in graph.get(node, {}):
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, depth + 1))
    return visited


# ---------------------------------------------------------------------------
# Main Explorer class
# ---------------------------------------------------------------------------

class Explorer:
    SENSOR_K: int = 4

    def reset(
        self,
        starts: List[int],
        observations: List[Observation],
        seed: int | None = None,
    ) -> None:
        self.graph: Dict[int, Dict[int, float]] = defaultdict(dict)
        self.pos: Dict[int, Tuple[float, float, float]] = {}

        self.n_agents = len(starts)
        self.current: List[int] = list(starts)
        self.paths: List[List[int]] = [[] for _ in range(self.n_agents)]
        self.targets: List[Optional[int]] = [None] * self.n_agents
        self.dist_travelled: List[float] = [0.0] * self.n_agents

        # physically_visited: nodes any UAV has actually stood on
        self.physically_visited: Set[int] = set(starts)
        self.observed: Set[int] = set()
        self.surveilled: Set[int] = set()

        self._stall_count: List[int] = [0] * self.n_agents
        self._last_pos: List[int] = list(starts)

        for obs in observations:
            self._merge_obs(obs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, observations: List[Observation], phase: str) -> List[int]:
        for obs in observations:
            self._merge_obs(obs)
            self.current[obs.agent_id] = obs.position
            # Track physical visits from the obs.visited field (simulator ground truth)
            self.physically_visited.update(obs.visited)
            self.physically_visited.add(obs.position)

        if phase == "surveil":
            for obs in observations:
                nearby = bfs_within_k(self.graph, obs.position, self.SENSOR_K)
                self.surveilled.update(nearby)

        # Stall guard: force a move after 3 ticks stuck with no path
        for i in range(self.n_agents):
            if self.current[i] == self._last_pos[i] and not self.paths[i]:
                self._stall_count[i] += 1
            else:
                self._stall_count[i] = 0
            self._last_pos[i] = self.current[i]

            if self._stall_count[i] >= 3:
                nbrs = [n for n in self.graph.get(self.current[i], {})
                        if n != self.current[i]]
                if nbrs:
                    # Pick neighbour furthest from other agents
                    others = [self.current[j] for j in range(self.n_agents) if j != i]
                    best_nbr = max(nbrs, key=lambda n: min(
                        abs(self.pos.get(n, (0,0,0))[0] - self.pos.get(o, (0,0,0))[0]) +
                        abs(self.pos.get(n, (0,0,0))[1] - self.pos.get(o, (0,0,0))[1])
                        for o in others
                    ) if others else 0)
                    self.paths[i] = [best_nbr]
                    self.targets[i] = best_nbr
                self._stall_count[i] = 0

        if phase == "explore":
            actions = self._explore_actions()
        else:
            actions = self._surveil_actions()

        actions = self._resolve_collisions(actions)

        for i, (cur, nxt) in enumerate(zip(self.current, actions)):
            if nxt != cur and nxt in self.graph.get(cur, {}):
                self.dist_travelled[i] += self.graph[cur][nxt]

        return actions

    # ------------------------------------------------------------------
    # Observation merging
    # ------------------------------------------------------------------

    def _merge_obs(self, obs: Observation) -> None:
        for n in obs.nodes:
            self.pos[n.id] = (n.x, n.y, n.z)
            if n.id not in self.graph:
                self.graph[n.id] = {}
            self.observed.add(n.id)
        for e in obs.edges:
            self.graph[e.u][e.v] = e.cost
            self.graph[e.v][e.u] = e.cost
        self.observed.add(obs.position)
        self.observed.update(n.id for n in obs.nodes)

    # ------------------------------------------------------------------
    # Frontier computation
    # ------------------------------------------------------------------

    def _frontier_nodes(self) -> Set[int]:
        """
        A frontier node is a KNOWN node that has at least one neighbour
        which has NOT been physically visited yet.

        This works regardless of graph degree — it purely tracks whether
        we have stood at a node and scanned outward from it.
        """
        frontiers: Set[int] = set()
        for u in self.graph:
            for v in self.graph[u]:
                if v not in self.physically_visited:
                    frontiers.add(u)
                    break
        return frontiers

    def _score_frontier(
        self,
        node: int,
        positions: List[int],
        dist_maps: List[Dict[int, float]],
        claimed: Set[int],
        agent_idx: int,
    ) -> float:
        if node in claimed:
            return -float("inf")

        my_dist = dist_maps[agent_idx].get(node, float("inf"))
        if my_dist == float("inf"):
            return -float("inf")

        # Count unvisited neighbours visible from this frontier node
        unvisited_nbrs = sum(
            1 for v in self.graph.get(node, {})
            if v not in self.physically_visited
        )

        # Dispersion: prefer targets far from other agents
        spread = 0.0
        for j in range(self.n_agents):
            if j != agent_idx:
                spread += dist_maps[j].get(node, 0.0)

        # Mild makespan balance penalty
        balance_penalty = self.dist_travelled[agent_idx] * 0.02

        return (
            unvisited_nbrs * 8.0
            + spread * 0.2
            - my_dist * 1.0
            - balance_penalty
        )

    # ------------------------------------------------------------------
    # Explore actions
    # ------------------------------------------------------------------

    def _explore_actions(self) -> List[int]:
        positions = list(self.current)
        dist_maps = [dijkstra(self.graph, pos)[0] for pos in positions]
        frontiers = self._frontier_nodes()

        # Invalidate stale targets (no longer a frontier, or already reached)
        claimed: Set[int] = set()
        for i in range(self.n_agents):
            t = self.targets[i]
            still_valid = (
                t is not None
                and t != positions[i]
                and t in self.graph
                and t in frontiers
                and dist_maps[i].get(t, float("inf")) < float("inf")
            )
            if still_valid:
                claimed.add(t)
            else:
                self.targets[i] = None
                self.paths[i] = []

        # Assign new targets
        for i in range(self.n_agents):
            if self.targets[i] is not None:
                continue

            candidates = frontiers - claimed

            if not candidates:
                # No frontiers: graph nearly fully explored, drift to
                # any unvisited node reachable from here
                candidates = {
                    n for n in self.graph
                    if n not in self.physically_visited
                    and dist_maps[i].get(n, float("inf")) < float("inf")
                    and n not in claimed
                }

            if not candidates:
                # Truly nothing left — stay put
                self.targets[i] = None
                self.paths[i] = []
                continue

            best = max(
                candidates,
                key=lambda n: self._score_frontier(n, positions, dist_maps, claimed, i),
            )
            self.targets[i] = best
            claimed.add(best)
            _, prev = dijkstra(self.graph, positions[i], {best})
            self.paths[i] = reconstruct_path(prev, best)
            if self.paths[i] and self.paths[i][0] == positions[i]:
                self.paths[i].pop(0)

        return self._follow_paths(positions)

    # ------------------------------------------------------------------
    # Surveil actions
    # ------------------------------------------------------------------

    def _surveil_coverage_value(self, node: int) -> int:
        reachable = bfs_within_k(self.graph, node, self.SENSOR_K)
        return len(reachable - self.surveilled)

    def _surveil_actions(self) -> List[int]:
        positions = list(self.current)
        all_nodes = list(self.graph.keys())

        # Apply sensor at current positions
        for pos in positions:
            self.surveilled.update(bfs_within_k(self.graph, pos, self.SENSOR_K))

        # Invalidate targets that are now fully covered or reached
        for i in range(self.n_agents):
            t = self.targets[i]
            if t is None or t == positions[i] or self._surveil_coverage_value(t) == 0:
                self.targets[i] = None
                self.paths[i] = []

        claimed: Set[int] = set(t for t in self.targets if t is not None)
        dist_maps = [dijkstra(self.graph, pos)[0] for pos in positions]

        for i in range(self.n_agents):
            if self.targets[i] is not None:
                continue

            best_score = -1.0
            best_node = None
            for node in all_nodes:
                if node in claimed:
                    continue
                cov = self._surveil_coverage_value(node)
                if cov == 0:
                    continue
                d = dist_maps[i].get(node, float("inf"))
                if d == float("inf"):
                    continue
                score = cov / (d + 1.0) - self.dist_travelled[i] * 0.005
                if score > best_score:
                    best_score = score
                    best_node = node

            if best_node is None:
                # Fallback: nearest uncovered node
                reachable = [
                    n for n in all_nodes
                    if dist_maps[i].get(n, float("inf")) < float("inf")
                    and n not in claimed
                    and self._surveil_coverage_value(n) > 0
                ]
                if reachable:
                    best_node = min(reachable, key=lambda n: dist_maps[i][n])

            self.targets[i] = best_node
            if best_node is not None:
                claimed.add(best_node)
                _, prev = dijkstra(self.graph, positions[i], {best_node})
                self.paths[i] = reconstruct_path(prev, best_node)
                if self.paths[i] and self.paths[i][0] == positions[i]:
                    self.paths[i].pop(0)

        return self._follow_paths(positions)

    # ------------------------------------------------------------------
    # Shared path-following
    # ------------------------------------------------------------------

    def _follow_paths(self, positions: List[int]) -> List[int]:
        actions = []
        for i, pos in enumerate(positions):
            path = self.paths[i]
            while path and path[0] == pos:
                path.pop(0)

            if path:
                next_hop = path[0]
                if next_hop in self.graph.get(pos, {}):
                    actions.append(next_hop)
                else:
                    # Edge disappeared from map — replan next tick
                    self.paths[i] = []
                    self.targets[i] = None
                    actions.append(pos)
            else:
                actions.append(pos)

        return actions

    # ------------------------------------------------------------------
    # Collision resolution
    # ------------------------------------------------------------------

    def _resolve_collisions(self, actions: List[int]) -> List[int]:
        resolved = list(actions)
        claimed_nodes: Set[int] = set()

        # Lower agent_id wins vertex conflicts
        for i in range(self.n_agents):
            nxt = resolved[i]
            if nxt == self.current[i]:
                continue
            if nxt in claimed_nodes:
                resolved[i] = self.current[i]
                self.paths[i] = []
                self.targets[i] = None
            else:
                claimed_nodes.add(nxt)

        # Block edge swaps (A→B and B→A same tick)
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if (
                    resolved[i] == self.current[j]
                    and resolved[j] == self.current[i]
                    and resolved[i] != self.current[i]
                    and resolved[j] != self.current[j]
                ):
                    resolved[j] = self.current[j]
                    self.paths[j] = []
                    self.targets[j] = None

        return resolved
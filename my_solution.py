from __future__ import annotations

import heapq
import math
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

    # Weights for _score_explore — tuned by optimize_weights.py
    W_EDGE_RATIO: float = 42.28
    W_ISOLATION: float = 22.67
    W_MAX_EDGE: float = 123.67
    W_DIST: float = 5.01

    # Weights for _surveil_actions — tuned by optimize_weights.py
    W_SURV_COV: float = 1.0
    W_SURV_DIST: float = 1.0

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

        # Only nodes a UAV has physically stood on (not just seen)
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
            # obs.visited is the simulator's ground-truth set of physically visited nodes
            self.physically_visited.update(obs.visited)
            self.physically_visited.add(obs.position)

        if phase == "surveil":
            for obs in observations:
                self.surveilled.update(bfs_within_k(self.graph, obs.position, self.SENSOR_K))

        # Stall guard: counts ticks where agent has no movement (with or without path)
        for i in range(self.n_agents):
            moved = self.current[i] != self._last_pos[i]
            has_path = bool(self.paths[i])
            if not moved:
                self._stall_count[i] += 1
            else:
                self._stall_count[i] = 0
            self._last_pos[i] = self.current[i]

            # Short threshold for truly stuck (no path); longer for collision-blocked
            threshold = 3 if not has_path else 5
            if self._stall_count[i] >= threshold:
                self.paths[i] = []
                self.targets[i] = None
                if not has_path:
                    self._force_unstick(i, phase)
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

    def _force_unstick(self, i: int, phase: str = "explore") -> None:
        pos = self.current[i]
        nbrs = [n for n in self.graph.get(pos, {}) if n != pos]
        if not nbrs:
            return
        if phase == "surveil":
            # Head toward the neighbor with the most unsurveilled nodes in sensor range
            best = max(nbrs, key=lambda n: len(
                bfs_within_k(self.graph, n, self.SENSOR_K) - self.surveilled
            ))
        else:
            # Head toward the neighbor furthest from all other agents
            others_xyz = [self.pos.get(self.current[j], (0.0, 0.0, 0.0))
                          for j in range(self.n_agents) if j != i]
            def spread(n: int) -> float:
                nx, ny, nz = self.pos.get(n, (0.0, 0.0, 0.0))
                if not others_xyz:
                    return 0.0
                return min(
                    (nx - ox)**2 + (ny - oy)**2 + (nz - oz)**2
                    for ox, oy, oz in others_xyz
                )
            best = max(nbrs, key=spread)
        self.paths[i] = [best]
        self.targets[i] = best

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
        """Known nodes adjacent to at least one physically-unvisited node."""
        frontiers: Set[int] = set()
        for u in self.graph:
            for v in self.graph[u]:
                if v not in self.physically_visited:
                    frontiers.add(u)
                    break
        return frontiers

    # ------------------------------------------------------------------
    # Explore actions
    # ------------------------------------------------------------------

    def _explore_actions(self) -> List[int]:
        positions = list(self.current)

        # Run all 3 Dijkstras once and reuse
        dist_maps = [dijkstra(self.graph, pos)[0] for pos in positions]

        frontiers = self._frontier_nodes()

        # Validate existing targets; keep claimed set
        claimed: Set[int] = set()
        for i in range(self.n_agents):
            t = self.targets[i]
            still_valid = (
                t is not None
                and t != positions[i]
                and t in self.graph
                and dist_maps[i].get(t, float("inf")) < float("inf")
            )
            if still_valid:
                claimed.add(t)
            else:
                self.targets[i] = None
                self.paths[i] = []

        # Assign new targets using a sequential greedy auction:
        # agents bid in order of how much distance they've already travelled
        # (most-travelled agent picks last, keeping makespan balanced)
        order = sorted(range(self.n_agents), key=lambda i: self.dist_travelled[i])

        for i in order:
            if self.targets[i] is not None:
                continue

            candidates = frontiers - claimed
            convoy = False
            if not candidates:
                # All frontiers claimed — convoy toward best existing frontier
                # so idle agents are nearby when it opens new territory
                candidates = {
                    n for n in frontiers
                    if dist_maps[i].get(n, float("inf")) < float("inf")
                }
                convoy = True
            if not candidates:
                self.targets[i] = None
                self.paths[i] = []
                continue

            best = max(candidates, key=lambda n: self._score_explore(
                n, positions, dist_maps, set() if convoy else claimed, i))

            self.targets[i] = best
            if not convoy:
                claimed.add(best)
            _, prev = dijkstra(self.graph, positions[i], {best})
            self.paths[i] = reconstruct_path(prev, best)
            if self.paths[i] and self.paths[i][0] == positions[i]:
                self.paths[i].pop(0)

        return self._follow_paths(positions)

    def _score_explore(
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

        reachable = bfs_within_k(self.graph, node, 1)
        visited_nbrs = sum(1 for v in reachable if v in self.physically_visited)
        unvisited_nbrs = sum(1 for v in reachable if v not in self.physically_visited)

        # Edge-ratio: high when mostly unvisited neighbors = true exploration boundary.
        # Interior nodes (surrounded by visited nodes) score low; edge nodes score high.
        edge_ratio = unvisited_nbrs / (visited_nbrs + 1.0)

        # Isolation from other agents: far = unexplored territory for multi-agent spreading
        min_other_dist = min(
            (dist_maps[j].get(node, float("inf"))
             for j in range(self.n_agents) if j != agent_idx),
            default=0.0
        )
        isolation = min(min_other_dist, 100.0)

        # Max edge cost: long edges signal a passage or corridor to undiscovered regions
        max_edge = max(self.graph.get(node, {}).values(), default=0.0)

        return (
            edge_ratio * self.W_EDGE_RATIO
            + isolation * self.W_ISOLATION
            + max_edge * self.W_MAX_EDGE
            - math.log(my_dist + 1.0) * self.W_DIST
        )

    # ------------------------------------------------------------------
    # Surveil actions
    # ------------------------------------------------------------------

    def _surveil_actions(self) -> List[int]:
        positions = list(self.current)

        # Apply sensor at rest
        for pos in positions:
            self.surveilled.update(bfs_within_k(self.graph, pos, self.SENSOR_K))

        # Only precompute coverage for nodes adjacent to unsurveilled area
        survey_frontier = {
            u for u in self.graph
            for v in self.graph[u]
            if v not in self.surveilled
        }
        cov_cache: Dict[int, int] = {}
        for node in survey_frontier:
            reachable = bfs_within_k(self.graph, node, self.SENSOR_K)
            cov_cache[node] = len(reachable - self.surveilled)

        # Invalidate targets that are now fully covered or reached
        for i in range(self.n_agents):
            t = self.targets[i]
            if t is None or t == positions[i] or cov_cache.get(t, 0) == 0:
                self.targets[i] = None
                self.paths[i] = []

        claimed: Set[int] = {t for t in self.targets if t is not None}
        dist_maps = [dijkstra(self.graph, pos)[0] for pos in positions]

        order = sorted(range(self.n_agents), key=lambda i: self.dist_travelled[i])

        for i in order:
            if self.targets[i] is not None:
                continue

            best_score = -float("inf")
            best_node = None
            for node in survey_frontier:
                if node in claimed:
                    continue
                cov = cov_cache.get(node, 0)
                if cov == 0:
                    continue
                d = dist_maps[i].get(node, float("inf"))
                if d == float("inf"):
                    continue
                score = (cov * self.W_SURV_COV) / (d * self.W_SURV_DIST + 1.0)
                if score > best_score:
                    best_score = score
                    best_node = node

            if best_node is None:
                # Fallback: closest unclaimed survey frontier with any coverage
                reachable = [
                    n for n in survey_frontier
                    if dist_maps[i].get(n, float("inf")) < float("inf")
                    and n not in claimed
                    and cov_cache.get(n, 0) > 0
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

        for i in range(self.n_agents):
            nxt = resolved[i]
            if nxt == self.current[i]:
                continue
            if nxt in claimed_nodes:
                # Vertex conflict: just wait this tick; keep path for next tick
                resolved[i] = self.current[i]
            else:
                claimed_nodes.add(nxt)

        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if (
                    resolved[i] == self.current[j]
                    and resolved[j] == self.current[i]
                    and resolved[i] != self.current[i]
                    and resolved[j] != self.current[j]
                ):
                    # Edge-swap deadlock: lower-priority agent re-plans
                    resolved[j] = self.current[j]
                    self.paths[j] = []
                    self.targets[j] = None

        return resolved
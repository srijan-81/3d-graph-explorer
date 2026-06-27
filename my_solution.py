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
    """Return (dist, prev) dicts from source over the known graph.

    If `targets` is given, stop early once all targets are settled.
    """
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
    """Return node list from source→target (inclusive) or [] if unreachable."""
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
    """BFS up to k hops; returns set of reachable node ids (incl. source)."""
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
    # Sensor horizon used for surveillance coverage approximation.
    # Must match params.toml [eval] k (default 4).
    SENSOR_K: int = 4

    def reset(
        self,
        starts: List[int],
        observations: List[Observation],
        seed: int | None = None,
    ) -> None:
        # Shared map: adjacency {node_id: {neighbour: cost}}
        self.graph: Dict[int, Dict[int, float]] = defaultdict(dict)
        # 3-D positions of every known node
        self.pos: Dict[int, Tuple[float, float, float]] = {}

        # Per-agent state
        self.n_agents = len(starts)
        self.current: List[int] = list(starts)
        # Planned path (list of node ids to follow, first element = next hop)
        self.paths: List[List[int]] = [[] for _ in range(self.n_agents)]
        # Assigned target node
        self.targets: List[Optional[int]] = [None] * self.n_agents
        # Cumulative distance travelled per agent (for makespan balancing)
        self.dist_travelled: List[float] = [0.0] * self.n_agents

        # Global observed / surveilled sets
        self.observed: Set[int] = set()
        self.surveilled: Set[int] = set()

        # Merge initial observations
        for obs in observations:
            self._merge_obs(obs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, observations: List[Observation], phase: str) -> List[int]:
        # 1. Merge new observations into shared map
        for obs in observations:
            self._merge_obs(obs)
            self.current[obs.agent_id] = obs.position

        # 2. Update surveilled set during surveil phase
        if phase == "surveil":
            for obs in observations:
                nearby = bfs_within_k(self.graph, obs.position, self.SENSOR_K)
                self.surveilled.update(nearby)

        # 3. Decide targets and compute next-hop actions
        if phase == "explore":
            actions = self._explore_actions()
        else:
            actions = self._surveil_actions()

        # 4. Collision resolution (same-node conflicts)
        actions = self._resolve_collisions(actions)

        # 5. Update distance tracking
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
        """Nodes in the known graph that border at least one unknown node.

        A node u is a frontier node if it has a neighbour v in the known graph
        that itself has zero known outgoing edges *other than* back to u —
        meaning v is a leaf that we haven't explored beyond. More practically:
        any node whose neighbor set contains nodes we haven't yet dispatched
        sensors to (not in self.observed with edges > 1).
        """
        # Classic definition: known nodes adjacent to at least one node whose
        # own neighbourhood hasn't been fully revealed. Since we can only detect
        # neighbours when within k hops, any leaf node (degree 1 in our map)
        # could have hidden branches if we haven't physically visited it.
        # We conservatively treat every node that hasn't been *physically
        # visited by a UAV* (i.e., not in any obs.visited) as a potential
        # frontier.
        frontiers: Set[int] = set()
        for node, nbrs in self.graph.items():
            for nb in nbrs:
                # If a neighbour looks like it might have unexplored edges
                # (i.e. it has very few known connections), it's a frontier.
                if len(self.graph.get(nb, {})) <= 2:
                    frontiers.add(node)
                    break
        return frontiers

    def _score_frontier(
        self,
        node: int,
        agent_positions: List[int],
        dist_from_agents: List[Dict[int, float]],
        claimed: Set[int],
        agent_idx: int,
    ) -> float:
        """Higher score = better target (we'll pick max)."""
        if node in claimed:
            return -float("inf")

        # Information gain: unexplored neighbours (low known degree = more to find)
        info_gain = max(0, 4 - len(self.graph.get(node, {})))

        # Distance cost for this agent
        my_dist = dist_from_agents[agent_idx].get(node, float("inf"))
        if my_dist == float("inf"):
            return -float("inf")

        # Spread bonus: distance from OTHER agents (we want dispersion)
        spread = 0.0
        for j, other in enumerate(agent_positions):
            if j != agent_idx:
                spread += dist_from_agents[j].get(node, 0.0)

        # Makespan balance: penalise if this agent is already far ahead
        balance_penalty = self.dist_travelled[agent_idx]

        # Combined score (tune weights as needed)
        score = (
            info_gain * 5.0
            + spread * 0.3
            - my_dist * 1.0
            - balance_penalty * 0.05
        )
        return score

    # ------------------------------------------------------------------
    # Explore actions
    # ------------------------------------------------------------------

    def _explore_actions(self) -> List[int]:
        positions = list(self.current)

        # Run Dijkstra from each agent's position
        dist_maps = [
            dijkstra(self.graph, pos)[0] for pos in positions
        ]

        frontiers = self._frontier_nodes()

        # (Re-)assign targets that are invalid or reached
        claimed: Set[int] = set()
        for i in range(self.n_agents):
            t = self.targets[i]
            if t is not None and t != positions[i] and t in self.graph:
                claimed.add(t)

        for i in range(self.n_agents):
            t = self.targets[i]
            need_new = (
                t is None
                or t == positions[i]
                or t not in self.graph
                or not frontiers  # nothing left — just keep moving
            )
            if need_new:
                # Pick best frontier for this agent
                candidates = frontiers - claimed
                if not candidates:
                    # Fallback: any unvisited-looking node
                    candidates = {
                        n for n in self.graph
                        if dist_maps[i].get(n, float("inf")) > 0
                    } - claimed
                if not candidates:
                    self.targets[i] = None
                    self.paths[i] = []
                    continue

                best = max(
                    candidates,
                    key=lambda n: self._score_frontier(
                        n, positions, dist_maps, claimed, i
                    ),
                )
                self.targets[i] = best
                claimed.add(best)
                # Pre-compute path
                _, prev = dijkstra(self.graph, positions[i], {best})
                self.paths[i] = reconstruct_path(prev, best)
                # Strip current position from front
                if self.paths[i] and self.paths[i][0] == positions[i]:
                    self.paths[i].pop(0)

        # Build actions: follow pre-computed path or wait
        return self._follow_paths(positions)

    # ------------------------------------------------------------------
    # Surveil actions
    # ------------------------------------------------------------------

    def _surveil_coverage_value(self, node: int) -> int:
        """How many un-surveilled nodes would be covered by visiting `node`."""
        reachable = bfs_within_k(self.graph, node, self.SENSOR_K)
        return len(reachable - self.surveilled)

    def _surveil_actions(self) -> List[int]:
        positions = list(self.current)
        claimed: Set[int] = set()

        # Find nodes worth visiting (coverage value > 0)
        all_nodes = list(self.graph.keys())
        uncovered = set(all_nodes) - self.surveilled

        # Update surveilled from current positions (sensor fires at rest too)
        for pos in positions:
            nearby = bfs_within_k(self.graph, pos, self.SENSOR_K)
            self.surveilled.update(nearby)

        # Re-assign targets
        for i in range(self.n_agents):
            t = self.targets[i]
            at_target = (t is None or t == positions[i])
            target_covered = (
                t is not None
                and self._surveil_coverage_value(t) == 0
            )
            if at_target or target_covered:
                self.targets[i] = None
                self.paths[i] = []

        for i in range(self.n_agents):
            if self.targets[i] is not None:
                claimed.add(self.targets[i])

        dist_maps = [dijkstra(self.graph, pos)[0] for pos in positions]

        for i in range(self.n_agents):
            if self.targets[i] is not None:
                continue  # still en route

            # Score all candidate nodes
            best_score = -1
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
                # Score: coverage per unit distance, balanced by makespan
                score = cov / (d + 1.0) - self.dist_travelled[i] * 0.01
                if score > best_score:
                    best_score = score
                    best_node = node

            if best_node is None:
                # No valuable target — sweep any unvisited node
                reachable = [
                    n for n in all_nodes
                    if dist_maps[i].get(n, float("inf")) < float("inf")
                    and n not in claimed
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
            # Advance path: drop any leading nodes already passed
            while path and path[0] == pos:
                path.pop(0)

            if path:
                next_hop = path[0]
                # Validate the hop is still a known neighbour
                if next_hop in self.graph.get(pos, {}):
                    actions.append(next_hop)
                else:
                    # Path is stale; clear and wait
                    self.paths[i] = []
                    self.targets[i] = None
                    actions.append(pos)
            else:
                actions.append(pos)  # wait

        return actions

    # ------------------------------------------------------------------
    # Collision resolution
    # ------------------------------------------------------------------

    def _resolve_collisions(self, actions: List[int]) -> List[int]:
        """Lower agent_id wins vertex conflicts; both halt on edge swaps."""
        resolved = list(actions)
        claimed_nodes: Set[int] = set()

        for i in range(self.n_agents):
            nxt = resolved[i]
            if nxt == self.current[i]:
                continue  # waiting — no conflict
            if nxt in claimed_nodes:
                # Someone else is heading here with higher priority
                resolved[i] = self.current[i]
                self.paths[i] = []  # force re-plan next tick
            else:
                claimed_nodes.add(nxt)

        # Edge-swap check: agent i goes A→B while agent j goes B→A
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if (
                    resolved[i] == self.current[j]
                    and resolved[j] == self.current[i]
                    and resolved[i] != self.current[i]
                    and resolved[j] != self.current[j]
                ):
                    # Block both; lower id gets priority (wait the other)
                    resolved[j] = self.current[j]
                    self.paths[j] = []

        return resolved

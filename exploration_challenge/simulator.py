"""The world model: ground-truth graph, drone state, movement, coverage, phases.

Official challenge: **3 UAVs** with lockstep per-hop movement and collision
blocking. Coverage unions across agents; score is makespan (max per-agent
distance) — lowest wins (fastest explore + surveil).

Coverage semantics
------------------
* Exploration uses *observed* coverage: a node counts once it is detected in any
  depth-``k`` scan (including nodes you fly past without stopping). With
  ``drop_prob > 0``, a node may be missed on one scan and detected on a later
  re-scan. Exploration ends when ``observed_fraction >= explore_threshold``.
* Surveillance starts fresh at the explore->surveil transition: its coverage
  counter is reset to empty, and a node counts again only once it re-enters a
  depth-``k`` ("visual") observation made *during* the surveillance phase.
  Surveillance ends when ``surveil_fraction >= surveil_threshold`` (fraction of all
  nodes re-observed). What exploration discovered or visited beforehand does not
  pre-credit surveillance -- the only carry-over is map knowledge, which makes the
  re-observation pass cheaper than starting blind.
  The episode ends as soon as both thresholds are met; no return-to-start is required.
"""

from __future__ import annotations

import random

import networkx as nx

from ._internal.config import eval_params
from ._internal.sensor import build_observation, los_ball, visible_adj
from .observation import Observation


class InvalidActionError(Exception):
    """Raised when a policy returns an unreachable target or an unknown edge."""


def _move_blocked(
    agent_id: int,
    frm: int,
    to: int,
    intended: dict[int, tuple[int, int]],
    positions: dict[int, int],
    committed: dict[int, tuple[int, int]],
    destinations: dict[int, int],
) -> bool:
    """Return True when ``agent_id`` cannot move ``frm`` -> ``to`` this tick."""
    if to in destinations:
        return True
    if any(u == to and v == frm for u, v in committed.values()):
        return True
    if frm == to:
        return False
    for other_id, pos in positions.items():
        if other_id == agent_id or pos != to:
            continue
        o_frm, o_to = intended[other_id]
        if o_frm == o_to or o_to == frm:
            return True
    return False


def _coverage_threshold(
    name: str,
    override: float | None,
    default: float,
) -> float:
    """Resolve explore/surveil threshold: explicit override or ``[eval]`` default."""
    if override is not None:
        return float(override)
    return float(eval_params().get(name, default))


class AgentState:
    def __init__(self, agent_id: int, start: int) -> None:
        self.id = agent_id
        self.position = start
        self.distance = 0.0
        self.visited: set[int] = {start}


class Simulator:
    def __init__(
        self,
        world: nx.Graph,
        k: int,
        starts: list[int],
        max_turn_deg: float | None = None,
        drop_prob: float = 0.0,
        seed: int | None = None,
        explore_threshold: float | None = None,
        surveil_threshold: float | None = None,
    ) -> None:
        self.world = world
        self.k = int(k)
        if max_turn_deg is None:
            max_turn_deg = float(eval_params().get("max_turn_deg", 75.0))
        self.max_turn_deg = float(max_turn_deg)
        self.drop_prob = float(drop_prob)
        self._rng = random.Random(seed)
        if len(set(starts)) != len(starts):
            raise ValueError("start nodes must be distinct")
        for node in starts:
            if node not in world:
                raise ValueError(f"start node {node!r} not present in graph")
        self.starts = [int(s) for s in starts]
        self.start = self.starts[0]
        self.explore_threshold = _coverage_threshold(
            "explore_threshold", explore_threshold, 0.9
        )
        self.surveil_threshold = _coverage_threshold(
            "surveil_threshold", surveil_threshold, 0.9
        )
        self.n_total = world.number_of_nodes()

        self.agents = [AgentState(i, self.starts[i]) for i in range(len(self.starts))]

        # Ground-truth bookkeeping maintained by the simulator (not trusted to the
        # policy): which nodes have been observed, and the observed adjacency used
        # to validate one-hop moves.
        self.observed: set[int] = set()
        self.known_adj: dict[int, dict[int, float]] = {}

        # Nodes re-observed during the surveillance phase. Empty until the
        # explore->surveil transition, then accrues independently of `observed`.
        self.surveilled: set[int] = set()

        self.phase = "explore"
        self.dist_explore: float | None = None
        self.steps = 0
        self._wait_counts: dict[int, int] = {a.id: 0 for a in self.agents}

        for a in self.agents:
            self._reveal_from(a.position)
        self.check_transition()

    # --- sensing ---------------------------------------------------------------
    def observe(self, agent_id: int) -> Observation:
        """Full depth-``k`` Observation for a decision point (also reveals)."""
        agent = self.agents[agent_id]
        visible = los_ball(
            self.world,
            agent.position,
            self.k,
            self.max_turn_deg,
            drop_prob=self.drop_prob,
            rng=self._rng,
        )
        obs = build_observation(
            self.world,
            agent.position,
            self.k,
            agent_id,
            self.visited_union(),
            self.max_turn_deg,
            drop_prob=self.drop_prob,
            rng=self._rng,
            visible=visible,
        )
        self._reveal_visible(visible)
        return obs

    def _reveal_from(self, position: int) -> None:
        """Cheaply fold the depth-``k`` ball at ``position`` into observed/known.

        Used on every traversed node during movement, so it avoids constructing
        Observation dataclasses -- just updates the ground-truth bookkeeping.
        """
        visible = los_ball(
            self.world,
            position,
            self.k,
            self.max_turn_deg,
            drop_prob=self.drop_prob,
            rng=self._rng,
        )
        self._reveal_visible(visible)

    def _reveal_visible(self, visible: set[int]) -> None:
        self.observed.update(visible)
        if self.phase == "surveil":
            self.surveilled.update(visible)
        for n, nbrs in visible_adj(self.world, visible).items():
            self.known_adj.setdefault(n, {}).update(nbrs)

    # --- coverage metrics ------------------------------------------------------
    def observed_fraction(self) -> float:
        return len(self.observed) / self.n_total

    def visited_union(self) -> set[int]:
        if len(self.agents) == 1:
            return self.agents[0].visited
        return set().union(*(a.visited for a in self.agents))

    def surveil_fraction(self) -> float:
        # Zero throughout exploration; only nodes re-observed during the
        # surveillance phase count, against the full node set.
        if self.n_total == 0:
            return 0.0
        return len(self.surveilled) / self.n_total

    def total_distance(self) -> float:
        return sum(a.distance for a in self.agents)

    def makespan_distance(self) -> float:
        if not self.agents:
            return 0.0
        return max(a.distance for a in self.agents)

    @property
    def deadlock(self) -> bool:
        """True when any agent has been blocked/waiting for five consecutive ticks."""
        return any(count >= 5 for count in self._wait_counts.values())

    # --- phase / termination ---------------------------------------------------
    def check_transition(self) -> None:
        if self.phase == "explore" and self.observed_fraction() >= self.explore_threshold:
            self.phase = "surveil"
            self.dist_explore = self.makespan_distance()
            self.surveilled = set()
            return
        if self.phase == "surveil" and self.surveil_fraction() >= self.surveil_threshold:
            self.phase = "done"

    def is_done(self) -> bool:
        return self.phase == "done"

    # --- action handling -------------------------------------------------------
    def resolve_action(self, agent_id: int, action) -> int:
        """Validate a one-hop action: a known neighbour id, or wait at current node."""
        if isinstance(action, (list, tuple)):
            raise InvalidActionError(
                "actions must be int node ids (one known neighbour per tick), not paths"
            )

        agent = self.agents[agent_id]
        nxt = int(action)
        pos = agent.position
        if nxt == pos:
            return pos
        if nxt not in self.known_adj.get(pos, {}):
            raise InvalidActionError(
                f"next node {nxt} is not a known neighbour of current position {pos}"
            )
        return nxt

    def execute_path(self, agent_id: int, path: list[int], on_node=None) -> None:
        """Fly ``agent`` along ``path``, streaming observations at each node.

        Stops early (handing control back to the policy) at the explore->surveil
        transition or once the episode is done.
        """
        agent = self.agents[agent_id]
        for u, v in zip(path, path[1:]):
            if not self.world.has_edge(u, v):
                raise InvalidActionError(f"edge {u}-{v} does not exist in the world")
            agent.distance += self.world.edges[u, v]["cost"]
            agent.position = v
            agent.visited.add(v)
            self._reveal_from(v)
            # Hand control back to the policy at the phase transition or when done.
            phase_before = self.phase
            self.check_transition()
            if on_node is not None:
                on_node()
            if self.phase != phase_before or self.is_done():
                return

    def step_agents(self, next_hops: dict[int, int]) -> dict[int, bool]:
        """Advance all agents one hop in lockstep with collision blocking.

        ``next_hops`` maps agent id to the neighbour to move to (or the current
        node to wait). Lower agent ids win vertex conflicts; edge swaps block
        both movers for the tick. Returns ``{agent_id: moved}``.
        """
        intended: dict[int, tuple[int, int]] = {}
        for agent in self.agents:
            frm = agent.position
            to = int(next_hops.get(agent.id, frm))
            if to != frm:
                if not self.world.has_edge(frm, to):
                    raise InvalidActionError(f"edge {frm}-{to} does not exist in the world")
                if to not in self.known_adj.get(frm, {}):
                    raise InvalidActionError(f"edge {frm}-{to} is not in the known graph")
            intended[agent.id] = (frm, to)

        committed: dict[int, tuple[int, int]] = {}
        destinations: dict[int, int] = {}
        moved: dict[int, bool] = {a.id: False for a in self.agents}
        positions = {a.id: a.position for a in self.agents}

        for agent in sorted(self.agents, key=lambda a: a.id):
            frm, to = intended[agent.id]
            if _move_blocked(agent.id, frm, to, intended, positions, committed, destinations):
                self._wait_counts[agent.id] += 1
                continue

            committed[agent.id] = (frm, to)
            destinations[to] = agent.id
            if frm == to:
                self._wait_counts[agent.id] = 0

        for agent in self.agents:
            if agent.id not in committed:
                continue
            frm, to = committed[agent.id]
            if frm == to:
                continue
            agent.distance += self.world.edges[frm, to]["cost"]
            agent.position = to
            agent.visited.add(to)
            self._reveal_from(to)
            moved[agent.id] = True
            self._wait_counts[agent.id] = 0

        self.check_transition()
        return moved

    # --- results ---------------------------------------------------------------
    def result(self) -> dict:
        makespan = self.makespan_distance()
        total = self.total_distance()
        explore_done = self.dist_explore is not None
        surveil_done = self.is_done()
        dist_surveil = (
            (makespan - self.dist_explore) if (explore_done and surveil_done) else None
        )
        score = makespan if (explore_done and surveil_done) else float("inf")
        return {
            "name": self.world.graph.get("name"),
            "n_total": self.n_total,
            "n_agents": len(self.agents),
            "k": self.k,
            "start": self.start,
            "starts": list(self.starts),
            "explore_completed": explore_done,
            "surveil_completed": surveil_done,
            "dist_explore": self.dist_explore,
            "dist_surveil": dist_surveil,
            "makespan_distance": makespan,
            "total_distance": total,
            "score": score,
            "steps": self.steps,
            "observed_fraction": self.observed_fraction(),
            "surveil_fraction": self.surveil_fraction(),
        }

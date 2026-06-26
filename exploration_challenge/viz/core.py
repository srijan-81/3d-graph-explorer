"""Rerun-based 3D visualizer orchestration."""

from __future__ import annotations

import heapq
import math
import random
import time
from collections import deque

import networkx as nx

from ..graph_io import node_xyz, undirected_edge_key
from .mesh import build_quadcopter_mesh
from .rerun_compat import HAVE_RERUN, clear_entity, log_scalar, rr, set_step
from .styles import (
    DARK_BLUE,
    EDGE_RADIUS,
    FIREWORK_COLORS,
    GREY,
    KNOWN_EDGE_RADIUS,
    LIGHT_BLUE,
    NODE_RADIUS_GREY,
    NODE_RADIUS_OBSERVED,
    NODE_RADIUS_VISITED,
    OBSERVED_COLOR,
    STEPS_RATE_WINDOW_SEC,
    SURVEILLED_COLOR,
    TRAIL_COLOR,
    TRAIL_RADIUS,
    blue_ramp,
)

DRONE_COLORS = [
    (220, 45, 45),
    (255, 105, 180),
    (170, 60, 220),
]


class Visualizer:
    def __init__(
        self,
        app_id: str = "exploration_challenge",
        spawn: bool = True,
        *,
        reduced: bool = False,
    ) -> None:
        self.enabled = HAVE_RERUN
        self.reduced = reduced
        self._t = 0
        self._world: nx.Graph | None = None
        self._order: list[int] = []
        self._prev_agent_pos: dict[int, int | None] = {}
        self._agent_yaw: dict[int, float] = {}
        self._trail_last_node: dict[int, int | None] = {}
        self._drone_entities_ready = 0
        self._last_phase = "explore"
        self._started: float | None = None
        self._step_samples: deque[tuple[float, int]] = deque()
        self._last_rate_steps = -1
        self._positions: list[list[float]] = []
        self._pos_by_node: dict[int, list[float]] = {}
        self._node_index: dict[int, int] = {}
        self._colors: list[tuple[int, int, int]] = []
        self._radii: list[float] = []
        self._dist: dict[int, int] = {}
        self._ramp_max = 0
        self._prev_observed: set[int] = set()
        self._prev_visited: set[int] = set()
        self._prev_surveilled: set[int] = set()
        self._logged_edges: set[tuple[int, int]] = set()
        if self.enabled:
            rr.init(app_id, spawn=spawn)

    def setup(self, world: nx.Graph) -> None:
        if not self.enabled:
            return
        self._world = world
        self._order = list(world.nodes)
        self._positions = [list(node_xyz(world, n)) for n in self._order]
        self._pos_by_node = dict(zip(self._order, self._positions))
        self._node_index = {n: i for i, n in enumerate(self._order)}
        n = len(self._positions)
        self._colors = [GREY] * n
        self._radii = [NODE_RADIUS_GREY] * n
        self._dist = {}
        self._ramp_max = 0
        self._prev_observed = set()
        self._prev_visited = set()
        self._prev_surveilled = set()
        self._logged_edges.clear()
        self._prev_agent_pos = {}
        self._agent_yaw = {}
        self._trail_last_node = {}
        self._drone_entities_ready = 0
        self._last_phase = "explore"
        self._started = time.monotonic()
        self._reset_rate_window(0)

        set_step(self._t)
        strips = [
            [self._pos_by_node[u], self._pos_by_node[v]] for u, v in world.edges
        ]
        rr.log("world/all_edges", rr.LineStrips3D(strips, colors=[(70, 70, 70)] * len(strips),
                                                  radii=EDGE_RADIUS))
        clear_entity("world/known_edges", recursive=True)
        clear_entity("world/drone", recursive=True)
        clear_entity("world/drone_trail", recursive=True)

        rr.log(
            "world/nodes",
            rr.Points3D(
                self._positions,
                colors=self._colors,
                radii=self._radii,
            ),
        )

        series = getattr(rr, "SeriesLines", None)
        if series is not None:
            rr.log("metrics/coverage",
                   series(colors=[OBSERVED_COLOR, SURVEILLED_COLOR],
                          names=["observed", "surveilled"], widths=[2.0, 2.0]),
                   static=True)

    def on_step(self, sim) -> None:
        """Single evaluator tick: one hop and one planning step."""
        if not self.enabled or self._world is None:
            return
        if self.reduced:
            self._advance_timeline()
            self._apply_motion(sim)
            self._log_coverage(sim.observed_fraction(), sim.surveil_fraction())
            self._log_status(sim)
        else:
            self.log(sim)

    def log(self, sim) -> None:
        """Full frame: fog-of-war, known edges, metrics, and drone pose."""
        if not self.enabled or self._world is None:
            return
        self._advance_timeline()
        self._update_scene(sim)
        self._apply_motion(sim)
        self._log_coverage(sim.observed_fraction(), sim.surveil_fraction())
        self._log_status(sim)

    def _advance_timeline(self) -> None:
        set_step(self._t)
        self._t += 1

    def _update_scene(self, sim) -> None:
        """Incrementally refresh fog-of-war nodes and newly discovered edges."""
        observed = sim.observed
        visited = sim.visited_union()
        surveilled = sim.surveilled

        new_observed = observed - self._prev_observed
        new_visited = visited - self._prev_visited
        new_surveilled = surveilled - self._prev_surveilled
        new_edges = self._collect_new_edges(sim.known_adj)

        dist_changed = self._relax_distances(sim, new_visited, new_observed, new_edges)

        changed: set[int] = set(new_observed) | set(new_visited) | set(new_surveilled) | dist_changed
        nodes_dirty = False
        for node in changed:
            idx = self._node_index[node]
            color, radius = self._node_style(sim, node)
            if self._colors[idx] != color or self._radii[idx] != radius:
                self._colors[idx] = color
                self._radii[idx] = radius
                nodes_dirty = True

        if nodes_dirty:
            rr.log(
                "world/nodes",
                rr.Points3D(self._positions, colors=self._colors, radii=self._radii),
            )

        if new_edges:
            strips = [
                [self._pos_by_node[u], self._pos_by_node[v]] for u, v in new_edges
            ]
            step = self._t - 1
            rr.log(
                f"world/known_edges/{step}",
                rr.LineStrips3D(
                    strips,
                    colors=[(160, 170, 190)] * len(strips),
                    radii=KNOWN_EDGE_RADIUS,
                ),
            )

        self._prev_observed = set(observed)
        self._prev_visited = set(visited)
        self._prev_surveilled = set(surveilled)

    def _collect_new_edges(
        self, known_adj: dict[int, dict[int, float]]
    ) -> list[tuple[int, int]]:
        """Return undirected edge keys discovered since the last scene update."""
        new_edges: list[tuple[int, int]] = []
        for u, nbrs in known_adj.items():
            for v in nbrs:
                key = undirected_edge_key(u, v)
                if key in self._logged_edges:
                    continue
                self._logged_edges.add(key)
                new_edges.append(key)
        return new_edges

    def _relax_distances(
        self,
        sim,
        new_visited: set[int],
        new_observed: set[int],
        new_edges: list[tuple[int, int]],
    ) -> set[int]:
        """Relax hop distances from visited nodes; return nodes whose dist changed."""
        observed = sim.observed
        changed: set[int] = set()
        queue: list[tuple[int, int]] = []

        def improve(node: int, dist: int) -> None:
            prev = self._dist.get(node)
            if prev is not None and dist >= prev:
                return
            self._dist[node] = dist
            if dist > self._ramp_max:
                self._ramp_max = dist
            changed.add(node)
            heapq.heappush(queue, (dist, node))

        for node in new_visited:
            improve(node, 0)

        for u, v in new_edges:
            for a, b in ((u, v), (v, u)):
                if b not in observed:
                    continue
                if a in self._dist:
                    improve(b, self._dist[a] + 1)

        for node in new_observed:
            best = math.inf
            for nbr in sim.known_adj.get(node, {}):
                if nbr in observed and nbr in self._dist:
                    best = min(best, self._dist[nbr] + 1)
            if best < math.inf:
                improve(node, int(best))

        while queue:
            dist, node = heapq.heappop(queue)
            if dist > self._dist.get(node, math.inf):
                continue
            for nbr in sim.known_adj.get(node, {}):
                if nbr not in observed:
                    continue
                improve(nbr, dist + 1)

        return changed

    def _node_style(self, sim, node: int) -> tuple[tuple[int, int, int], float]:
        """Return display color and radius for one node."""
        in_surveillance = sim.phase in ("surveil", "done")
        if in_surveillance and node in sim.surveilled:
            return SURVEILLED_COLOR, NODE_RADIUS_VISITED
        if node in self._dist:
            dist = self._dist[node]
            if dist == 0:
                return DARK_BLUE, NODE_RADIUS_VISITED
            ramp_t = dist / self._ramp_max if self._ramp_max > 0 else 0.0
            return blue_ramp(ramp_t), NODE_RADIUS_OBSERVED
        if node in sim.observed:
            return LIGHT_BLUE, NODE_RADIUS_OBSERVED
        return GREY, NODE_RADIUS_GREY

    def _drone_entity(self, agent_id: int, n_agents: int) -> str:
        if n_agents == 1:
            return "world/drone"
        return f"world/drone/{agent_id}"

    def _trail_entity(self, agent_id: int, n_agents: int) -> str:
        if n_agents == 1:
            return "world/drone_trail"
        return f"world/drone_trail/{agent_id}"

    def _ensure_drone_entities(self, n_agents: int) -> None:
        if not self.enabled or n_agents <= self._drone_entities_ready:
            return
        verts, tris, colors = build_quadcopter_mesh()
        for agent_id in range(self._drone_entities_ready, n_agents):
            rr.log(
                self._drone_entity(agent_id, n_agents),
                rr.Mesh3D(
                    vertex_positions=verts,
                    triangle_indices=tris,
                    vertex_colors=colors,
                ),
                static=True,
            )
        self._drone_entities_ready = n_agents

    def _apply_motion(self, sim) -> None:
        """Update drone pose and trail for the current simulator state."""
        n_agents = len(sim.agents)
        self._ensure_drone_entities(n_agents)
        if sim.phase == "surveil" and self._last_phase == "explore":
            self._reset_trail(n_agents)
        self._last_phase = sim.phase
        for agent in sim.agents:
            agent_id = agent.id
            self._append_trail_segment(agent_id, agent.position, n_agents)
            self._update_uav(agent_id, agent.position, n_agents)

    def celebrate(self, sim) -> None:
        """Play a short firework burst in Rerun when the episode completes."""
        if not self.enabled or self._world is None:
            return

        if self._t > 0:
            set_step(self._t - 1)
        self._log_status(sim)

        center = self._drone_pos(sim.agents[0].position)
        center[2] += 0.35
        rng = random.Random(7)

        bursts: list[tuple[list[float], list[float], tuple[int, int, int]]] = []
        for _ in range(8):
            origin = [
                center[0] + rng.uniform(-0.8, 0.8),
                center[1] + rng.uniform(-0.8, 0.8),
                center[2] + rng.uniform(0.0, 1.0),
            ]
            for _ in range(28):
                color = rng.choice(FIREWORK_COLORS)
                speed = rng.uniform(2.0, 4.5)
                theta = rng.uniform(0.0, 2.0 * math.pi)
                phi = rng.uniform(0.15, math.pi - 0.15)
                vel = [
                    speed * math.sin(phi) * math.cos(theta),
                    speed * math.sin(phi) * math.sin(theta),
                    speed * math.cos(phi),
                ]
                bursts.append((origin, vel, color))

        gravity = 2.8
        n_frames = 64
        for frame in range(n_frames):
            set_step(self._t)
            self._t += 1
            t = frame * 0.045
            fade = max(0.0, 1.0 - frame / (n_frames - 4))
            positions: list[list[float]] = []
            colors: list[tuple[int, int, int]] = []
            radii: list[float] = []
            for origin, vel, color in bursts:
                positions.append([
                    origin[0] + vel[0] * t,
                    origin[1] + vel[1] * t,
                    origin[2] + vel[2] * t - 0.5 * gravity * t * t,
                ])
                brightness = fade ** 0.6
                colors.append(tuple(min(255, int(c * brightness)) for c in color))
                radii.append(0.12 * fade)
            rr.log("celebration/fireworks", rr.Points3D(positions, colors=colors, radii=radii))

        set_step(self._t - 1)
        clear_entity("celebration/fireworks")
        self._reset_rate_window(0)

    def _update_uav(self, agent_id: int, agent_pos: int, n_agents: int) -> None:
        """Place the quadcopter mesh at the agent, oriented along travel direction."""
        pos = self._drone_pos(agent_pos)
        prev = self._prev_agent_pos.get(agent_id)

        if prev is not None and prev != agent_pos:
            prev_pos = self._drone_pos(prev)
            dx = pos[0] - prev_pos[0]
            dy = pos[1] - prev_pos[1]
            if dx * dx + dy * dy > 1e-8:
                self._agent_yaw[agent_id] = math.atan2(dy, dx)

        yaw = self._agent_yaw.get(agent_id, 0.0)
        rr.log(
            self._drone_entity(agent_id, n_agents),
            rr.Transform3D(
                translation=pos,
                rotation=rr.RotationAxisAngle(
                    axis=[0.0, 0.0, 1.0],
                    angle=rr.Angle(rad=yaw),
                ),
            ),
        )
        self._prev_agent_pos[agent_id] = agent_pos

    def _reset_trail(self, n_agents: int) -> None:
        """Clear flight trails (e.g. at the explore→surveillance transition)."""
        self._trail_last_node = {i: None for i in range(n_agents)}
        for agent_id in range(n_agents):
            clear_entity(self._trail_entity(agent_id, n_agents), recursive=True)

    def _append_trail_segment(self, agent_id: int, node: int, n_agents: int) -> None:
        """Append one trail segment when the drone reaches a new node."""
        last = self._trail_last_node.get(agent_id)
        if last == node:
            return
        if last is not None:
            color = DRONE_COLORS[agent_id % len(DRONE_COLORS)]
            step = self._t - 1
            rr.log(
                f"{self._trail_entity(agent_id, n_agents)}/{step}",
                rr.LineStrips3D(
                    [[self._drone_pos(last), self._drone_pos(node)]],
                    colors=[color],
                    radii=TRAIL_RADIUS,
                ),
            )
        self._trail_last_node[agent_id] = node

    def _log_status(self, sim) -> None:
        """Live status readout; overwritten in place each step (not a scrolling log)."""
        doc = getattr(rr, "TextDocument", None)
        if doc is None:
            return
        if sim.phase == "done":
            phase = "Done"
        elif sim.phase == "surveil":
            phase = "Surveillance"
        else:
            phase = "Exploration"
        elapsed = max(time.monotonic() - (self._started or time.monotonic()), 1e-6)
        steps_per_sec = self._recent_steps_per_sec(sim.steps)
        graph_name = self._world.graph.get("name") if self._world is not None else None
        text = (
            f"## graph: {graph_name or '—'}\n"
            f"## phase: {phase}\n"
            f"## observed: {sim.observed_fraction():.1%} / {sim.explore_threshold:.1%}\n"
            f"## surveilled: {sim.surveil_fraction():.1%} / {sim.surveil_threshold:.1%}\n"
            f"## distance (makespan): {sim.makespan_distance():.1f}\n"
            f"## nodes visited: {len(sim.visited_union())}\n"
            f"## time passed: {elapsed:.1f} s\n"
            f"## planning steps: {sim.steps} ({steps_per_sec:.1f}/s)"
        )
        markdown = getattr(getattr(rr, "MediaType", None), "MARKDOWN", None)
        rr.log("status", doc(text, **({"media_type": markdown} if markdown else {})))

    def _reset_rate_window(self, steps: int) -> None:
        """Start a fresh sliding window for recent planning-step rate."""
        self._step_samples.clear()
        self._step_samples.append((time.monotonic(), steps))
        self._last_rate_steps = steps

    def _recent_steps_per_sec(self, steps: int) -> float:
        """Planning steps per second over a recent wall-clock window."""
        if steps < self._last_rate_steps:
            self._reset_rate_window(steps)
        now = time.monotonic()
        self._step_samples.append((now, steps))
        self._last_rate_steps = steps
        cutoff = now - STEPS_RATE_WINDOW_SEC
        while len(self._step_samples) >= 2 and self._step_samples[1][0] <= cutoff:
            self._step_samples.popleft()
        if len(self._step_samples) < 2:
            t0, s0 = self._step_samples[0]
            return max(steps - s0, 0) / max(now - t0, 1e-6)
        t0, s0 = self._step_samples[0]
        t1, s1 = self._step_samples[-1]
        return max(s1 - s0, 0) / max(t1 - t0, 1e-6)

    def _log_coverage(self, observed: float, surveilled: float) -> None:
        """Log observed + surveilled fraction as two lines on one shared plot."""
        scalars = getattr(rr, "Scalars", None)
        if scalars is not None:
            rr.log("metrics/coverage", scalars([observed, surveilled]))
            return
        log_scalar("metrics/observed_fraction", observed)
        log_scalar("metrics/surveilled_fraction", surveilled)

    def _drone_pos(self, node: int) -> list[float]:
        """Return a copy of the node position, raised for drone/trail rendering."""
        x, y, z = self._pos_by_node[node]
        return [x, y, z + 0.10]

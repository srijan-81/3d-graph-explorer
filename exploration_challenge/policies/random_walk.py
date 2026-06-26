"""Random-walk starter policy — copy this file and implement ``Explorer``.

Save as ``submission.py`` in the repo root to have ``run_eval.py`` pick it up
automatically, set ``submission`` in ``params.toml`` ``[eval]``, or pass
``--submission my_solution.py``.

Your job: coordinate **three UAVs** to (1) observe >= explore_threshold of the
graph as cheaply as possible, then (2) re-observe >= surveil_threshold during a
second pass. **Lowest makespan flight distance wins** — finish explore + surveil
fastest (slowest UAV sets the score).

The evaluator always calls the same API (`n_agents = 3` by default; return three
actions each tick):

    explorer = Explorer()
    explorer.reset(starts, observations, seed)   # once, before the run
    while not done:
        actions = explorer.step(observations, phase)  # three actions (one per UAV)

Each action is an **int** node id: the **next hop** only — a known neighbour of
that UAV's current position, or its current position to wait. You cannot name a
distant target and have the evaluator route toward it.

With three UAVs, observations and actions are aligned by ``agent_id``. UAVs
move in lockstep; collisions and edge swaps are blocked by the simulator.

You only ever see what's within ``k`` hops of each UAV's current node (``k``,
``max_turn_deg``, and ``drop_prob`` come from ``params.toml`` / CLI), plus each
drone's visited history. The sensor may randomly miss nodes on any scan
(``drop_prob``); revisiting an area gives fresh detection chances. See
``docs/RULES.md``, ``docs/graph_format.md``, and ``exploration_challenge/observation.py``.

Replace ``reset`` / ``step`` with your own logic.
"""

from __future__ import annotations

import random

from exploration_challenge.observation import Observation


class Explorer:
    def reset(
        self,
        starts: list[int],
        observations: list[Observation],
        seed: int | None = None,
    ) -> None:
        """Called once at the start. Initialise your state here."""
        self.visited: set[int] = set()
        self.surveil_seen: set[int] = set()
        self.rng = random.Random(seed)
        for obs in observations:
            self._note_visit(obs)

    def step(self, observations: list[Observation], phase: str) -> list[int]:
        """Return one next-hop node id per UAV (known neighbour, or wait)."""
        actions: list[int] = []
        for obs in observations:
            self._note_visit(obs)
            if phase == "surveil":
                self.surveil_seen.update(n.id for n in obs.nodes)

            neighbors = list(obs.neighbors(obs.position))
            if phase == "explore":
                choices = [n for n in neighbors if n not in self.visited]
            else:
                choices = [n for n in neighbors if n not in self.surveil_seen]
            if not choices:
                choices = neighbors
            if choices:
                actions.append(self.rng.choice(choices))
            else:
                actions.append(obs.position)
        return actions

    def _note_visit(self, obs: Observation) -> None:
        self.visited.update(obs.visited)
        self.visited.add(obs.position)

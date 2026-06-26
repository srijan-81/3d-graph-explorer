"""Run an ``Explorer`` policy over graphs and seeds; compute scores and live stats.

Official challenge uses **3 UAVs**. The score for a completed episode is makespan
flight distance (max per-agent distance across explore + surveil). **Lower is
better** — finish both phases fastest. Incomplete episodes score ``inf``.
"""

from __future__ import annotations

import json
import os
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from typing import Any, Callable

import networkx as nx

from ._internal.config import eval_params
from ._internal.progress import ProgressBar
from ._internal.seeding import policy_seed, sensor_seed, start_seed
from .simulator import InvalidActionError, Simulator

EXPORT_VERSION = 1
MAX_AGENTS = 3


def _validate_n_agents(n_agents: int) -> int:
    n = int(n_agents)
    if not 1 <= n <= MAX_AGENTS:
        raise ValueError(f"n_agents must be between 1 and {MAX_AGENTS}, got {n}")
    return n


def _from_eval(name: str, value, *, cast=float):
    """Use ``value`` when set, else the matching key from ``[eval]`` in params.toml."""
    if value is not None:
        return cast(value)
    return cast(eval_params()[name])


def _resolve_starts(
    world: nx.Graph,
    seed: int | None,
    start: int | None,
    n_agents: int,
) -> list[int]:
    """Pick distinct start nodes: agent 0 honors fixed override/config, rest from start RNG."""
    nodes = sorted(world.nodes)
    if not nodes:
        raise ValueError("graph has no nodes")
    if n_agents > len(nodes):
        raise ValueError(
            f"cannot place {n_agents} agents on graph with {len(nodes)} nodes"
        )

    rng = random.Random(start_seed(seed))

    if start is not None:
        first = int(start)
    else:
        cfg_start = eval_params().get("start")
        if cfg_start is not None:
            first = int(cfg_start)
        else:
            first = rng.choice(nodes)

    if first not in world:
        raise ValueError(f"start node {first!r} not present in graph")

    if n_agents == 1:
        return [first]

    pool = [n for n in nodes if n != first]
    rng.shuffle(pool)
    return [first, *pool[: n_agents - 1]]


ExplorerFactory = Callable[[], object]


def _actions_to_next_hops(sim: Simulator, actions: list) -> dict[int, int]:
    """Validate per-agent one-hop actions for ``step_agents``."""
    return {agent_id: sim.resolve_action(agent_id, action) for agent_id, action in enumerate(actions)}


def _notify_viz_step(viz, sim, step_delay: float = 0) -> None:
    if viz is not None:
        viz.on_step(sim)
    if step_delay > 0:
        time.sleep(step_delay)


def _reset_explorer(
    explorer,
    starts: list[int],
    observations: list,
    seed: int | None,
) -> bool:
    """Return False when ``reset()`` raises (episode should score ``inf``)."""
    pseed = policy_seed(seed)
    try:
        try:
            explorer.reset(starts, observations, pseed)
        except TypeError:
            explorer.reset(starts, observations)
    except Exception:
        return False
    return True


def _run_loop(
    sim: Simulator,
    explorer,
    *,
    viz,
    live: bool,
    max_steps: int,
    step_delay: float = 0.0,
) -> None:
    n_agents = len(sim.agents)
    last_phase = sim.phase
    all_stalls = 0

    while not sim.is_done() and sim.steps < max_steps:
        observations = [sim.observe(i) for i in range(n_agents)]
        try:
            actions = explorer.step(observations, sim.phase)
        except Exception as exc:
            if live:
                _finish_status_line()
                print(f"  ! policy error ({exc}); aborting episode")
            break

        if not isinstance(actions, (list, tuple)) or len(actions) != n_agents:
            if live:
                _finish_status_line()
                print(
                    f"  ! step must return {n_agents} actions, "
                    f"got {actions!r}; aborting episode"
                )
            break

        next_hops: dict[int, int] = {}
        try:
            next_hops = _actions_to_next_hops(sim, list(actions))
        except InvalidActionError as exc:
            if live:
                _finish_status_line()
                print(f"  ! invalid action ({exc}); aborting episode")
            break

        moved = sim.step_agents(next_hops)
        sim.steps += 1
        _notify_viz_step(viz, sim, step_delay)

        if not any(moved.values()):
            all_stalls += 1
        else:
            all_stalls = 0

        if sim.deadlock or all_stalls >= 5:
            if live:
                _finish_status_line()
                print("  ! policy stalled (no movement); aborting episode")
            break

        if sim.phase != last_phase:
            _print_status(sim, live, prefix=">>> surveillance phase")
            last_phase = sim.phase
        else:
            _print_status(sim, live)


def run_episode(
    world: nx.Graph,
    make_explorer: ExplorerFactory,
    seed: int | None = None,
    k: int | None = None,
    start: int | None = None,
    n_agents: int = 1,
    max_turn_deg: float | None = None,
    drop_prob: float | None = None,
    explore_threshold: float | None = None,
    surveil_threshold: float | None = None,
    viz=None,
    live: bool = True,
    max_steps: int = 20000,
    step_delay: float = 0.0,
    celebrate: bool = False,
) -> dict:
    n_agents = _validate_n_agents(n_agents)
    starts = _resolve_starts(world, seed, start, n_agents)
    sim = Simulator(
        world,
        k=_from_eval("k", k, cast=int),
        starts=starts,
        max_turn_deg=_from_eval("max_turn_deg", max_turn_deg),
        drop_prob=_from_eval("drop_prob", drop_prob),
        seed=sensor_seed(seed),
        explore_threshold=explore_threshold,
        surveil_threshold=surveil_threshold,
    )
    explorer = make_explorer()

    observations = [sim.observe(i) for i in range(n_agents)]
    if not _reset_explorer(explorer, starts, observations, seed):
        if live:
            _finish_status_line()
            print("  ! policy error during reset(); aborting episode")
        return sim.result()

    if viz is not None:
        viz.setup(world)

    _notify_viz_step(viz, sim, step_delay)
    if live:
        _reset_status_tracking()
        if n_agents == 1:
            print(f"  [start node {starts[0]}]")
        else:
            print(f"  [start nodes {starts}]")
    _print_status(sim, live, prefix="start")

    _run_loop(
        sim,
        explorer,
        viz=viz,
        live=live,
        max_steps=max_steps,
        step_delay=step_delay,
    )

    if viz is not None and sim.is_done() and celebrate:
        viz.celebrate(sim)

    if live and sim.is_done():
        _print_status(sim, live, prefix=">>> done")
    elif live:
        _finish_status_line()

    return sim.result()


_STATUS_LINE_ACTIVE = False
_STATUS_LINE_LEN = 0
_LAST_LOGGED_STEP = -1
_LAST_LOGGED_OBSERVED: float | None = None
_LAST_LOGGED_SURVEILLED: float | None = None
_STATUS_THROTTLE_STEPS = 25


def _reset_status_tracking() -> None:
    global _LAST_LOGGED_STEP, _LAST_LOGGED_OBSERVED, _LAST_LOGGED_SURVEILLED
    _finish_status_line()
    _LAST_LOGGED_STEP = -1
    _LAST_LOGGED_OBSERVED = None
    _LAST_LOGGED_SURVEILLED = None


def _finish_status_line() -> None:
    global _STATUS_LINE_ACTIVE, _STATUS_LINE_LEN
    if _STATUS_LINE_ACTIVE:
        sys.stdout.write("\n")
        sys.stdout.flush()
        _STATUS_LINE_ACTIVE = False
        _STATUS_LINE_LEN = 0


def _format_status(sim: Simulator, prefix: str = "") -> str:
    tag = prefix or sim.phase
    return (
        f"  [{tag:<22}] step={sim.steps:>4}  "
        f"observed={sim.observed_fraction():6.1%}  "
        f"surveilled={sim.surveil_fraction():6.1%}  "
        f"dist={sim.makespan_distance():8.1f}"
    )


def _print_status(sim: Simulator, live: bool, prefix: str = "") -> None:
    if not live:
        return

    line = _format_status(sim, prefix)
    if prefix:
        _finish_status_line()
        print(line)
        return

    if sys.stdout.isatty():
        global _STATUS_LINE_ACTIVE, _STATUS_LINE_LEN
        pad = max(_STATUS_LINE_LEN - len(line), 0)
        sys.stdout.write("\r" + line + " " * pad)
        sys.stdout.flush()
        _STATUS_LINE_ACTIVE = True
        _STATUS_LINE_LEN = len(line)
        return

    global _LAST_LOGGED_STEP, _LAST_LOGGED_OBSERVED, _LAST_LOGGED_SURVEILLED
    observed = sim.observed_fraction()
    surveilled = sim.surveil_fraction()
    if (
        _LAST_LOGGED_STEP < 0
        or sim.steps - _LAST_LOGGED_STEP >= _STATUS_THROTTLE_STEPS
        or abs(observed - (_LAST_LOGGED_OBSERVED or 0.0)) >= 0.005
        or abs(surveilled - (_LAST_LOGGED_SURVEILLED or 0.0)) >= 0.005
    ):
        print(line)
        _LAST_LOGGED_STEP = sim.steps
        _LAST_LOGGED_OBSERVED = observed
        _LAST_LOGGED_SURVEILLED = surveilled


def run_graph(
    world: nx.Graph,
    make_explorer: ExplorerFactory,
    seeds: list[int] | None = None,
    k: int | None = None,
    start: int | None = None,
    n_agents: int = 1,
    max_turn_deg: float | None = None,
    drop_prob: float | None = None,
    viz=None,
    live: bool = True,
    max_steps: int = 20000,
    step_delay: float = 0.0,
    progress: ProgressBar | None = None,
    celebrate_on_finish: bool = False,
) -> dict:
    """Run one graph over one or more seeds and aggregate."""
    n_agents = _validate_n_agents(n_agents)
    seeds = seeds if seeds else [None]
    runs = []
    for i, seed in enumerate(seeds):
        runs.append(
            run_episode(
                world,
                make_explorer,
                seed=seed,
                k=k,
                start=start,
                n_agents=n_agents,
                max_turn_deg=max_turn_deg,
                drop_prob=drop_prob,
                viz=viz,
                live=live,
                max_steps=max_steps,
                step_delay=step_delay,
                celebrate=celebrate_on_finish and i == len(seeds) - 1,
            )
        )
        if progress is not None:
            progress.set_postfix(graph=world.graph.get("name"), seed=seed)
            progress.update()

    scores = [r["score"] for r in runs]
    completed = [s for s in scores if s != float("inf")]
    mean_score = statistics.mean(scores) if scores else float("inf")
    return {
        "name": world.graph.get("name"),
        "n_total": world.number_of_nodes(),
        "n_agents": n_agents,
        "k": _from_eval("k", k, cast=int),
        "seeds": list(seeds),
        "runs": runs,
        "mean_score": mean_score,
        "stdev_score": statistics.pstdev(completed) if len(completed) > 1 else 0.0,
        "completion_rate": len(completed) / len(runs) if runs else 0.0,
    }


def run_suite(
    worlds: list[nx.Graph],
    make_explorer: ExplorerFactory,
    seeds: list[int] | None = None,
    k: int | None = None,
    start: int | None = None,
    n_agents: int = 1,
    max_turn_deg: float | None = None,
    drop_prob: float | None = None,
    viz=None,
    live: bool = True,
    max_steps: int = 20000,
    step_delay: float = 0.0,
    show_progress: bool = False,
) -> dict:
    """Run a set of graphs; the suite score is the sum of per-graph mean scores."""
    n_agents = _validate_n_agents(n_agents)
    k_val = _from_eval("k", k, cast=int)
    seeds = seeds if seeds else [None]
    per_graph = []
    progress = ProgressBar(
        "evaluating",
        total=len(worlds) * len(seeds),
        disable=not show_progress,
    )
    try:
        for i, world in enumerate(worlds):
            if live:
                _finish_status_line()
                print(
                    f"\n=== graph '{world.graph.get('name')}' "
                    f"(n={world.number_of_nodes()}, k={k_val}, agents={n_agents}) ==="
                )
            per_graph.append(
                run_graph(
                    world,
                    make_explorer,
                    seeds,
                    k,
                    start,
                    n_agents,
                    max_turn_deg,
                    drop_prob,
                    viz,
                    live,
                    max_steps,
                    step_delay,
                    progress=progress,
                    celebrate_on_finish=i == len(worlds) - 1,
                )
            )
    finally:
        progress.close()

    total = sum(g["mean_score"] for g in per_graph)
    return {"per_graph": per_graph, "total_score": total}


def print_summary(suite: dict) -> None:
    print("\n" + "=" * 64)
    print(f"{'graph':<24}{'n':>5}{'k':>4}{'mean score':>14}{'complete':>12}")
    print("-" * 64)
    for g in suite["per_graph"]:
        score = f"{g['mean_score']:.1f}" if g["mean_score"] != float("inf") else "inf"
        print(f"{str(g['name']):<24}{g['n_total']:>5}{g['k']:>4}{score:>14}"
              f"{g['completion_rate']:>11.0%}")
    print("-" * 64)
    total = suite["total_score"]
    total_str = f"{total:.1f}" if total != float("inf") else "inf"
    print(f"{'TOTAL (sum of means)':<24}{'':>5}{'':>4}{total_str:>14}")
    print("=" * 64)


def _sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, float) and (obj == float("inf") or obj == float("-inf")):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def export_suite(
    suite: dict,
    *,
    submission: str | None,
    eval_settings: dict[str, Any],
    graphs: str,
    created_at: str | None = None,
) -> dict:
    """Wrap a ``run_suite()`` result with metadata for JSON export."""
    if created_at is None:
        created_at = (
            datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        )
    payload = {
        "version": EXPORT_VERSION,
        "created_at": created_at,
        "submission": submission,
        "eval": eval_settings,
        "graphs": graphs,
        "per_graph": suite["per_graph"],
        "total_score": suite["total_score"],
    }
    return _sanitize_for_json(payload)


def write_suite_json(path: str, payload: dict) -> None:
    """Write an export payload to ``path``, creating parent directories."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

"""
Optuna optimizer targeting solution_v2.py weights.
Searches all tunable parameters simultaneously.

Usage:
    uv run optimize_v2.py
    uv run optimize_v2.py --trials 100 --seeds 1
    uv run optimize_v2.py --trials 60 --seeds 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import optuna
from exploration_challenge.evaluator import run_suite
from exploration_challenge.graph_io import load_graph
import my_solution

GRAPHS_DIR = Path("graphs/train")
N_AGENTS = 3

SEARCH_SPACE = {
    "W_EDGE_RATIO": (0.0,  150.0),
    "W_ISOLATION":  (0.0,   50.0),
    "W_MAX_EDGE":   (0.0,  250.0),
    "W_DIST":       (10.0, 150.0),
    "W_SURV_COV":   (0.1,   15.0),
    "W_SURV_DIST":  (0.0,    5.0),
    "TERR_EXPLORE": (1.0,    3.0),
    "TERR_SURVEIL": (1.0,   15.0),
}


def evaluate(weights: dict, seeds: list[int]) -> float:
    for k, v in weights.items():
        setattr(my_solution.Explorer, k, v)

    paths = sorted(GRAPHS_DIR.glob("*.json"))
    worlds = [load_graph(str(p)) for p in paths]
    names = [p.stem for p in paths]

    total = 0.0
    for world, name in zip(worlds, names):
        print(f"       {name}...", end="\r", flush=True)
        result = run_suite(
            [world],
            my_solution.Explorer,
            seeds=seeds,
            n_agents=N_AGENTS,
            live=False,
            max_steps=600,
        )
        s = result["total_score"]
        if s == float("inf"):
            return float("inf")
        total += s
    return total


def objective(trial: optuna.Trial, seeds: list[int]) -> float:
    weights = {
        k: trial.suggest_float(k, lo, hi)
        for k, (lo, hi) in SEARCH_SPACE.items()
    }
    return evaluate(weights, seeds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--seed",   type=int, default=0)
    parser.add_argument("--seeds",  type=int, default=1, help="eval seeds per graph")
    args = parser.parse_args()

    eval_seeds = list(range(args.seeds))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )

    # Seed with current values so TPE has a good starting point
    current = {k: getattr(my_solution.Explorer, k) for k in SEARCH_SPACE}
    study.enqueue_trial(current)

    n_graphs = len(list(GRAPHS_DIR.glob("*.json")))
    print(f"Running {args.trials} trials | {n_graphs} graphs | seeds={eval_seeds}")
    header = f"{'Trial':>6}  {'Score':>9}  {'Best':>9}  " + "  ".join(f"{k[:8]:>8}" for k in SEARCH_SPACE)
    print(header)
    print("-" * len(header))

    def callback(study: optuna.Study, trial: optuna.Trial) -> None:
        w = trial.params
        score = trial.value
        best = study.best_value
        marker = " *" if score == best else ""
        vals = "  ".join(f"{w[k]:>8.2f}" for k in SEARCH_SPACE)
        print(f"{trial.number+1:>6}  {score:>9.1f}  {best:>9.1f}  {vals}{marker}")
        sys.stdout.flush()

    study.optimize(lambda t: objective(t, eval_seeds), n_trials=args.trials, callbacks=[callback])

    best = study.best_params
    print("\n=== Best weights ===")
    for k, v in best.items():
        print(f"    {k}: float = {v:.4f}")
    print(f"Total score: {study.best_value:.1f}")


if __name__ == "__main__":
    main()

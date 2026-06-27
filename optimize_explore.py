"""
Optuna-based optimizer for Explorer exploration phase weights only.
Surveillance weights are frozen at their current values in my_solution.py.

Usage:
    uv run optimize_explore.py
    uv run optimize_explore.py --trials 100 --seed 42
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
EVAL_SEEDS = [0]

SEARCH_SPACE = {
    "W_EDGE_RATIO": (0.0, 150.0),
    "W_ISOLATION":  (0.0,  30.0),
    "W_MAX_EDGE":   (0.0, 200.0),
    "W_DIST":       (0.0,  20.0),
}


def evaluate(weights: dict) -> float:
    for k, v in weights.items():
        setattr(my_solution.Explorer, k, v)

    paths = sorted(GRAPHS_DIR.glob("*.json"))
    worlds = [load_graph(str(p)) for p in paths]
    names = [p.stem for p in paths]

    total = 0.0
    for world, name in zip(worlds, names):
        print(f"       evaluating {name}...", end="\r", flush=True)
        result = run_suite(
            [world],
            my_solution.Explorer,
            seeds=EVAL_SEEDS,
            n_agents=N_AGENTS,
            live=False,
            max_steps=1000,
        )
        total += result["total_score"]
    return total


def objective(trial: optuna.Trial) -> float:
    weights = {
        k: trial.suggest_float(k, lo, hi)
        for k, (lo, hi) in SEARCH_SPACE.items()
    }
    return evaluate(weights)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )

    current = {k: getattr(my_solution.Explorer, k) for k in SEARCH_SPACE}
    study.enqueue_trial(current)

    n_graphs = len(list(GRAPHS_DIR.glob("*.json")))
    print(f"Running {args.trials} Optuna trials over {n_graphs} graphs (explore phase only)...")
    print(f"{'Trial':>7}  {'Score':>10}  {'Best':>10}  {'EDGE_RATIO':>12}  {'ISOLATION':>10}  {'MAX_EDGE':>10}  {'DIST':>6}")
    print("-" * 80)

    def callback(study: optuna.Study, trial: optuna.Trial) -> None:
        w = trial.params
        score = trial.value
        best = study.best_value
        marker = "  <-- best" if score == best else ""
        print(
            f"{trial.number + 1:>7}  {score:>10.1f}  {best:>10.1f}"
            f"  {w['W_EDGE_RATIO']:>12.2f}"
            f"  {w['W_ISOLATION']:>10.2f}"
            f"  {w['W_MAX_EDGE']:>10.2f}"
            f"  {w['W_DIST']:>6.2f}"
            f"{marker}"
        )
        sys.stdout.flush()

    study.optimize(objective, n_trials=args.trials, callbacks=[callback])

    best = study.best_params
    print("\n=== Best exploration weights found ===")
    for k, v in best.items():
        print(f"  Explorer.{k} = {v:.4f}")
    print(f"  Total score: {study.best_value:.1f}")
    print("\nPaste these into my_solution.py to use them.")


if __name__ == "__main__":
    main()

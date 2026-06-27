"""
Optuna-based optimizer for Explorer surveillance phase weights only.
Exploration weights are frozen at their current values in my_solution.py.

Usage:
    uv run optimize_surveil.py
    uv run optimize_surveil.py --trials 100 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import optuna
from exploration_challenge.evaluator import run_suite
from exploration_challenge.graph_io import load_graph
import my_solution

GRAPHS_DIR = Path("graphs_test")
N_AGENTS = 3
EVAL_SEEDS = [0]

SEARCH_SPACE = {
    "W_SURV_COV":  (0.1, 10.0),
    "W_SURV_DIST": (0.0,  5.0),
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
            max_steps=100,
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
    print(f"Running {args.trials} Optuna trials over {n_graphs} graphs (surveil phase only)...")
    print(f"{'Trial':>7}  {'Score':>10}  {'Best':>10}  {'S_COV':>7}  {'S_DIST':>7}")
    print("-" * 55)

    def callback(study: optuna.Study, trial: optuna.Trial) -> None:
        w = trial.params
        score = trial.value
        best = study.best_value
        marker = "  <-- best" if score == best else ""
        print(
            f"{trial.number + 1:>7}  {score:>10.1f}  {best:>10.1f}"
            f"  {w['W_SURV_COV']:>7.2f}"
            f"  {w['W_SURV_DIST']:>7.2f}"
            f"{marker}"
        )
        sys.stdout.flush()

    study.optimize(objective, n_trials=args.trials, callbacks=[callback])

    best = study.best_params
    print("\n=== Best surveillance weights found ===")
    for k, v in best.items():
        print(f"  Explorer.{k} = {v:.4f}")
    print(f"  Total score: {study.best_value:.1f}")
    print("\nPaste these into my_solution.py to use them.")


if __name__ == "__main__":
    main()

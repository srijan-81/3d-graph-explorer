"""
Random-search optimizer for Explorer._score_explore weights.

Usage:
    uv run optimize_weights.py
    uv run optimize_weights.py --iterations 100 --seed 42
"""

from __future__ import annotations

import argparse
import random
import sys
import json
from pathlib import Path

from exploration_challenge.evaluator import run_suite
from exploration_challenge.graph_io import load_graph
import my_solution

GRAPHS_DIR = Path("graphs_test")
N_AGENTS = 3
EVAL_SEED = 0  # fixed seed so runs are comparable

SEARCH_SPACE = {
    "W_EDGE_RATIO": (0.0, 150.0),
    "W_ISOLATION":  (0.0,  30.0),
    "W_MAX_EDGE":   (0.0, 200.0),
    "W_DIST":       (0.0,  20.0),
}


def evaluate(weights: dict, iteration: int) -> float:
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
            seeds=[EVAL_SEED],
            n_agents=N_AGENTS,
            live=False,
            max_steps=100,
        )
        total += result["total_score"]
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50,
                        help="Number of random weight combinations to try")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for sampling weights")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    best_score = float("inf")
    best_weights = None

    print(f"Running {args.iterations} iterations over {len(list(GRAPHS_DIR.glob('*.json')))} graphs...")
    print(f"{'Iter':>5}  {'Score':>10}  {'Best':>10}  {'EDGE_RATIO':>12}  {'ISOLATION':>10}  {'MAX_EDGE':>10}  {'DIST':>6}")
    print("-" * 80)

    for i in range(args.iterations):
        if i == 0:
            # First iteration: use current defaults as baseline
            weights = {k: getattr(my_solution.Explorer, k) for k in SEARCH_SPACE}
        else:
            weights = {k: rng.uniform(lo, hi) for k, (lo, hi) in SEARCH_SPACE.items()}

        score = evaluate(weights, i + 1)
        marker = ""
        if score < best_score:
            best_score = score
            best_weights = dict(weights)
            marker = "  <-- best"

        print(
            f"{i+1:>5}  {score:>10.1f}  {best_score:>10.1f}"
            f"  {weights['W_EDGE_RATIO']:>12.2f}"
            f"  {weights['W_ISOLATION']:>10.2f}"
            f"  {weights['W_MAX_EDGE']:>10.2f}"
            f"  {weights['W_DIST']:>6.2f}"
            f"{marker}"
        )
        sys.stdout.flush()

    print("\n=== Best weights found ===")
    for k, v in best_weights.items():
        print(f"  Explorer.{k} = {v:.2f}")
    print(f"  Total score: {best_score:.1f}")
    print("\nPaste these into my_solution.py to use them.")


if __name__ == "__main__":
    main()

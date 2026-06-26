# Results export format

Evaluation results can be written to JSON with ``run_eval.py --output`` (or
``-o``). The schema mirrors the in-memory structures built by
``Simulator.result()`` → ``run_graph()`` → ``run_suite()`` in
``exploration_challenge/evaluator.py``.

## Writing results

```bash
uv run run_eval.py --submission my_solution.py --graphs graphs/train \
  --quiet --output results/eval.json
```

The terminal summary table is still printed unless ``--quiet`` is set. JSON
export is independent of ``--quiet``.

## Top-level fields

| Field | Type | Meaning |
|-------|------|---------|
| `version` | int | Export format version (currently `1`). |
| `created_at` | string | ISO-8601 UTC timestamp when the file was written. |
| `submission` | string \| null | Resolved policy path or module (`--submission`, `[eval].submission`, or `submission.py`); `null` for the default baseline. |
| `eval` | object | Eval settings used for this run (see below). |
| `graphs` | string | Graph path(s) passed to ``--graphs`` (space-separated when multiple). |
| `per_graph` | array | One aggregate entry per graph (see below). |
| `total_score` | number \| null | Sum of per-graph ``mean_score`` values. Lower is better (fastest explore + surveil). `null` when any mean is infinite. |

### `eval` object

| Field | Type | Meaning |
|-------|------|---------|
| `k` | int | Observation depth (graph hops). |
| `max_turn_deg` | number | Line-of-sight turn limit per hop (degrees). |
| `drop_prob` | number | Per-scan probability each candidate node is missed (`0` = perfect sensor). |
| `n_agents` | int | Number of UAVs (official challenge: `3`). |
| `seeds` | array[int] | Seed ids used (`0 .. N-1`, `N` from ``--seeds`` or ``params.toml``). |
| `start` | int \| null | Fixed start node for agent 0, or `null` for random distinct starts per seed. |
| `max_steps` | int | Policy decision cap per episode. |

Phase thresholds (`explore_threshold`, `surveil_threshold`) are read from
``[eval]`` in ``exploration_challenge/params.toml`` at eval time.

## Per-graph aggregate (`per_graph[]`)

| Field | Type | Meaning |
|-------|------|---------|
| `name` | string | Graph name from JSON metadata. |
| `n_total` | int | Number of nodes in the graph. |
| `n_agents` | int | Number of UAVs in each episode. |
| `k` | int | Observation depth used. |
| `seeds` | array[int] | Seeds run for this graph. |
| `runs` | array | One episode result per seed (see below). |
| `mean_score` | number \| null | Arithmetic mean of episode scores (including incomplete runs). |
| `stdev_score` | number | Population stdev of **completed** (finite) runs only, `0` when fewer than two complete. |
| `completion_rate` | number | Fraction of runs that finished both phases (`0.0`–`1.0`). |

## Episode result (`runs[]`)

| Field | Type | Meaning |
|-------|------|---------|
| `name` | string | Graph name. |
| `n_total` | int | Number of nodes. |
| `n_agents` | int | Number of UAVs. |
| `k` | int | Observation depth. |
| `start` | int | Start node for agent 0. |
| `starts` | array[int] | Distinct start node per UAV. |
| `explore_completed` | bool | Exploration phase finished (`dist_explore` is set). |
| `surveil_completed` | bool | Surveillance phase finished (episode done). |
| `dist_explore` | number \| null | Makespan distance at explore→surveil transition. |
| `dist_surveil` | number \| null | Surveillance leg makespan (`makespan_distance - dist_explore`). |
| `makespan_distance` | number | Max per-agent distance flown (score metric). |
| `total_distance` | number | Sum of per-agent distances (telemetry). |
| `score` | number \| null | `makespan_distance` when both phases complete, `null` when incomplete (`inf` in Python). |
| `steps` | int | Policy ticks taken (one `step()` call per tick; shared clock in multi-agent). |
| `observed_fraction` | number | Final exploration coverage (`0.0`–`1.0`). |
| `surveil_fraction` | number | Final surveillance coverage (`0.0`–`1.0`). |

When both phases complete, ``score == makespan_distance`` (max per-agent distance;
with 3 UAVs this is the slowest drone's total flight distance). Incomplete episodes
(stuck, invalid move, stall, or step cap) serialize ``score`` and ``mean_score``
as `null`.

## Example (abbreviated)

```json
{
  "version": 1,
  "created_at": "2026-06-17T12:00:00+00:00",
  "submission": "my_solution.py",
  "eval": {
    "k": 4,
    "max_turn_deg": 75.0,
    "drop_prob": 0.0,
    "n_agents": 3,
    "seeds": [0],
    "start": null,
    "max_steps": 1000
  },
  "graphs": "graphs/train",
  "per_graph": [
    {
      "name": "basic",
      "n_total": 751,
      "k": 4,
      "seeds": [0],
      "runs": [
        {
          "name": "basic",
          "n_total": 751,
          "n_agents": 3,
          "k": 4,
          "start": 42,
          "starts": [42],
          "explore_completed": true,
          "surveil_completed": true,
          "dist_explore": 310.5,
          "dist_surveil": 280.2,
          "makespan_distance": 590.7,
          "total_distance": 590.7,
          "score": 590.7,
          "steps": 87,
          "observed_fraction": 0.92,
          "surveil_fraction": 0.91
        }
      ],
      "mean_score": 590.7,
      "stdev_score": 0.0,
      "completion_rate": 1.0
    }
  ],
  "total_score": 590.7
}
```

The terminal summary table shows ``mean score`` and ``complete`` (completion
rate) per graph. The JSON export adds ``stdev_score`` per graph.

"""Load challenge configuration from TOML files.

``exploration_challenge/params.toml`` holds ``[eval]`` defaults.
CLI flags on ``run_eval.py`` override ``[eval]`` at runtime.
"""

from __future__ import annotations

import importlib.resources
import tomllib
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def _load_eval() -> dict[str, Any]:
    with importlib.resources.files("exploration_challenge").joinpath("params.toml").open("rb") as f:
        return tomllib.load(f)["eval"]


def eval_params() -> dict[str, Any]:
    return dict(_load_eval())

"""Reproducible RNG stream derivation for evaluation.

Seeding contract
----------------
Given the same master ``seed``, every submission sees the same environment
(start nodes and sensor-drop stream). Policy randomness uses a separate
derived stream so it never consumes environment draws.

Streams (all from ``derive_seed(master, tag)`` via stable ``zlib.crc32``):

* ``"start"`` — distinct start-node selection (shared across algorithms).
* ``"sensor"`` — simulator sensor-drop draws (``drop_prob > 0``).
* ``"policy"`` — passed to ``Explorer.reset``.

Note: with ``drop_prob > 0``, drop outcomes at a node still depend on visit
order, so two policies may diverge at specific scans. Average over ``--seeds``
for fair comparison.
"""

from __future__ import annotations

import zlib


def derive_seed(seed: int | None, tag: str) -> int:
    """Derive a stable 32-bit sub-seed from a master seed and stream tag."""
    master = 0 if seed is None else int(seed)
    return zlib.crc32(f"{master}:{tag}".encode()) & 0xFFFFFFFF


def start_seed(seed: int | None) -> int:
    return derive_seed(seed, "start")


def sensor_seed(seed: int | None) -> int:
    return derive_seed(seed, "sensor")


def policy_seed(seed: int | None) -> int:
    return derive_seed(seed, "policy")

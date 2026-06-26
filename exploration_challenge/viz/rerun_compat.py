"""Rerun import guard and version-compat logging helpers."""

from __future__ import annotations

try:
    import rerun as rr

    HAVE_RERUN = True
except Exception:  # pragma: no cover - import guard
    rr = None  # type: ignore[assignment]
    HAVE_RERUN = False


def log_scalar(path: str, value: float) -> None:
    """Log a time-series value across Rerun versions (Scalars vs Scalar)."""
    for archetype in ("Scalars", "Scalar"):
        ctor = getattr(rr, archetype, None)
        if ctor is not None:
            rr.log(path, ctor(value))
            return


def set_step(t: int) -> None:
    """Advance the timeline across Rerun versions (set_time vs set_time_sequence)."""
    if hasattr(rr, "set_time_sequence"):  # rerun < 0.23
        rr.set_time_sequence("step", t)
    else:  # rerun >= 0.23
        rr.set_time("step", sequence=t)


def clear_entity(path: str, *, recursive: bool = False) -> None:
    """Remove an entity from the viewer, optionally including descendants."""
    clear = getattr(rr, "Clear", None)
    if clear is None:
        rr.log(path, rr.Points3D([], colors=[], radii=[]))
        return
    if recursive:
        mode = getattr(clear, "recursive", None)
        rr.log(path, mode() if mode is not None else clear(recursive=True))
    else:
        mode = getattr(clear, "flat", None)
        rr.log(path, mode() if mode is not None else clear(recursive=False))

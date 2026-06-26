"""Optional Rerun-based 3D visualizer.

Streams the world graph, fog-of-war discovery state, the drone, and live coverage
metrics to the Rerun viewer with a scrubable timeline. If ``rerun`` is not
installed, ``Visualizer`` degrades to a no-op so scoring never depends on the GUI.

Update modes (CLI ``--viz``, ``--viz-reduced``):
  * **Default (``--viz``):** full scene refresh each planning step; fog-of-war,
    known edges, drone pose, and metrics update as the UAV explores.
  * **Reduced (``--viz-reduced``):** static grey graph drawn once; only drone
    motion, coverage plot, and status text update thereafter.

Color legend:
  grey        unobserved nodes (fog of war)
  blue ramp   observed nodes during exploration, shaded by hop-distance from
              the nearest visited node: dark blue = directly visited, fading
              to light blue the farther away a node is. The ramp denominator
              grows monotonically (never shrinks) so only nodes whose own
              distance changes are recoloured.
  light green nodes re-observed during surveillance (same green as the
              metrics surveilled line); sized like visited exploration nodes.
  red trail   flight path behind the drone; resets at the surveillance phase.
  quadcopter mesh marks the current UAV position
"""

from .core import Visualizer

__all__ = ["Visualizer"]

"""Procedural mesh primitives for the drone visualizer."""

from __future__ import annotations

import math


def append_box(
    verts: list[list[float]],
    tris: list[list[int]],
    colors: list[tuple[int, int, int]],
    center: tuple[float, float, float],
    half: tuple[float, float, float],
    color: tuple[int, int, int],
) -> None:
    cx, cy, cz = center
    hx, hy, hz = half
    base = len(verts)
    corners = [
        [cx - hx, cy - hy, cz - hz], [cx + hx, cy - hy, cz - hz],
        [cx + hx, cy + hy, cz - hz], [cx - hx, cy + hy, cz - hz],
        [cx - hx, cy - hy, cz + hz], [cx + hx, cy - hy, cz + hz],
        [cx + hx, cy + hy, cz + hz], [cx - hx, cy + hy, cz + hz],
    ]
    verts.extend(corners)
    colors.extend([color] * 8)
    for face in (
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ):
        tris.append([base + i for i in face])


def append_disc(
    verts: list[list[float]],
    tris: list[list[int]],
    colors: list[tuple[int, int, int]],
    center: tuple[float, float, float],
    radius: float,
    height: float,
    color: tuple[int, int, int],
    segments: int = 12,
) -> None:
    cx, cy, cz = center
    base = len(verts)
    verts.append([cx, cy, cz + height / 2])
    colors.append(color)
    top_center = base
    verts.append([cx, cy, cz - height / 2])
    colors.append(color)
    bot_center = base + 1
    for i in range(segments):
        ang = 2 * math.pi * i / segments
        x = cx + radius * math.cos(ang)
        y = cy + radius * math.sin(ang)
        verts.append([x, y, cz + height / 2])
        colors.append(color)
        verts.append([x, y, cz - height / 2])
        colors.append(color)
    for i in range(segments):
        top_a = base + 2 + 2 * i
        top_b = base + 2 + 2 * ((i + 1) % segments)
        bot_a = top_a + 1
        bot_b = top_b + 1
        tris.append([top_center, top_a, top_b])
        tris.append([bot_center, bot_b, bot_a])
        tris.append([top_a, bot_a, bot_b])
        tris.append([top_a, bot_b, top_b])


def build_quadcopter_mesh() -> tuple[list[list[float]], list[list[int]], list[tuple[int, int, int]]]:
    """Return a small +‑layout quadcopter mesh centred at the origin."""
    verts: list[list[float]] = []
    tris: list[list[int]] = []
    colors: list[tuple[int, int, int]] = []
    body_color = (75, 78, 85)
    arm_color = (105, 108, 115)
    rotor_color = (220, 45, 45)

    append_box(verts, tris, colors, (0.0, 0.0, 0.02), (0.06, 0.06, 0.03), body_color)

    arm_len = 0.28
    arm_thick = 0.018
    for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        if sx:
            half = (arm_len / 2, arm_thick, arm_thick)
            center = (sx * arm_len / 2, 0.0, 0.02)
            tip = (sx * arm_len, 0.0, 0.05)
        else:
            half = (arm_thick, arm_len / 2, arm_thick)
            center = (0.0, sy * arm_len / 2, 0.02)
            tip = (0.0, sy * arm_len, 0.05)
        append_box(verts, tris, colors, center, half, arm_color)
        append_disc(verts, tris, colors, tip, 0.08, 0.012, rotor_color)

    return verts, tris, colors

"""Pure geometry helpers used by violation rules."""

from __future__ import annotations

import math


def segments_intersect(
    p1: tuple[float, float],
    p2: tuple[float, float],
    q1: tuple[float, float],
    q2: tuple[float, float],
) -> bool:
    def orient(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    o1 = orient(p1, p2, q1)
    o2 = orient(p1, p2, q2)
    o3 = orient(q1, q2, p1)
    o4 = orient(q1, q2, p2)
    return o1 * o2 < 0 and o3 * o4 < 0


def angle_diff_deg(a: float, b: float) -> float:
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)


def movement_angle_deg(p1: tuple[float, float], p2: tuple[float, float]) -> float | None:
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if math.hypot(dx, dy) < 1e-6:
        return None
    # 0 degrees points to the right. Image y grows downward, so invert dy.
    return math.degrees(math.atan2(-dy, dx)) % 360


def point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi)
        if intersects:
            inside = not inside
        j = i
    return inside

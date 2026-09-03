"""Geometry of PG(5,2) minus the standard PG(3,2) solid."""
from __future__ import annotations

from itertools import combinations


def points_f2(m: int) -> list[int]:
    """Return all vectors of F_2^m as integers."""
    if m < 0:
        raise ValueError("m must be nonnegative")
    return list(range(1 << m))


def nonzero_points_f2(m: int) -> list[int]:
    """Return all nonzero vectors of F_2^m as integers."""
    return list(range(1, 1 << m))


def span_from_basis(basis: list[int]) -> set[int]:
    """Return the F_2-span of integer-encoded basis vectors."""
    span = {0}
    for b in basis:
        span |= {x ^ int(b) for x in list(span)}
    return span


def standard_subspace_u() -> set[int]:
    """Return U=span(e0,e1,e2,e3) inside F_2^6."""
    return span_from_basis([1 << i for i in range(4)])


def compose_u_w(u: int, w: int) -> int:
    """Compose u in F_2^4 and w in F_2^2 into F_2^6."""
    if not (0 <= u < 16):
        raise ValueError("u must be in 0..15")
    if not (0 <= w < 4):
        raise ValueError("w must be in 0..3")
    return int(u) | (int(w) << 4)


def decompose_x_as_u_w(x: int) -> tuple[int, int]:
    """Return the standard U/W coordinates of x in F_2^6."""
    if not (0 <= x < 64):
        raise ValueError("x must be in 0..63")
    return int(x) & 0xF, (int(x) >> 4) & 0x3


def fg48_fiber_order() -> list[int]:
    """Return X ordered as U x {01} | U x {10} | U x {11}."""
    return [compose_u_w(u, w) for w in (1, 2, 3) for u in range(16)]


def fg48_points() -> list[int]:
    """Return X = F_2^6 \\ U in the canonical fiber order."""
    return fg48_fiber_order()


def line_through(a: int, b: int) -> tuple[int, int, int]:
    """Return the binary projective line {a,b,a+b}."""
    if a == 0 or b == 0 or a == b:
        raise ValueError("a and b must be distinct nonzero vectors")
    return tuple(sorted((int(a), int(b), int(a) ^ int(b))))


def all_projective_lines_pg(m: int) -> list[tuple[int, int, int]]:
    """Return all projective lines in PG(m-1,2)."""
    lines = {line_through(a, b) for a, b in combinations(nonzero_points_f2(m), 2)}
    return sorted(lines)


def restrict_line_to_x(line: tuple[int, int, int], X: set[int]) -> tuple[int, ...]:
    """Return surviving points of a projective line after restriction to X."""
    return tuple(x for x in line if x in X)


def restricted_lines_fg48() -> dict[int, list[tuple[int, ...]]]:
    """Group PG(5,2) lines by the number of surviving FG48 points."""
    X = set(fg48_points())
    grouped: dict[int, list[tuple[int, ...]]] = {0: [], 1: [], 2: [], 3: []}
    for line in all_projective_lines_pg(6):
        restricted = restrict_line_to_x(line, X)
        grouped[len(restricted)].append(restricted)
    return {k: grouped[k] for k in sorted(grouped) if grouped[k]}

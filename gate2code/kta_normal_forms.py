"""KTA76 Table I normal-form support generators, d = 16 instantiation.

Verbatim source and instantiation analysis:
docs/notes/kta_table1_transcription.md.  Five classes have weight in
[32, 40) at effective level m(f) = r + 4 with m(f) >= 9: rows 11, 15, 19,
20, 21.  Each generator returns (s_eff, sorted support points), with the
support enumerated directly from the printed polynomial (the enumeration,
not the printed weight formula, is authoritative).

Variable convention: X_i is the i-th coordinate bit, X1 = most significant.
"""
from __future__ import annotations


def _bits(x: int, m: int) -> list:
    return [(x >> (m - 1 - i)) & 1 for i in range(m)]


def _support(f, m: int) -> list:
    return [x for x in range(1 << m) if f(_bits(x, m))]


def row11():
    """X1..X5 + (X1+1)X6X7X8X9; r=5, m=9, weight 32."""
    def f(b):
        return (b[0] & b[1] & b[2] & b[3] & b[4]) ^ \
               ((b[0] ^ 1) & b[5] & b[6] & b[7] & b[8])
    return 9, _support(f, 9)


def row15():
    """X1X2X3X4X5 + X6X7(X3X4X8 + X2(X5+X8)X9); r=5, m=9, weight 38."""
    def f(b):
        head = b[0] & b[1] & b[2] & b[3] & b[4]
        tail = b[5] & b[6] & ((b[2] & b[3] & b[7]) ^
                              (b[1] & (b[4] ^ b[7]) & b[8]))
        return head ^ tail
    return 9, _support(f, 9)


def row19():
    """X1..X6 + X7X8X9X10(X3X4 + X5(X6+1)); r=6, m=10, weight 38."""
    def f(b):
        head = b[0] & b[1] & b[2] & b[3] & b[4] & b[5]
        tail = b[6] & b[7] & b[8] & b[9] & \
            ((b[2] & b[3]) ^ (b[4] & (b[5] ^ 1)))
        return head ^ tail
    return 10, _support(f, 10)


def row20():
    """X1..X5 + X6X7X8(X9(X5+1) + X3X4); r=5, m=9, weight 36."""
    def f(b):
        head = b[0] & b[1] & b[2] & b[3] & b[4]
        tail = b[5] & b[6] & b[7] & ((b[8] & (b[4] ^ 1)) ^ (b[2] & b[3]))
        return head ^ tail
    return 9, _support(f, 9)


def row21():
    """X1..X5 + X6X7X8(X9X3 + X4X5); r=5, m=9, weight 38."""
    def f(b):
        head = b[0] & b[1] & b[2] & b[3] & b[4]
        tail = b[5] & b[6] & b[7] & ((b[8] & b[2]) ^ (b[3] & b[4]))
        return head ^ tail
    return 9, _support(f, 9)


ROWS = {"11": (row11, 32), "15": (row15, 38), "19": (row19, 38),
        "20": (row20, 36), "21": (row21, 38)}


def instance(row: str, branch: str):
    """(s_eff, support) for a Table I row and spectrum branch.

    branch 'plain': the support as enumerated (must avoid the origin).
    branch 'punctured': translate one support point to the origin and
    delete it (all translates are handled by the positioning audit).
    """
    gen, expected_w = ROWS[row]
    s_eff, pts = gen()
    if len(pts) != expected_w:
        raise ValueError(
            f"row {row}: enumerated weight {len(pts)} != printed "
            f"{expected_w} -- transcription discrepancy, do not proceed")
    if branch == "plain":
        if 0 in pts:
            raise ValueError(f"row {row}: support contains the origin")
        return s_eff, pts
    t = pts[0]
    shifted = sorted(p ^ t for p in pts)
    return s_eff, [p for p in shifted if p != 0]

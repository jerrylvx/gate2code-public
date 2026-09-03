"""G-matrix construction for synthillation codes.

Implements the gate-synthesis → distillation code constructions from
Campbell & Howard, including:
- Case 1–4 constructions (Sec. III C of the paper)
- Controlled-unitary construction (Sec. III D)
- Sub-additivity construction (Sec. III E)
- Convenience wrappers (Case10, Case11)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gate2code.gf2 import GF2Array, to_gf2
from gate2code.lempel import bmat


@dataclass
class GInfo:
    """Container for a G-matrix and its logical-qubit partition index.

    Attributes
    ----------
    G : GF2Array
        The n × m binary matrix (rows are generators).
    k : int
        Number of logical qubits; the first *k* rows span G₁,
        the remaining rows span G₀.
    """

    G: GF2Array
    k: int

    @property
    def n_rows(self) -> int:
        return self.G.shape[0]

    @property
    def n_cols(self) -> int:
        return self.G.shape[1]


# ---------------------------------------------------------------------------
# Helper blocks
# ---------------------------------------------------------------------------


def _ones_row(n: int) -> np.ndarray:
    return np.ones((1, n), dtype=int)


def _zeros_block(rows: int, cols: int) -> np.ndarray:
    return np.zeros((rows, cols), dtype=int)


def _c_mat(A: np.ndarray) -> np.ndarray:
    """Column parity vector replicated 4 times → shape (n_A, 4)."""
    # c = Mod[Total[Transpose[A]], 2]  → column-wise sum mod 2 of A → shape (nrows,)
    # Then Transpose[c].{{1,1,1,1}} → outer product c ⊗ [1,1,1,1] → shape (nrows, 4)
    c = A.sum(axis=1) % 2  # shape (n_rows_A,)
    return np.outer(c, [1, 1, 1, 1]).astype(int)


# ---------------------------------------------------------------------------
# Case constructions  (Section III C)
# ---------------------------------------------------------------------------


def left_block_12(A: GF2Array) -> np.ndarray:
    """Left block used in Case 1 and Case 2."""
    A_int = np.asarray(A, dtype=int)
    B_int = np.asarray(bmat(A), dtype=int)
    n = A_int.shape[0]
    b_cols = B_int.shape[1]

    # ArrayFlatten[{{A, B, B}, {1, 1, 0}, {0, 1, 1}}]
    # Bottom rows: "1" means all-ones row, "0" means all-zeros row
    a_cols = A_int.shape[1]
    top = np.hstack([A_int, B_int, B_int])
    mid = np.hstack([_ones_row(a_cols), _ones_row(b_cols), _zeros_block(1, b_cols)])
    bot = np.hstack([_zeros_block(1, a_cols), _ones_row(b_cols), _ones_row(b_cols)])
    return np.vstack([top, mid, bot])


def left_block_34(A: GF2Array) -> np.ndarray:
    """Left block used in Case 3 and Case 4."""
    A_int = np.asarray(A, dtype=int)
    B_int = np.asarray(bmat(A), dtype=int)
    n = A_int.shape[0]
    b_cols = B_int.shape[1]
    a_cols = A_int.shape[1]

    # ArrayFlatten[{{A, B, B}, {1, 1, 0}, {1, 0, 1}}]
    top = np.hstack([A_int, B_int, B_int])
    mid = np.hstack([_ones_row(a_cols), _ones_row(b_cols), _zeros_block(1, b_cols)])
    bot = np.hstack([_ones_row(a_cols), _zeros_block(1, b_cols), _ones_row(b_cols)])
    return np.vstack([top, mid, bot])


# Fixed lower-right sub-blocks from the paper
_RIGHT_LOWER_1 = np.array(
    [
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [1, 1, 1, 1],
    ],
    dtype=int,
)


def case1(A: GF2Array) -> GInfo:
    """Case 1 construction (Sec. III C)."""
    A_int = np.asarray(A, dtype=int)
    k = A_int.shape[0]
    left = left_block_12(A)
    right_upper = _c_mat(A_int)
    right_lower = np.hstack(
        [_RIGHT_LOWER_1, _RIGHT_LOWER_1]
    )  # RightLower1, RightLower2
    right = np.vstack(
        [
            (
                np.hstack(
                    [
                        right_upper,
                        _zeros_block(k, right_lower.shape[1] - right_upper.shape[1]),
                    ]
                )
                if right_upper.shape[1] < right_lower.shape[1]
                else (
                    np.vstack(
                        [
                            np.hstack(
                                [
                                    right_upper,
                                    _zeros_block(
                                        right_upper.shape[0], right_lower.shape[1]
                                    ),
                                ]
                            ),
                        ]
                    )
                    if False
                    else _build_right(right_upper, _RIGHT_LOWER_1, _RIGHT_LOWER_1)
                )
            ),
        ]
    )
    G = np.hstack([left, right])
    return GInfo(G=to_gf2(G % 2), k=k)


def _build_right(
    upper: np.ndarray, lower1: np.ndarray, lower2: np.ndarray
) -> np.ndarray:
    """Build the right block: {{upper, 0}, {lower1, lower2}}."""
    n_upper_rows = upper.shape[0]
    n_lower_rows = lower1.shape[0]
    upper_cols = upper.shape[1]
    lower_cols = lower1.shape[1] + lower2.shape[1]

    right_top = np.hstack([upper, _zeros_block(n_upper_rows, lower2.shape[1])])
    right_bot = np.hstack([lower1, lower2])
    return np.vstack([right_top, right_bot])


def case2(A: GF2Array) -> GInfo:
    """Case 2 construction."""
    A_int = np.asarray(A, dtype=int)
    k = A_int.shape[0]
    left = left_block_12(A)
    right_upper = _c_mat(A_int)
    right_lower_2 = np.array(
        [
            [1, 0, 0, 1, 1],
            [0, 1, 0, 1, 1],
            [1, 1, 1, 1, 0],
        ],
        dtype=int,
    )
    right = _build_right(right_upper, _RIGHT_LOWER_1, right_lower_2)
    G = np.hstack([left, right]) % 2
    return GInfo(G=to_gf2(G), k=k)


def case3(A: GF2Array) -> GInfo:
    """Case 3 construction."""
    A_int = np.asarray(A, dtype=int)
    k = A_int.shape[0]
    left = left_block_34(A)
    right_upper = _c_mat(A_int)
    right_lower_2 = np.array(
        [
            [1, 0, 0, 1, 1],
            [0, 1, 0, 1, 1],
            [1, 1, 1, 1, 0],
        ],
        dtype=int,
    )
    right = _build_right(right_upper, _RIGHT_LOWER_1, right_lower_2)
    G = np.hstack([left, right]) % 2
    return GInfo(G=to_gf2(G), k=k)


def case4(A: GF2Array) -> GInfo:
    """Case 4 construction."""
    A_int = np.asarray(A, dtype=int)
    k = A_int.shape[0]
    left = left_block_34(A)
    right_upper = _c_mat(A_int)
    right_lower_2 = np.array(
        [
            [1, 0, 0, 1, 1, 1, 0],
            [0, 1, 0, 1, 1, 0, 1],
            [1, 1, 1, 1, 0, 1, 1],
        ],
        dtype=int,
    )
    right = _build_right(right_upper, _RIGHT_LOWER_1, right_lower_2)
    G = np.hstack([left, right]) % 2
    return GInfo(G=to_gf2(G), k=k)


# ---------------------------------------------------------------------------
# Convenience wrappers (Case10, Case11 from Synthillation2.nb)
# ---------------------------------------------------------------------------


def case10(A: GF2Array) -> GInfo:
    """Append all-ones syndrome row with two extra ancilla columns.

    ``G = [[A, 0, 0], [1, 1, 1]]``  with k = nrows(A).
    """
    A_int = np.asarray(A, dtype=int)
    k = A_int.shape[0]
    a_cols = A_int.shape[1]
    top = np.hstack([A_int, _zeros_block(k, 1), _zeros_block(k, 1)])
    bot = np.hstack([_ones_row(a_cols), np.array([[1]]), np.array([[1]])])
    G = np.vstack([top, bot]) % 2
    return GInfo(G=to_gf2(G), k=k)


def case11(A: GF2Array) -> GInfo:
    """Append all-ones syndrome row with one extra ancilla column.

    ``G = [[A, 0], [1, 1]]``  with k = nrows(A).
    """
    A_int = np.asarray(A, dtype=int)
    k = A_int.shape[0]
    a_cols = A_int.shape[1]
    top = np.hstack([A_int, _zeros_block(k, 1)])
    bot = np.hstack([_ones_row(a_cols), np.array([[1]])])
    G = np.vstack([top, bot]) % 2
    return GInfo(G=to_gf2(G), k=k)


# ---------------------------------------------------------------------------
# Controlled-unitary  (Section III D)
# ---------------------------------------------------------------------------


def cont_u(A: GF2Array) -> GF2Array:
    """Gate-synthesis matrix for controlled-V² given a matrix A for V.

    If V has gate-synthesis matrix A, then controlled-V² has gate-synthesis
    matrix ``contU(A)``.  Uses the Lempel factor B = Bmat(A).
    """
    A_int = np.asarray(A, dtype=int)
    B_int = np.asarray(bmat(A), dtype=int)
    n = B_int.shape[0]  # rows
    b_cols = B_int.shape[1]

    ell = b_cols % 2  # parity of number of columns of B

    if ell == 1:
        # ArrayFlatten[{{B, B, 0}, {1, 0, 1}}]
        top = np.hstack([B_int, B_int, _zeros_block(n, 1)])
        bot = np.hstack([_ones_row(b_cols), _zeros_block(1, b_cols), np.array([[1]])])
        result = np.vstack([top, bot])
    else:
        # ArrayFlatten[{{B, B}, {1, 0}}]
        top = np.hstack([B_int, B_int])
        bot = np.hstack([_ones_row(b_cols), _zeros_block(1, b_cols)])
        result = np.vstack([top, bot])

    return to_gf2(result % 2)


# ---------------------------------------------------------------------------
# Sub-additivity  (Section III E)
# ---------------------------------------------------------------------------


def sub_add(A1: GF2Array, A2: GF2Array) -> GF2Array:
    """Gate-synthesis matrix for U₁ ⊗ U₂ via sub-additivity.

    Given gate-synthesis matrices A1 (for U₁) and A2 (for U₂),
    constructs a joint gate-synthesis matrix.
    """
    A1_int = np.asarray(A1, dtype=int)
    A2_int = np.asarray(A2, dtype=int)
    n1 = A1_int.shape[0]
    col1 = A1_int.shape[1]
    col2 = A2_int.shape[1]
    n2 = A2_int.shape[0]

    # z = first column of A2, as a column vector
    z = A2_int[:, 0:1]  # shape (n2, 1)

    # Astar = remaining columns of A2 (if any)
    if col2 > 1:
        Astar = A2_int[:, 1:]
    else:
        Astar = np.zeros((n2, 0), dtype=int)

    # R = z ⊗ ones(col1)  → shape (n2, col1)
    vec1 = np.ones((1, col1), dtype=int)
    R = z @ vec1  # outer product: (n2,1) @ (1,col1) = (n2,col1)

    # ArrayFlatten[{{A1, 0}, {R, Astar}}]
    top = np.hstack([A1_int, _zeros_block(n1, Astar.shape[1])])
    bot = np.hstack([R, Astar])
    result = np.vstack([top, bot]) % 2

    return to_gf2(result)

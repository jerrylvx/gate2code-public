"""Shared rank-seven cubic-class data over the binary field."""
import numpy as np

M = 7
N = 1 << M

CLASSES = {
    "f1": [(1, 2, 3)],
    "f2": [(1, 2, 3), (1, 4, 5)],
    "f3": [(1, 2, 3), (4, 5, 6)],
    "f4": [(1, 2, 6), (2, 3, 4), (1, 3, 5)],
    "f5": [(1, 2, 3), (4, 5, 6), (1, 4, 7)],
    "f6": [(1, 2, 5), (1, 4, 7), (1, 3, 6), (2, 3, 4)],
    "f7": [(1, 4, 6), (1, 5, 7), (2, 4, 5), (3, 6, 7)],
    "f8": [(1, 2, 3), (1, 4, 5), (1, 6, 7)],
    "f9": [(1, 2, 3), (4, 5, 6), (1, 4, 7), (2, 5, 7), (3, 6, 7)],
    "f10": [(1, 2, 6), (1, 3, 5), (2, 3, 4), (1, 5, 6), (3, 4, 5), (2, 4, 6)],
    "f11": [(1, 2, 6), (1, 3, 5), (2, 3, 4), (1, 5, 6), (3, 4, 5), (2, 4, 6), (1, 4, 7)],
}

x = np.arange(N, dtype=np.uint8)
BITS = np.stack([(x >> (M - i)) & 1 for i in range(1, M + 1)], axis=1)


def truth(monos):
    table = np.zeros(N, dtype=np.uint8)
    for mono in monos:
        product = np.ones(N, dtype=np.uint8)
        for variable in mono:
            product &= BITS[:, variable - 1]
        table ^= product
    return table


QUAD_MONOS = [(i, j) for i in range(1, 8) for j in range(i + 1, 8)]
QUAD_TT = np.stack([BITS[:, i - 1] & BITS[:, j - 1] for i, j in QUAD_MONOS], axis=0)


def wht_batch(table):
    transformed = 1 - 2 * table.astype(np.int16)
    width = 1
    while width < N:
        transformed = transformed.reshape(-1, N // (2 * width), 2, width)
        total = transformed[:, :, 0, :] + transformed[:, :, 1, :]
        difference = transformed[:, :, 0, :] - transformed[:, :, 1, :]
        transformed = np.stack([total, difference], axis=2)
        width *= 2
    return transformed.reshape(-1, N)

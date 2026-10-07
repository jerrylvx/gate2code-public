"""Numba kernels: orbit BFS of a group (given by generator matrices) on F_2^d (d <= ~26)."""
import numpy as np
from numba import njit


@njit(cache=True)
def _apply(cols, c):
    y = 0
    i = 0
    while c:
        if c & 1:
            y ^= cols[i]
        c >>= 1
        i += 1
    return y


@njit(cache=True)
def orbit_bfs(gens, d, members):
    """gens: (k, d) int64 column images. members: int64 array of the elements to cover
    (an invariant subset of F_2^d, given as a boolean mask of length 2^d).
    Returns orbit_id (-1 for non-members), parent, gen_used, reps (list order)."""
    N = 1 << d
    orbit = -np.ones(N, dtype=np.int64)
    parent = -np.ones(N, dtype=np.int64)
    gused = -np.ones(N, dtype=np.int8)
    reps = np.empty(N, dtype=np.int64)
    sizes = np.empty(N, dtype=np.int64)
    queue = np.empty(N, dtype=np.int64)
    nrep = 0
    k = gens.shape[0]
    for x in range(N):
        if not members[x] or orbit[x] >= 0:
            continue
        orbit[x] = nrep
        head = 0
        tail = 0
        queue[tail] = x
        tail += 1
        while head < tail:
            y = queue[head]
            head += 1
            for t in range(k):
                z = _apply(gens[t], y)
                if orbit[z] < 0:
                    orbit[z] = nrep
                    parent[z] = y
                    gused[z] = t
                    queue[tail] = z
                    tail += 1
        reps[nrep] = x
        sizes[nrep] = tail
        nrep += 1
    return orbit, parent, gused, reps[:nrep], sizes[:nrep]

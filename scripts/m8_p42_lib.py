"""m=8 P_{4,2} machinery: H-sets, stabilizer reduction, support assembly."""
import numpy as np

M6, N6 = 6, 64
x6 = np.arange(N6, dtype=np.int64)
B6 = np.stack([((x6 >> (M6 - i)) & 1) for i in range(1, M6 + 1)],
              axis=1).astype(np.uint8)

def truth6(monos):
    t = np.zeros(N6, dtype=np.uint8)
    for mono in monos:
        p = np.ones(N6, dtype=np.uint8)
        for v in mono:
            p &= B6[:, v - 1]
        t ^= p
    return t

def pack(tt):
    return int.from_bytes(np.packbits(tt).tobytes(), "big")

def unpack(v):
    return np.unpackbits(np.frombuffer(v.to_bytes(8, "big"), dtype=np.uint8))

def point_perm_linear(A):
    """Permutation of [0,64) induced by x -> A x (A: 6x6 GF(2), row 0 = x1)."""
    out = np.zeros(N6, dtype=np.int64)
    for x in range(N6):
        bits = [(x >> (M6 - i)) & 1 for i in range(1, M6 + 1)]
        y = (A @ np.array(bits)) % 2
        out[x] = sum(int(y[i]) << (M6 - 1 - i) for i in range(M6))
    return out

def point_perm_translate(b):
    return x6 ^ b

def gl_gens():
    C = np.zeros((6, 6), dtype=np.uint8)         # coordinate cycle
    for i in range(6):
        C[i, (i + 1) % 6] = 1
    T = np.eye(6, dtype=np.uint8)                # transvection x1 += x2
    T[0, 1] = 1
    return [C, T]

def agl_point_perms():
    perms = [point_perm_linear(A) for A in gl_gens()]
    perms.append(point_perm_translate(32))       # x += e1
    return perms

def orbit_bfs(seed_tt, perms, cap=None):
    """Orbit of a truth table under permutations acting on inputs."""
    seen = {pack(seed_tt)}
    frontier = [seed_tt]
    while frontier:
        nxt = []
        for tt in frontier:
            for p in perms:
                t2 = tt[p]
                k = pack(t2)
                if k not in seen:
                    seen.add(k)
                    nxt.append(t2)
        frontier = nxt
        if cap and len(seen) > cap:
            raise RuntimeError("orbit cap exceeded")
    return seen

def all_3flats():
    """Truth tables of all 11160 affine 3-flat indicators in F_2^6."""
    flat = truth6([(1, 2, 3)])  # {x1=x2=x3=1}: an affine 3-flat
    return orbit_bfs(flat, agl_point_perms(), cap=20000)

def weight12_class():
    rep = B6[:, 0] & (truth6([(2, 3)]) ^ truth6([(4, 5)]))
    assert rep.sum() == 12
    return orbit_bfs(rep, agl_point_perms(), cap=3_000_000)

def stab_perms_g8():
    """Generators of a subgroup of Stab(x1x2x3) acting on points."""
    perms = []
    # permutations of {x1,x2,x3} and of {x4,x5,x6}
    for (i, j) in [(0, 1), (1, 2), (3, 4), (4, 5)]:
        A = np.eye(6, dtype=np.uint8)
        A[[i, j]] = A[[j, i]]
        perms.append(point_perm_linear(A))
    # transvections x4 += x1 and x1 += x4 is NOT stabilizing; x4 += x5 is
    A = np.eye(6, dtype=np.uint8); A[3, 4] = 1
    perms.append(point_perm_linear(A))
    # x4 += x1 keeps x1x2x3 (only support coords x1..x3 matter): check via assert in caller
    A = np.eye(6, dtype=np.uint8); A[3, 0] = 1
    perms.append(point_perm_linear(A))
    # translations in x4, x5, x6
    for b in (4, 2, 1):
        perms.append(point_perm_translate(b))
    return perms

def stab_perms_g12():
    """Generators of a subgroup of Stab(x1(x2x3+x4x5))."""
    perms = []
    for (i, j) in [(1, 2), (3, 4)]:              # x2<->x3, x4<->x5
        A = np.eye(6, dtype=np.uint8)
        A[[i, j]] = A[[j, i]]
        perms.append(point_perm_linear(A))
    A = np.eye(6, dtype=np.uint8)                # (x2x3)<->(x4x5)
    A[[1, 3]] = A[[3, 1]]; A[[2, 4]] = A[[4, 2]]
    perms.append(point_perm_linear(A))
    # symplectic-type: x2 += x4, x5 += x3 preserves x2x3 + x4x5
    A = np.eye(6, dtype=np.uint8); A[1, 3] = 1; A[4, 2] = 1
    perms.append(point_perm_linear(A))
    # x4 += x2, x3 += x5
    A = np.eye(6, dtype=np.uint8); A[3, 1] = 1; A[2, 4] = 1
    perms.append(point_perm_linear(A))
    # x2 += x1 (g12 = x1 x2 x3 + ... : x1(x2+x1)x3 = x1x2x3 + x1x3 ... NOT stabilizing; skip)
    # x6 shears
    for (i, j) in [(5, 0), (5, 1), (5, 2), (5, 3), (5, 4)]:
        A = np.eye(6, dtype=np.uint8); A[i, j] = 1
        perms.append(point_perm_linear(A))
    perms.append(point_perm_translate(1))        # x += e6
    return perms

def orbit_reps(keys, perms):
    """Partition a set of packed truth tables into orbits under perms;
    return one (minimal) representative per orbit."""
    keys = set(keys)
    reps = []
    while keys:
        k = min(keys)
        orb = orbit_bfs(unpack(k), perms)
        orb &= keys | orb  # closure may leave the set if gens not stabilizing the SET... they do stabilize weight
        keys -= orb
        reps.append(k)
    return reps

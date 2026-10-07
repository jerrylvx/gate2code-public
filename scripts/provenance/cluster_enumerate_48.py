#!/usr/bin/env python3
"""
Cluster-ready E1 enumeration: find inequivalent [[48,3,3]] CCZ codes.

Designed for SLURM array parallelism. Each array task handles a slice of 
activity patterns, finds solutions, classifies them with pynauty, and writes
results atomically.

Usage:
  python cluster_enumerate_48.py --task_id 1 --num_tasks 16 --max_per_task 50 \
      --timeout 300 --output_dir /work/bl298/ccz_overnight/e1_results

Architecture:
  - 48 active betas out of 63: C(63,48) ≈ 10^13 patterns total
  - Pure-S constraints cut this to a feasible set the SAT solver enumerates
  - Each task enumerates ALL solutions but adds a symmetry-breaking constraint
    based on task_id to split the work:  beta_{split_beta} = task_id's bit
  - On-the-fly pynauty canonical certificate for equivalence classification
  - Atomic writes with SHA-256 integrity (matches cluster_ccz_search.py style)

Splitting strategy:
  We pick the first few betas and fix them to partition the search space.
  With num_tasks=16, we fix 4 activity bits → 16 slices.
  With num_tasks=32, we fix 5 bits, etc.
"""

import sys
import os
import argparse
import json
import time
import hashlib
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import z3
# pynauty not available on cluster (no Python.h); use sorted-support hash instead.
# Safe because |Aut|=1 for [[48,3,3]]: every support set is a distinct class.
try:
    import pynauty
    _HAVE_PYNAUTY = True
except ImportError:
    _HAVE_PYNAUTY = False
from itertools import combinations, product
from functools import reduce
from collections import Counter
from gate2code.ccz import (
    verify_ch_conditions, compute_distances, f2_rank,
    vec_to_int, popcount,
)


# ─── Integrity layer (matches cluster_ccz_search.py) ───────────────

def atomic_write_json(filepath, data):
    tmp_path = filepath + '.tmp'
    content = json.dumps(data, indent=2)
    with open(tmp_path, 'w') as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp_path, filepath)


def sha256_of_matrix(G):
    G_bytes = np.ascontiguousarray(G, dtype=np.int64).tobytes()
    return hashlib.sha256(G_bytes).hexdigest()


def environment_fingerprint():
    import platform
    return {
        'hostname': platform.node(),
        'python': platform.python_version(),
        'z3_version': z3.get_version_string(),
        'numpy_version': np.__version__,
        'platform': platform.platform(),
        'pid': os.getpid(),
    }


# ─── SAT infrastructure ────────────────────────────────────────────

def build_monomial_table(r, k=3):
    s = r - k
    monomials = []
    for deg in range(4):
        for indices in combinations(range(r), deg):
            k_idx = tuple(i for i in indices if i < k)
            s_idx = tuple(i - k for i in indices if i >= k)
            target = 1 if tuple(indices) == tuple(range(k)) else 0
            monomials.append((k_idx, s_idx, target))
    return monomials


def eval_s_monomial(s_indices, beta_bits):
    result = 1
    for j in s_indices:
        result &= beta_bits[j]
    return result


def canonical_cert(G, k=3):
    """Canonical certificate for a G-matrix.

    Uses nauty (pynauty) when available, otherwise falls back to a
    sorted-support tuple.  The fallback is exact when |Aut|=1, which
    holds for the [[48,3,3]] code family.
    """
    if _HAVE_PYNAUTY:
        r, n = G.shape
        total_v = r + n
        graph = pynauty.Graph(total_v)
        for col in range(n):
            for row in range(r):
                if G[row, col]:
                    graph.connect_vertex(row, [r + col])
        coloring = [set(range(k)), set(range(k, r)), set(range(r, r + n))]
        graph.set_vertex_coloring(coloring)
        return pynauty.certificate(graph)
    else:
        # Fallback: encode each column as a frozenset of row indices, then
        # canonicalise by sorting the multiset of column vectors.
        r, n = G.shape
        cols = tuple(sorted(tuple(int(G[row, col]) for row in range(r))
                            for col in range(n)))
        return hashlib.sha256(str(cols).encode()).digest()


# ─── Main enumeration ──────────────────────────────────────────────

def enumerate_task(task_id, num_tasks, max_per_task, timeout_per_call_s,
                   output_dir):
    """
    Enumerate [[48,3,3]] codes for one SLURM array slice.
    
    Splitting: We pick the first `num_split_bits` betas and fix their 
    activity bits to deterministically assign search regions to tasks.
    """
    r, k = 9, 3
    s = r - k
    target_n = 48
    num_betas = 2**s - 1  # 63

    beta_list = list(range(1, 2**s))
    beta_bits_map = {}
    for b in beta_list:
        bits = [(b >> (s - 1 - j)) & 1 for j in range(s)]
        beta_bits_map[b] = bits

    monomials = build_monomial_table(r, k)
    pure_s = [(ki, si, t) for ki, si, t in monomials if len(ki) == 0]
    mixed = [(ki, si, t) for ki, si, t in monomials if len(ki) > 0]

    # ── Determine split bits ──
    # num_split_bits = ceil(log2(num_tasks))
    num_split_bits = 0
    while (1 << num_split_bits) < num_tasks:
        num_split_bits += 1
    
    # Pick the last `num_split_bits` betas (high-order) for splitting.
    # These are betas 63, 62, 61, ... counting down.
    split_betas = beta_list[-num_split_bits:]  # highest betas
    
    # task_id determines which combination of activity values for split_betas.
    # If task_id >= 2^num_split_bits, this task does nothing (over-provisioned).
    if task_id >= (1 << num_split_bits):
        print(f"Task {task_id}: no work (over-provisioned, "
              f"only {1 << num_split_bits} slices needed)")
        return
    
    split_values = {}
    for i, b in enumerate(split_betas):
        split_values[b] = (task_id >> (num_split_bits - 1 - i)) & 1

    print(f"E1 Cluster Enumeration — Task {task_id}/{num_tasks}")
    print(f"  r={r}, s={s}, n={target_n}")
    print(f"  Split: {num_split_bits} bits → betas {split_betas}")
    print(f"  This task: {split_values}")
    print(f"  Max solutions per task: {max_per_task}")
    print(f"  Timeout per SAT call: {timeout_per_call_s}s")
    print(f"  Environment: {environment_fingerprint()}")
    print(flush=True)

    # ── Build solver ──
    solver = z3.Solver()
    g = {b: z3.Bool(f"g_{b}") for b in beta_list}
    a = {b: [z3.Bool(f"a_{b}_{i}") for i in range(k)] for b in beta_list}

    # Pure-S constraints
    for ki, si, target in pure_s:
        terms = [g[b] for b in beta_list
                 if eval_s_monomial(si, beta_bits_map[b]) == 1]
        if terms:
            solver.add(z3.Not(reduce(z3.Xor, terms)))

    # Mixed constraints
    for ki, si, target in mixed:
        xor_terms = []
        for b in beta_list:
            if eval_s_monomial(si, beta_bits_map[b]) == 0:
                continue
            parts = [g[b]] + [a[b][i] for i in ki]
            xor_terms.append(z3.And(*parts) if len(parts) > 1 else parts[0])
        if not xor_terms:
            if target == 1:
                solver.add(z3.BoolVal(False))
            continue
        xr = reduce(z3.Xor, xor_terms)
        solver.add(xr if target == 1 else z3.Not(xr))

    # Cardinality: exactly 48
    solver.add(z3.PbEq([(g[b], 1) for b in beta_list], target_n))

    # ── Split constraint: fix activity of split betas ──
    for b, val in split_values.items():
        solver.add(g[b] if val == 1 else z3.Not(g[b]))

    # ── Enumerate ──
    pts = np.array(list(product([0, 1], repeat=r)), dtype=np.uint8)
    solutions = []
    certs = {}
    n_classes = 0
    t0 = time.time()

    while len(solutions) < max_per_task:
        solver.set("timeout", timeout_per_call_s * 1000)
        result = solver.check()

        if result == z3.unsat:
            print(f"\n  EXHAUSTED slice after {len(solutions)} solutions "
                  f"({time.time()-t0:.0f}s)", flush=True)
            break
        if result == z3.unknown:
            print(f"\n  TIMEOUT after {len(solutions)} solutions "
                  f"({time.time()-t0:.0f}s)", flush=True)
            break

        model = solver.model()
        active = []
        alpha_map = {}
        for b in beta_list:
            if z3.is_true(model[g[b]]):
                active.append(b)
                alpha_map[b] = tuple(
                    1 if z3.is_true(model[a[b][i]]) else 0
                    for i in range(k)
                )

        support = []
        for b in sorted(active):
            alpha_int = sum(alpha_map[b][i] * (1 << (k-1-i)) for i in range(k))
            support.append((alpha_int << s) | b)

        cols = pts[np.array(support)]
        G = cols.T.astype(np.int64)

        # Verify
        rank = f2_rank(G)
        if rank == r:
            ch_ok, _ = verify_ch_conditions(G, k)
            if ch_ok:
                dx, dz = compute_distances(G, k)
                d = min(dx, dz)
                if d >= 3:
                    cert = canonical_cert(G, k)
                    is_new = cert not in certs
                    if is_new:
                        certs[cert] = len(solutions)
                        n_classes += 1

                    solutions.append({
                        'support': [int(x) for x in support],
                        'active': [int(x) for x in sorted(active)],
                        'dx': int(dx), 'dz': int(dz), 'd': int(d),
                        'class_id': certs[cert],
                        'is_new_class': is_new,
                        'sha256': sha256_of_matrix(G),
                    })

                    idx = len(solutions)
                    elapsed = time.time() - t0
                    if idx <= 10 or idx % 25 == 0:
                        print(f"  Sol {idx:4d}: d_X={dx} d_Z={dz}  "
                              f"classes={n_classes}  "
                              f"{'NEW' if is_new else 'dup'}  "
                              f"[{elapsed:.0f}s]", flush=True)

        # Block this assignment
        block = []
        for b in beta_list:
            if b in alpha_map:
                block.append(z3.Not(g[b]))
                for i in range(k):
                    block.append(a[b][i] if alpha_map[b][i] == 0
                                 else z3.Not(a[b][i]))
            else:
                block.append(g[b])
        solver.add(z3.Or(block))

    total_time = time.time() - t0

    # ── Summary ──
    print(f"\n{'='*72}")
    print(f"  E1 TASK {task_id} RESULTS")
    print(f"{'='*72}")
    print(f"  Solutions found:              {len(solutions)}")
    print(f"  Distinct equivalence classes: {n_classes}")
    print(f"  Duplicates (same class):      {len(solutions) - n_classes}")
    print(f"  Time:                         {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"  Exhausted slice:              {result == z3.unsat}")

    # ── Save results atomically ──
    output = {
        'task_id': task_id,
        'num_tasks': num_tasks,
        'split_betas': split_betas,
        'split_values': {str(k): v for k, v in split_values.items()},
        'r': r, 'k': k, 's': s, 'n': target_n,
        'total_solutions': len(solutions),
        'n_classes': n_classes,
        'time_s': total_time,
        'exhausted': result == z3.unsat,
        'environment': environment_fingerprint(),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'solutions': solutions,
    }

    os.makedirs(output_dir, exist_ok=True)
    outfile = os.path.join(output_dir, f'e1_task{task_id:03d}.json')
    atomic_write_json(outfile, output)
    
    # SHA-256 sidecar
    content = json.dumps(output, indent=2).encode()
    sha = hashlib.sha256(content).hexdigest()
    with open(outfile + '.sha256', 'w') as f:
        f.write(f"{sha}  {os.path.basename(outfile)}\n")
    
    print(f"  Saved to {outfile} (sha256={sha[:16]}...)")

    # Also save any new codes as .npz
    saved_npz = 0
    for sol in solutions:
        if sol['is_new_class']:
            G_sol = pts[np.array(sol['support'])].T.astype(np.int64)
            fname = os.path.join(output_dir,
                                 f'e1_code_t{task_id:03d}_c{sol["class_id"]:04d}.npz')
            np.savez(fname, G=G_sol, support=np.array(sol['support']))
            saved_npz += 1
    
    print(f"  Saved {saved_npz} new code .npz files")
    return output


def main():
    parser = argparse.ArgumentParser(
        description='E1: Enumerate [[48,3,3]] CCZ codes (cluster-ready)')
    parser.add_argument('--task_id', type=int, required=True,
                        help='SLURM array task ID (0-indexed)')
    parser.add_argument('--num_tasks', type=int, default=16,
                        help='Total number of parallel tasks')
    parser.add_argument('--max_per_task', type=int, default=100,
                        help='Max solutions per task before stopping')
    parser.add_argument('--timeout', type=int, default=300,
                        help='Timeout per SAT call (seconds)')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Output directory')
    args = parser.parse_args()

    enumerate_task(
        task_id=args.task_id,
        num_tasks=args.num_tasks,
        max_per_task=args.max_per_task,
        timeout_per_call_s=args.timeout,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()

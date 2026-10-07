"""Group the label-map classes by the code C1 they generate: two classes span equivalent codes
with CH structure iff they are related by H and a GL(3,2) change of logical basis (the triple-product
parity is the determinant form, so every basis of span(k) is again a CH triple). Records, per code,
how many of its label-permutation classes admit single-qubit transversal patterns.
Usage: python code_groups.py 4 2
"""
import sys, pickle
from itertools import product
from collections import Counter
from pathlib import Path
from classify import run
HERE = Path(__file__).resolve().parent


def gl3():
    out = []
    for cols in product(range(1, 8), repeat=3):
        a, b, c = cols
        if len({a, b, c, a ^ b, a ^ c, b ^ c, a ^ b ^ c} - {0}) == 7:
            out.append(cols)
    return out


def main(p, q):
    results, canon, sup = run(p, q)
    ana = pickle.load(open(HERE / f"analysis_p{p}q{q}.pkl", "rb"))
    classes = results["classes"]
    key_to_class = {m: ci for ci, c in enumerate(classes) for m in c["members"]}
    M = gl3()
    assert len(M) == 168
    parent = list(range(len(classes)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for ci, c in enumerate(classes):
        k = c["rep"]
        for m in M:
            new = []
            for col in m:  # new label = sum of old labels selected by bits of col
                v = 0
                for i in range(3):
                    if (col >> i) & 1:
                        v ^= k[i]
                new.append(v)
            cj = key_to_class[canon(*new)]
            a, b = find(ci), find(cj)
            if a != b:
                parent[a] = b
    groups = {}
    for ci in range(len(classes)):
        groups.setdefault(find(ci), []).append(ci)
    rows = []
    for g, mem in groups.items():
        nat = [ci for ci in mem if ana["classes"][ci]["native"]]
        rows.append((len(mem), len(nat), sorted(mem)[:3], nat))
    print("codes C1 (GL(3,2) x H orbits):", len(groups))
    print("codes admitting a single-qubit transversal basis:", sum(1 for r in rows if r[1]))
    for r in sorted(rows, key=lambda r: -r[1]):
        if r[1]:
            print("  code with", r[0], "classes,", r[1], "single-qubit transversal classes:", r[3])
    print("class-count distribution per code:", sorted(Counter(r[0] for r in rows).items()))
    pickle.dump({"groups": list(groups.values())}, open(HERE / f"codes_p{p}q{q}.pkl", "wb"))


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))

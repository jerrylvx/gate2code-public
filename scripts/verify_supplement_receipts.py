#!/usr/bin/env python3
"""Check the retained exact receipts cited by Supplementary Note C."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def load(name: str) -> dict:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def main() -> None:
    s5 = load("s5_corrected_closure.json")["summary"]
    assert s5["activities_checked"] == 63
    assert s5["verdict_summary"] == {
        "n=15:unsat": 31,
        "n=16:unsat": 31,
        "n=31:unknown": 1,
    }
    simplex = load("s5_simplex_linearized_decision.json")
    assert simplex["result"] == "unsat"
    assert simplex["pairs_checked"] == 49_741_825

    weight32 = load("s6_weight32_exact.json")
    assert weight32["affine_classes"] == 3
    assert weight32["origin_positions_verified"] == 192
    assert weight32["unpunctured_tests"] == weight32["punctured_tests"] == 2
    assert weight32["sampling"] is False and weight32["all_unsat_or_reduced"]
    assert [row["quadratic_rank"] for row in weight32["classes"]] == [0, 2, 4]
    for row in weight32["classes"]:
        assert len(row["origin_transports"]) == 64
        assert all(t["label_space_verified"] for t in row["origin_transports"])
    assert [[t["pairs_checked"] for t in row["tests"]]
            for row in weight32["classes"]] == [[], [12_042_241, 3_006_465], [2_224_129, 555_009]]
    assert all(t["result"] == "unsat" for row in weight32["classes"] for t in row["tests"])

    cover = load("m8_h_rep_completeness_audit.json")
    assert cover["all_complete"] is True
    expected_cover = {
        "h8_g8": (20, 11_160),
        "h12_g8": (88, 1_749_888),
        "h12_g12": (2167, 1_749_888),
    }
    for name, (n_reps, class_size) in expected_cover.items():
        row = cover["reports"][name]
        assert row["complete"] is True
        assert row["n_reps"] == n_reps
        assert row["sum_orbit_sizes"] == class_size
        assert row["missing_count"] == 0
        assert row["overlap_count"] == 0
        assert row["outside_target_count"] == 0

    h14_reps = load("m8_h14_reps.json")
    assert h14_reps["cover_complete"] is True
    assert h14_reps["n_reps"] == 124
    assert sum(h14_reps["orbit_sizes"]) == h14_reps["class_size"] == 22_855_680

    tasks = [load(f"m8_floor_manifest.task{i}.json") for i in range(4)]
    assert {row["task"] for row in tasks} == {0, 1, 2, 3}
    assert all(row["ntasks"] == 4 for row in tasks)
    base_systems = sum(row["total_surviving"] for row in tasks)
    base_third_label_solves = sum(row["real_decides"] for row in tasks)
    assert base_systems == 16_798_112
    assert base_third_label_solves == 3_523_584

    h14 = load("m8_h14_sweep_final.json")
    assert h14["all_unsat"] is True
    assert h14["n_reps"] == 124
    assert h14["decided"] == 840_384
    assert not h14["non_unsat"]
    assert not h14["deferred"]

    combined_systems = base_systems + h14["decided"]
    combined_third_label_solves = base_third_label_solves + h14["decided"]
    k3_rejections = base_systems - base_third_label_solves
    assert combined_systems == 17_638_496
    assert combined_third_label_solves == 4_363_968
    assert k3_rejections == 13_274_528

    result = {
        "rank6_weight32": {"affine_classes": 3, "analytic_reductions": 1,
                           "exact_label_tests": 4, "origin_positions_verified": 192,
                           "result": "all excluded"},
        "s5": {
            "activities": 63,
            "simplex_pairs": 49_741_825,
            "simplex_result": "unsat",
        },
        "rank8": {
            "systems": combined_systems,
            "k3_rejections": k3_rejections,
            "third_label_systems": combined_third_label_solves,
            "third_label_result": "all inconsistent",
            "weight14_representatives": 124,
            "missing": 0,
            "deferred": 0,
            "survivors": 0,
        },
        "orbit_cover_complete": True,
        "all_checks_passed": True,
    }
    path = REPORTS / "supplement_receipt_audit.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

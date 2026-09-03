#!/usr/bin/env python3
"""Decide the rank-five simplex case with the validated linear solver."""
import json
import sys
import time

sys.path.insert(0, ".")

from gate2code.alpha_linear import decide_activity


t0 = time.time()
result = decide_activity(list(range(1, 32)), 5)
result["instance"] = "s=5 n=31 simplex"
result["seconds"] = round(time.time() - t0, 1)
print(json.dumps({k: v for k, v in result.items() if not k.startswith("witness")}))
with open("reports/s5_simplex_linearized_decision.json", "w", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2)
    handle.write("\n")

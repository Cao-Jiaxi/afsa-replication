"""Batch 6 of analysis/gap/EXTENSION.md (section 16): lambda_var 10 and 1000 without no-signal positions, and seeds
3-9 of `nosig0` and of the reference in separate `_s3-9` folders (so the 3-seed runs already scored do not change).
Trains with the harness's own configs; extension.py is not edited while it is being audited."""
import time

import extension as ex  # noqa: E402
from reeval_weights import recover  # noqa: E402

from afsa.panel import load_panel  # noqa: E402
from afsa.run import run_config  # noqa: E402

ex.TRIALS["lv10"] = {"nosignal_hedge": False, "lambda_var": 10.0, "ref": "nosig0"}
ex.TRIALS["lv1000"] = {"nosignal_hedge": False, "lambda_var": 1000.0, "ref": "nosig0"}
JOBS = [("lv10", (0, 1, 2), ""), ("lv1000", (0, 1, 2), ""),
        ("nosig0", tuple(range(3, 10)), "s3-9"), ("reference", tuple(range(3, 10)), "s3-9")]

panel = load_panel(ex.config("reference"), "cuda")                          # the same month blocks for every job
start = time.time()
for name, seeds, tag in JOBS:
    cfg = ex.config(name)
    years = list(range(ex.first_year(cfg), ex.LAST + 1))
    t0 = time.time()
    folder = run_config(cfg, list(seeds), years, panel, None, tag=tag, verbose=False)
    n, worst = recover(folder, panel, years=years, tolerance=ex.REPLAY_TOLERANCE)
    print(f"{name} seeds {seeds[0]}-{seeds[-1]}: {folder.name}, {n} weight files "
          f"(largest replay difference {worst:.1e}), {time.time() - t0:.0f} s", flush=True)
print(f"BATCH6 DONE in {time.time() - start:.0f} s", flush=True)

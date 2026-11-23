"""Benchmark by facility type, ASHRAE G14 calibration and comparison-corrected GB."""
import os, sys, time, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("XGB_DEVICE", "cpu")
from common import RES
import s04_analysis as A, s06_benchmark as BM, x01_ml_audit as X, x03_review_round1 as X3
SEED = int(json.loads((RES / "B2_benchmark_meta.json").read_text())["seed"])
if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    B, E = A.load(); Bp, e = BM.population(B, E, "elec", X3.USES); e = e.merge(Bp[["pid", "use_class"]], on="pid", how="left")
    pools = BM.date_pools(B, X3.USES, "use"); R, AS = [], []; t0 = time.time()
    print("population", len(Bp), Bp.use_class.value_counts().to_dict(), flush=True)
    for rep in range(reps):
        pseudo, _ = BM.draw(Bp, e, pools, np.random.default_rng([SEED, rep]))
        P = BM.make_panel(e, pseudo).reset_index(drop=True)
        R += X3.rows_for(dict(rep=rep, n_pseudo=len(pseudo), **X3.drift_by_type(P)), Bp, P, ("naive", "dd", "gb", "gb_did", "imp"), SEED, rep)
        um = Bp.set_index("pid").use_class
        AS += [dict(r, use=um.get(r["pid"])) for r in X3.ashrae_rows(P, rep)]
        pd.DataFrame(R).to_csv(RES / "x08_ep_reps.csv", index=False); pd.DataFrame(AS).to_csv(RES / "x08_ep_ashrae.csv", index=False)
        if (rep + 1) % 10 == 0: print(f"rep {rep + 1} {time.time() - t0:.0f}s", flush=True)

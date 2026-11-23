"""Calendar profile of waiting health posts by selection year."""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_k, "1")
import sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES
import s04_analysis as A
import r3_descriptive as D

B, E = A.load(); out = []
for y in (2023, 2024, 2025):
    P, meta = D.calendar_profile(B[(B.year == y) | ~B.role.isin(D.CR)], E, "보건"); out.append(P.assign(selection_year=y, n_buildings_total=meta["n_buildings"]))
P = pd.concat(out); P.to_csv(RES / "r3_calendar_by_selection_year.csv", index=False)
print(P.pivot_table(index="quarter", columns="selection_year", values=["pct", "n_buildings"]).round(3).to_string())

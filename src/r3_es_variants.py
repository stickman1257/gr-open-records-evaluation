"""Sample variants of the main event study."""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_k, "1")
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES
import s04_analysis as A
import r3_analysis as R

B, E = A.load(); Bi = B.drop_duplicates("pid").set_index("pid")
G = pd.read_csv(RES / "r3_building_gaps.csv"); head = set(G.pid); R.TREATED_HEAD.update(head)
hc = A.heating_carrier(B[B.pid.isin(head)], E)
elec = set(hc[hc == "elec"].index); noren = set(Bi[Bi.renewable == 0].index); noboil = set(Bi[Bi.boiler == 0].index)
multi = set(E[(E.carrier == "elec") & (E.n_items > 1)].pid); annex = set(Bi[Bi.reg_n_title > Bi.reg_n_main].index)
big = set(G[G.pct <= -0.5].pid); CR = ("control", "aux_2023")
ctrl = Bi[Bi.role.isin(CR) & (Bi.use_class == "보건")]
V = [("main", head, None), ("controls_2023_2024", head, set(ctrl[ctrl.year == 2025].index)), ("controls_2024_only", head, set(ctrl[ctrl.year != 2024].index)),
     ("without_fall_ge50", head - big, None), ("clean_elec_heat_no_renewable", head & elec & noren, None), ("clean_plus_no_boiler", head & elec & noren & noboil, None),
     ("no_renewable", head & noren, None), ("elec_heat", head & elec, None), ("single_meter_no_annex", head - multi - annex, multi | annex)]
rows = []
for name, tr, dc in V:
    o, ev, S = R.event_long(B, E, cluster="sigungu", treated=tr, drop_ctrl=dc)
    p = o["post_1_12"]; rows.append(dict(variant=name, pct=p["pct"], pct_lo=p["pct_ci"][0], pct_hi=p["pct_ci"][1], theta_log=p["theta_log"], se_log=p["se_log"],
                                         n_treated=o["n_treated"], n_control=o["n_control"], n_clusters=o["n_clusters"], pretrend_p=o["pretrend"]["p_F"],
                                         label="post hoc"))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(RES / "r3_es_variants.csv", index=False)

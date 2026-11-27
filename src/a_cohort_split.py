"""Completion-cohort split of the electricity estimate."""
import os, sys, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES
import s04_analysis as A
from a_interim_checks import estimate, rows_from, SAMPLES, HC, DC

LABEL = "added after the interim look, before the final data; cohort split = exploratory decomposition of the headline"
NB = int(os.environ.get("IMP_NBOOT", "200")); NBH = int(os.environ.get("IMP_NBOOT_HEAD", "1000"))
CUT = 202311


def main():
    B, E = A.load(); rows = []
    ctrl = E.role.isin(["control", "aux_2023"])
    Epre = E[~(ctrl & (E.carrier == "elec") & (E.ym >= CUT))]
    out = dict(label=LABEL, generated=time.strftime("%Y-%m-%d %H:%M"), cut_month=CUT, rows={})
    for s, uses in SAMPLES.items():
        out["rows"][s] = {}
        for spec, Ev, nb in [("primary", E, NBH)] + ([("controls_pre_202311", Epre, NB)] if s != "daycare" else []):
            try:
                r = estimate(B, Ev, "elec", nboot=nb, uses=uses)
                if r is None: raise RuntimeError("no treated rows")
                rr = rows_from(r[0], sample=s, spec=spec, status="ok")
                for x in rr: x["label"] = LABEL
                rows += rr; out["rows"][s][spec] = {x["level"]: {k: x[k] for k in ["att_log", "pct", "lo_log", "hi_log", "pct_lo", "pct_hi", "n_treated", "n_control", "support_rows", "nboot"]} for x in rr}
                a = rr[0]; print(f"{s} {spec}: {a['pct']:+.3f} [{a['pct_lo']:+.3f}, {a['pct_hi']:+.3f}] nT {a['n_treated']} nC {a['n_control']} support {a['support_rows']:.2f}", flush=True)
            except Exception as ex:
                rows.append(dict(sample=s, spec=spec, level="all", status=f"{type(ex).__name__}: {ex}"[:200], label=LABEL)); out["rows"][s][spec] = dict(status=f"{type(ex).__name__}: {ex}"[:200])
    S = A.imputation_setup(B, E, "elec", **A.PRIMARY_IMP); D = S["D"]; c = D[D.ctrl == 1].assign(use=lambda d: d.pid.map(B.drop_duplicates("pid").set_index("pid").use_class))
    n = c.groupby(["use", "ym"]).pid.nunique(); cp = {}
    for u, en in [(HC, "health_centre"), (DC, "daycare")]:
        if u not in n.index.get_level_values(0): continue
        x = n[u]; pre = x[(x.index < CUT) & (x.index % 100 != 1) & (x.index != 202110)]; post = x[(x.index >= CUT) & (x.index <= 202410)]
        cp[en] = dict(median_buildings_per_month_before_202311_excl_january=float(pre.median()) if len(pre) else None,
                      median_buildings_per_month_202311_202410=float(post.median()) if len(post) else None,
                      n_202310=int(x.get(202310, 0)), n_202311=int(x.get(CUT, 0)), last_month_with_ge10_controls=int(x[x >= 10].index.max()) if (x >= 10).any() else None)
    out["control_pool_by_period"] = cp
    pd.DataFrame(rows).to_csv(RES / "a_cohort_split.csv", index=False)
    (RES / "a_cohort_split.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps(cp, ensure_ascii=False))

if __name__ == "__main__":
    main()

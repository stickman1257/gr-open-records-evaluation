"""Stacked DiD variants on the real sample."""
import os, sys, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES
import s04_analysis as A

LABEL = "post hoc (added 11/16 after the final run)"
SAMPLES = {"health_centre": ("보건",), "pooled": ("보건", "어린이집")}


def main():
    B, E = A.load(); rows = []; out = dict(label=LABEL, generated=time.strftime("%Y-%m-%d %H:%M"), definition=__doc__, samples={})
    for s, uses in SAMPLES.items():
        out["samples"][s] = {}
        ri = A.imputation_did(B, E, "elec", nboot=int(os.environ.get("IMP_NBOOT", "200")), fe="sido", uses=uses)
        pt = ri["pretrend"] or {}
        out["samples"][s]["imputation_province_month"] = dict(pct=ri["pct"], pct_ci=ri["pct_ci"], n_treated=ri["n_treated"], n_control=ri["n_control"], support_rows=ri["support_rows"],
                                                              pretrend_p=ri["pretrend_p"], pretrend_coef=pt.get("coef"), pretrend_terms=pt.get("terms"), nboot=ri["nboot"], label=LABEL)
        print(f"{s} imputation, province x month: {100 * ri['pct']:+.1f}% ({100 * ri['pct_ci'][0]:+.1f}, {100 * ri['pct_ci'][1]:+.1f}); T {ri['n_treated']}; support {ri['support_rows']:.3f}; pre-trend p (s04 definition) {ri['pretrend_p']}", flush=True)
        for name, kw in {"did_province_month": dict(), "did_province_month_anticip3": dict(anticip=3)}.items():
            base = dict(fe="sido", ctrl_trim=3, ctrl_roles=("control", "aux_2023"), uses=uses, **kw)
            r = A.stacked_did(B, E, "elec", **base); r = r[r.term == "post"].iloc[0]
            ev = A.stacked_did(B, E, "elec", event=True, **base)
            d = dict(sample=s, spec=name, label=LABEL, coef_cohort_w=float(r.coef_cohort_w), pct=float(r.pct_cohort_w), pct_lo=float(r.pct_cohort_w_lo), pct_hi=float(r.pct_cohort_w_hi),
                     ci_excludes_zero=bool(r.pct_cohort_w_hi < 0 or r.pct_cohort_w_lo > 0), n_treated=int(r.n_treated), n_control=int(r.n_control), support_share=float(r.support_share),
                     pretrend_wald_p=float(ev.pretrend_wald_p.iloc[0]) if ev is not None and "pretrend_wald_p" in ev else None,
                     pretrend_p_s04_definition=ri["pretrend_p"],
                     event_terms={t: float(c) for t, c in zip(ev.term, ev.coef)} if ev is not None else None, se_convention=str(r.se_convention))
            out["samples"][s][name] = d; rows.append({k: v for k, v in d.items() if not isinstance(v, dict)})
            print(f"{s} {name}: {100 * d['pct']:+.1f}% ({100 * d['pct_lo']:+.1f}, {100 * d['pct_hi']:+.1f}); T {d['n_treated']}, C {d['n_control']}; support {d['support_share']:.3f}; pre-trend p {d['pretrend_wald_p']}", flush=True)
    pd.DataFrame(rows).to_csv(RES / "a_r6_did_real.csv", index=False)
    (RES / "a_r6_did_real.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()

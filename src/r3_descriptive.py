"""Calendar profile of waiting health posts (Fig. 5) and descriptives."""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_k, "1")
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES
import s04_analysis as A
import r3_analysis as R3

CR = ("control", "aux_2023")


def calendar_profile(B, E, use):
    """y = a_i + l_t + b HDD18 + c CDD24 (t = calendar month; b, c identified across provinces within a month). The weather- and"""
    import pyfixest as pf
    Bg = B[(B.link_grade == "A_cand") & (B.use_class == use) & B.role.isin(CR)]
    e = A.carrier_frame(E, "elec"); e = e[e.pid.isin(Bg.pid)].copy(); e = e[e.ym < e.start_ym.map(lambda v: A.ym_add(int(v), -3))]
    e = e.dropna(subset=["HDD18", "CDD24"]).copy(); e["y"] = np.log(e.kwh / e.gr_gfa); e["t"] = e.ym.astype(int)
    months = sorted(e.t.unique()); ref = 202007
    m = pf.feols("y ~ HDD18 + CDD24 + i(t, ref=%d) | pid" % ref, data=e, vcov={"CRV1": "pid"})
    co = m.coef(); V = pd.DataFrame(np.asarray(m._vcov), index=m._coefnames, columns=m._coefnames)
    name = {t: [n for n in co.index if n.endswith(f"::{t}") or n.endswith(f"[T.{t}]")] for t in months}
    lam = pd.Series({t: (float(co[name[t][0]]) if name[t] else 0.0) for t in months})
    base = [t for t in months if 202007 <= t <= 202106]; moy = {t: t % 100 for t in months}
    rows = []
    for q in sorted({(t // 100, (t % 100 - 1) // 3 + 1) for t in months}):
        tq = [t for t in months if t // 100 == q[0] and (t % 100 - 1) // 3 + 1 == q[1]]
        w = pd.Series(0.0, index=months)
        for t in tq:
            w[t] += 1 / len(tq); bm = [s for s in base if moy[s] == moy[t]]
            for s in bm: w[s] -= 1 / len(tq) / len(bm)
        wn = pd.Series({name[t][0]: v for t, v in w.items() if name[t] and v != 0})
        est = float((w * lam).sum()); se = float(np.sqrt(wn.values @ V.loc[wn.index, wn.index].values @ wn.values)) if len(wn) else 0.0
        rows.append(dict(use=use, quarter=f"{q[0]}Q{q[1]}", level=est, se=se, pct=float(np.exp(est) - 1), pct_lo=float(np.exp(est - 1.96 * se) - 1), pct_hi=float(np.exp(est + 1.96 * se) - 1),
                         n_buildings=int(e[e.t.isin(tq)].pid.nunique()), n_obs=int(e.t.isin(tq).sum())))
    return pd.DataFrame(rows), dict(hdd18=float(co["HDD18"]), cdd24=float(co["CDD24"]), n_buildings=int(e.pid.nunique()), n_obs=int(len(e)),
                                    method=calendar_profile.__doc__)


def main():
    B, E = A.load(); out = {"label": "post hoc"}
    K = pd.read_csv(RES / "ep_scope_locator_kalis.csv"); F = pd.read_csv(RES / "ep_scope_locator_flags.csv")
    out["locator_vs_statistics"] = K.to_dict("records"); out["locator_flags"] = F.to_dict("records")
    Bi = B.drop_duplicates("pid")
    C = Bi[Bi.role.isin(CR) & (Bi.use_class == "보건") & (Bi.link_grade == "A_cand")]
    nonblank = lambda s: (s.notna() & (s.astype(str).str.strip() != "")).mean()
    out["comparison_health_records"] = dict(by_year=C.groupby("year").size().to_dict(),
                                            share_with=dict(period=float(nonblank(C.period_txt)), cost=float(nonblank(C.cost_txt)), measures=float(nonblank(C.measures_txt))))
    S0 = R3.imp_setup(B, E); f0 = S0["fit"](); m0 = S0["post"] & f0["ok"]; head = set(S0["pids"][pd.unique(S0["pi"][m0])])
    for tag, yrs in (("controls_2024_only", (2024,)), ("controls_2023_2024", (2023, 2024))):
        drop = set(Bi[Bi.role.isin(CR) & ~Bi.year.isin(yrs)].pid)
        S = R3.imp_setup(B, E, treated_keep=head, drop=drop); r = R3.imp_estimate(S, B, kinds=("plain_sig",)); out[tag] = r; print(tag, r, flush=True)
    prof = []
    for use in ("보건", "어린이집"):
        P, meta = calendar_profile(B, E, use); prof.append(P); out[f"calendar_{use}"] = meta
    P = pd.concat(prof); P.to_csv(RES / "r3_calendar_profile.csv", index=False); print(P.to_string())
    (RES / "r3_descriptive.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()

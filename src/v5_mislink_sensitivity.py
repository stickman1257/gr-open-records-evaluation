"""Mislinkage simulation with the imputation estimator."""
import json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES
import s04_analysis as A
from s04_analysis import ym_add, MIN_WIN

PIS = [0.0, 0.05, 0.10, 0.20, 0.30]; SHARES = [0.5, 0.75]; NREP = int(os.environ.get("V5_NREP", "200")); SEED = 0
LABEL = "exploratory, post hoc (verification round V5)"
TAG = os.environ.get("V5_TAG", "")
KW = dict(uses=("보건",), fe="use")


def solve_d(att, pi, s):
    f = lambda d: (1 - pi) * np.log1p(d) + pi * np.log1p(s * d) - att
    lo, hi = -0.999, 0.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if f(mid) > 0: hi = mid
        else: lo = mid
    return (lo + hi) / 2


if __name__ == "__main__":
    g = [r for r in json.load(open(RES / "G7_imputation.json", encoding="utf-8")) if r["spec"] == "imputation_health_only" and r["carrier"] == "elec"][0]
    att, lo, hi = g["att_log"], g["ci_log"][0], g["ci_log"][1]
    pct = lambda v: float(np.exp(v) - 1)
    rows = []
    for pi in PIS:
        r = dict(pi=pi, a_true_pct=pct(att / (1 - pi)), a_true_pct_lo=pct(lo / (1 - pi)), a_true_pct_hi=pct(hi / (1 - pi)))
        for s in SHARES:
            for nm, v in [("", att), ("_lo", lo), ("_hi", hi)]: r[f"b_s{s}_true_pct{nm}"] = float(solve_d(v, pi, s))
        rows.append(r)
    AN = pd.DataFrame(rows); AN.to_csv(RES / f"v5_mislink_analytic{TAG}.csv", index=False)

    B, E = A.load()
    S0 = A.imputation_setup(B, E, "elec", **KW); f0 = S0["fit"](); att0 = S0["att"](f0)[0]
    assert abs(att0 - att) < 1e-6, (att0, att)
    T = S0["T"][["pid", "start_ym", "end_ym"]].copy(); C = S0["C"]
    Ec = A.carrier_frame(E, "elec")
    don = Ec[Ec.pid.isin(C.pid)]; don = don[don.ym < don.start_ym.map(lambda v: ym_add(int(v), -3))]
    dmon = {p: set(d.ym) for p, d in don.groupby("pid")}; drow = {p: d for p, d in don.groupby("pid")}
    elig = {}
    for r in T.itertuples():
        s_, e_ = int(r.start_ym), int(r.end_ym)
        pre = {ym_add(s_, -k) for k in range(1, 13)}; post = {ym_add(e_, k) for k in range(1, 13)}
        elig[r.pid] = [p for p, m in dmon.items() if len(m & pre) >= MIN_WIN and len(m & post) >= MIN_WIN]
    ne = np.array([len(v) for v in elig.values()])
    Ti = T.set_index("pid"); tp = T.pid.to_numpy(); n = len(tp)
    E_other = E[~((E.carrier == "elec") & E.pid.isin(tp))]
    E_t = {p: d for p, d in E[(E.carrier == "elec") & E.pid.isin(tp)].groupby("pid")}
    rng = np.random.default_rng(SEED); draws = []; t0 = time.time()
    for pi in PIS[1:]:
        k = int(round(pi * n))
        for rep in range(NREP):
            sel = rng.choice(tp, k, replace=False); parts = [E_other] + [E_t[p] for p in tp if p not in set(sel)]
            for p in sel:
                d = drow[elig[p][rng.integers(len(elig[p]))]].copy()
                d["pid"] = p; d["role"] = "treated"; d["start_ym"] = Ti.start_ym[p]; d["end_ym"] = Ti.end_ym[p]
                d["phase"] = np.select([d.ym < d.start_ym, d.ym > d.end_ym], ["pre", "post"], "constr")
                parts.append(d)
            S = A.imputation_setup(B, pd.concat(parts, ignore_index=True), "elec", **KW)
            f = S["fit"](); a = S["att"](f)[0]; m = S["post"] & f["ok"]
            draws.append(dict(pi=pi, rep=rep, n_replaced=k, att_log=a, n_treated=int(pd.unique(S["pi"][m]).size)))
        pd.DataFrame(draws).to_csv(RES / f"v5_mislink_sim_draws{TAG}.csv", index=False)
        print(pi, round(time.time() - t0), "s", flush=True)
    D = pd.DataFrame(draws); D.to_csv(RES / f"v5_mislink_sim_draws{TAG}.csv", index=False)
    q = lambda v, p_: float(np.percentile(v, p_))
    sim = [dict(pi=0.0, n_replaced=0, reps=1, att_pct_median=pct(att0), att_pct_p2_5=pct(att0), att_pct_p97_5=pct(att0),
                att_log_median=att0, linear_attenuation_pct=pct(att0), n_treated_min=n, n_treated_max=n)]
    for pi, d in D.groupby("pi"):
        sim.append(dict(pi=pi, n_replaced=int(d.n_replaced.iloc[0]), reps=len(d), att_pct_median=pct(d.att_log.median()),
                        att_pct_p2_5=pct(q(d.att_log, 2.5)), att_pct_p97_5=pct(q(d.att_log, 97.5)), att_log_median=float(d.att_log.median()),
                        linear_attenuation_pct=pct((1 - pi) * att0), n_treated_min=int(d.n_treated.min()), n_treated_max=int(d.n_treated.max())))
    SIM = pd.DataFrame(sim); SIM.to_csv(RES / f"v5_mislink_sim{TAG}.csv", index=False)
    aud = json.load(open(RES / "v5_linkage_audit.json", encoding="utf-8")) if (RES / "v5_linkage_audit.json").exists() else {}
    sg = json.load(open(RES / "a_signature_eval.json", encoding="utf-8"))["overall"]
    out = dict(label=LABEL, source="src/v5_mislink_sensitivity.py",
               headline=dict(file="results/G7_imputation.json", spec="imputation_health_only", att_log=att, pct=pct(att), ci_log=[lo, hi],
                             pct_ci=[pct(lo), pct(hi)], n_treated=g["n_treated"], n_control=g["n_control"]),
               mechanisms=__doc__.split("No value of pi is asserted.")[1].split("Outputs:")[0].strip(),
               analytic=AN.to_dict("records"), simulation=SIM.to_dict("records"),
               simulation_design=dict(reps_per_pi=NREP, seed=SEED, n_treated=n, n_donor_buildings=len(dmon),
                                      eligible_donors_per_treated=dict(min=int(ne.min()), median=float(np.median(ne)), max=int(ne.max()))),
               basis_for_pi=dict(
                   note="No true pi is asserted. The audit classes rest on records that cannot show who uses the meter; the signature bounds are upper bounds under the assumption that a mislinked meter behaves like a donor meter.",
                   audit_conflicting_share_sample={k: v["conflicting"] for k, v in (aud.get("sample") or {}).items()},
                   audit_conflicting_share_frame={k: v["conflicting"] for k, v in (aud.get("census") or {}).items()},
                   audit_headline_health_treated={k: (aud.get("census_headline_health_treated") or {}).get(k) for k in ["n", "record-consistent", "unresolved", "conflicting"]},
                   signature_upper_bounds=dict(file="results/a_signature_eval.json", at_R_le_0p7=sg["bound_mislink_at_0p7"], at_null_median=sg["bound_mislink_at_null_median"])))
    (RES / f"v5_mislink_sensitivity{TAG}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print(AN.round(4).to_string()); print(SIM.round(4).to_string())

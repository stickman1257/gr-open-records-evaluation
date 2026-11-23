"""Long stacked event study (main estimate), trend extrapolation, honestdid inputs."""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_k, "1")
import sys, json, time, multiprocessing as mp
import numpy as np, pandas as pd
from scipy import stats
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, PROC
import s04_analysis as A
from s03_energy import ym_add

LABEL = "post hoc (Energy Policy review R3; rules recorded in commit e580a0f before the run)"
HC = ("보건",); CR = ("control", "aux_2023")
NB = int(os.environ.get("R3_NBOOT", "1000")); NPROC = int(os.environ.get("R3_NPROC", "10"))
PRE_LO, POST_HI, REF = -36, 24, (-6, -4)
mi = lambda v: (np.asarray(v).astype(int) // 100) * 12 + np.asarray(v).astype(int) % 100
pct = lambda a: float(np.exp(a) - 1)


def long_stack(B, E, pre_lo=PRE_LO, post_hi=POST_HI, treated=None, drop_ctrl=None, ctrl_trim=3):
    Ec = A.carrier_frame(E, "elec").copy(); Ec["y"] = np.log(Ec.kwh / Ec.gr_gfa)
    Bg = B[(B.link_grade == "A_cand") & B.use_class.isin(HC)]
    T = Bg[Bg.role == "treated"].copy(); T = T[T.pid.isin(A.complete_windows(Ec, T))]
    if treated is not None: T = T[T.pid.isin(set(treated))]
    C = Bg[Bg.role.isin(CR)]
    if drop_ctrl is not None: C = C[~C.pid.isin(set(drop_ctrl))]
    T["cohort"] = (T.end_ym // 100) * 10 + ((T.end_ym % 100) > 6).astype(int)
    stacks = []
    for coh, tc in T.groupby("cohort"):
        if len(tc) < 3: continue
        lo = ym_add(int(tc.start_ym.min()), pre_lo); hi = ym_add(int(tc.end_ym.max()), post_hi)
        et = Ec[Ec.pid.isin(tc.pid)].copy()
        rs = mi(et.ym) - mi(et.start_ym); re_ = mi(et.ym) - mi(et.end_ym)
        keep = ((rs >= pre_lo) & (rs <= -1)) | ((re_ >= 1) & (re_ <= post_hi))
        et = et[keep].copy(); et["rel"] = np.where(et.ym < et.start_ym, (mi(et.ym) - mi(et.start_ym)), (mi(et.ym) - mi(et.end_ym)))
        ec = Ec[Ec.pid.isin(C.pid) & (Ec.ym >= lo) & (Ec.ym <= hi)]
        ec = ec[ec.ym < ec.start_ym.map(lambda v: ym_add(int(v), -ctrl_trim))].copy(); ec["rel"] = 0
        st = pd.concat([et.assign(treat=1), ec.assign(treat=0)]); st["stack"] = coh; stacks.append(st)
    S = pd.concat(stacks, ignore_index=True)
    S["unit_fe"] = S.pid + "|" + S["stack"].astype(str)
    S["time_fe"] = A.time_key(S, B, "use") + "|" + S.ym.astype(str) + "|" + S["stack"].astype(str)
    n0 = int((S.treat == 1).sum()); has_c = set(S[S.treat == 0].time_fe)
    S = S[(S.treat == 0) | S.time_fe.isin(has_c)].copy(); support = float((S.treat == 1).sum() / max(n0, 1))
    S = S[S.groupby("unit_fe").y.transform("size") > 1].copy()
    bins = [(a, a + 2) for a in range(pre_lo, 0, 3)] + [(a, a + 2) for a in range(1, post_hi + 1, 3)]
    x = []
    for a, b in bins:
        if (a, b) == REF: continue
        c = f"ev_{a}_{b}"; S[c] = ((S.treat == 1) & (S.rel >= a) & (S.rel <= b)).astype(int)
        if S[c].sum() > 0: x.append(c)
    return S, x, support


def cl_of(S, B, cluster):
    if cluster == "sigungu": return S.pid.map(B.drop_duplicates("pid").set_index("pid").sigunguCd.astype(str)).fillna(S.pid).to_numpy()
    return S.pid.to_numpy()


def event_long(B, E, cluster="sigungu", **kw):
    S, x, support = long_stack(B, E, **kw)
    F = A.fe_fit(S, x, cl_of(S, B, cluster))
    b, V = F["b"], F["cov"]; G = int(pd.Series(cl_of(S, B, cluster)).nunique())
    nb = {c: int(S[(S[c] == 1)].pid.nunique()) for c in x}
    ev = pd.DataFrame(dict(term=x, coef=b, se=np.sqrt(np.diag(V)), n_treated_bld=[nb[c] for c in x]))
    ev["a"] = ev.term.str.extract(r"ev_(-?\d+)_")[0].astype(int); ev["b"] = ev.term.str.extract(r"_(-?\d+)$")[0].astype(int)
    ev["lo"] = ev.coef - F["crit"] * ev.se; ev["hi"] = ev.coef + F["crit"] * ev.se
    for c in ("coef", "lo", "hi"): ev[f"pct_{c}"] = np.exp(ev[c]) - 1
    leads = [i for i, t in enumerate(x) if ev.b.iloc[i] <= -7]
    post12 = [i for i, t in enumerate(x) if 1 <= ev.a.iloc[i] <= 12]; post24 = [i for i, t in enumerate(x) if 13 <= ev.a.iloc[i] <= 24]
    def lin(ix):
        l = np.zeros(len(x)); l[ix] = 1 / len(ix); th = float(l @ b); se = float(np.sqrt(l @ V @ l))
        return dict(theta_log=th, se_log=se, pct=pct(th), pct_ci=[pct(th - F["crit"] * se), pct(th + F["crit"] * se)], bins=[x[i] for i in ix])
    bl = b[leads]; Vl = V[np.ix_(leads, leads)]; q = len(leads); W = float(bl @ np.linalg.solve(Vl, bl))
    t = (ev.a.iloc[leads].to_numpy() + 1 + 5) / 3.0; Vi = np.linalg.inv(Vl); g = float(t @ Vi @ bl / (t @ Vi @ t)); g_se = float(1 / np.sqrt(t @ Vi @ t))
    wg = np.zeros(len(x)); wg[leads] = (t @ Vi) / (t @ Vi @ t)
    out = dict(cluster=cluster, n_clusters=G, n_treated=int(S[S.treat == 1].pid.nunique()), n_control=int(S[S.treat == 0].pid.nunique()), support_share=support,
               n_obs=int(len(S)), se_convention=F["convention"], crit=F["crit"], post_1_12=lin(post12), post_13_24=lin(post24) if post24 else None,
               pretrend=dict(leads=[x[i] for i in leads], q=q, wald=W, p_F=float(stats.f.sf(W / q, q, max(G - 1, 1))), p_chi2=float(stats.chi2.sf(W, q))),
               lead_slope_per_3m=dict(g=g, se=g_se), _b=b, _V=V, _x=x, _wg=wg, _post12=post12)
    return out, ev, S


def single_post(B, E, cluster="sigungu"):
    """Conventional stacked DiD on the long sample: one post coefficient, pre months s-36..s-4 as baseline (-3..-1 dropped)."""
    S, x, support = long_stack(B, E, post_hi=12)
    S = S[~((S.treat == 1) & (S.rel >= -3) & (S.rel <= -1))].copy()
    S["post"] = ((S.treat == 1) & (S.rel >= 1)).astype(int)
    F = A.fe_fit(S, ["post"], cl_of(S, B, cluster)); b = float(F["b"][0]); se = float(np.sqrt(F["cov"][0, 0]))
    return dict(theta_log=b, se_log=se, pct=pct(b), pct_ci=[pct(b - F["crit"] * se), pct(b + F["crit"] * se)], n_treated=int(S[S.treat == 1].pid.nunique()),
                n_control=int(S[S.treat == 0].pid.nunique()), cluster=cluster)


def trend_adjusted(out, B, E):
    """E3b: theta - g x distance, distance = mean calendar distance (in 3-month units) of post months 1..12 from the reference centre."""
    T = B[B.pid.isin(TREATED_HEAD)]; dur = (mi(T.end_ym) - mi(T.start_ym)).astype(float)
    dist = (np.mean(dur) + 6.5 + 5.0) / 3.0
    b, V, x = out["_b"], out["_V"], out["_x"]; l = np.zeros(len(x)); l[out["_post12"]] = 0.25
    a = l - dist * out["_wg"]; th = float(a @ b); se = float(np.sqrt(a @ V @ a)); c = out["crit"]
    return dict(distance_3m=float(dist), mean_construction_months=float(np.mean(dur)), median_construction_months=float(np.median(dur)),
                slope_per_3m=out["lead_slope_per_3m"], bias_log=float(out["lead_slope_per_3m"]["g"] * dist), theta_adj_log=th, se_log=se, pct=pct(th),
                pct_ci=[pct(th - c * se), pct(th + c * se)])


def imp_setup(B, E, fit_lo=None, treated_keep=None, drop=None):
    return A.imputation_setup(B, E, "elec", uses=HC, fe="use", ctrl_roles=CR, ctrl_trim=3, anticip=3, fit_lo=fit_lo, treated_keep=treated_keep, drop=drop)


def trend_fit(S, w=None, trend=True, tol=1e-10):
    """Imputation fit with an optional common linear trend z = months since the start of works for treated rows (0 for controls)."""
    D = S["D"]; pi, ki = S["pi"], S["ki"]; nP, nK = len(S["pids"]), len(S["tks"])
    z = np.where(D.ctrl.to_numpy() == 1, 0.0, (mi(D.ym) - mi(D.start_ym)).astype(float)) / 12.0
    w = np.ones(nP) if w is None else np.asarray(w, float); wr = w[pi]
    F = ~S["post"] & ~S["ant"] & (wr > 0); y = D.y.to_numpy(float)
    n = int(F.sum()); sw = np.sqrt(wr[F]); r = np.arange(n)
    cols = [np.r_[r, r], np.r_[pi[F], nP + ki[F]]]; vals = np.r_[sw, sw]; ncol = nP + nK
    if trend: cols = [np.r_[cols[0], r], np.r_[cols[1], np.full(n, ncol)]]; vals = np.r_[vals, sw * z[F]]; ncol += 1
    X = coo_matrix((vals, (cols[0], cols[1])), shape=(n, ncol)).tocsr()
    mu = float(np.average(y[F], weights=wr[F])); sol = lsqr(X, sw * (y[F] - mu), atol=tol, btol=tol, iter_lim=30000)[0]
    dp = np.bincount(pi[F], wr[F], nP); dk = np.bincount(ki[F], wr[F], nK)
    a = np.where(dp > 0, sol[:nP] + mu, np.nan); l = np.where(dk > 0, sol[nP:nP + nK], np.nan); gam = float(sol[-1]) if trend else 0.0
    res = y - a[pi] - l[ki] - gam * z
    base = S["fit"](w, tol=1e-9)
    return dict(res=res, ok=base["ok"] & np.isfinite(res), wr=wr, gamma_per_year=gam)


_G = {}
def _boot(args):
    seed, kind = args; S = _G["S"]; rng = np.random.default_rng([seed, 17])
    w = A.boot_weights(S, rng, cluster=_G["cl"] if kind.endswith("sig") else None)
    if kind.startswith("trend"): f = trend_fit(S, w, trend=True)
    else: f = S["fit"](w, tol=1e-9)
    return S["att"](f)[0]


def imp_estimate(S, B, kinds=("plain_pid", "plain_sig")):
    _G["S"] = S; _G["cl"] = pd.Series(S["pids"]).map(B.drop_duplicates("pid").set_index("pid").sigunguCd.astype(str)).fillna("NA").to_numpy()
    f0 = S["fit"](); a0 = S["att"](f0)[0]; ft = trend_fit(S, trend=True); at = S["att"](ft)[0]
    m = S["post"] & f0["ok"]; out = dict(n_treated=int(pd.unique(S["pi"][m]).size), n_control=int(pd.unique(S["pi"][f0["F"] & S["isc"]]).size),
                                          att_log=a0, pct=pct(a0), trend_att_log=at, trend_pct=pct(at), gamma_per_year=ft["gamma_per_year"],
                                          n_clusters_sig=int(pd.Series(_G["cl"]).nunique()))
    with mp.get_context("fork").Pool(NPROC) as p:
        for k in kinds:
            d = np.array(p.map(_boot, [(s, k) for s in range(NB)], chunksize=10), float); d = d[np.isfinite(d)]
            out[f"ci_{k}"] = [pct(np.percentile(d, 2.5)), pct(np.percentile(d, 97.5))]; out[f"nboot_{k}"] = int(len(d)); out[f"se_log_{k}"] = float(d.std(ddof=1))
    return out


def rm_bounds(b, V, pre_ix, ref_pos, post_ix, steps, Mbar, crit=1.96, nsim=100000, seed=3):
    """Delta^RM identified set and a projection (simultaneous) interval. b: event-study vector (reference omitted); pre_ix: indices of the"""
    k = len(b); l = np.zeros(k); l[post_ix] = 1 / len(post_ix); c = float(np.mean(steps))
    Ds = []
    for i in range(len(pre_ix)):
        d = np.zeros(k)
        if i + 1 < len(pre_ix): d[pre_ix[i + 1]] += 1; d[pre_ix[i]] -= 1
        else: d[pre_ix[i]] -= 1
        Ds.append(d)
    vecs_up = [l + s * Mbar * c * d for d in Ds for s in (1, -1)]; vecs_lo = [l - s * Mbar * c * d for d in Ds for s in (1, -1)]
    U = np.array(vecs_up); L = np.array(vecs_lo); maxd = max(abs(d @ b) for d in Ds)
    rng = np.random.default_rng(seed); Z = rng.multivariate_normal(np.zeros(k), V, nsim, method="eigh")
    def band(M, upper):
        se = np.sqrt(np.einsum("ij,jk,ik->i", M, V, M)); t = (Z @ M.T) / se; q = float(np.quantile((-t if upper else t).max(1), 0.95 if False else 0.975))
        return float((M @ b + q * se).max()) if upper else float((M @ b - q * se).min())
    th = float(l @ b)
    return dict(Mbar=Mbar, c=c, maxd=float(maxd), set_lo=th - Mbar * c * maxd, set_hi=th + Mbar * c * maxd, ci_lo=band(L, False), ci_hi=band(U, True))


def own_breakdown(b, V, pre_ix, post_ix, steps):
    grid = np.round(np.arange(0, 3.0001, 0.01), 2); bd = None
    for Mb in grid:
        r = rm_bounds(b, V, pre_ix, None, post_ix, steps, float(Mb), nsim=40000)
        if r["ci_hi"] < 0 or r["ci_lo"] > 0: bd = float(Mb)
        else: return dict(breakdown_last_excluding_zero=bd, breakdown_first_including_zero=float(Mb))
    return dict(breakdown_last_excluding_zero=bd, breakdown_first_including_zero=None)


TREATED_HEAD = set()

def main():
    t0 = time.time(); B, E = A.load(); res = {"label": LABEL}; rows = []
    S0 = imp_setup(B, E); f0 = S0["fit"](); m0 = S0["post"] & f0["ok"]
    TREATED_HEAD.update(S0["pids"][pd.unique(S0["pi"][m0])]); res["n_headline_treated"] = len(TREATED_HEAD)
    evs = []
    for cl in ("sigungu", "pid"):
        o, ev, S = event_long(B, E, cluster=cl, treated=TREATED_HEAD)
        evs.append(ev.assign(cluster=cl)); o["trend_adjusted"] = trend_adjusted(o, B, E)
        if cl == "sigungu":
            gap_m = float(np.median(mi(B[B.pid.isin(TREATED_HEAD)].end_ym) - mi(B[B.pid.isin(TREATED_HEAD)].start_ym) + 1))
            g = int(round(gap_m / 3.0)); x = o["_x"]
            pre_ix = [i for i, t in enumerate(x) if int(t.split("_")[2]) <= -7]; post_ix = o["_post12"]; ant = x.index("ev_-3_-1")
            np.savez(RES / "r3_es_inputs.npz", b=o["_b"], V=o["_V"], terms=np.array(x), pre_ix=np.array(pre_ix), ant_ix=ant, post_ix=np.array(post_ix))
            own = {}
            for tag, steps in (("adjacent", [2, 3, 4, 5]), (f"gap_{g}_bins", [2 + g, 3 + g, 4 + g, 5 + g])):
                own[tag] = dict(steps=steps, **own_breakdown(o["_b"], o["_V"], pre_ix, post_ix, steps),
                                identified_set_breakdown=float(abs(o["post_1_12"]["theta_log"]) / (np.mean(steps) * rm_bounds(o["_b"], o["_V"], pre_ix, None, post_ix, steps, 1.0, nsim=1000)["maxd"])))
            o["own_rm"] = dict(median_construction_months_inclusive=gap_m, gap_bins=g, **own)
        res[f"event_long_{cl}"] = {k: v for k, v in o.items() if not k.startswith("_")}
        print(cl, "theta", round(o["post_1_12"]["pct"] * 100, 1), o["post_1_12"]["pct_ci"], "pretrend p", round(o["pretrend"]["p_F"], 3), "trend-adj", round(o["trend_adjusted"]["pct"] * 100, 1), flush=True)
    pd.concat(evs).to_csv(RES / "r3_event_long.csv", index=False)
    res["single_post_sigungu"] = single_post(B, E, "sigungu"); res["single_post_pid"] = single_post(B, E, "pid")
    for tag, kw in (("w12", {}), ("full", dict(fit_lo=72, treated_keep=TREATED_HEAD))):
        S = imp_setup(B, E, **kw)
        r = imp_estimate(S, B, kinds=("plain_pid", "plain_sig", "trend_pid", "trend_sig"))
        res[f"imputation_{tag}"] = r; print("imputation", tag, r, flush=True)
    hc = A.heating_carrier(B[B.pid.isin(TREATED_HEAD)], E)
    Bi = B.drop_duplicates("pid").set_index("pid")
    elec = set(hc[hc == "elec"].index); noren = set(Bi[Bi.renewable == 0].index); noboil = set(Bi[Bi.boiler == 0].index); boil = set(Bi[Bi.boiler == 1].index)
    multi = set(E[(E.carrier == "elec") & (E.n_items > 1)].pid); annex = set(Bi[Bi.reg_n_title > Bi.reg_n_main].index)
    subs = {"clean_elec_heat_no_renewable": elec & noren, "clean_plus_no_boiler": elec & noren & noboil, "boiler_entry": boil, "no_boiler_entry": noboil,
            "elec_heat": elec, "no_renewable": noren, "single_meter_no_annex": TREATED_HEAD - multi - annex}
    res["subsample_sizes"] = {k: len(v & TREATED_HEAD) for k, v in subs.items()}
    res["shared_meter_flags"] = dict(multi_record_months=len(multi & TREATED_HEAD), annex_buildings=len(annex & TREATED_HEAD), either=len((multi | annex) & TREATED_HEAD),
                                     controls_flagged=len((multi | annex) & set(Bi[Bi.role.isin(CR) & Bi.use_class.isin(HC)].index)))
    for k, v in subs.items():
        keep = v & TREATED_HEAD
        if len(keep) < 20: res[f"sub_{k}"] = dict(n=len(keep), status="not estimated (< 20 buildings)"); continue
        drop = None
        if k == "single_meter_no_annex": drop = (multi | annex)
        S = imp_setup(B, E, treated_keep=keep, drop=drop)
        r = imp_estimate(S, B, kinds=("plain_sig",)); res[f"sub_{k}"] = r; print(k, r, flush=True)
    bt = S0["building_tau"](f0).reset_index(); bt["pct"] = np.exp(bt.tau_mean) - 1
    Ec = A.carrier_frame(E, "elec"); Ec = Ec[Ec.pid.isin(TREATED_HEAD)].copy()
    pre_med = Ec[(Ec.ym < Ec.start_ym) & (mi(Ec.ym) >= mi(Ec.start_ym) - 12)].groupby("pid").kwh.median()
    po = Ec[(Ec.ym > Ec.end_ym) & (mi(Ec.ym) <= mi(Ec.end_ym) + 12)].copy(); po["low"] = po.kwh < 0.25 * po.pid.map(pre_med)
    low = po.groupby("pid").low.sum(); bt["low_post_months"] = bt.pid.map(low).fillna(0).astype(int)
    bt.to_csv(RES / "r3_building_gaps.csv", index=False)
    big = set(bt[bt.pct <= -0.5].pid)
    S = imp_setup(B, E, treated_keep=TREATED_HEAD - big); r = imp_estimate(S, B, kinds=("plain_sig",))
    hcs = hc.reindex(bt.pid).to_numpy(); bt["heat"] = hcs
    for m_ in ("hvac", "boiler", "renewable", "win", "wall"): bt[m_] = bt.pid.map(Bi[m_]).to_numpy()
    comp = {g: dict(n=int(len(d)), heat_elec=float((d.heat == "elec").mean()), hvac=float(d.hvac.mean()), boiler=float(d.boiler.mean()),
                    renewable=float(d.renewable.mean()), low_post_months_mean=float(d.low_post_months.mean()), with_3plus_low=int((d.low_post_months >= 3).sum()))
            for g, d in (("fall_ge50", bt[bt.pct <= -0.5]), ("fall_30_50", bt[(bt.pct > -0.5) & (bt.pct <= -0.3)]), ("other", bt[bt.pct > -0.3]))}
    Cc = A.carrier_frame(E, "elec"); Cc = Cc[Cc.pid.isin(set(Bi[Bi.role.isin(CR) & Bi.use_class.isin(HC) & (Bi.link_grade == "A_cand")].index))]
    Cc = Cc[Cc.ym < Cc.start_ym.map(lambda v: ym_add(int(v), -3))].sort_values(["pid", "ym"]); ref_low = []
    for pid, d in Cc.groupby("pid"):
        if len(d) < 24: continue
        last, prev = d.tail(12), d.iloc[-24:-12]; ref_low.append(int((last.kwh < 0.25 * prev.kwh.median()).sum()))
    ref_low = np.array(ref_low)
    res["distribution"] = dict(n=len(bt), fall_ge30=int((bt.pct <= -0.3).sum()), fall_ge50=int((bt.pct <= -0.5).sum()), rise_ge30=int((bt.pct >= 0.3).sum()),
                               median_pct=float(bt.pct.median()), buildings_with_low_post_month=int((bt.low_post_months > 0).sum()),
                               buildings_with_3plus_low_post_months=int((bt.low_post_months >= 3).sum()), without_fall_ge50=r, composition=comp,
                               comparison_reference=dict(n=int(len(ref_low)), share_with_low_month=float((ref_low > 0).mean()), share_with_3plus_low=float((ref_low >= 3).mean())))
    bt.to_csv(RES / "r3_building_gaps.csv", index=False)
    res["runtime_min"] = round((time.time() - t0) / 60, 1)
    (RES / "r3_main.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float)); print("done", res["runtime_min"], "min")


if __name__ == "__main__":
    main()

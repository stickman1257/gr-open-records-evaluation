"""Step 4: panel, quality rules, imputation and stacked DiD estimators."""
import os, sys, re, json
import numpy as np, pandas as pd, statsmodels.api as sm
sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, PROC, RES, sido_of, fit_dd_model, predict_dd
from s03_energy import ym_add, ym_range

MIN_WIN = int(os.environ.get("MIN_WIN_MONTHS", "11"))

def _mi(v): return (v // 100) * 12 + v % 100

def complete_windows(Ec, B, min_obs=None, window=12):
    """pids of treated buildings (role == 'treated' in B) with >= min_obs observed months of the carrier frame Ec in the pre"""
    min_obs = MIN_WIN if min_obs is None else int(min_obs)
    t = Ec[Ec.pid.isin(B[B.role == "treated"].pid) & Ec.start_ym.notna() & Ec.end_ym.notna()]
    if t.empty: return set()
    rs = _mi(t.ym) - _mi(t.start_ym.astype(int)); re_ = _mi(t.ym) - _mi(t.end_ym.astype(int))
    pre = t[(rs >= -window) & (rs <= -1)].groupby("pid").ym.nunique(); post = t[(re_ >= 1) & (re_ <= window)].groupby("pid").ym.nunique()
    return set(pre[pre >= min_obs].index) & set(post[post >= min_obs].index)

MEAS_MAP = {
    "wall": [r"단열\(벽", r"외벽"], "roof": [r"단열\(지붕", r"지붕\s*단열"], "cool_roof": [r"쿨루프", r"Cool Roof"], "win": [r"창호"],
    "shade": [r"일사조절", r"차양"], "hrv": [r"폐열\s*회수", r"전열\s*교환", r"열\s*회수"], "vent_generic": [r"환기"], "led": [r"조명", r"LED"],
    "hvac": [r"냉.?난방기", r"EHP", r"GHP"], "boiler": [r"보일러"], "renewable": [r"신.?재생", r"태양광"], "pv": [r"태양광"],
    "floor": [r"단열\(바닥", r"바닥단열"], "door": [r"^문$|,문,|문\b"],
}
SIM6 = ["wall", "roof", "win", "shade", "hrv", "led"]

def parse_measures(txt):
    t = str(txt)
    out = {k: int(any(re.search(p, t) for p in pats)) for k, pats in MEAS_MAP.items()}
    ents = [e.strip() for e in re.split(r"[,\n]", t)]
    out["wall"] = int(bool(re.search(r"단열\(벽", t)) or any(("외벽" in e) and not re.search(r"누수|방수", e) for e in ents))
    out["led"] = int(bool(re.search(r"조명|LED", re.sub(r"조명\s*제어\s*(장치)?", "", t))))
    return out

def regrade_buildings(B):
    """Recompute use class and link grade at analysis time with s02_link.regrade()."""
    import s02_link as L2
    B = B.copy()
    if "area_txt" not in B:
        g = pd.read_csv(PROC / "gr_projects_raw.csv", usecols=["year", "seq", "area_txt"]).drop_duplicates(["year", "seq"])
        B = B.merge(g, on=["year", "seq"], how="left")
    if "use_class" not in B: B["use_class"] = B.name_list.apply(L2.use_class)
    res = [L2.regrade(r.name_list, r.reg_use, r.reg_area, r.area_txt, r.reg_n_main, r.reg_is_collective, r.bdKdcd, r.reg_area_share) for r in B.itertuples()]
    B["use_class_v1"] = B.use_class; B["link_grade_v1"] = B.link_grade
    B["use_class"] = [x[0] for x in res]; B["link_grade"] = [x[1] for x in res]; B["regrade_flags"] = [";".join(x[2]) for x in res]
    return B

def original_grades(B):
    """Buildings with the link grades and use classes as stored in the file (before the regrade) - robustness only."""
    return B.assign(link_grade=B.link_grade_v1, use_class=B.use_class_v1) if "link_grade_v1" in B else B

def zone(sido, addr):
    """Approximate Korean energy-code climate zone (archetype climate)."""
    a = str(addr)
    if sido == "강원":
        return "중부2_서울" if re.search(r"강릉|동해|속초|삼척|고성|양양", a) else "중부1_춘천"
    if sido == "경기" and re.search(r"연천|포천|가평|남양주|의정부|양주|동두천|파주", a): return "중부1_춘천"
    if sido == "충북" and "제천" in a: return "중부1_춘천"
    if sido == "경북" and re.search(r"봉화|청송", a): return "중부1_춘천"
    if sido in ("서울", "인천", "대전", "세종", "경기", "충북", "충남", "전북"): return "중부2_서울"
    if sido == "경북" and re.search(r"영주|문경|안동|의성|영양|상주|군위|예천|김천|영덕", a): return "중부2_서울"
    if sido == "경남" and re.search(r"거창|함양", a): return "중부2_서울"
    if sido == "전남" and re.search(r"구례", a): return "중부2_서울"
    return "남부_부산"

def era(by):
    try: y = int(str(by)[:4])
    except Exception: return None
    return "E1" if y <= 2000 else ("E2" if y <= 2010 else "E3")

REG_SIDO = {"11": "서울", "26": "부산", "27": "대구", "28": "인천", "29": "광주", "30": "대전", "31": "울산", "36": "세종", "41": "경기", "42": "강원",
            "43": "충북", "44": "충남", "45": "전북", "46": "전남", "47": "경북", "48": "경남", "50": "제주"}

def load(dedup=None):
    """D23 (PI 11/14 02:19, approved by the user 11/14): one physical building (reg_pk) = one unit, earliest record kept"""
    global DEDUP_DROPPED
    if dedup is None: dedup = os.environ.get("DEDUP", "1") != "0"
    B = pd.read_csv(PROC / "analysis_buildings.csv", dtype={"sigunguCd": str})
    E = pd.read_csv(PROC / "energy_monthly.csv")
    W = pd.read_csv(PROC / "weather_monthly_sido.csv")
    B["sido"] = B.addr_query.apply(sido_of)
    na = B.sido.isna()
    if na.any(): B.loc[na, "sido"] = B.loc[na, "bdMgtSn"].astype(str).str[:2].map(REG_SIDO)
    B = regrade_buildings(B)
    if dedup:
        from a_dedup import dedup_physical
        B, DEDUP_DROPPED = dedup_physical(B)
    else:
        DEDUP_DROPPED = None
    B["zone"] = [zone(s, a) for s, a in zip(B.sido, B.addr_query)]
    M = pd.DataFrame([parse_measures(t) for t in B.measures_txt]); B = pd.concat([B.reset_index(drop=True), M], axis=1)
    B["pred_saving"] = pd.to_numeric(B.pred_saving_txt.astype(str).str.extract(r"([\d\.]+)\s*%")[0], errors="coerce")
    B["area_m2"] = pd.to_numeric(B.get("reg_area"), errors="coerce").fillna(B.gr_gfa)
    B["gr_gfa"] = B["area_m2"]
    E = E[E.status == "ok"].merge(B[["pid", "sido", "role", "start_ym", "end_ym", "gr_gfa"]], on="pid")
    E = E.merge(W, on=["sido", "ym"], how="left")
    E["phase"] = np.select([E.ym < E.start_ym, E.ym > E.end_ym], ["pre", "post"], "constr")
    E.loc[E.role.isin(["control", "aux_2023"]), "phase"] = np.where(E.loc[E.role.isin(["control", "aux_2023"]), "ym"]
                                                                    < E.loc[E.role.isin(["control", "aux_2023"]), "start_ym"], "pre", "constr")
    qa = []
    for (pid, car), d in E[E.phase == "pre"].groupby(["pid", "carrier"]):
        d = d.sort_values("ym").tail(12); inten = (d.kwh / d.gr_gfa).mean()
        cv = d.kwh.std() / d.kwh.mean() if d.kwh.mean() > 0 else 0
        if len(d) >= 6 and (cv < 0.01 or (car == "elec" and (inten < 0.5 or inten > 60))): qa.append((pid, car))
    QA_EXCL.update(qa)
    E = E[~pd.Series(list(zip(E.pid, E.carrier)), index=E.index).isin(qa)]
    return B, E

QA_EXCL = set()
DEDUP_DROPPED = None

def building_metrics(B, E):
    rows = []
    for (pid, car), d in E[E.role == "treated"].groupby(["pid", "carrier"]):
        b = B.set_index("pid").loc[pid]; s, e = int(b.start_ym), int(b.end_ym)
        pre12 = d[d.ym.isin(ym_range(ym_add(s, -12), ym_add(s, -1)))]; post12 = d[d.ym.isin(ym_range(ym_add(e, 1), ym_add(e, 12)))]
        r = dict(pid=pid, carrier=car, n_pre12=len(pre12), n_post12=len(post12))
        if len(pre12) >= MIN_WIN and len(post12) >= MIN_WIN and pre12.kwh.sum() > 0:
            cal = set(pre12.ym % 100) & set(post12.ym % 100); pm = pre12[(pre12.ym % 100).isin(cal)]; qm = post12[(post12.ym % 100).isin(cal)]
            r["n_match"] = len(cal)
            r["pre12"] = pm.kwh.sum() * 12 / len(cal); r["post12"] = qm.kwh.sum() * 12 / len(cal); r["chg_unadj"] = r["post12"] / r["pre12"] - 1
            base = d[(d.ym < s) & (d.ym >= ym_add(s, -24))]
            m = fit_dd_model(base)
            if m:
                pred = predict_dd(m, post12).sum()
                r.update(dd_form=m["form"], cvrmse=m["cvrmse"], nmbe=m["nmbe"], n_base=m["n"], sav_norm=1 - post12.kwh.sum() / pred,
                         pred_post12=pred)
        rows.append(r)
    return pd.DataFrame(rows)

def demean(df, y, x_cols, fe_cols, tol=1e-9, it=200):
    """Alternating projections to absorb several fixed effects (unbalanced panel)."""
    Z = df[[y] + x_cols].astype(float).copy()
    for _ in range(it):
        prev = Z.copy()
        for f in fe_cols: Z = Z - Z.groupby(df[f]).transform("mean")
        if (Z - prev).abs().to_numpy().max() < tol: break
    return Z

from common import COVID_MONTHS

def jumpy_buildings(E, carrier="elec", k=3.0):
    """Buildings with any monthly value > k x their own median or < median / k (pre-specified robustness filter)."""
    out = set()
    for pid, d in E[(E.carrier == carrier) & (E.kwh > 0)].groupby("pid"):
        m = d.kwh.median()
        if ((d.kwh > k * m) | (d.kwh < m / k)).any(): out.add(pid)
    return out

def unstable_meter_buildings(E, carrier="elec"):
    """Buildings whose number of meter records per month (n_items) changes over the observed months (review R9)."""
    if "n_items" not in E: return set()
    g = E[(E.carrier == carrier) & (E.kwh > 0)].groupby("pid").n_items.nunique()
    return set(g[g > 1].index)

def carrier_frame(E, carrier):
    """Monthly rows for 'elec', 'gas' or 'total'. total = elec + gas (both in kWh as delivered by the API) for buildings"""
    if carrier != "total":
        return E[(E.carrier == carrier) & (E.kwh > 0)].copy()
    el = E[(E.carrier == "elec") & (E.kwh > 0)]; ga = E[(E.carrier == "gas") & E.kwh.notna() & (E.kwh >= 0)]
    m = el.merge(ga[["pid", "ym", "kwh"]], on=["pid", "ym"], suffixes=("", "_gas"))
    m["kwh"] = m.kwh + m.kwh_gas; m["carrier"] = "total"
    return m.drop(columns=["kwh_gas"])

def time_key(S, B, fe):
    u = S.pid.map(B.set_index("pid").use_class).fillna("NA")
    if fe == "sido": return S.sido.fillna("NA")
    if fe == "use": return u
    if fe == "use_sido": return u + "|" + S.sido.fillna("NA")
    return pd.Series("all", index=S.index)

SE_PYFIXEST = "pyfixest default (CRV1; singleton cells dropped; small-sample factor G/(G-1) x (N-1)/(N-k) with non-nested absorbed effects counted in k; t(G-1) intervals)"
SE_STATSMODELS = "statsmodels cluster on demeaned data (absorbed effects not counted; normal intervals) - fallback, pyfixest unavailable"

def fe_fit(S, x, g):
    """y ~ x | unit_fe + time_fe with cluster-robust covariance (clusters g)."""
    from scipy import stats as _st
    k = len(x)
    try:
        import pyfixest as pf, warnings
        nm = [f"x{i}" for i in range(k)]
        d = pd.DataFrame({"y": S["y"].to_numpy(float), "unit_fe": S["unit_fe"].to_numpy(), "time_fe": S["time_fe"].to_numpy(), "cl": np.asarray(g).astype(str)})
        for n_, c in zip(nm, x): d[n_] = S[c].to_numpy(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = pf.feols("y ~ " + " + ".join(nm) + " | unit_fe + time_fe", data=d, vcov={"CRV1": "cl"},
                         demeaner=pf.MapDemeaner(fixef_tol=1e-10, fixef_maxiter=100000))
        names = list(m._coefnames); V0 = np.asarray(m._vcov); co = m.coef()
        b = np.full(k, np.nan); V = np.full((k, k), np.nan)
        pos = [nm.index(n_) for n_ in names]
        b[pos] = co.reindex(names).to_numpy(float); V[np.ix_(pos, pos)] = V0
        df = max(int(d.cl.nunique()) - 1, 1)
        return dict(b=b, cov=V, crit=float(_st.t.ppf(0.975, df)), pval=lambda t: 2 * _st.t.sf(np.abs(t), df), nobs=int(m._N), convention=SE_PYFIXEST)
    except Exception as ex:
        Z = demean(S, "y", x, ["unit_fe", "time_fe"])
        res = sm.OLS(Z["y"], Z[x]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(g)[0]})
        return dict(b=res.params.to_numpy(float), cov=np.asarray(res.cov_params()), crit=1.959964, pval=lambda t: 2 * _st.norm.sf(np.abs(t)),
                    nobs=int(res.nobs), convention=SE_STATSMODELS + f" [{type(ex).__name__}: {ex}]"[:120])


def stacked_did(B, E, carrier, window=12, measure_terms=None, event=False, grades=("A_cand",), fe="sido",
                drop=None, drop_months=None, ctrl_roles=("control",), uses=None, ctrl_trim=0, donut=0,
                cluster="pid", support=True, return_sample=False, min_win=None, ctrl_until=None, anticip=0):
    """Stacked DiD (Cengiz et al. 2019). Stacks = completion half-year cohorts."""
    Ec = carrier_frame(E, carrier)
    win_ok = None if min_win == 0 else complete_windows(Ec, B, min_win)
    if drop_months: Ec = Ec[~Ec.ym.isin(drop_months)]
    Ec["y"] = np.log(Ec.kwh / Ec.gr_gfa)
    Bg = B[B.link_grade.isin(grades)]
    if uses is not None: Bg = Bg[Bg.use_class.isin(uses)]
    if drop is not None: Bg = Bg[~Bg.pid.isin(drop)]
    T = Bg[Bg.role == "treated"].copy(); C = Bg[Bg.role.isin(ctrl_roles)].copy()
    if win_ok is not None: T = T[T.pid.isin(win_ok)]
    T["cohort"] = (T.end_ym // 100) * 10 + ((T.end_ym % 100) > 6).astype(int)
    stacks = []
    for coh, tc in T.groupby("cohort"):
        if len(tc) < 3: continue
        lo = ym_add(int(tc.start_ym.min()), -window); hi = ym_add(int(tc.end_ym.max()), window)
        et = Ec[Ec.pid.isin(tc.pid)].copy()
        s_lo = et.start_ym.map(lambda v: ym_add(int(v), -window)); s_hi = et.start_ym.map(lambda v: ym_add(int(v), -max(donut, anticip)))
        e_lo = et.end_ym.map(lambda v: ym_add(int(v), donut)); e_hi = et.end_ym.map(lambda v: ym_add(int(v), window))
        et = et[((et.ym >= s_lo) & (et.ym < s_hi)) | ((et.ym > e_lo) & (et.ym <= e_hi))]
        ec = Ec[Ec.pid.isin(C.pid) & (Ec.ym >= lo) & (Ec.ym <= hi)]
        ec = ec[ec.ym < ec.start_ym.map(lambda v: ym_add(int(v), -ctrl_trim))]
        if ctrl_until is not None: ec = ec[ec.ym <= int(ctrl_until)]
        st = pd.concat([et.assign(treat=1), ec.assign(treat=0)])
        st["post"] = ((st.treat == 1) & (st.ym > st.end_ym)).astype(int); st["stack"] = coh
        if event:
            rel = np.where(st.ym < st.start_ym, -((st.start_ym // 100 * 12 + st.start_ym % 100) - (st.ym // 100 * 12 + st.ym % 100)),
                           (st.ym // 100 * 12 + st.ym % 100) - (st.end_ym // 100 * 12 + st.end_ym % 100))
            st["rel"] = np.where(st.treat == 1, rel, 0)
        stacks.append(st)
    if not stacks: return None
    S = pd.concat(stacks, ignore_index=True)
    S["unit_fe"] = S.pid + "|" + S["stack"].astype(str)
    S["time_fe"] = time_key(S, B, fe) + "|" + S.ym.astype(str) + "|" + S["stack"].astype(str)
    n_treat_rows = int((S.treat == 1).sum())
    if support:
        has_c = S[S.treat == 0].time_fe.unique()
        S = S[(S.treat == 0) | S.time_fe.isin(has_c)].copy()
    support_share = float((S.treat == 1).sum() / max(n_treat_rows, 1))
    S = S[S.groupby("unit_fe").y.transform("size") > 1]
    x = ["post"]
    if measure_terms:
        for m in measure_terms:
            S = S.merge(B[["pid", m]], on="pid", how="left"); S[f"post_{m}"] = S.post * S[m].fillna(0); x.append(f"post_{m}")
    if event:
        bins = [(-12, -10), (-9, -7), (-6, -4), (-3, -1), (1, 3), (4, 6), (7, 9), (10, 12)]
        x = []
        for a, b in bins:
            if (a, b) == (-6, -4): continue
            c = f"ev_{a}_{b}"; S[c] = ((S.treat == 1) & (S.rel >= a) & (S.rel <= b)).astype(int); x.append(c)
    if return_sample: return S, x
    if cluster == "sigungu":
        g = S.pid.map(B.set_index("pid").sigunguCd.astype(str)).fillna(S.pid)
    else:
        g = S.pid
    F = fe_fit(S, x, g)
    q = F["crit"]; se = np.sqrt(np.diag(F["cov"]))
    out = pd.DataFrame({"term": x, "coef": F["b"], "se": se, "lo": F["b"] - q * se, "hi": F["b"] + q * se, "p": F["pval"](F["b"] / se)})
    out["carrier"] = carrier; out["n_obs"] = int(len(S)); out["n_obs_inference"] = F["nobs"]; out["n_treated"] = S[S.treat == 1].pid.nunique()
    out["n_control"] = S[S.treat == 0].pid.nunique(); out["n_stacks"] = S["stack"].nunique()
    out["support_share"] = support_share; out["n_clusters"] = int(pd.Series(g).nunique()); out["se_convention"] = F["convention"]
    out["pct_effect"] = np.exp(out.coef) - 1; out["pct_lo"] = np.exp(out.lo) - 1; out["pct_hi"] = np.exp(out.hi) - 1
    if not event and not measure_terms:
        try:
            xs = []
            for c in sorted(S["stack"].unique()):
                col = f"post_c{c}"; S[col] = ((S.post == 1) & (S["stack"] == c)).astype(int)
                if S[col].sum() > 0: xs.append((c, col))
            cols = [c for _, c in xs]
            Fc = fe_fit(S, cols, g)
            if not np.isfinite(Fc["b"]).all(): raise RuntimeError("a cohort coefficient is not estimable")
            nb = np.array([S[(S[col] == 1)].pid.nunique() for _, col in xs], float); w = nb / nb.sum()
            b = float(w @ Fc["b"]); sew = float(np.sqrt(w @ Fc["cov"] @ w)); qc = Fc["crit"]
            out["coef_cohort_w"] = b; out["se_cohort_w"] = sew; out["lo_cohort_w"] = b - qc * sew; out["hi_cohort_w"] = b + qc * sew
            out["pct_cohort_w"] = np.exp(b) - 1; out["pct_cohort_w_lo"] = np.exp(b - qc * sew) - 1; out["pct_cohort_w_hi"] = np.exp(b + qc * sew) - 1
            out["se_convention_cohort_w"] = Fc["convention"]
            out["cohort_coefs"] = json.dumps([dict(cohort=int(c), coef=float(v), n_bld=int(n), weight=float(x_)) for (c, _), v, n, x_ in zip(xs, Fc["b"], nb, w)])
        except Exception as ex:
            out["coef_cohort_w"] = np.nan; out["se_cohort_w"] = np.nan; out["cohort_coefs"] = f"failed: {type(ex).__name__}: {ex}"[:150]
    if event:
        pre = [c for c in x if c.startswith("ev_-") and c != "ev_-3_-1"]
        ix = [x.index(c) for c in pre]
        try:
            from scipy import stats as _st
            bb = F["b"][ix]; V = F["cov"][np.ix_(ix, ix)]
            out["pretrend_wald_p"] = float(_st.chi2.sf(float(bb @ np.linalg.solve(V, bb)), len(ix)))
        except Exception: out["pretrend_wald_p"] = np.nan
    if measure_terms:
        from statsmodels.stats.multitest import multipletests
        mm = out.term.str.startswith("post_")
        mm = mm & out.p.notna()
        if mm.any(): out.loc[mm, "p_holm"] = multipletests(out.loc[mm, "p"], method="holm")[1]
    return out


def imputation_setup(B, E, carrier, window=12, grades=("A_cand",), uses=("보건", "어린이집"), fe="use",
                     ctrl_roles=("control", "aux_2023"), ctrl_trim=3, donut=0, anticip=3, min_ctrl=1, drop=None,
                     min_win=None, treated_keep=None, fit_lo=None):
    """Data and weighted fit for the imputation estimator (Borusyak, Jaravel & Spiess 2024)."""
    Ec = carrier_frame(E, carrier); Ec["y"] = np.log(Ec.kwh / Ec.gr_gfa)
    Bg = B[B.link_grade.isin(grades)]
    if uses is not None: Bg = Bg[Bg.use_class.isin(uses)]
    if drop is not None: Bg = Bg[~Bg.pid.isin(drop)]
    T = Bg[Bg.role == "treated"]; C = Bg[Bg.role.isin(ctrl_roles)]
    if min_win != 0: T = T[T.pid.isin(complete_windows(Ec, T, min_win))]
    if treated_keep is not None: T = T[T.pid.isin(set(treated_keep))]
    flo = int(fit_lo) if fit_lo else window
    if fit_lo:
        t0 = Ec[Ec.pid.isin(T.pid)]; rs0 = _mi(t0.ym) - _mi(t0.start_ym.astype(int)); hi0 = -(max(anticip, donut) + 1)
        nfit = t0[(rs0 >= -flo) & (rs0 <= hi0)].groupby("pid").ym.nunique()
        T = T[T.pid.isin(set(nfit[nfit >= min(MIN_WIN, flo + hi0 + 1)].index))]
    et = Ec[Ec.pid.isin(T.pid)].copy()
    s_lo = et.start_ym.map(lambda v: ym_add(int(v), -flo)); s_hi = et.start_ym.map(lambda v: ym_add(int(v), -donut))
    e_lo = et.end_ym.map(lambda v: ym_add(int(v), donut)); e_hi = et.end_ym.map(lambda v: ym_add(int(v), window))
    et = et[((et.ym >= s_lo) & (et.ym < s_hi)) | ((et.ym > e_lo) & (et.ym <= e_hi))]
    et["post"] = (et.ym > et.end_ym).astype(int)
    et["ant"] = ((et.post == 0) & (et.ym >= et.start_ym.map(lambda v: ym_add(int(v), -anticip)))).astype(int)
    et["ctrl"] = 0
    ec = Ec[Ec.pid.isin(C.pid)].copy(); ec = ec[ec.ym < ec.start_ym.map(lambda v: ym_add(int(v), -ctrl_trim))]
    ec["post"] = 0; ec["ant"] = 0; ec["ctrl"] = 1
    D = pd.concat([et, ec], ignore_index=True)
    if D.empty or et.empty: return None
    D["tk"] = time_key(D, B, fe) + "|" + D.ym.astype(str)
    D["cohort"] = np.where(D.ctrl == 0, (D.end_ym // 100) * 10 + ((D.end_ym % 100) > 6).astype(int), 0).astype(int)
    idx = lambda ym: (ym // 100) * 12 + ym % 100
    D["rel"] = np.where(D.ctrl == 1, 0, np.where(D.post == 1, idx(D.ym) - idx(D.end_ym), idx(D.ym) - idx(D.start_ym))).astype(int)
    pi, pids = pd.factorize(D.pid); ki, tks = pd.factorize(D.tk); ci, cohs = pd.factorize(D.cohort)
    y = D.y.to_numpy(float); post = D.post.to_numpy() == 1; ant = D.ant.to_numpy() == 1; isc = D.ctrl.to_numpy() == 1
    nP, nK = len(pids), len(tks)
    Bi = B.drop_duplicates("pid").set_index("pid")
    use_p = Bi.use_class.reindex(pids).fillna("NA").to_numpy(); ctrl_p = np.zeros(nP, bool); ctrl_p[pi[isc]] = True
    state = {}

    def fit(w=None, tol=1e-12, it=3000):
        w = np.ones(nP) if w is None else np.asarray(w, float)
        wr = w[pi]; F = ~post & ~ant & (wr > 0)
        pf, kf, yf, wf = pi[F], ki[F], y[F], wr[F]
        dp = np.bincount(pf, wf, nP); dk = np.bincount(kf, wf, nK)
        from scipy.sparse import coo_matrix as _coo
        from scipy.sparse.linalg import lsqr
        n = len(yf); sw = np.sqrt(wf)
        X = _coo((np.r_[sw, sw], (np.r_[np.arange(n), np.arange(n)], np.r_[pf, nP + kf])), shape=(n, nP + nK)).tocsr()
        mu = float(np.average(yf, weights=wf)) if n else 0.0
        sol = lsqr(X, sw * (yf - mu), atol=tol, btol=tol, iter_lim=it * 10)[0] if n else np.zeros(nP + nK)
        a = np.where(dp > 0, sol[:nP] + mu, np.nan); l = np.where(dk > 0, sol[nP:], np.nan)
        nctrl = np.bincount(ki[F & isc], minlength=nK)
        sup = (nctrl >= min_ctrl)[ki]
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        g = coo_matrix((np.ones(len(pf)), (pf, nP + kf)), shape=(nP + nK, nP + nK))
        comp = connected_components(g, directed=False)[1]
        conn = comp[pi] == comp[nP + ki]
        res = y - a[pi] - l[ki]
        ok = sup & conn & np.isfinite(res) & (wr > 0)
        return dict(res=res, ok=ok, wr=wr, a=a, l=l, nctrl=nctrl, F=F)

    def att(f, mask=None):
        """Cohort ATTs (mean tau over supported treated post building-months) averaged with treated-count weights."""
        m = post & f["ok"] if mask is None else mask & f["ok"]
        if not m.any(): return np.nan, np.nan, []
        num = np.bincount(ci[m], f["wr"][m] * f["res"][m], len(cohs)); den = np.bincount(ci[m], f["wr"][m], len(cohs))
        seen = pd.DataFrame({"c": ci[m], "p": pi[m]}).drop_duplicates()
        w_p = np.zeros(nP); w_p[pi[m]] = f["wr"][m]
        nb = np.bincount(seen.c.to_numpy(), w_p[seen.p.to_numpy()], len(cohs))
        k = den > 0; ac = num[k] / den[k]
        coh = [dict(cohort=int(cohs[j]), att=float(num[j] / den[j]), n_bld=float(nb[j]), weight=float(nb[j] / nb[k].sum()))
               for j in np.flatnonzero(k)]
        return float((ac * nb[k]).sum() / nb[k].sum()), float((f["wr"][m] * f["res"][m]).sum() / f["wr"][m].sum()), coh

    def building_tau(f):
        m = post & f["ok"]
        g = pd.DataFrame({"pid": pids[pi[m]], "tau": f["res"][m], "cohort": cohs[ci[m]]}).groupby("pid")
        return g.agg(tau_mean=("tau", "mean"), n_post=("tau", "size"), cohort=("cohort", "first"))

    return dict(D=D, pids=pids, pi=pi, ki=ki, tks=tks, post=post, ant=ant, isc=isc, use_p=use_p, ctrl_p=ctrl_p, fit=fit, att=att,
                building_tau=building_tau, n_treated_total=int(et.pid.nunique()), carrier=carrier, min_ctrl=min_ctrl,
                T=T, C=C, anticip=max(anticip, donut), window=window)


def boot_weights(S, rng, cluster=None):
    """Bootstrap multiplicities per building, stratified by role group (treated / control pool) x use type."""
    w = np.zeros(len(S["pids"]))
    strata = pd.Series(S["use_p"]).astype(str) + "|" + pd.Series(S["ctrl_p"]).astype(str)
    for _, ix in strata.groupby(strata).groups.items():
        ix = np.asarray(ix)
        if cluster is None:
            np.add.at(w, rng.choice(ix, len(ix), replace=True), 1)
        else:
            cl = pd.Series(np.asarray(cluster)[ix]); g = cl.groupby(cl).groups; keys = list(g)
            for k in rng.choice(len(keys), len(keys), replace=True): np.add.at(w, ix[np.asarray(g[keys[k]])], 1)
    return w


def imputation_pretrend(S, f, B):
    """Pre-trend test in the spirit of Borusyak et al. (2024): on untreated observations only (the fit sample), regress"""
    D = S["D"][f["F"]].copy()
    hi = -(S["anticip"] + 1); lo = -S["window"]
    edges = [(a, min(a + 2, hi)) for a in range(lo, hi + 1, 3)]
    if len(edges) < 2: return None
    x = []
    for a, b in edges[1:]:
        c = f"lead_{a}_{b}"; D[c] = ((D.ctrl == 0) & (D.rel >= a) & (D.rel <= b)).astype(int)
        if D[c].sum() > 0: x.append(c)
    D = D[D.groupby("pid").y.transform("size") > 1]
    if not x or D.empty: return None
    try:
        from scipy import stats as _st
        D = D.assign(unit_fe=D.pid, time_fe=D.tk).reset_index(drop=True)
        F = fe_fit(D, x, D.pid)
        b, V = F["b"], F["cov"]; k = np.isfinite(b)
        if not k.any(): return dict(error="no estimable lead term")
        G = int(D.pid.nunique()); q = int(k.sum()); bb = b[k]; W = float(bb @ np.linalg.solve(V[np.ix_(k, k)], bb))
        pj = float(_st.f.sf(W / q, q, max(G - 1, 1))) if F["convention"] == SE_PYFIXEST else float(_st.chi2.sf(W, q))
        se = np.sqrt(np.diag(V))
        return dict(reference=f"lead_{edges[0][0]}_{edges[0][1]}", terms=x, coef=[float(v) for v in b], se=[float(v) for v in se],
                    p=[float(v) for v in F["pval"](b / se)], p_joint=pj, wald=W, df=[q, G - 1], n_obs=int(F["nobs"]), n_buildings=G,
                    se_convention=F["convention"], joint_test="Wald / q against F(q, G-1)" if F["convention"] == SE_PYFIXEST else "Wald against chi2(q)")
    except Exception as ex:
        return dict(error=f"{type(ex).__name__}: {ex}"[:160])


def imputation_did(B, E, carrier, window=12, grades=("A_cand",), uses=("보건", "어린이집"), fe="use",
                   ctrl_roles=("control", "aux_2023"), ctrl_trim=3, nboot=200, seed=0, donut=0, return_tau=False,
                   anticip=3, min_ctrl=1, drop=None, pretrend=True, min_win=None, treated_keep=None, return_draws=False, fit_lo=None):
    """Imputation estimator; see imputation_setup for the fit sample and the support rule."""
    S = imputation_setup(B, E, carrier, window=window, grades=grades, uses=uses, fe=fe, ctrl_roles=ctrl_roles, ctrl_trim=ctrl_trim,
                         donut=donut, anticip=anticip, min_ctrl=min_ctrl, drop=drop, min_win=min_win, treated_keep=treated_keep, fit_lo=fit_lo)
    if S is None: raise RuntimeError("imputation_did: no treated rows" + (f" (no treated building with >= {MIN_WIN} of the 12 fit months s-{fit_lo}..s-{max(anticip, donut) + 1}; needs pass-2 months)" if fit_lo else ""))
    f = S["fit"](); a, a_row, coh = S["att"](f)
    post, ant = S["post"], S["ant"]
    gap = S["att"](f, ant)[1] if ant.any() else np.nan
    mon = (S["D"].ym.to_numpy() % 100)
    SEAS = {"heating_nov_mar": np.isin(mon, [11, 12, 1, 2, 3]), "cooling_jun_sep": np.isin(mon, [6, 7, 8, 9]), "shoulder_apr_may_oct": np.isin(mon, [4, 5, 10])}
    nP_ = len(S["pids"])
    def med_bld(ff):
        mm = post & ff["ok"]
        if not mm.any(): return np.nan
        cnt = np.bincount(S["pi"][mm], ff["wr"][mm], nP_); tb = np.bincount(S["pi"][mm], ff["wr"][mm] * ff["res"][mm], nP_)
        k = cnt > 0; v = tb[k] / cnt[k]; mult = np.zeros(nP_); mult[S["pi"][mm]] = ff["wr"][mm]; wv = mult[k]
        o = np.argsort(v); cw = np.cumsum(wv[o]); return float(v[o][np.searchsorted(cw, cw[-1] / 2)])
    seas0 = {k: S["att"](f, post & v)[0] for k, v in SEAS.items()}; med0 = med_bld(f)
    rng = np.random.default_rng(seed); bs, bg, bseas, bmed = [], [], {k: [] for k in SEAS}, []
    for _ in range(nboot):
        fb = S["fit"](boot_weights(S, rng), tol=1e-9)
        bs.append(S["att"](fb)[0]); bg.append(S["att"](fb, ant)[1] if ant.any() else np.nan)
        for k, v in SEAS.items(): bseas[k].append(S["att"](fb, post & v)[0])
        bmed.append(med_bld(fb))
    bs = np.array(bs, float); bs = bs[np.isfinite(bs)]; bg = np.array(bg, float); bg = bg[np.isfinite(bg)]
    pc = lambda v, q: float(np.percentile(v, q)) if len(v) else np.nan
    fin = lambda v: (lambda x: x[np.isfinite(x)])(np.array(v, float))
    m = post & f["ok"]; D = S["D"]
    fitc = f["F"] & S["isc"]
    out = dict(carrier=carrier, att_log=float(a), pct=float(np.exp(a) - 1), ci_log=[pc(bs, 2.5), pc(bs, 97.5)],
               pct_ci=[float(np.exp(pc(bs, 2.5)) - 1), float(np.exp(pc(bs, 97.5)) - 1)],
               att_log_rowmean=float(a_row), aggregation="cohort ATTs, treated-count weights", nboot_ok=int(len(bs)),
               n_treated=int(pd.unique(S["pi"][m]).size), n_treated_total=S["n_treated_total"],
               n_control=int(pd.unique(S["pi"][fitc]).size),
               n_control_by_role={r: int(n) for r, n in B[B.pid.isin(S["pids"][pd.unique(S["pi"][fitc])])].role.value_counts().items()},
               support_rows=float(m.sum() / max(int(post.sum()), 1)), min_ctrl=int(min_ctrl), anticip=int(S["anticip"]),
               min_win=int(MIN_WIN if min_win is None else min_win), bootstrap="buildings, stratified by role (treated / control pool) x use type, full refit",
               n_treated_by_use={str(u): int(n) for u, n in pd.Series(S["use_p"][pd.unique(S["pi"][m])]).value_counts().items()},
               n_control_by_use={str(u): int(n) for u, n in pd.Series(S["use_p"][pd.unique(S["pi"][fitc])]).value_counts().items()},
               anticipation_gap=float(gap), anticipation_gap_ci=[pc(bg, 2.5), pc(bg, 97.5)],
               pretrend=imputation_pretrend(S, f, B) if pretrend else None, cohorts=coh)
    out["pretrend_p"] = (out["pretrend"] or {}).get("p_joint")
    out["nboot"] = int(nboot); out["fit_window"] = f"s-{fit_lo or window}..s-{S['anticip'] + 1}"
    bm = fin(bmed)
    out["median_bld_log"] = float(med0); out["median_bld_pct"] = float(np.exp(med0) - 1) if np.isfinite(med0) else np.nan
    out["median_bld_pct_ci"] = [float(np.exp(pc(bm, 2.5)) - 1), float(np.exp(pc(bm, 97.5)) - 1)]
    out["season"] = {}
    for k, v in SEAS.items():
        b = fin(bseas[k]); mk = post & v & f["ok"]
        out["season"][k] = dict(att_log=float(seas0[k]), pct=float(np.exp(seas0[k]) - 1) if np.isfinite(seas0[k]) else np.nan,
                                pct_ci=[float(np.exp(pc(b, 2.5)) - 1), float(np.exp(pc(b, 97.5)) - 1)],
                                n_treated=int(pd.unique(S["pi"][mk]).size), n_rows=int(mk.sum()), label="exploratory (D21)")
        out[f"pct_{k}"] = out["season"][k]["pct"]; out[f"pct_{k}_lo"], out[f"pct_{k}_hi"] = out["season"][k]["pct_ci"]
    if return_draws: out["_draws"] = bs
    if return_tau:
        P = D[m].copy(); P["tau"] = f["res"][m]
        return out, P
    return out


def pre_intensity(B, E, carrier, min_months=None):
    """Annualised pre-period intensity (kWh/m2/y) of treated buildings: 12 x mean monthly kWh over the observed months in"""
    e = carrier_frame(E[E.role == "treated"], carrier)
    e = e[(e.ym >= e.start_ym.map(lambda v: ym_add(int(v), -12))) & (e.ym < e.start_ym)]
    g = e.groupby("pid").agg(n=("kwh", "size"), m=("kwh", "mean"), area=("gr_gfa", "first"))
    g = g[g.n >= (MIN_WIN if min_months is None else min_months)]
    return (12 * g.m / g.area).rename("pre_int"), g.area


PRIMARY_IMP = dict(uses=("보건", "어린이집"), fe="use", ctrl_roles=("control", "aux_2023"), ctrl_trim=3)

def program_ratio_chain(B, E, R, carrier="elec", nboot=200, seed=0, min_report=30, uses=None, sample="pooled"):
    """Metered-to-simulated ratio Q (exploratory; review R2):"""
    S = imputation_setup(B, E, carrier, **(dict(PRIMARY_IMP, uses=tuple(uses)) if uses else PRIMARY_IMP))
    if S is None: return [], pd.Series(dtype=float), None
    pids, pi, post = S["pids"], S["pi"], S["post"]; nP = len(pids)
    pint, area = pre_intensity(B, E, carrier)
    Bi = B.drop_duplicates("pid").set_index("pid")
    r = R[R.carrier == carrier].drop_duplicates("pid").set_index("pid")
    col = lambda c: pd.to_numeric(r[c], errors="coerce").reindex(pids).to_numpy(float) if c in r else np.full(nP, np.nan)
    phi, phe, phs = col("pred_total"), col("phi_env"), col("phi_sys")
    pin = pint.reindex(pids).to_numpy(float); ar = area.reindex(pids).to_numpy(float)
    flag = {k: Bi[k].reindex(pids).fillna(0).to_numpy(float) for k in ["renewable", "hvac", "boiler"]}
    ps = pd.to_numeric(Bi.pred_saving.reindex(pids), errors="coerce").to_numpy(float); ps = np.where((ps > 0) & (ps <= 100), ps, np.nan)
    end = pd.to_numeric(Bi.end_ym.reindex(pids), errors="coerce").fillna(0).astype(int)
    coh = np.where(S["ctrl_p"], Bi.year.reindex(pids).astype(str), ((end // 100) * 10 + ((end % 100) > 6)).astype(str))
    cluster = (Bi.sido.reindex(pids).fillna("NA").astype(str) + "|" + coh).to_numpy()
    SUB = ["all", "no_renewable", "no_renewable_no_hvac_boiler"]

    def stats(f, w):
        m = post & f["ok"]
        cnt = np.bincount(pi[m], minlength=nP)
        tb = np.bincount(pi[m], f["res"][m], nP) / np.where(cnt > 0, cnt, np.nan)
        sav = (1 - np.exp(tb)) * pin
        base = np.isfinite(sav) & np.isfinite(phi) & (phi > 0) & np.isfinite(ar) & (w > 0)
        out = {}
        for name, k in zip(SUB, [base, base & (flag["renewable"] == 0), base & (flag["renewable"] == 0) & (flag["hvac"] == 0) & (flag["boiler"] == 0)]):
            den = (w[k] * phi[k] * ar[k]).sum()
            out[name] = ((w[k] * sav[k] * ar[k]).sum() / den if den > 0 else np.nan, int(k.sum()),
                         float((w[k] * sav[k] * ar[k]).sum()), float(den))
        k = np.isfinite(tb) & np.isfinite(ps) & (w > 0)
        out["vs_program_prediction"] = (100 * (np.exp((w[k] * tb[k]).sum() / w[k].sum()) - 1) if k.any() else np.nan, int(k.sum()),
                                        float(np.median(np.repeat(ps[k], w[k].astype(int)))) if k.any() else np.nan, np.nan)
        k = base & np.isfinite(phe) & np.isfinite(phs); rho = (np.nan, np.nan, int(k.sum()))
        if k.sum() >= 15:
            X = np.column_stack([phe[k], phs[k], flag["hvac"][k], flag["renewable"][k]]); sw = np.sqrt(w[k])
            try:
                c = np.linalg.lstsq(X * sw[:, None], sav[k] * sw, rcond=None)[0]; rho = (float(c[0]), float(c[1]), int(k.sum()))
            except Exception: pass
        return out, sav, rho
    f0 = S["fit"](); st0, sav0, rho0 = stats(f0, np.ones(nP))
    draws = {}
    for tag, cl, sd in [("building", None, seed), ("cluster", cluster, seed + 1)]:
        rng = np.random.default_rng(sd); bs = {k: [] for k in st0}; br = []
        for _ in range(nboot):
            w = boot_weights(S, rng, cl); sb, _, rb = stats(S["fit"](w, tol=1e-9), w)
            for k in sb: bs[k].append(sb[k][0])
            br.append(rb[:2])
        draws[tag] = (bs, np.array(br, float).reshape(-1, 2))
    def ci(v):
        b = np.array(v, float); b = b[np.isfinite(b)]
        return ([float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] if len(b) else [np.nan, np.nan]), int(len(b))
    rows = []
    for k, (v, n, num, den) in st0.items():
        c, nb = ci(draws["building"][0][k]); cc, _ = ci(draws["cluster"][0][k])
        row = dict(sample=sample, carrier=carrier, subset=k, n=n, nboot_ok=nb, ci=c, ci_cluster=cc, reportable=bool(n >= min_report), estimable=bool(n >= min_report),
                   bootstrap="ci: buildings resampled within role x use type (same as the ATT), imputation refitted in every draw; "
                             "ci_cluster: province x cohort clusters within the same strata")
        if k == "vs_program_prediction":
            row.update(att_pct=float(v), median_predicted_saving_pct=num,
                       note="ATT (%) of this carrier (metered site energy) vs program-predicted saving (%) of regulated primary energy; selected subsample")
        else:
            row.update(ratio=float(v), realised_kwh=num, predicted_kwh=den,
                       note="could be read as a realisation rate" if k == "no_renewable_no_hvac_boiler" else "ratio of attributable change to the simulated saving of the measures in the decision set (wall, roof, window, shading, LED; "
                            "heat-recovery ventilation is not identifiable from the records and is 0 for every building in the primary coding); "
                            "HVAC, boiler and renewable-entry effects enter the numerator only")
        rows.append(row)
    sav = pd.Series(sav0, index=pids, name="sav_imp").dropna()
    bd = draws["building"][1]; okd = bd[np.isfinite(bd).all(1)] if len(bd) else bd
    k = np.isfinite(phe) & np.isfinite(phs) & np.isfinite(sav0)
    rho = dict(sample=sample, carrier=carrier, n=rho0[2], rho_env=rho0[0], rho_sys=rho0[1],
               ci_env=ci(bd[:, 0])[0] if len(bd) else None, ci_sys=ci(bd[:, 1])[0] if len(bd) else None, nboot_ok=int(len(okd)),
               corr_phi_env_phi_sys=float(np.corrcoef(phe[k], phs[k])[0, 1]) if k.sum() >= 3 else np.nan,
               draws=bd, q_draws=np.column_stack([np.array(draws["building"][0][c], float) for c in SUB]),
               label="exploratory; least squares of R_i on phi_env, phi_sys, hvac, renewable (no intercept), chain bootstrap (buildings, role x use strata)")
    return rows, sav, rho


def building_did(B, E, carrier, window=12, min_months=9, min_same_prov=3):
    """Per treated building: change in mean log(kWh/day) from its pre window [s-12,s-1] to post window [e+1,e+12]"""
    Ec = E[(E.carrier == carrier) & (E.kwh > 0)].copy(); Ec["y"] = np.log(Ec.kwh / Ec.days)
    Bi = B.set_index("pid")
    C = B[(B.role == "control") & (B.link_grade == "A_cand")]
    ctrl = {pid: d.set_index("ym").y for pid, d in Ec[Ec.pid.isin(C.pid)].groupby("pid")}
    cstart = C.set_index("pid").start_ym.to_dict(); cprov = C.set_index("pid").sido.to_dict()
    rows = []
    for pid, d in Ec[Ec.pid.isin(B[(B.role == "treated") & (B.link_grade == "A_cand")].pid)].groupby("pid"):
        b = Bi.loc[pid]; s0, e0 = int(b.start_ym), int(b.end_ym)
        P = ym_range(ym_add(s0, -window), ym_add(s0, -1)); Q = ym_range(ym_add(e0, 1), ym_add(e0, window))
        yi = d.set_index("ym").y
        if yi.reindex(P).notna().sum() < min(window, MIN_WIN) or yi.reindex(Q).notna().sum() < min(window, MIN_WIN): continue
        cal = {m % 100 for m in yi.reindex(P).dropna().index} & {m % 100 for m in yi.reindex(Q).dropna().index}
        P = [m for m in P if m % 100 in cal]; Q = [m for m in Q if m % 100 in cal]
        di = yi.reindex(Q).mean() - yi.reindex(P).mean()
        def cchg(pids):
            out = []
            for j in pids:
                yj = ctrl[j]; lim = cstart.get(j, 999999)
                pj = [m for m in P if m < lim]; qj = [m for m in Q if m < lim]
                a, c = yj.reindex(pj).dropna(), yj.reindex(qj).dropna()
                if len(a) >= min_months and len(c) >= min_months: out.append(c.mean() - a.mean())
            return out
        same = cchg([j for j in ctrl if cprov.get(j) == b.sido]); allc = cchg(list(ctrl))
        cc = same if len(same) >= min_same_prov else allc
        if not cc: continue
        delta = di - float(np.mean(cc))
        pre_kwh = np.exp(yi.reindex(P)).mul(E[(E.pid == pid) & (E.carrier == carrier)].set_index("ym").days.reindex(P)).sum() * 12 / len(P)
        rows.append(dict(pid=pid, carrier=carrier, delta=delta, n_ctrl=len(cc), ctrl_scope="province" if cc is same else "all",
                         pre12=float(pre_kwh), area=float(b.gr_gfa), sav_did=float((1 - np.exp(delta)) * pre_kwh / b.gr_gfa)))
    return pd.DataFrame(rows)

def heating_carrier(B, E, gas_ratio=2.0, elec_ratio=1.3):
    """Pre-period heating class: gas (winter/summer gas ratio >= gas_ratio), elec (winter/shoulder electricity ratio >= elec_ratio),"""
    out = {}
    for pid, b in B.set_index("pid").iterrows():
        e = E[(E.pid == pid) & (E.phase == "pre")]
        g = e[e.carrier == "gas"]; el = e[e.carrier == "elec"]
        g = g[g.kwh.notna() & (g.kwh >= 0)]
        if len(g) >= 10 and g.kwh.sum() > 0:
            gw = g[(g.ym % 100).isin([12, 1, 2])].kwh.mean(); gs = g[(g.ym % 100).isin([6, 7, 8])].kwh.mean()
            if gw > 0 and gs >= 0 and (gs == 0 or gw / gs >= gas_ratio): out[pid] = "gas"; continue
        if len(el) >= 10:
            w = el[(el.ym % 100).isin([12, 1, 2])].kwh.mean(); sh = el[(el.ym % 100).isin([4, 5, 10])].kwh.mean()
            out[pid] = "elec" if (sh > 0 and w / sh >= elec_ratio) else "other"
        else: out[pid] = "unknown"
    return pd.Series(out, name="heat_carrier")

COP_HEAT = 2.5
HC = {}
BD = {}
def realisation(B, M1, carrier_cols=("elec", "gas")):
    """For each treated building with per-building normalised savings (kWh/m2/y), regress measured savings on"""
    import c01_decision as C
    g = C.load_gt(); contrib = C.build_contrib(g)
    rows = []
    for r in M1.itertuples():
        b = B.set_index("pid").loc[r.pid]
        use = {"어린이집": "어린이집", "경로당": "경로당", "보건": "보건소"}.get(b.use_class)
        er = era(b.built_year)
        if use is None or er is None or b.zone is None or b.link_grade != "A_cand": continue
        key = f"{use}|{er}|{b.zone}"
        if key not in contrib: continue
        c = contrib[key]; bb = c["b"]
        lv = {v: int(b[v]) for v in SIM6}
        j = np.flatnonzero(np.all(np.column_stack([bb[v].to_numpy() == lv[v] for v in SIM6]), axis=1))
        if len(j) != 1: continue
        hc = HC.get(r.pid, "unknown")
        if r.carrier == "elec":
            phi = c["phi_e"][j[0]].copy()
            if hc == "elec": phi = phi + c["phi_g"][j[0]] * C.ETA / COP_HEAT
        else:
            if hc != "gas": continue
            phi = c["phi_g"][j[0]]
        meas = r.sav_norm * r.pre12 / b.gr_gfa if pd.notna(getattr(r, "sav_norm", np.nan)) and pd.notna(getattr(r, "pre12", np.nan)) else np.nan
        sd = BD.get((r.pid, r.carrier), np.nan)
        if pd.isna(meas) and pd.isna(sd): continue
        rows.append(dict(pid=r.pid, carrier=r.carrier, key=key, heat_carrier=hc, measured_dd=meas, measured=sd, pred_total=float(phi.sum()),
                         phi_env=float(phi[:4].sum()), phi_sys=float(phi[4:].sum()),
                         **{f"phi_{v}": float(phi[k]) for k, v in enumerate(SIM6)}, **{v: int(b[v]) for v in ["hvac", "boiler", "renewable", "cool_roof", "vent_generic", "floor"]}))
    return pd.DataFrame(rows)


def fit_rho(R, carrier="elec", nboot=1000, seed=0, ycol="measured"):
    """measured_i = rho_env*phi_env_i + rho_sys*phi_sys_i + b_hvac*hvac_i + b_ren*renewable_i + e_i (no intercept)."""
    import statsmodels.api as sm
    d = R[R.carrier == carrier].dropna(subset=[ycol]).copy()
    X = d[["phi_env", "phi_sys", "hvac", "renewable"]].astype(float); y = d[ycol]
    fit = sm.OLS(y, X).fit(cov_type="HC3")
    rng = np.random.default_rng(seed); bs = []
    for _ in range(nboot):
        i = rng.integers(0, len(d), len(d))
        try: bs.append(np.linalg.lstsq(X.values[i], y.values[i], rcond=None)[0])
        except Exception: pass
    bs = np.array(bs)
    return dict(n=int(len(d)), coef=fit.params.to_dict(), se=fit.bse.to_dict(),
                ci=dict(zip(X.columns, [[float(np.percentile(bs[:, k], 2.5)), float(np.percentile(bs[:, k], 97.5))] for k in range(X.shape[1])])),
                boot=bs)

def program_ratio(R, B, carrier="elec", nboot=1000, seed=0):
    """Program-level realised-to-predicted ratio: sum_i saving_i*area_i / sum_i phi_total_i*area_i (kWh),"""
    d = R[(R.carrier == carrier)].dropna(subset=["measured"]).merge(B[["pid", "gr_gfa", "sido", "end_ym"]], on="pid")
    if len(d) < 10: return None
    d["num"] = d.measured * d.gr_gfa; d["den"] = d.pred_total * d.gr_gfa
    d["cl"] = d.sido.fillna("NA") + "|" + (d.end_ym // 100).astype(str)
    ratio = d.num.sum() / d.den.sum()
    rng = np.random.default_rng(seed); cls = d.cl.unique(); bs = []
    G = {c: d[d.cl == c][["num", "den"]].sum() for c in cls}
    for _ in range(nboot):
        pick = rng.choice(cls, len(cls), replace=True)
        num = sum(G[c].num for c in pick); den = sum(G[c].den for c in pick)
        if den > 0: bs.append(num / den)
    return dict(carrier=carrier, n=int(len(d)), n_clusters=int(len(cls)), ratio=float(ratio),
                ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                predicted_kwh=float(d.den.sum()), realised_kwh=float(d.num.sum()))

def fit_rho_variants(R, carrier="elec"):
    """Exploratory: (rho_env, rho_sys) with and without intercept and with use fixed effects, in percent."""
    import statsmodels.formula.api as smf
    d = R[R.carrier == carrier].dropna(subset=["measured"]).copy()
    if len(d) < 20: return []
    d["use"] = d.key.str.split("|").str[0]; out = []
    for name, f in [("no_intercept", "measured ~ 0 + phi_env + phi_sys + hvac + renewable"),
                    ("intercept", "measured ~ phi_env + phi_sys + hvac + renewable"),
                    ("use_fe", "measured ~ 0 + C(use) + phi_env + phi_sys + hvac + renewable")]:
        r = smf.ols(f, data=d).fit(cov_type="HC3")
        for t in ["phi_env", "phi_sys"]:
            out.append(dict(carrier=carrier, spec=name, term=t, rho_pct=100 * r.params[t], lo=100 * r.conf_int().loc[t, 0],
                            hi=100 * r.conf_int().loc[t, 1], n=int(r.nobs)))
    return out

if __name__ == "__main__":
    B, E = load()
    W = ("보건", "어린이집"); CR = ("control", "aux_2023")
    M1 = building_metrics(B, E); M1.to_csv(RES / "G1_building_metrics.csv", index=False)
    prim = dict(uses=W, ctrl_roles=CR, fe="use", ctrl_trim=3, anticip=3)
    prim_ev = dict(prim, anticip=0)
    NOTE_PLAN = ("feasible version of the originally planned specification: 2024-25 selections as controls with control months up to 2024-06, "
                 "province x month x stack effects, no support rule. Deviation: no senior-centre controls were collected, so the planned "
                 "comparison (all later-selected controls including senior centres) is not estimable; window rule, quality gate and s.e. convention as elsewhere")
    rows = []
    for car in ["elec", "total", "gas"]:
        for name, kw, note in [("primary_within_use", prim, "primary stacked specification; treated months -3..-1 excluded"),
                               ("primary_within_use_with_anticipation_months", prim_ev, "as primary but treated months -3..-1 kept (earlier specification)"),
                               ("originally_planned_feasible", dict(fe="sido", support=False, ctrl_until=202406), NOTE_PLAN)]:
            try:
                r = stacked_did(B, E, car, **kw)
                if r is not None: rows.append(r.assign(spec=name, note=note))
            except Exception as ex: print("stacked failed", name, car, type(ex).__name__, ex)
    D = pd.concat(rows); D.to_csv(RES / "G2_did.csv", index=False); print(D[["spec", "carrier", "pct_effect", "pct_lo", "pct_hi", "n_treated", "n_control", "support_share"]].to_string())
    imp = []; NB = int(os.environ.get("IMP_NBOOT", "200")); NBH = int(os.environ.get("IMP_NBOOT_HEAD", "1000"))
    HEAD = {"imputation_use", "imputation_health_only", "imputation_daycare_only"}
    JUMPY = jumpy_buildings(E); HC_ = ("보건",); MIN_FITWIN_N = 30
    IMP_SPECS = [("imputation_use", dict(fe="use")),
                 ("imputation_health_only", dict(fe="use", uses=("보건",))), ("imputation_daycare_only", dict(fe="use", uses=("어린이집",))),
                 ("imputation_controls_2024_25_only", dict(fe="use", ctrl_roles=("control",))),
                 ("imputation_drop_renewable", dict(fe="use", drop=set(B[B.renewable == 1].pid))),
                 ("imputation_donut3", dict(fe="use", donut=3)),
                 ("imputation_use_sido", dict(fe="use_sido")),
                 ("imputation_use_minctrl5", dict(fe="use", min_ctrl=5)), ("imputation_use_minctrl10", dict(fe="use", min_ctrl=10)),
                 ("imputation_strict_12_12", dict(fe="use", min_win=12)),
                 ("imputation_use_no_anticip_holdout", dict(fe="use", anticip=0)),
                 ("imputation_drop_jumpy", dict(fe="use", drop=JUMPY)),
                 ("imputation_fit_s15_s4", dict(fe="use", fit_lo=15)),
                 ("imputation_health_controls_2024_25_only", dict(fe="use", uses=HC_, ctrl_roles=("control",))),
                 ("imputation_health_drop_renewable", dict(fe="use", uses=HC_, drop=set(B[B.renewable == 1].pid))),
                 ("imputation_health_donut3", dict(fe="use", uses=HC_, donut=3)),
                 ("imputation_health_use_sido", dict(fe="use_sido", uses=HC_)),
                 ("imputation_health_minctrl5", dict(fe="use", uses=HC_, min_ctrl=5)), ("imputation_health_minctrl10", dict(fe="use", uses=HC_, min_ctrl=10)),
                 ("imputation_health_strict_12_12", dict(fe="use", uses=HC_, min_win=12)),
                 ("imputation_health_drop_jumpy", dict(fe="use", uses=HC_, drop=JUMPY)),
                 ("imputation_health_fit_s15_s4", dict(fe="use", uses=HC_, fit_lo=15)),
                 ("imputation_health_no_anticip_holdout", dict(fe="use", uses=HC_, anticip=0))]
    for car in ["elec", "total"]:
        for name, kw in IMP_SPECS:
            try:
                r_ = dict(spec=name, status="ok", **imputation_did(B, E, car, nboot=NBH if name in HEAD else NB, **kw))
                if "fit_s15" in name and r_["n_treated_total"] < MIN_FITWIN_N:
                    r_ = dict(spec=name, carrier=car, status=f"not estimable (n<{MIN_FITWIN_N}): {r_['n_treated_total']} treated buildings with >= {MIN_WIN} of the 12 fit months s-15..s-4",
                              n_treated_total=r_["n_treated_total"], fit_window=r_["fit_window"])
                imp.append(r_)
            except Exception as ex:
                st = (f"not estimable (n<{MIN_FITWIN_N}): 0 qualifying treated buildings; " if "fit_s15" in name else "not estimable: " if "no treated rows" in str(ex) else "") + f"{type(ex).__name__}: {ex}"
                print("imputation failed", car, name, ex); imp.append(dict(spec=name, carrier=car, status=st[:260]))
    (RES / "G7_imputation.json").write_text(json.dumps(imp, indent=1, ensure_ascii=False, default=float))
    pd.DataFrame([{k: v for k, v in r.items() if k not in ("cohorts", "pretrend", "n_control_by_role", "n_treated_by_use", "n_control_by_use", "season")} for r in imp]).to_csv(RES / "G7_imputation.csv", index=False)
    print(pd.read_csv(RES / "G7_imputation.csv").to_string())
    ev = []
    for utag, ukw in [("pooled", prim_ev), ("health_centre", dict(prim_ev, uses=("보건",))), ("daycare", dict(prim_ev, uses=("어린이집",)))]:
        for c in ["elec", "total"]:
            try:
                x_ = stacked_did(B, E, c, event=True, **ukw)
                if x_ is not None: ev.append(x_.assign(spec="primary_within_use", use=utag))
            except Exception as ex: print("event study failed", utag, c, type(ex).__name__, ex)
    if ev: pd.concat(ev).to_csv(RES / "G3_event_study.csv", index=False)
    mt = [x for x in [stacked_did(B, E, c, measure_terms=[m_ for m_ in SIM6 + ["hvac", "boiler", "renewable", "cool_roof", "vent_generic"] if B[m_].sum() > 0], **prim) for c in ["elec", "total"]] if x is not None]
    if mt: pd.concat(mt).to_csv(RES / "G4_measures.csv", index=False)
    rob = []
    specs = [("A+B", dict(prim, grades=("A_cand", "B"))), ("controls_2024_25_only", dict(prim, ctrl_roles=("control",))),
             ("fe_use_province", dict(prim, fe="use_sido")), ("fe_month_only", dict(prim, fe="none")),
             ("drop_renewable", dict(prim, drop=set(B[B.renewable == 1].pid))), ("window24", dict(prim, window=24)),
             ("drop_covid_months", dict(prim, drop_months=COVID_MONTHS)), ("drop_jumpy", dict(prim, drop=jumpy_buildings(E))),
             ("donut3", dict(prim, donut=3)), ("no_ctrl_trim", dict(prim, ctrl_trim=0)), ("cluster_municipality", dict(prim, cluster="sigungu")),
             ("health_only", dict(prim, uses=("보건",))), ("daycare_only", dict(prim, uses=("어린이집",))),
             ("stable_meter_count", dict(prim, drop=unstable_meter_buildings(E))),
             ("strict_12_12", dict(prim, min_win=12))]
    for name, kw in specs:
        for c in ["elec", "total"]:
            try:
                r = stacked_did(B, E, c, **kw)
                if r is not None: rob.append(r.assign(spec=name))
            except Exception as ex: print("rob fail", name, c, ex)
    if rob: pd.concat(rob).to_csv(RES / "G6_robustness.csv", index=False)
    HC.update(heating_carrier(B, E).to_dict()); pd.Series(HC, name="heat_carrier").to_csv(RES / "G5_heat_carrier.csv")
    (RES / "G0_qa_excluded.json").write_text(json.dumps(sorted([list(x) for x in QA_EXCL])))

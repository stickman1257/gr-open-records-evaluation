"""Linkage and sample flow (Fig. 1)."""
import os, sys, time, pathlib
import numpy as np, pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
PROC, RES, FIGS = ROOT / "data/processed", ROOT / "results", ROOT / "paper/figs"
sys.path.insert(0, str(HERE))
_argv, sys.argv = sys.argv, sys.argv[:1]
from s03_energy import parse_period, ym_add, ym_range
sys.argv = _argv

SCOPE = [2021, 2022, 2023, 2024, 2025]
COHORTS = {"sel_2021_22": [2021, 2022], "sel_2023": [2023], "sel_2024_25": [2024, 2025], "sel_2026_out_of_scope": [2026]}
COLS = ["all"] + list(COHORTS)
ADDR_FAIL = {"ADDR_NOT_FOUND", "ADDR_AMBIGUOUS", "JUSO_ERROR"}
PENDING = "pending"


def mtime(p): return time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime)) if p.exists() else None


def read(name, **kw):
    p = PROC / name
    return (pd.read_csv(p, **kw), f"data/processed/{name} ({mtime(p)})") if p.exists() else (None, f"data/processed/{name} (absent)")


def load_detail():
    f = PROC / "gr_projects_raw.csv"
    if f.exists(): return pd.read_csv(f), f"data/processed/gr_projects_raw.csv ({mtime(f)})"
    try:
        argv, sys.argv = sys.argv, sys.argv[:1]
        import s01_gr_sites as s1
        g = s1.table_from_cache()
        return g, "s01_gr_sites.table_from_cache() [cached detail pages; collection running]"
    except Exception as e:
        return None, f"detail cache unavailable ({type(e).__name__})"
    finally:
        sys.argv = argv


def counts(df):
    """Counts of unique projects per cohort column."""
    if df is None: return {c: PENDING for c in COLS}
    k = df.drop_duplicates(["year", "seq"])
    out = {"all": int(len(k))}
    for c, ys in COHORTS.items(): out[c] = int(k.year.isin(ys).sum())
    return out


def main():
    rows = []
    def add(no, stage, df, status, source, note=""):
        rows.append(dict(stage_no=no, stage=stage, **counts(df), status=PENDING if df is None else status, source=source, note=note))

    L, sL = read("gr_list_all.csv")
    add("1", "Listed projects (program locator)", L, "complete", sL)

    G, sG = load_detail()
    if G is not None and L is not None:
        n_scope_listed = int(L.year.isin(SCOPE).sum())
        part = "complete" if G.year.isin(SCOPE).sum() >= n_scope_listed else "partial"
    else: part = "partial"
    add("2", "Detail record retrieved", G, part, sG, "detail pages not requested for 2026")
    Gs = G[G.year.isin(SCOPE)] if G is not None else None
    add("3", "Selection year in scope (2021-2025)", Gs, part, sG)
    if Gs is not None:
        se = Gs.period_txt.apply(parse_period)
        Gs = Gs.assign(start_ym=[x[0] for x in se], end_ym=[x[1] for x in se])
        add("3a", "  of which construction start/end dates parseable", Gs[Gs.start_ym.notna()], part, sG,
            "s03_energy.parse_period; informational (not a sequential filter until stage 7)")

    P, sP = read("gr_parcels.csv")
    if P is not None and Gs is not None:
        linkpart = "complete" if len(P.drop_duplicates(["year", "seq"])) >= len(Gs) else "partial"
    else: linkpart = "partial"
    add("4", "Linkage attempted (address geocoding run)", P, linkpart, sP, "s02_link.py run covers only part of stage 3 until re-run")
    if P is not None:
        Pm = P[~P.match_status.isin(ADDR_FAIL)]
        add("5", "Address matched to a unique parcel", Pm, linkpart, sP,
            "dropped: " + ", ".join(f"{k}={v}" for k, v in P[P.match_status.isin(ADDR_FAIL)].match_status.value_counts().items()))
        Pr = P[P.match_status == "MATCHED"]
        add("6", "Building register record found", Pr, linkpart, sP,
            "dropped: " + ", ".join(f"{k}={v}" for k, v in Pm[Pm.match_status != "MATCHED"].match_status.value_counts().items()))
        for no, gname in [("7a", "A_cand"), ("7b", "B"), ("7c", "C")]:
            add(no, f"Link grade {gname}" + (" (excluded)" if gname == "C" else ""), Pr[Pr.link_grade == gname], linkpart, sP)
        Pab = Pr[Pr.link_grade.isin(["A_cand", "B"])]
        add("7", "Link grade A_cand or B", Pab, linkpart, sP)
        try:
            import s02_link as L2
            if G is None: raise RuntimeError("detail records absent (area_txt needed)")
            Pg = Pr.merge(G[["year", "seq", "area_txt", "name_list"]].drop_duplicates(["year", "seq"]), on=["year", "seq"], how="left", suffixes=("", "_g"))
            g2 = [L2.regrade(x.name_list, x.reg_use, x.reg_area, x.area_txt, x.reg_n_main, x.reg_is_collective, x.bdKdcd, x.reg_area_share)[1]
                  for x in Pg.itertuples()]
            Pg = Pg.assign(link_grade_v1=Pg.link_grade, link_grade=g2)
            tr = Pg[Pg.link_grade_v1 != Pg.link_grade].groupby(["link_grade_v1", "link_grade"]).size()
            add("7e", "Link grade A_cand after the regrade (analysis grade)", Pg[Pg.link_grade == "A_cand"], linkpart, sP + " + detail records",
                "s02_link.regrade on every matched project (as src/pi_regrade_all.py); transitions: " +
                (", ".join(f"{a}->{b} {int(n)}" for (a, b), n in tr.items()) or "none"))
            add("7eb", "Link grade B after the regrade", Pg[Pg.link_grade == "B"], linkpart, sP + " + detail records")
            add("7ec", "Link grade C after the regrade (excluded)", Pg[Pg.link_grade == "C"], linkpart, sP + " + detail records")
        except Exception as ex:
            add("7e", "Link grade A_cand after the regrade (analysis grade)", None, PENDING, sP, f"{type(ex).__name__}: {ex}"[:150])
        if Gs is not None:
            Pd = Pab.merge(Gs.loc[Gs.start_ym.notna(), ["year", "seq"]], on=["year", "seq"])
            add("7d", "  of which construction start/end dates parseable", Pd, linkpart, sP + " + detail records",
                "remaining gap to stage 8 = completion too late for 12 post months (treated_short) or control started before 2024-07 (control_early)")
    else:
        for no, st in [("5", "Address matched to a unique parcel"), ("6", "Building register record found"), ("7a", "Link grade A_cand"),
                       ("7b", "Link grade B"), ("7c", "Link grade C (excluded)"), ("7", "Link grade A_cand or B"),
                       ("7e", "Link grade A_cand after the regrade (analysis grade)")]:
            add(no, st, None, PENDING, sP)

    B, sB = read("analysis_buildings.csv")
    roles = [("treated", "Role: treated (dates parsed, completion leaves >=12 post months)"),
             ("control", "Role: not-yet-treated control (2024-25 selections, start >= 2024-07)"),
             ("aux_2023", "Role: 2023 selection, used only before its own start (control pool)")]
    if B is not None:
        stale = "" if P is None else ("" if (PROC / "analysis_buildings.csv").stat().st_mtime >= (PROC / "gr_parcels.csv").stat().st_mtime
                                      else "STALE: older than gr_parcels.csv; ")
        Br = B[B.role.isin(["treated", "control", "aux_2023"])]
        add("8", "Role assigned (treated, control or 2023 selection), grades A_cand+B", Br, linkpart, sB,
            stale + "written by s03_energy.py at the end of a run; dropped A/B projects = unparseable dates, treated_short, control_early")
        for i, (r, lab) in enumerate(roles):
            add(f"8{'abd'[i]}", "  " + lab, Br[Br.role == r], linkpart, sB)
        add("8c", "  of which link grade A_cand as stored (before the regrade)", Br[Br.link_grade == "A_cand"], linkpart, sB)
        try:
            import s04_analysis as A4r
            Rg = A4r.regrade_buildings(Br)
            ch = {}
            for r_ in ["treated", "control", "aux_2023"]:
                x = Rg[Rg.role == r_]
                ch[r_] = (f"{r_}: A {int((x.link_grade_v1 == 'A_cand').sum())} -> {int((x.link_grade == 'A_cand').sum())}, "
                          f"A->B {int(((x.link_grade_v1 == 'A_cand') & (x.link_grade == 'B')).sum())}, A->C {int(((x.link_grade_v1 == 'A_cand') & (x.link_grade == 'C')).sum())}, "
                          f"B->A {int(((x.link_grade_v1 == 'B') & (x.link_grade == 'A_cand')).sum())}, use class changed {int((x.use_class_v1 != x.use_class).sum())} "
                          f"(to health centre {int(((x.use_class_v1 != x.use_class) & (x.use_class == '보건')).sum())})")
            add("8e", "  of which link grade A_cand after the regrade (analysis grade)", Rg[Rg.link_grade == "A_cand"], linkpart, sB,
                "s02_link.regrade applied at load time; " + "; ".join(ch.values()))
            try:
                from a_dedup import dedup_physical
                Rd, Dd = dedup_physical(Rg)
                add("8f", "  of which grade A_cand after D23 (one physical building = one unit)", Rd[Rd.link_grade == "A_cand"], linkpart, sB,
                    f"dropped later records: {len(Dd)} ({Dd.role.value_counts().to_dict()}) of {Dd.reg_pk.nunique()} physical buildings; results/a_dedup.csv")
            except Exception as ex_:
                add("8f", "  of which grade A_cand after D23", None, PENDING, sB, f"{type(ex_).__name__}: {ex_}"[:150])
            for u_, en_ in [("보건", "health centres"), ("어린이집", "daycare centres")]:
                add("8e" + en_[0], f"    {en_} (regraded use class)", Rg[(Rg.link_grade == "A_cand") & (Rg.use_class == u_)], linkpart, sB)
            Br = Rg
        except Exception as ex:
            add("8e", "  of which link grade A_cand after the regrade (analysis grade)", None, PENDING, sB, f"{type(ex).__name__}: {ex}"[:150])
    else:
        Br = None
        add("8", "Role assigned (treated, control or 2023 selection), grades A_cand+B", None, PENDING, sB)

    E, sE = read("energy_monthly.csv")
    if E is not None and Br is not None:
        El = E[E.carrier == "elec"]
        n_probe = El.pid.nunique(); epart = "complete" if n_probe >= Br.pid.nunique() else "partial"
        passes = sorted((int(x) if str(x).isdigit() else x) for x in El.pass_.dropna().unique() if str(x).isdigit()) + (["bulk"] if (El.pass_.astype(str) == "bulk").any() else [])
        note_e = f"passes present in file: {passes}; pass 0 = 2 probe months per building, pass 1 = 12 pre + 12 post"
        Bq = Br[Br.pid.isin(El.pid)]
        add("9", "Electricity meter queried (any month requested)", Bq, epart, sE, note_e)
        obs = El[(El.status == "ok") & (El.kwh > 0)].merge(Br[["pid", "start_ym", "end_ym", "role"]], on="pid")
        n_zero = int(((El.status == "ok") & ~(El.kwh > 0)).sum())
        Bo = Br[Br.pid.isin(obs.pid)]
        add("10", "Electricity data available (>=1 observed month)", Bo, epart, sE, f"ok months with kWh<=0 (excluded): {n_zero}")
        if "link_grade_v1" in Bo:
            add("10c", "  of which link grade A_cand as stored (before the regrade)", Bo[Bo.link_grade_v1 == "A_cand"], epart, sE,
                "until 11/13 23:0x this row was filtered on the regraded grade although it was not labelled so; it now uses the stored grade (pairs with 7a, 8c)")
            add("10e", "  of which link grade A_cand after the regrade (analysis grade)", Bo[Bo.link_grade == "A_cand"], epart, sE,
                "pairs with 7e, 8e")
        else:
            add("10c", "  of which link grade A_cand as stored (before the regrade)", Bo[Bo.link_grade == "A_cand"], epart, sE)
            add("10e", "  of which link grade A_cand after the regrade (analysis grade)", None, PENDING, sE, "regrade not available (see stage 8e)")
        npre = obs[obs.ym < obs.start_ym].groupby("pid").size(); npost = obs[obs.ym > obs.end_ym].groupby("pid").size()
        Br = Br.assign(n_pre=Br.pid.map(npre).fillna(0), n_post=Br.pid.map(npost).fillna(0))
        full = []
        o = obs.set_index("pid")
        for b in Br[Br.role == "treated"].itertuples():
            if b.pid not in o.index: continue
            ym = set(np.atleast_1d(o.loc[[b.pid], "ym"].to_numpy())); s, e = int(b.start_ym), int(b.end_ym)
            if set(ym_range(ym_add(s, -12), ym_add(s, -1))) <= ym and set(ym_range(ym_add(e, 1), ym_add(e, 12))) <= ym: full.append(b.pid)
        T = Br[(Br.role == "treated") & (Br.n_pre >= 12) & (Br.n_post >= 12)]
        C = Br[(Br.role == "control") & (Br.n_pre >= 12)]
        st = epart if 1 in passes else "partial"
        nn = "pending: pass 1 (12 pre + 12 post months) not yet in energy_monthly.csv" if 1 not in passes else ""
        pend = lambda df: df if 1 in passes else None
        add("11", "Panel-ready: treated >=12 pre & >=12 post months; controls >=12 pre months", pend(pd.concat([T, C])), st, sE, nn)
        add("11a", "  treated: >=12 pre & >=12 post observed months", pend(T), st, sE, nn)
        add("11b", "  treated: complete 12+12 window (months s-12..s-1 and e+1..e+12)", pend(Br[Br.pid.isin(full)]), st, sE, nn)
        add("11c", "  controls: >=12 observed months before own start", pend(C), st, sE, nn)
        A = pd.concat([T, C]); A = A[A.link_grade == "A_cand"]
        add("12", "Primary sample (analysis_plan.md): stage 11, link grade A_cand", pend(A), st, sE,
            nn + ("; " if nn else "") + "primary treated = sel_2021_22 column; sel_2023 auxiliary; grade = regraded (analysis grade) when stage 8e is available")
        try:
            import s04_analysis as A4
            B4, E4 = A4.load()
            okw = A4.complete_windows(A4.carrier_frame(E4, "elec"), B4)
            oks = A4.complete_windows(A4.carrier_frame(E4, "elec"), B4, 12)
            add("11d", f"  treated: >= {A4.MIN_WIN} of 12 months observed in each window, QA gate passed (rule of 11/13)", pend(Br[Br.pid.isin(okw)]), st, sE,
                "2021-10 electricity is empty in the API for most parcels; strict 12 + 12 after the QA gate: %d" % len(oks))
            S4 = A4.imputation_setup(B4, E4, "elec", **A4.PRIMARY_IMP)
            if S4 is not None:
                f4 = S4["fit"](); m4 = S4["post"] & f4["ok"]; c4 = f4["F"] & S4["isc"]
                tp = set(S4["pids"][pd.unique(S4["pi"][m4])]); cp = set(S4["pids"][pd.unique(S4["pi"][c4])])
                Ba = B4[B4.pid.isin(tp | cp)]
                add("13", "Analysis sample (paper): health centres and daycares, grade A_cand, imputation frame", pend(Ba), st, sE,
                    "treated = sel_2021_22 with >= 1 supported post month; control pool = sel_2024_25 and sel_2023 buildings with >= 1 untreated month in the fit")
                for u, en in [("보건", "health centres"), ("어린이집", "daycare centres")]:
                    add("13" + en[0], f"  of which {en}", pend(Ba[Ba.use_class == u]), st, sE)
        except Exception as ex:
            add("13", "Analysis sample (paper): health centres and daycares, grade A_cand, imputation frame", None, PENDING, sE, f"{type(ex).__name__}: {ex}"[:150])
    else:
        for no, stg in [("9", "Electricity meter queried (any month requested)"), ("10", "Electricity data available (>=1 observed month)"),
                        ("11", "Panel-ready: treated >=12 pre & >=12 post months; controls >=12 pre months"),
                        ("10e", "  of which link grade A_cand after the regrade (analysis grade)"),
                       ("12", "Primary sample (analysis_plan.md): stage 11, link grade A_cand")]:
            add(no, stg, None, PENDING, sE)

    R = pd.DataFrame(rows)
    RES.mkdir(exist_ok=True); R.to_csv(RES / "a_attrition.csv", index=False)
    write_tex(R)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
    print("DESCRIPTIVE linkage funnel (counts at run time; 'partial' = upstream collection/linkage incomplete)")
    print(R.drop(columns=["source", "note"]).to_string(index=False))
    for r in R.itertuples():
        if r.note: print(f"  [{r.stage_no}] {r.note}")
    for s in dict.fromkeys(R.source): print("  src:", s)
    return R


def write_tex(R):
    esc = lambda s: str(s).replace("_", r"\_").replace("&", r"\&").replace(">=", r"$\geq$").replace("%", r"\%")
    def cell(v, status):
        if v == PENDING: return r"\textit{pending}"
        return f"{int(v):,}".replace(",", r"{,}") + (r"$^{\dagger}$" if status == "partial" else "")
    lines = ["% Auto-generated by src/a_attrition.py from results/a_attrition.csv on " + time.strftime("%Y-%m-%d %H:%M") +
             " -- do not edit by hand. Requires booktabs.",
             "% Descriptive linkage funnel. Dagger = upstream collection/linkage still partial at generation time.",
             r"\begin{tabular}{lrrrr}", r"\toprule",
             r" & & \multicolumn{3}{c}{Selection-year cohort} \\ \cmidrule(lr){3-5}",
             r"Stage & All & 2021--22 & 2023 & 2024--25 \\",
             r" & & (treated) & (auxiliary) & (controls) \\", r"\midrule"]
    for r in R.itertuples():
        ind = r"\quad " if r.stage.startswith("  ") else ""
        vals = [r.all, r.sel_2021_22, r.sel_2023, r.sel_2024_25]
        lines.append(f"{ind}{esc(r.stage.strip())} & " + " & ".join(cell(v, r.status) for v in vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              "% Note for caption: 'All' in the first two rows includes 2026 selections (out of scope: " +
              str(R.sel_2026_out_of_scope.iloc[0]) + " listed). $\\dagger$ partial: counts will grow as collection completes."]
    FIGS.mkdir(parents=True, exist_ok=True)
    (FIGS / "a_attrition.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

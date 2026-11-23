"""Programme descriptives from the project records."""
import os, sys, re, json, time, pathlib
import numpy as np, pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
PROC, RES, FIGS = ROOT / "data/processed", ROOT / "results", ROOT / "paper/figs"
sys.path.insert(0, str(HERE))
_argv, sys.argv = sys.argv, sys.argv[:1]
from s03_energy import parse_period
from s02_link import parse_area, use_class
from s04_analysis import parse_measures, MEAS_MAP
sys.argv = _argv

MEAS = list(MEAS_MAP)
MEAS_LABEL = {"wall": "Wall insulation", "roof": "Roof insulation / cool roof", "floor": "Floor insulation", "win": "Windows",
              "door": "Doors", "shade": "Solar shading", "hrv": "Ventilation (HRV)", "led": "Lighting (LED)",
              "hvac": "Heating/cooling units", "boiler": "Boiler", "pv": "Photovoltaics stated", "renewable": "Renewable energy entry",
              "cool_roof": "Cool-roof coating", "vent_generic": "Ventilation device (generic)"}
MEAS_LABEL.update({"hrv": "Heat recovery stated", "roof": "Roof insulation"})
USE_EN = {"어린이집": "daycare", "경로당": "senior_centre", "보건": "health_centre", "기타": "other"}
DATE = re.compile(r"(\d{4})[-.]?(\d{2})[-.]?(\d{2})")


def load_projects():
    f = PROC / "gr_projects_raw.csv"
    if f.exists():
        return pd.read_csv(f), "data/processed/gr_projects_raw.csv"
    argv, sys.argv = sys.argv, sys.argv[:1]
    try:
        import s01_gr_sites as s1
        g = s1.table_from_cache()
    finally:
        sys.argv = argv
    return g, "s01_gr_sites.table_from_cache() [data/raw/gr_sites/detail/*.html; collection still running]"


def parse_cost(s):
    m = re.search(r"([\d,]+)\s*원", str(s))
    if not m: return np.nan
    d = m.group(1).replace(",", "")
    return float(d) if d else np.nan


def parse_pred(s):
    m = re.search(r"(-?[\d\.]+)\s*%", str(s))
    try: return float(m.group(1)) if m else np.nan
    except ValueError: return np.nan


def duration_months(s, ok):
    if not ok: return np.nan
    d = DATE.findall(str(s))
    try:
        a, b = [pd.Timestamp(year=int(x[0]), month=int(x[1]), day=int(x[2])) for x in d[:2]]
    except Exception:
        return np.nan
    return (b - a).days / 30.4375


def period_problem(s):
    """Why parse_period failed (diagnostic label only)."""
    t = str(s).strip()
    if re.fullmatch(r"~?", t) or t in ("nan", "None"): return "blank"
    if re.search(r"[가-힣]", t): return "text (e.g. cancelled)"
    d = DATE.findall(t)
    if len(d) >= 2: return "end before start"
    shape = re.sub(r"\d", "9", t)
    if re.fullmatch(r"9{6}\s*~\s*9{6}|9{4}-99\s*~\s*9{4}-99", shape): return "year-month only"
    if re.fullmatch(r"9{4}\s*~\s*9{4}", shape): return "year only"
    return "other format"


def build(g):
    d = g.copy()
    se = d.period_txt.apply(parse_period)
    d["start_ym"] = [x[0] for x in se]; d["end_ym"] = [x[1] for x in se]
    d["period_ok"] = d.start_ym.notna()
    d["period_problem"] = [None if ok else period_problem(s) for s, ok in zip(d.period_txt, d.period_ok)]
    d["duration_months"] = [duration_months(s, ok) for s, ok in zip(d.period_txt, d.period_ok)]
    d["gfa_m2"] = d.area_txt.apply(parse_area)
    fp = d.area_txt.astype(str).str.extract(r"([\d,\.]+)\s*㎡")[0].str.replace(",", "")
    d["footprint_m2"] = pd.to_numeric(fp, errors="coerce")
    d.loc[d.gfa_m2 <= 0, "gfa_m2"] = np.nan
    d["built_year_num"] = pd.to_numeric(d.built_year, errors="coerce")
    d.loc[~d.built_year_num.between(1900, 2025), "built_year_num"] = np.nan
    d["cost_krw"] = d.cost_txt.apply(parse_cost)
    d.loc[d.cost_krw <= 0, "cost_krw"] = np.nan
    d["cost_per_m2"] = d.cost_krw / d.gfa_m2
    d["pred_saving"] = d.pred_saving_txt.apply(parse_pred)
    d["pred_saving_valid"] = d.pred_saving.where((d.pred_saving > 0) & (d.pred_saving <= 100))
    M = pd.DataFrame([parse_measures(t) for t in d.measures_txt], index=d.index)
    M.loc[d.measures_txt.isna() | (d.measures_txt.astype(str).str.strip() == "")] = np.nan
    d = pd.concat([d, M], axis=1)
    d["n_measures"] = d[MEAS].sum(axis=1, min_count=1)
    d["use_class"] = d.name_list.apply(use_class)
    d["use_en"] = d.use_class.map(USE_EN)
    lc = np.log10(d.cost_per_m2)
    d["cost_log10_dev"] = lc - lc.median()
    d["cost_outlier"] = d.cost_log10_dev.abs() > 1
    return d


def summarise(d, gtype, gname):
    rows = []
    def add(metric, value, n, unit=""):
        rows.append(dict(group_type=gtype, group=gname, metric=metric, value=value, n=int(n), unit=unit))
    N = len(d)
    add("n_projects", N, N, "count")
    add("share_signature_project", (d.project_type == "시그니처 사업").mean() if N else np.nan, N, "share")
    for u in USE_EN.values():
        add(f"share_use_{u}", (d.use_en == u).mean() if N else np.nan, N, "share")
    mm = d.dropna(subset=["wall"])
    for m in MEAS:
        add(f"share_measure_{m}", mm[m].mean() if len(mm) else np.nan, len(mm), "share")
    add("median_n_measures", mm.n_measures.median() if len(mm) else np.nan, len(mm), "count of 11 coded measures")
    for col, name, unit in [("gfa_m2", "gfa", "m2"), ("built_year_num", "built_year", "year"), ("cost_krw", "cost", "KRW"),
                            ("cost_per_m2", "cost_per_m2", "KRW/m2"), ("duration_months", "duration", "months"),
                            ("pred_saving_valid", "pred_saving", "% (program-reported simulated prediction)")]:
        x = d[col].dropna()
        for q, lab in [(0.5, "median"), (0.25, "p25"), (0.75, "p75")]:
            add(f"{lab}_{name}", x.quantile(q) if len(x) else np.nan, len(x), unit)
    x = d.cost_per_m2[~d.cost_outlier].dropna()
    add("median_cost_per_m2_excl_outliers", x.median() if len(x) else np.nan, len(x), "KRW/m2")
    add("share_period_parsed", d.period_ok.mean() if N else np.nan, N, "share")
    add("share_pred_saving_present", d.pred_saving_valid.notna().mean() if N else np.nan, N, "share")
    return rows


def flags(d):
    F = []
    def flag(mask, reason):
        for r in d[mask].itertuples():
            F.append(dict(year=r.year, seq=r.seq, name_list=r.name_list, reason=reason, period_txt=r.period_txt, area_txt=r.area_txt,
                          cost_txt=r.cost_txt, pred_saving_txt=r.pred_saving_txt, cost_per_m2=r.cost_per_m2,
                          duration_months=r.duration_months))
    flag(~d.period_ok, "period unparseable")
    flag(d.period_ok & d.duration_months.isna(), "period: invalid calendar date")
    flag(d.duration_months > 36, "duration > 36 months")
    flag(d.duration_months < 1, "duration < 1 month")
    flag(d.period_ok & ((d.start_ym < 202001) | (d.end_ym > 202712)), "period outside 2020-2027")
    flag(d.period_ok & (d.start_ym // 100 < d.year - 1), "start more than 1 calendar year before selection year")
    flag(d.gfa_m2.isna(), "floor area missing/zero")
    flag(d.footprint_m2 > 1.5 * d.gfa_m2, "footprint > 1.5 x gross floor area (fields possibly swapped)")
    flag(d.cost_krw.isna(), "cost missing/zero")
    flag(d.cost_outlier.fillna(False), "cost per m2 outlier (>10x or <1/10 of median)")
    flag(d.pred_saving.notna() & d.pred_saving_valid.isna(), "predicted saving outside (0,100]")
    flag(d.built_year_num.isna(), "built year missing/implausible")
    flag(d.wall.isna(), "measures text missing")
    return pd.DataFrame(F)


def figure(d, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mm = d.dropna(subset=["wall"])
    years = sorted(mm.year.unique())
    T = mm.groupby("year")[MEAS].mean().T * 100
    order = T.mean(axis=1).sort_values(ascending=False).index
    T = T.loc[order, years]
    n = mm.groupby("year").size()
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    im = ax.imshow(T.values, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(years))); ax.set_xticklabels([f"{y}\n(n={n[y]})" for y in years], fontsize=7.5)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([MEAS_LABEL.get(m, m) for m in order], fontsize=7.5)
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            v = T.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5, color="white" if v > 55 else "#222222")
    ax.set_xticks(np.arange(-.5, len(years)), minor=True); ax.set_yticks(np.arange(-.5, len(order)), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5); ax.tick_params(which="both", length=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xlabel("Selection year", fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02); cb.set_label("Projects with measure (%)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7); cb.outline.set_visible(False)
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path); plt.close(fig)
    return T


if __name__ == "__main__":
    g, src = load_projects()
    d = build(g)
    rows = summarise(d, "all", "all")
    for y, dy in d.groupby("year"): rows += summarise(dy, "selection_year", str(y))
    for u, du in d.groupby("use_en"): rows += summarise(du, "facility_type", u)
    for (y, u), dd in d.groupby(["year", "use_en"]): rows += summarise(dd, "year_x_facility", f"{y}|{u}")
    S = pd.DataFrame(rows)
    RES.mkdir(exist_ok=True)
    S.to_csv(RES / "a_program_descriptives.csv", index=False)
    F = flags(d); F.to_csv(RES / "a_program_descriptives_flags.csv", index=False)
    T = figure(d, FIGS / "a_measures_by_year.pdf")

    listed = None
    if (PROC / "gr_list_all.csv").exists():
        listed = pd.read_csv(PROC / "gr_list_all.csv").groupby("year").size().to_dict()
    diag = dict(
        period_unparseable_by_reason=d.loc[~d.period_ok, "period_problem"].value_counts().to_dict(),
        period_unparseable_by_year=d.loc[~d.period_ok].groupby("year").size().to_dict(),
        flags_by_reason=F.reason.value_counts().to_dict() if len(F) else {},
        cost_per_m2_outlier_rule="abs(log10(cost/m2) - log10(median)) > 1",
        cost_per_m2_quantiles={str(q): float(d.cost_per_m2.quantile(q)) for q in [0, .01, .05, .25, .5, .75, .95, .99, 1]},
        pred_saving_raw_out_of_range=sorted(d.loc[d.pred_saving.notna() & d.pred_saving_valid.isna(), "pred_saving"].tolist()),
    )
    wide = {gt: {gn: {r.metric: (None if pd.isna(r.value) else float(r.value)) for r in x.itertuples()} |
                      {"_n": {r.metric: r.n for r in x.itertuples()}}
                 for gn, x in sg.groupby("group")} for gt, sg in S.groupby("group_type")}
    out = dict(label="DESCRIPTIVE program-record statistics; not effect estimates. Predicted saving = program-reported "
                     "simulated prediction, not metered.",
               source=src, generated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), n_projects_with_detail=int(len(d)),
               detail_by_year={int(k): int(v) for k, v in d.groupby("year").size().items()},
               listed_by_year=listed, coverage_note="Detail pages are still being collected; years with detail < listed "
               "are incomplete and their statistics will change on re-run.", diagnostics=diag, stats=wide)
    (RES / "a_program_descriptives.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

    pd.set_option("display.width", 220)
    print("DESCRIPTIVE. source:", src); print("detail rows", len(d), "by year", out["detail_by_year"], "| listed", listed)
    show = ["n_projects", "median_gfa", "median_built_year", "median_cost", "median_cost_per_m2", "median_duration",
            "share_period_parsed", "share_pred_saving_present", "median_pred_saving", "median_n_measures"]
    for gt in ["all", "selection_year", "facility_type"]:
        x = S[(S.group_type == gt) & S.metric.isin(show)].pivot(index="group", columns="metric", values="value")[show]
        print(f"\n[{gt}]"); print(x.round(3).to_string())
    print("\nshare with measure (%), by selection year"); print(T.round(1).to_string())
    x = S[(S.group_type == "facility_type") & S.metric.str.startswith("share_measure_")].pivot(index="metric", columns="group", values="value")
    print("\nshare with measure, by facility type"); print((x * 100).round(1).to_string())
    print("\nperiod unparseable:", diag["period_unparseable_by_reason"]); print("flags:", diag["flags_by_reason"])
    print("cost/m2 quantiles:", {k: round(v) for k, v in diag["cost_per_m2_quantiles"].items()})
    print("pred saving out of range:", diag["pred_saving_raw_out_of_range"])

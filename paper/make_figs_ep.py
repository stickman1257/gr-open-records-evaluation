"""Figures 1-5 of the manuscript, drawn from results/."""
import os, pathlib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "paper" / "figs"; FIG.mkdir(exist_ok=True)
FINAL = True
INK, MID, LIGHT = "black", "0.45", "0.82"
W = 6.3
plt.rcParams.update({"font.size": 9, "font.family": "serif", "pdf.fonttype": 42})


def _clean(ax):
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.tick_params(length=2, width=0.5, labelsize=7)
    for sp in ax.spines.values(): sp.set_linewidth(0.5)

def _finish(fig, name):
    if not FINAL:
        fig.text(0.5, 0.5, "DEBUG", rotation=14, ha="center", va="center", fontsize=30, color="0.35", alpha=0.35, zorder=100)
    fig.savefig(FIG / name); plt.close(fig)


def fig_flow():
    """Fig. 1: two chains with their own denominators. All boxes count project records (stage 8e, not 8f: the footnote of the"""
    A = pd.read_csv(RES / "a_attrition.csv", dtype={"stage_no": str}).set_index("stage_no")
    g = lambda k, c: int(A.loc[k, c])
    fig = plt.figure(figsize=(W, 2.75)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis("off")

    def chain(y, title, boxes, h=9.6, gap=2.2, x0=0.6):
        ax.text(x0, y + h / 2 + 1.2, title, fontsize=8.5, ha="left", va="bottom", style="italic")
        w = (98.8 - gap * (len(boxes) - 1)) / len(boxes); x = x0
        for i, (n, lab, share) in enumerate(boxes):
            ax.add_patch(plt.Rectangle((x, y - h / 2), w, h, fill=False, lw=0.7, ec=INK))
            ax.text(x + w / 2, y + 2.2, f"{n:,}", fontsize=10, ha="center", va="center", weight="bold")
            ax.text(x + w / 2, y - 1.9, lab, fontsize=6.6, ha="center", va="center", linespacing=1.0)
            if share:
                ax.text(x + w / 2, y - h / 2 - 1.5, share, fontsize=7.2, ha="center", va="top", color=MID)
            if i < len(boxes) - 1:
                ax.annotate("", xy=(x + w + gap, y), xytext=(x + w, y), arrowprops=dict(arrowstyle="-|>", lw=0.7, color=INK, mutation_scale=7))
            x += w + gap

    n0, n1, n2 = g("2", "all"), g("6", "all"), g("7e", "all")
    chain(33.5, "(a) Linkage: all project records selected 2021–25",
          [(n0, "project records", ""), (n1, "matched to a parcel\nwith a register record", f"{100 * n1 / n0:.1f}% of records"),
           (n2, "grade A link", f"{100 * n2 / n1:.1f}% of matched")], gap=5)
    m0, m1, m2, m3, m4, m5 = (g(k, "sel_2021_22") for k in ("2", "6", "7e", "8e", "10e", "13"))
    chain(11.5, "(b) Evaluation sample: project records selected 2021–22",
          [(m0, "project records", ""), (m1, "matched", f"{100 * m1 / m0:.1f}%"), (m2, "grade A link", f"{100 * m2 / m0:.1f}%"),
           (m3, "dates, 12 months\nafter completion", f"{100 * m3 / m0:.1f}%"), (m4, "electricity\nreturned", f"{100 * m4 / m0:.1f}%"),
           (m5, f"{g('13h', 'sel_2021_22')} health posts,\n{g('13d', 'sel_2021_22')} daycares", f"{100 * m5 / m0:.1f}%")])
    ax.text(99.4, 0.3, "Shares under chain (b): of the 1,490 records selected in 2021–22.", fontsize=7.2, ha="right", va="bottom", color=MID)
    _finish(fig, "fig_flow.pdf")


def fig_event_long():
    """Fig. 2 (R3): (a) long stacked event study, health posts, municipality-clustered 95% intervals (results/r3_event_long.csv);"""
    fig = plt.figure(figsize=(W, 2.6))
    a = fig.add_axes([0.075, 0.20, 0.56, 0.68]); b = fig.add_axes([0.715, 0.20, 0.27, 0.68])
    S = pd.read_csv(RES / "r3_event_long.csv"); S = S[S.cluster == "sigungu"].copy(); S["x"] = (S.a + S.b) / 2
    for k in ("pct_coef", "pct_lo", "pct_hi"): S[k] = 100 * S[k]
    a.axhline(0, color=MID, lw=0.6); a.axvspan(-0.6, 0.6, color=LIGHT, lw=0, zorder=0)
    pre, ant, post = S.x < -3.5, (S.x > -3.5) & (S.x < 0), S.x > 0
    for m, mk, fc, lab in [(pre, "D", "white", "before the works"), (ant, "s", "0.6", "months −3 to −1"), (post, "o", INK, "after completion")]:
        e = S[m]; a.errorbar(e.x, e.pct_coef, yerr=[e.pct_coef - e.pct_lo, e.pct_hi - e.pct_coef], fmt=mk, ms=4, mfc=fc, mec=INK, mew=0.8, ecolor=INK, elinewidth=0.7, capsize=1.8, label=lab)
    a.plot([-5], [0], marker="x", color=INK, ms=5, mew=0.9); a.annotate("reference", (-5, 0), xytext=(-5, 9), fontsize=7, ha="center")
    a.set_xticks([-35, -29, -23, -17, -11, -5, 2, 8, 14, 20]); a.set_xticklabels(["−35", "−29", "−23", "−17", "−11", "−5", "+2", "+8", "+14", "+20"])
    a.set_xlim(-37, 25); a.set_ylim(-40, 25); a.set_xlabel("months before start of works / after completion", fontsize=8); a.set_ylabel("change in electricity (%)", fontsize=8)
    a.legend(frameon=False, fontsize=7, loc="lower left", handletextpad=0.2, borderaxespad=0.2, ncol=3); a.set_title("(a) Health posts against comparison buildings", fontsize=9, loc="left")
    _clean(a); a.tick_params(labelsize=8)
    G = pd.read_csv(RES / "r3_building_gaps.csv"); v = 100 * G.pct.clip(-80, 80)
    b.hist(v, bins=np.arange(-80, 81, 10), color=LIGHT, edgecolor=INK, lw=0.6); b.axvline(0, color=MID, lw=0.6)
    b.axvline(100 * float(G.pct.median()), color=INK, lw=0.8, ls=(0, (2, 2)))
    b.set_xlabel("building-level change (%)", fontsize=8); b.set_ylabel("buildings", fontsize=8); b.set_title("(b) Distribution", fontsize=9, loc="left")
    b.set_xticks([-80, -40, 0, 40, 80]); _clean(b); b.tick_params(labelsize=8)
    _finish(fig, "fig_event_long.pdf")


def fig_sensitivity_r3():
    """Fig. 3 (R3): honestdid Delta^RM intervals on the long event study (results/r3_honestdid.csv)."""
    H = pd.read_csv(RES / "r3_honestdid.csv").sort_values("Mbar")
    fig = plt.figure(figsize=(W * 0.55, 2.3)); c = fig.add_axes([0.16, 0.21, 0.8, 0.68])
    lo, hi = 100 * (np.exp(H.lb) - 1), 100 * (np.exp(H.ub) - 1)
    c.fill_between(H.Mbar, lo, hi, color=LIGHT, lw=0); c.plot(H.Mbar, lo, color=INK, lw=0.9); c.plot(H.Mbar, hi, color=INK, lw=0.9); c.axhline(0, color=MID, lw=0.6)
    bd = H.breakdown_Mbar.iloc[0]
    if pd.notna(bd):
        c.axvline(float(bd), color=INK, lw=0.7, ls=(0, (2, 2))); c.text(float(bd) + 0.02, -57, f"$\\bar M^*$ = {float(bd):.2f}", fontsize=8, va="bottom", ha="left")
    c.set_xlim(0, float(H.Mbar.max())); c.set_ylim(-60, 40); c.set_xlabel("allowed violation $\\bar M$", fontsize=8); c.set_ylabel("mean change, months 1–12 (%)", fontsize=8)
    _clean(c); c.tick_params(labelsize=8)
    _finish(fig, "fig_sensitivity_r3.pdf")


def fig_benchmark_r3():
    """Fig. 4 (R3): estimate at a true effect of zero, 500 replications (results/r3_benchmark_summary.csv) for the five methods"""
    R = pd.read_csv(RES / "r3_benchmark_summary.csv"); R = R[R.subset.str.startswith("reps 0..4")].set_index("estimator")
    V = pd.read_csv(RES / "v6_benchmark_table.csv").set_index("estimator")
    rows = [("naive", "Before–after", 0, R), ("dd", "Degree-day baseline", 0, R), ("gb", "Gradient boosting (100 rep.)", 0, V),
            ("sdid", "Synthetic DiD (100 rep.)", 1, V), ("did_usefe", "Stacked DiD, type × month", 1, R), ("did", "Stacked DiD, province × month", 1, R),
            ("imp", "Imputation", 1, R), ("mvi", "Imputation, degree-day slopes (100 rep.)", 1, V)]
    fig = plt.figure(figsize=(W, 2.5)); a = fig.add_axes([0.41, 0.19, 0.57, 0.78]); y = np.arange(len(rows))[::-1]
    a.axvline(0, color=MID, lw=0.6); a.axhspan(4.5, 7.6, color="0.93", lw=0, zorder=0)
    for yi, (k, _, kind, T) in zip(y, rows):
        m, se = T.loc[k, "bias_log"], T.loc[k, "mc_se"]
        c = 100 * (np.exp(m) - 1); lo, hi = 100 * (np.exp(m - 1.96 * se) - 1), 100 * (np.exp(m + 1.96 * se) - 1)
        a.errorbar(c, yi, xerr=[[c - lo], [hi - c]], fmt="o", ms=5, mfc=["white", INK][kind], mec=INK, mew=0.8, ecolor=INK, elinewidth=0.8, capsize=2.5)
    a.text(2.8, 7.35, "no comparison group", fontsize=7.5, va="top", ha="right", style="italic")
    a.text(-14.6, -0.45, "with comparison group", fontsize=7.5, va="bottom", ha="left", style="italic")
    a.set_yticks(y); a.set_yticklabels([r[1] for r in rows], fontsize=8); a.set_ylim(-0.6, 7.6); a.set_xlim(-15, 3); a.set_xticks([-12, -9, -6, -3, 0, 3])
    a.set_xlabel("estimated change when the true effect is zero (%)", fontsize=8); _clean(a); a.tick_params(axis="y", length=0); a.tick_params(axis="x", labelsize=8)
    _finish(fig, "fig_benchmark_r3.pdf")


EVENTS = [(202204, "distancing ends, tariff rise 1"), (202210, "public-sector restriction"), (202305, "tariff rise 5")]


def fig_calendar():
    """Fig. 5: weather- and season-adjusted electricity of comparison health posts by quarter."""
    P = pd.read_csv(RES / "r3_calendar_profile.csv"); P = P[(P.use == "보건") & (P.n_buildings >= 20)].copy()
    P["t"] = P.quarter.str[:4].astype(int) + (P.quarter.str[-1].astype(int) - 1) / 4 + 0.125; summer = P.quarter.str.endswith("Q3")
    fig = plt.figure(figsize=(W, 2.4)); a = fig.add_axes([0.085, 0.2, 0.88, 0.72]); a.axhline(0, color=MID, lw=0.6)
    for m, fc, ec, lab in [(~summer, INK, INK, "comparison health posts"), (summer, "white", "0.55", "July–September (not reliable: cooling load not captured)")]:
        d = P[m]
        a.errorbar(d.t, 100 * d.pct, yerr=[100 * (d.pct - d.pct_lo), 100 * (d.pct_hi - d.pct)], fmt="o", ms=4, mfc=fc, mec=ec, mew=0.8, ecolor=ec, elinewidth=0.7, capsize=1.5, label=lab)
    for k, (ym, lab) in enumerate(EVENTS, 1):
        x = ym // 100 + ((ym % 100) - 1) / 12; a.axvline(x, color=MID, lw=0.6, ls=(0, (2, 2)))
        a.text(x, 1.01, f"({k})", transform=a.get_xaxis_transform(), fontsize=7, ha="center", va="bottom", color=MID)
    a.set_xlim(2020, 2025.8); a.set_ylim(-35, 35); a.set_ylabel("change against 2020Q3–2021Q2 (%)", fontsize=8); a.set_xlabel("calendar quarter", fontsize=8)
    a.legend(frameon=False, fontsize=7, loc="lower left"); _clean(a); a.tick_params(labelsize=8)
    _finish(fig, "fig_calendar.pdf")

if __name__ == "__main__":
    for f in (fig_flow, fig_event_long, fig_sensitivity_r3, fig_benchmark_r3, fig_calendar):
        try:
            f()
        except FileNotFoundError as ex:
            print(f"skipped {f.__name__}: {ex.filename} is not in results/ (building-level, rebuilt by run_all.sh)")
    print(sorted(p.name for p in FIG.glob("*.pdf")))

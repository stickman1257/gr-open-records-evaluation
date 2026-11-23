"""Electricity, gas and their sum for gas-metered daycare centres (Table 3)."""
import os, sys, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES
import s04_analysis as A
from a_interim_checks import DC, mi, pct

LABEL = "exploratory; added after the interim results were seen, before the final data"
ROW_LABEL = "daycares with both carriers observed in the same months"
GAS_METER_LABEL = "daycares with a confirmed gas meter"
MIN_TOTAL_MAIN, MIN_TOTAL_CTRL = 30, 10
NB = int(os.environ.get("IMP_NBOOT", "200"))
KEEP = ["pct", "pct_ci", "att_log", "ci_log", "n_treated", "n_control", "support_rows", "pretrend_p", "nboot", "median_bld_pct"]


def main():
    B, E = A.load(); Bi = B.drop_duplicates("pid").set_index("pid")
    el = E[(E.carrier == "elec") & (E.kwh > 0)][["pid", "ym"]]; ga = E[(E.carrier == "gas") & (E.kwh > 0)][["pid", "ym"]]
    both = set(map(tuple, el.merge(ga, on=["pid", "ym"]).to_numpy())); Eb = E[pd.Series(list(zip(E.pid, E.ym)), index=E.index).isin(both)]
    kw = dict(A.PRIMARY_IMP, uses=(DC,))
    out = dict(label=LABEL, row_label=ROW_LABEL, generated=time.strftime("%Y-%m-%d %H:%M"), nboot=NB,
               confirmed_gas_meter_label=GAS_METER_LABEL,
               definition="treated daycare centres on the panel of building-months with both carriers observed (> 0); primary imputation settings; "
                          "not the same group as 'confirmed gas meter' (>= 1 gas month with status ok)",
               both_carriers_same_months={}, elec_all_months={}); rows = []; tp = set(); tau = {}
    for car in ["elec", "gas", "total"]:
        try:
            r, P = A.imputation_did(B, Eb, car, nboot=NB, return_tau=True, **kw)
            out["both_carriers_same_months"][car] = dict({k: r.get(k) for k in KEEP}, status="ok", season={k: v["pct"] for k, v in r["season"].items()})
            rows.append(dict(block="both_carriers_same_months", sample=ROW_LABEL, carrier=car, status="ok", pct=r["pct"], pct_lo=r["pct_ci"][0], pct_hi=r["pct_ci"][1],
                             n_treated=r["n_treated"], n_control=r["n_control"], pretrend_p=r.get("pretrend_p"), label=LABEL))
            tau[car] = P.groupby("pid").tau.mean(); tp |= set(P.pid)
        except Exception as ex:
            out["both_carriers_same_months"][car] = dict(status=f"{type(ex).__name__}: {ex}"[:200])
            rows.append(dict(block="both_carriers_same_months", sample=ROW_LABEL, carrier=car, status=f"{type(ex).__name__}: {ex}"[:200], label=LABEL))
    S = A.imputation_setup(B, E, "elec", **kw)
    if S is not None:
        td = set(S["T"].pid); gas_any = set(E[E.carrier == "gas"].pid)
        for name, ids, desc in [("confirmed_gas_meter", td & gas_any, "treated daycares with >= 1 gas month with status ok (group of the level comparison)"),
                                ("no_confirmed_gas_meter", td - gas_any, "treated daycares without any gas month with status ok"),
                                ("both_carriers_same_months_sample", td & tp, "treated daycares of the both-carriers same-month sample (total-energy sample)"),
                                ("not_in_both_carriers_sample", td - tp, "all other treated daycares of the electricity sample")]:
            if len(ids) < 10:
                out["elec_all_months"][name] = dict(status=f"not estimated: {len(ids)} treated buildings (< 10)", description=desc); continue
            try:
                r = A.imputation_did(B, E, "elec", nboot=NB, pretrend=False, treated_keep=ids, **kw)
                out["elec_all_months"][name] = dict({k: r.get(k) for k in KEEP if k != "pretrend_p"}, status="ok", description=desc)
                rows.append(dict(block="elec_all_months", sample=name, carrier="elec", status="ok", pct=r["pct"], pct_lo=r["pct_ci"][0], pct_hi=r["pct_ci"][1],
                                 n_treated=r["n_treated"], n_control=r["n_control"], label=LABEL))
            except Exception as ex: out["elec_all_months"][name] = dict(status=f"{type(ex).__name__}: {ex}"[:200], description=desc)
    if tp:
        e = Eb[Eb.pid.isin(tp)]; e = e[(mi(e.ym) - mi(e.start_ym.astype(int))).between(-12, -1)]
        lv = e.assign(i=e.kwh / e.gr_gfa).groupby(["pid", "carrier"]).i.mean().unstack()
        if {"elec", "gas"} <= set(lv.columns):
            out["pre_level_kwh_m2_month"] = dict(elec=float(lv.elec.mean()), gas=float(lv.gas.mean()), gas_share_of_total=float((lv.gas / (lv.gas + lv.elec)).mean()), n_buildings=int(len(lv)))
        X = Bi.loc[sorted(tp)]; txt = X.measures_txt.astype(str)
        out["works_text_share"] = dict(n=int(len(X)), boiler=float(X.boiler.mean()), hvac_replacement=float(X.hvac.mean()), heat_source_change_stated=float(txt.str.contains("열원").mean()),
                                       electric_capacity_upgrade=float(txt.str.contains(r"전기\s*용량|전기\s*증설|수전").mean()), instant_water_heater=float(txt.str.contains("순간온수기").mean()),
                                       renewable_entry=float(X.renewable.mean()))
        if {"elec", "gas"} <= set(tau):
            bt = pd.DataFrame(tau).dropna()
            out["building_level"] = dict(n=int(len(bt)), corr_elec_gas=float(bt.elec.corr(bt.gas)), share_elec_down_gas_up=float(((bt.elec < 0) & (bt.gas > 0)).mean()),
                                         share_both_down=float(((bt.elec < 0) & (bt.gas < 0)).mean()))
    t = out["both_carriers_same_months"].get("total", {}); nt, nc = t.get("n_treated") or 0, t.get("n_control") or 0
    rule = dict(rule=f"main text only if >= {MIN_TOTAL_MAIN} treated buildings and >= {MIN_TOTAL_CTRL} gas-metered control-pool buildings in the fit",
                n_treated=int(nt), n_control=int(nc), condition_treated_ge30=bool(nt >= MIN_TOTAL_MAIN), condition_control_ge10=bool(nc >= MIN_TOTAL_CTRL),
                main_text=bool(nt >= MIN_TOTAL_MAIN and nc >= MIN_TOTAL_CTRL), checked_on="both_carriers_same_months.total of this file")
    tf = RES / "a_total_energy.json"
    if tf.exists():
        try:
            tj = json.loads(tf.read_text(encoding="utf-8"))
            rule.update(a_total_energy_json=dict(n_treated=tj.get("n_treated"), n_control=tj.get("n_control"), main_text=tj.get("main_text")),
                        agrees_with_a_total_energy_json=bool(tj.get("n_treated") == nt and tj.get("main_text") == rule["main_text"]),
                        note="a_total_energy.json counts the pooled total-energy row (carrier_frame 'total': zero-gas months kept); small differences in n_control are possible")
        except Exception as ex: rule["a_total_energy_json"] = f"{type(ex).__name__}: {ex}"[:120]
    out["total_energy_rule"] = rule
    (RES / "a_gas_heated_daycares.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    pd.DataFrame(rows).to_csv(RES / "a_gas_heated_daycares.csv", index=False)
    print(pd.DataFrame(rows).drop(columns=["label"]).round(4).to_string(index=False))

if __name__ == "__main__":
    main()

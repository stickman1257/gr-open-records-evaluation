"""Checks whether the building-energy service holds months before 2020-01."""
import json, os, sys, time
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, PROC
import s03_energy as s3

PROBE_MONTHS = [201801, 201807, 201901, 201907, 201912]
POS_MONTHS = list(range(202001, 202013))
N_PARCELS = 12

def main():
    E = pd.read_csv(PROC / "energy_monthly.csv")
    pos = E[E.ym.isin(POS_MONTHS) & (E.status == "ok")]
    by_car = {c: sorted(pos[pos.carrier == c].pid.unique())[:N_PARCELS] for c in ("elec", "gas")}
    pids = sorted(set(by_car["elec"]) | set(by_car["gas"]))
    d = s3.build_parcels().set_index("pid")
    store = s3.Store("_probe")
    rows, quota = [], False
    for carrier in ("elec", "gas"):
        for pid in by_car[carrier]:
            for ym in PROBE_MONTHS:
                if quota: break
                try:
                    st, kwh, n, code = s3.fetch_month(store, carrier, d.loc[pid].to_dict() | {"pid": pid}, ym)
                except s3.QuotaExceeded:
                    quota = True; st, kwh, n, code = "quota", None, 0, None
                rows.append(dict(pid=pid, carrier=carrier, ym=ym, status=st, n_items=int(n)))
    R = pd.DataFrame(rows)
    out = dict(
        source="src/pi_api_probe.py", run_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        design=dict(parcels=len(pids), parcels_by_carrier={c: len(v) for c, v in by_car.items()}, probe_months=PROBE_MONTHS,
                    positive_control="same carrier status ok in at least one month of 2020 (cache); first 12 parcels per carrier by id"),
        quota_hit=quota, new_calls=store.n_new,
        status_by_carrier_month={f"{c}_{ym}": g.status.value_counts().to_dict()
                                 for (c, ym), g in R.groupby(["carrier", "ym"])} if len(R) else {},
        any_ok_before_2020=bool((R.status == "ok").any()) if len(R) else None)
    (ROOT / "results" / "a_api_probe.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))

if __name__ == "__main__":
    main()

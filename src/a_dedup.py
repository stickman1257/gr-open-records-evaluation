"""One physical building = one unit; duplicate project records."""
import os, sys, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES

REASON = "same building-register key (reg_pk) as an earlier record; D23: earliest record kept (selection year, then start month)"


def dedup_physical(B):
    k = B.reg_pk.notna()
    o = B[k].assign(_s=pd.to_numeric(B.loc[k, "start_ym"], errors="coerce").fillna(999999)).sort_values(["reg_pk", "year", "_s", "pid"], kind="stable")
    g = o.groupby("reg_pk")
    first = g.pid.transform("first"); n = g.pid.transform("size"); roles = g.role.transform(lambda v: "+".join(sorted(map(str, v))))
    d = o[o.pid != first]; Bi = B.drop_duplicates("pid").set_index("pid"); kp = first[d.index].to_numpy()
    D = pd.DataFrame({"pid": d.pid.to_numpy(), "reg_pk": d.reg_pk.to_numpy(), "role": d.role.to_numpy(), "year": d.year.to_numpy(),
                      "start_ym": d.start_ym.to_numpy(), "end_ym": d.end_ym.to_numpy(), "kept_pid": kp, "kept_role": Bi.role.reindex(kp).to_numpy(),
                      "kept_year": Bi.year.reindex(kp).to_numpy(), "kept_start_ym": Bi.start_ym.reindex(kp).to_numpy(),
                      "kept_end_ym": Bi.end_ym.reindex(kp).to_numpy(), "records_of_building": n[d.index].to_numpy(),
                      "roles_of_building": roles[d.index].to_numpy(), "name": d.name_list.to_numpy(),
                      "use_class": d.use_class.to_numpy() if "use_class" in d else np.nan, "link_grade": d.link_grade.to_numpy(), "reason": REASON})
    return B[~B.pid.isin(set(d.pid))].reset_index(drop=True), D


def load(dedup=True):
    """s04_analysis.load() (regraded records, QA gate) and then the D23 rule. Returns B, E, dropped-record frame (empty if dedup=False)."""
    import s04_analysis as A
    B, E = A.load(dedup=False)
    if not dedup: return B, E, pd.DataFrame()
    B2, D = dedup_physical(B)
    return B2, E[E.pid.isin(set(B2.pid))].copy(), D


def summary(B, D):
    per = D.drop_duplicates("reg_pk")
    return dict(generated=time.strftime("%Y-%m-%d %H:%M"), rule=__doc__.split("\n\n")[1].replace("\n", " "),
                records_before=int(len(B) + len(D)), records_after=int(len(B)), records_dropped=int(len(D)),
                physical_buildings_with_more_than_one_record=int(per.reg_pk.nunique()),
                records_of_those_buildings=int(per.records_of_building.sum()),
                by_roles_of_building={str(k): int(v) for k, v in per.roles_of_building.value_counts().items()},
                dropped_by_role={str(k): int(v) for k, v in D.role.value_counts().items()},
                dropped_grade_A_by_use={f"{u}|{r}": int(v) for (u, r), v in D[D.link_grade == "A_cand"].groupby(["use_class", "role"]).size().items()},
                kept_by_role={str(k): int(v) for k, v in B.role.value_counts().items()},
                dropped_record_has_different_grade_or_use=int(sum(1 for r in D.itertuples() if False)),
                note="declared as a change made after the interim results were seen (PLAN.md, 11/14 02:19, D23)")


if __name__ == "__main__":
    B, E, D = load()
    D.to_csv(RES / "a_dedup.csv", index=False)
    s = summary(B, D); s.pop("dropped_record_has_different_grade_or_use")
    (RES / "a_dedup.json").write_text(json.dumps(s, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps(s, indent=1, ensure_ascii=False, default=float))
    print(D[["pid", "role", "year", "start_ym", "kept_pid", "kept_role", "kept_year", "kept_start_ym", "roles_of_building"]].to_string(index=False))

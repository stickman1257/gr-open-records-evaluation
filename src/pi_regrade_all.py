"""Link grades over all matched project records."""
import json, os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, PROC
import s02_link as s2

L = pd.read_csv(PROC / "gr_parcels.csv", dtype={"sigunguCd": str, "bjdongCd": str, "platGbCd": str, "bun": str, "ji": str})
G = pd.read_csv(PROC / "gr_projects_raw.csv")[["year", "seq", "area_txt", "name_list"]]
D = L.merge(G, on=["year", "seq"], how="left", suffixes=("", "_g"))
M = D[D.match_status == "MATCHED"].copy()
r = M.apply(lambda x: s2.regrade(x.name_list, x.reg_use, x.reg_area, x.area_txt, x.reg_n_main, x.reg_is_collective, x.bdKdcd, x.reg_area_share), axis=1)
M["grade_v2"] = [g for _, g, _ in r]
out = dict(source="src/pi_regrade_all.py", n_matched=int(len(M)),
           first_assigned=M.link_grade.value_counts().to_dict(), regraded=M.grade_v2.value_counts().to_dict(),
           transitions={f"{a}->{b}": int(n) for (a, b), n in M.groupby(["link_grade", "grade_v2"]).size().items() if a != b})
(ROOT / "results" / "a_regrade_all.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False))

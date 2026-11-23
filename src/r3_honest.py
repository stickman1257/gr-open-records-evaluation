"""Rambachan-Roth sensitivity analysis (Fig. 3); needs the honestdid package."""
import os, sys, json, time, pathlib, warnings, multiprocessing as mp
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import honestdid as H

ROOT = pathlib.Path(__file__).resolve().parents[1]; RES = ROOT / "results"; NPROC = int(os.environ.get("V4_NPROC", "11"))
with np.load(RES / "r3_es_inputs.npz", allow_pickle=False) as z:
    b0, V0, pre_ix, ant, post_ix = z["b"], z["V"], list(z["pre_ix"]), int(z["ant_ix"]), list(z["post_ix"])
IX = pre_ix + [ant] + post_ix; B = b0[IX]; S = V0[np.ix_(IX, IX)]; NPRE = len(pre_ix); NPOST = 1 + len(post_ix)
L = np.r_[0.0, np.full(len(post_ix), 1 / len(post_ix))]


def rm(mb):
    sd = float(np.sqrt(L @ S[NPRE:, NPRE:] @ L)); wid = False
    o = H.createSensitivityResults_relativeMagnitudes(B, S, NPRE, NPOST, Mbarvec=[mb], l_vec=L).iloc[0]; lb, ub = float(o.lb), float(o.ub)
    if lb <= -20 * sd + 1e-9 or ub >= 20 * sd - 1e-9:
        o = H.createSensitivityResults_relativeMagnitudes(B, S, NPRE, NPOST, Mbarvec=[mb], l_vec=L, gridPoints=4000, grid_lb=-80 * sd, grid_ub=80 * sd).iloc[0]
        lb, ub = float(o.lb), float(o.ub); wid = True
    return dict(Mbar=round(float(mb), 2), lb=lb, ub=ub, includes_zero=bool(lb <= 0 <= ub), grid_widened=wid, method="C-LF")


def main():
    t0 = time.time()
    with mp.get_context("fork").Pool(NPROC) as p:
        rows = p.map(rm, list(np.round(np.arange(0, 1.5001, 0.1), 2)), chunksize=1)
        inc = sorted(r["Mbar"] for r in rows if r["includes_zero"])
        if inc and inc[0] > 0: rows += p.map(rm, list(np.round(np.arange(inc[0] - 0.09, inc[0] - 0.005, 0.01), 2)), chunksize=1)
    D = pd.DataFrame(rows).sort_values("Mbar"); inc = D[D.includes_zero]; bd = float(inc.Mbar.min()) if len(inc) else None
    D["breakdown_Mbar"] = bd; D.to_csv(RES / "r3_honestdid.csv", index=False)
    sd = H.createSensitivityResults(B, S, NPRE, NPOST, l_vec=L)
    pd.DataFrame([dict(M=float(r.M), lb=float(r.lb), ub=float(r.ub), includes_zero=bool(r.lb <= 0 <= r.ub), method=r.method) for r in sd.itertuples()]).to_csv(RES / "r3_honestdid_sd.csv", index=False)
    orig = H.constructOriginalCS(B, S, NPRE, NPOST, l_vec=L)
    meta = dict(numPrePeriods=NPRE, numPostPeriods=NPOST, l_vec=L.tolist(), theta=float(L @ B[NPRE:]), original_ci=[float(orig.lb.iloc[0]), float(orig.ub.iloc[0])],
                breakdown_Mbar=bd, monotone=bool(D.includes_zero.is_monotonic_increasing), runtime_s=round(time.time() - t0, 1), definition=__doc__)
    (RES / "r3_honestdid_meta.json").write_text(json.dumps(meta, indent=1)); print(meta)


if __name__ == "__main__":
    main()

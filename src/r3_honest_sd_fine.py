"""Smoothness-bound sensitivity on a fine grid of M."""
import sys, pathlib, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import honestdid as H
import r3_honest as RH
o = H.createSensitivityResults(RH.B, RH.S, RH.NPRE, RH.NPOST, l_vec=RH.L, Mvec=list(np.round(np.arange(0, 0.0401, 0.002), 3)))
D = pd.DataFrame([dict(M=float(r.M), lb=float(r.lb), ub=float(r.ub), includes_zero=bool(r.lb <= 0 <= r.ub), method=r.method) for r in o.itertuples()])
D.to_csv(RH.RES / "r3_honestdid_sd_fine.csv", index=False); print(D.to_string())

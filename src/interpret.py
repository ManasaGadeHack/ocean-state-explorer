"""PC interpretation: correlation of PC scores with SST, 20C isotherm depth, latitude, longitude."""
import numpy as np
import pandas as pd
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"
z = np.load(R / "explorer.npz")
m = pd.read_csv(R / "explorer_meta.csv")
S = (z["X"].astype(float) - z["pmean"]) @ z["comps"][:4].T
rows = {f"PC{i+1}": {c: m[c].corr(pd.Series(S[:, i])) for c in ["sst", "d20", "lat", "lon"]} for i in range(4)}
out = pd.DataFrame(rows).T.round(2)
out.to_csv(R / "pc_correlations.csv")
print(out)
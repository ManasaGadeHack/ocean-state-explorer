"""NetCDF -> (samples x features) matrix. One sample = one (time, lat, lon) water column;
features = temperature then salinity at every depth level. Temporal train/test split, standardisation."""
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.preprocessing import StandardScaler
import config as C


def load_matrix(path, variables=("thetao", "so"), space_stride=C.SPACE_STRIDE, time_stride=C.TIME_STRIDE):
    ds = xr.open_dataset(path)
    ds = ds.isel(latitude=slice(None, None, space_stride), longitude=slice(None, None, space_stride),
                 time=slice(None, None, time_stride))
    depth = ds["depth"].values
    A = np.concatenate([ds[v].transpose("time", "latitude", "longitude", "depth").values for v in variables], axis=-1)
    T, Y, X, F = A.shape
    time, lat, lon = ds["time"].values, ds["latitude"].values, ds["longitude"].values
    A = A.reshape(-1, F)
    meta = pd.DataFrame({"time": np.repeat(time, Y * X),
                         "lat": np.tile(np.repeat(lat, X), T),
                         "lon": np.tile(lon, Y * T)})
    ok = np.isfinite(A).all(axis=1)          # drops land and shallow columns missing deep levels
    meta = meta[ok].reset_index(drop=True)
    meta["month"] = pd.DatetimeIndex(meta["time"]).month
    names = [f"{v}@{d:.0f}m" for v in variables for d in depth]
    return A[ok].astype(np.float64), meta, names, depth


def isotherm_depth(T, depth, level=20.0):
    """Depth of the `level` degC isotherm per row (thermocline proxy). NaN if not crossed."""
    out = np.full(len(T), np.nan)
    for i, row in enumerate(T):
        out[i] = np.interp(level, row[::-1], depth[::-1], left=np.nan, right=np.nan)
    return out


def prepare(path, variables=("thetao", "so")):
    X, meta, names, depth = load_matrix(path, variables)
    times = np.sort(meta["time"].unique())
    cut = times[int(C.TRAIN_FRAC * len(times))]
    tr = (meta["time"] < cut).values
    rng = np.random.default_rng(C.SEED)

    def sub(mask, n):
        idx = np.flatnonzero(mask)
        return rng.choice(idx, min(n, len(idx)), replace=False) if len(idx) > n else idx

    itr, ite = sub(tr, C.MAX_TRAIN), sub(~tr, C.MAX_TEST)
    scaler = StandardScaler().fit(X[itr])      # fit on train only (no leakage)
    return dict(Xtr=scaler.transform(X[itr]), Xte=scaler.transform(X[ite]),
                meta_tr=meta.iloc[itr].reset_index(drop=True), meta_te=meta.iloc[ite].reset_index(drop=True),
                scaler=scaler, names=names, depth=depth, nd=len(depth), cut=str(cut)[:10])

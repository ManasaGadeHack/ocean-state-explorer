"""Synthetic GLORYS-like file (4 hidden modes + noise) ONLY to test the pipeline before real data is downloaded.
Never use its numbers in the report. Usage: python src/make_synthetic.py out.nc"""
import sys
import numpy as np
import pandas as pd
import xarray as xr

rng = np.random.default_rng(0)
time = pd.date_range("2012-01-01", "2021-12-01", freq="MS")
lat, lon = np.arange(8, 18.01, 0.5), np.arange(82, 92.01, 0.5)
depth = np.array([0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000], float)
T, Y, X, Z = len(time), len(lat), len(lon), len(depth)
t = np.arange(T)[:, None, None, None]
la = ((lat - 8) / 10)[None, :, None, None]
lo = ((lon - 82) / 10)[None, None, :, None]
z = depth[None, None, None, :]


def ar1(n, phi=0.9):
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal()
    return (x / x.std())[:, None, None, None]


f1 = np.sin(2 * np.pi * t / 12 + 0.8 * la)                       # seasonal heating
f2 = ar1(T) * np.cos(2 * la + lo) + 0.4 * np.sin(2 * np.pi * t / 12)   # thermocline depth anomaly
f3 = ar1(T, 0.8) * (0.5 + la)                                    # surface freshening
f4 = (t / T) * (0.5 + lo)                                        # slow deep-water drift
Ts = 28 - 2 * la + 1.5 * f1
D = 90 + 35 * f2
temp = 5 + (Ts - 5) / (1 + np.exp((z - D) / 40)) + 0.5 * f4 * (z / 1000)
sal = 34.6 + 0.6 * np.exp(-z / 200) - 0.5 * f3 * np.exp(-z / 120) + 0.15 * f4 * (z / 1000)
temp = temp + rng.normal(0, 0.03, temp.shape)
sal = sal + rng.normal(0, 0.005, sal.shape)
temp, sal = [np.broadcast_to(a, (T, Y, X, Z)).copy() for a in (temp, sal)]
land = (lon[None, :] > 90) & (lat[:, None] > 14)                 # NaN corner = land
shelf = lon[None, :] < 83                                        # shallow strip: deep levels missing
for a in (temp, sal):
    a[:, land, :] = np.nan
    a[:, np.broadcast_to(shelf, (Y, X)), 12:] = np.nan
dims = ("time", "latitude", "longitude", "depth")
xr.Dataset({"thetao": (dims, temp.astype("float32")), "so": (dims, sal.astype("float32"))},
           coords=dict(time=time, latitude=lat, longitude=lon, depth=depth)).to_netcdf(sys.argv[1])
print("wrote", sys.argv[1])

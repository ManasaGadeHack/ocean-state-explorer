"""Central settings. Change region / period / sizes here only."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
RESULTS = ROOT / "results"

# Copernicus Marine GLORYS12 reanalysis, monthly means (verify ID with: copernicusmarine describe -c GLOBAL_MULTIYEAR_PHY_001_030)
DATASET_ID = "cmems_mod_glo_phy_my_0.083deg_P1M-m"
VARIABLES = ["thetao", "so"]            # temperature, salinity
BBOX = dict(lon=(82.0, 92.0), lat=(8.0, 18.0))   # Bay of Bengal box
DEPTH = (0.0, 1000.0)
TIME = ("2012-01-01", "2021-12-31")
RAW_FILE = DATA_DIR / "glorys_bob_monthly.nc"

# Sampling / split
SPACE_STRIDE = 2      # keep every 2nd grid point
TIME_STRIDE = 1
TRAIN_FRAC = 0.7      # first 70% of time steps = train, rest = test (temporal split)
MAX_TRAIN = 40000
MAX_TEST = 20000
SEED = 42

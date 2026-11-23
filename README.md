# Learning Compact Representations of Subsurface Ocean State

INT396 Unsupervised Learning | Section K3P24WV | Group G1 | CA-1
Members: Abhiram Alagandula (12406172), Sai Manasa Gade (12404684)
Problem source: Smart India Hackathon 2026, PS ID SIH26066 (inspired by, not a full solution).

**Question:** how many numbers does the subsurface ocean state (temperature + salinity, 0-1000 m) really need?
PCA vs a random-projection baseline at equal dimension; intrinsic dimension (TwoNN); optional Isomap/UMAP comparison.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
copernicusmarine login                    # free account at data.marine.copernicus.eu
python src/download_data.py               # GLORYS subset -> data/raw/ (region/period in src/config.py)
python src/experiments.py --manifold      # all metrics, figures, tables -> results/   (drop --manifold for a fast run)
streamlit run app/streamlit_app.py        # Ocean state explorer
```
Pipeline test without real data (numbers meaningless): `python src/make_synthetic.py data/raw/synthetic.nc && python src/experiments.py --data data/raw/synthetic.nc`

## Data
Copernicus Marine GLORYS12 global reanalysis (monthly), variables `thetao`, `so`. Not stored in the repo (see `src/config.py` for dataset ID, region, period). Cite and check the licence on the Copernicus product page.

## Layout
`src/` code | `notebooks/` EDA | `results/` tables + figures used in the report | `app/` Streamlit explorer | `report/`, `presentation/`



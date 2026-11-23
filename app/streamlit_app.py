"""Ocean state explorer. Run: streamlit run app/streamlit_app.py  (needs results/ from src/experiments.py)"""
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go

R = Path(__file__).resolve().parents[1] / "results"
st.set_page_config(page_title="Ocean State Explorer", layout="wide")


@st.cache_data
def load():
    z = np.load(R / "explorer.npz")
    meta = pd.read_csv(R / "explorer_meta.csv", parse_dates=["time"])
    return {k: z[k] for k in z.files}, meta, pd.read_csv(R / "metrics_pca.csv"), pd.read_csv(R / "metrics_rp.csv")


Z, meta, pca_m, rp_m = load()
X, comps, pmean, depth, nd = Z["X"].astype(float), Z["comps"], Z["pmean"], Z["depth"], int(Z["nd"])
scale, mean, evr = Z["scale"], Z["mean"], Z["evr"]
meta = meta.assign(idx=np.arange(len(meta)))
embs = [k[4:] for k in Z if k.startswith("emb_")]

st.title("Ocean State Explorer")
st.caption("Each dot = one ocean water column (temperature + salinity from the surface to 1000 m) compressed to a few numbers.")
k = st.sidebar.slider("Dimensions kept (k)", 1, len(comps), min(5, len(comps)))
proj = st.sidebar.selectbox("2D projection", ["PCA (PC1 vs PC2)"] + embs)
color = st.sidebar.selectbox("Colour by", ["sst", "d20", "month", "lat", "lon", "split"])
st.sidebar.metric(f"Variance kept by {k} PCs (train)", f"{100 * evr[:k].sum():.2f}%")

tab1, tab2, tab3 = st.tabs(["Explorer", "Metrics vs k", "Components"])

with tab1:
    S = (X - pmean) @ comps[:k].T
    Xh = S @ comps[:k] + pmean
    xy = (X - pmean) @ comps[:2].T if proj.startswith("PCA") else Z["emb_" + proj]
    df = meta.assign(x=xy[:, 0], y=xy[:, 1])
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = px.scatter(df, x="x", y="y", color=color, custom_data=["idx"], opacity=.7, height=520,
                         hover_data=["lat", "lon", "month", "sst"], labels=dict(x="dim 1", y="dim 2"))
        ev = st.plotly_chart(fig, on_select="rerun", selection_mode="points", key="sc", width="stretch")
        pts = ev.selection["points"] if ev and ev.selection else []
        if pts:
            st.session_state["i"] = int(pts[0]["customdata"][0])
    with c2:
        i = st.number_input("Sample index (or click a dot)", 0, len(df) - 1, st.session_state.get("i", 0))
        orig, rec = X[i] * scale + mean, Xh[i] * scale + mean
        st.write(f"**{df.time[i]:%b %Y}**, lat {df.lat[i]:.2f}, lon {df.lon[i]:.2f}, split: {df.split[i]}")
        f = make_subplots(1, 2, subplot_titles=("Temperature (degC)", "Salinity (PSU)"), shared_yaxes=True)
        for j, sl in enumerate([slice(0, nd), slice(nd, 2 * nd)], 1):
            f.add_trace(go.Scatter(x=orig[sl], y=depth, name="original", line=dict(color="#1f77b4"), showlegend=j == 1), 1, j)
            f.add_trace(go.Scatter(x=rec[sl], y=depth, name=f"reconstructed (k={k})", line=dict(color="#d62728", dash="dash"), showlegend=j == 1), 1, j)
        f.update_yaxes(autorange="reversed", title_text="Depth (m)", col=1)
        f.update_layout(height=430, margin=dict(t=40))
        st.plotly_chart(f, width="stretch")
        st.write(f"RMSE this profile: {np.sqrt(((orig[:nd] - rec[:nd]) ** 2).mean()):.3f} degC, "
                 f"{np.sqrt(((orig[nd:] - rec[nd:]) ** 2).mean()):.4f} PSU")

with tab2:
    cols = st.columns(2)
    for n, (m, t) in enumerate({"ve_test": "Variance explained (test)", "rmse_test": "Reconstruction RMSE (test)",
                                "trust": "Trustworthiness", "knn": "kNN overlap"}.items()):
        g = go.Figure()
        g.add_trace(go.Scatter(x=pca_m.k, y=pca_m[m], name="PCA", mode="lines+markers"))
        g.add_trace(go.Scatter(x=rp_m.k, y=rp_m[m], name="Random projection", mode="lines+markers",
                               error_y=dict(type="data", array=rp_m[m + "_std"])))
        g.add_vline(x=k, line_dash="dot")
        g.update_layout(title=t, xaxis_title="k", height=330, margin=dict(t=40))
        cols[n % 2].plotly_chart(g, width="stretch")

with tab3:
    npc = st.slider("Components to show", 1, min(6, len(comps)), 3)
    f = make_subplots(1, 2, subplot_titles=("Temperature loadings", "Salinity loadings"), shared_yaxes=True)
    for p in range(npc):
        for j, sl in enumerate([slice(0, nd), slice(nd, 2 * nd)], 1):
            f.add_trace(go.Scatter(x=comps[p, sl], y=depth, name=f"PC{p + 1} ({100 * evr[p]:.1f}%)", legendgroup=str(p),
                                   showlegend=j == 1, line=dict(color=px.colors.qualitative.Set2[p])), 1, j)
    f.update_yaxes(autorange="reversed", title_text="Depth (m)", col=1)
    f.update_layout(height=450)
    st.plotly_chart(f, width="stretch")
    
    st.subheader("Physical interpretation (hypotheses)")
    if (R / "pc_correlations.csv").exists():
        st.caption("Correlation of each PC score with surface temperature (sst), 20C isotherm depth (d20), latitude and longitude.")
        st.dataframe(pd.read_csv(R / "pc_correlations.csv", index_col=0))
    st.markdown(
        "- **PC1**: surface layer with a north-south gradient (linked to SST and latitude, not thermocline depth)\n"
        "- **PC2**: warm upper-ocean heat content (SST and thermocline depth)\n"
        "- **PC3**: thermocline depth, a surface-versus-subsurface dipole around 200 m\n"
        "- **PC4**: no clean physical label"
    )

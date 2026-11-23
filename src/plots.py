"""Report figures (saved to results/figures)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LAB = {"ve_test": "Variance explained (test)", "rmse_test": "Reconstruction RMSE (std. units, test)",
       "trust": "Trustworthiness (k=10)", "knn": "kNN overlap (k=10)"}


def scree(evr, path):
    cum, k = np.cumsum(evr), np.arange(1, len(evr) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].bar(k, evr * 100); ax[0].set(xlabel="Component", ylabel="Variance (%)", title="Scree plot")
    ax[1].plot(k, cum * 100, "o-")
    for t in (90, 95, 99):
        ax[1].axhline(t, ls=":", c="grey")
    ax[1].set(xlabel="Components k", ylabel="Cumulative variance (%)", title="Cumulative variance")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def metrics_vs_k(pca, rp, mani, path):
    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    for ax, m in zip(axs.ravel(), LAB):
        ax.plot(pca["k"], pca[m], "o-", label="PCA")
        ax.errorbar(rp["k"], rp[m], yerr=rp[m + "_std"], fmt="s--", capsize=2, label="Random projection (baseline)")
        if mani is not None:
            for name, g in mani[mani.method != "PCA"].groupby("method"):
                ax.plot(g["k"], g[m], "^:", label=name)
        ax.set(xlabel="Dimension k", title=LAB[m]); ax.grid(alpha=.3)
    axs[0, 0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def per_depth(df, depth, nd, k, path):
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True)
    for ax, (sl, ttl, unit) in zip(axs, [(slice(0, nd), "Temperature", "degC"), (slice(nd, 2 * nd), "Salinity", "PSU")]):
        ax.plot(df["pca"].values[sl], depth, "o-", label="PCA"); ax.plot(df["rp"].values[sl], depth, "s--", label="Random proj.")
        ax.set(xlabel=f"RMSE ({unit})", title=f"{ttl}, k={k}"); ax.grid(alpha=.3)
    axs[0].invert_yaxis(); axs[0].set_ylabel("Depth (m)"); axs[0].legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def pc_profiles(comps, depth, nd, path, npc=3):
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True)
    for i in range(npc):
        axs[0].plot(comps[i, :nd], depth, label=f"PC{i + 1}"); axs[1].plot(comps[i, nd:2 * nd], depth, label=f"PC{i + 1}")
    axs[0].set(title="Temperature loadings", xlabel="Loading", ylabel="Depth (m)"); axs[1].set(title="Salinity loadings", xlabel="Loading")
    axs[0].invert_yaxis(); axs[0].legend()
    for a in axs:
        a.axvline(0, c="grey", lw=.5); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def pc_maps(scores, meta, path, npc=3):
    df = meta[["lat", "lon"]].copy()
    fig, axs = plt.subplots(1, npc, figsize=(4.5 * npc, 4))
    for i, ax in enumerate(np.atleast_1d(axs)):
        df["s"] = scores[:, i]
        g = df.groupby(["lat", "lon"])["s"].mean().reset_index()
        sc = ax.scatter(g["lon"], g["lat"], c=g["s"], cmap="RdBu_r", s=14)
        ax.set(title=f"Mean PC{i + 1} score", xlabel="Lon", ylabel="Lat"); fig.colorbar(sc, ax=ax)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

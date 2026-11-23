"""Full study: random-projection baseline vs PCA (vs optional Isomap/UMAP), intrinsic dimension,
cluster validity, figures, and the data the Streamlit explorer needs.
Usage: python src/experiments.py [--data FILE] [--manifold]"""
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.random_projection import GaussianRandomProjection

import config as C
import metrics as M
import plots
from preprocess import isotherm_depth, prepare
from twonn import twonn, twonn_mle

METR = ["ve_test", "rmse_test", "trust", "knn"]


def pca_eval(pca, X, k, idx):
    W = pca.components_[:k]
    Z = (X - pca.mean_) @ W.T
    r = M.recon_stats(X, Z @ W + pca.mean_)
    return dict(ve_test=r["ve"], rmse_test=r["rmse"], trust=M.trust(X[idx], Z[idx]), knn=M.knn_overlap(X[idx], Z[idx])), Z


def sil(Z, c):
    return silhouette_score(Z, KMeans(c, n_init=10, random_state=C.SEED).fit_predict(Z))


def manifold_block(a, D, pca, k_star, rng):
    from sklearn.manifold import Isomap
    methods = {"Isomap": lambda m: Isomap(n_neighbors=a.iso_nn, n_components=m)}
    try:
        import umap
        methods["UMAP"] = lambda m: umap.UMAP(n_components=m, n_neighbors=15, random_state=C.SEED)
    except ImportError:
        print("umap-learn not installed: skipping UMAP")
    Xtr, Xte = D["Xtr"], D["Xte"]
    Xf = Xtr[rng.choice(len(Xtr), min(a.man_fit, len(Xtr)), replace=False)]
    Xt = Xte[rng.choice(len(Xte), min(a.man_test, len(Xte)), replace=False)]
    dims = sorted({m for m in (2, 3, 5, 8, k_star) if m <= Xtr.shape[1]})
    rows, est2 = [], {}
    for m in dims:
        o, Zp = pca_eval(pca, Xt, m, np.arange(len(Xt)))
        rows.append(dict(o, method="PCA", k=m, silhouette=sil(Zp, a.clusters)))
        for name, mk in methods.items():
            est = mk(m)
            Zf = est.fit_transform(Xf)
            Zt = est.transform(Xt)
            Xh = KNeighborsRegressor(10, weights="distance").fit(Zf, Xf).predict(Zt)   # learned inverse map
            r = M.recon_stats(Xt, Xh)
            rows.append(dict(method=name, k=m, ve_test=r["ve"], rmse_test=r["rmse"], trust=M.trust(Xt, Zt),
                             knn=M.knn_overlap(Xt, Zt), silhouette=sil(Zt, a.clusters)))
            if m == 2:
                est2[name] = est
            print(f"  {name} dim={m} done")
    return pd.DataFrame(rows), est2


def save_explorer(a, D, pca, est2, out):
    rng = np.random.default_rng(C.SEED + 1)
    itr = rng.choice(len(D["Xtr"]), min(a.exp_n, len(D["Xtr"])), replace=False)
    ite = rng.choice(len(D["Xte"]), min(a.exp_n, len(D["Xte"])), replace=False)
    Xe = np.vstack([D["Xtr"][itr], D["Xte"][ite]])
    meta = pd.concat([D["meta_tr"].iloc[itr].assign(split="train"), D["meta_te"].iloc[ite].assign(split="test")],
                     ignore_index=True)
    raw = D["scaler"].inverse_transform(Xe)
    meta["sst"], meta["d20"] = raw[:, 0], isotherm_depth(raw[:, :D["nd"]], D["depth"])
    emb = {f"emb_{n}": e.transform(Xe) for n, e in est2.items()}
    np.savez_compressed(out / "explorer.npz", X=Xe.astype(np.float32), scale=D["scaler"].scale_, mean=D["scaler"].mean_,
                        pmean=pca.mean_, comps=pca.components_, evr=pca.explained_variance_ratio_,
                        depth=D["depth"], nd=D["nd"], **emb)
    meta.to_csv(out / "explorer_meta.csv", index=False)


def main(a):
    out, figs = C.RESULTS, C.RESULTS / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(C.SEED)
    D = prepare(a.data)
    Xtr, Xte, d, nd = D["Xtr"], D["Xte"], D["Xtr"].shape[1], D["nd"]
    print(f"train {Xtr.shape}, test {Xte.shape}, temporal cut at {D['cut']}")
    ks = list(range(1, min(a.kmax, d) + 1))
    nb = rng.choice(len(Xte), min(a.nb_n, len(Xte)), replace=False)

    # ---- PCA
    pca = PCA(n_components=ks[-1], random_state=C.SEED).fit(Xtr)
    cum = np.cumsum(pca.explained_variance_ratio_)
    rows = []
    for k in ks:
        o, _ = pca_eval(pca, Xte, k, nb)
        rows.append(dict(k=k, ve_train=cum[k - 1], **o))
    pca_df = pd.DataFrame(rows)

    # ---- Baseline: Gaussian random projection at the same dimension, several seeds
    rp_runs = []
    for s in range(a.seeds):
        for k in ks:
            W = GaussianRandomProjection(n_components=k, random_state=s).fit(Xtr).components_
            Z, Xh = M.rp_reconstruct(W, Xte)
            r = M.recon_stats(Xte, Xh)
            rp_runs.append(dict(seed=s, k=k, ve_test=r["ve"], rmse_test=r["rmse"],
                                trust=M.trust(Xte[nb], Z[nb]), knn=M.knn_overlap(Xte[nb], Z[nb])))
    rp_runs = pd.DataFrame(rp_runs)
    g = rp_runs.groupby("k")[METR]
    rp_df = g.mean().join(g.std().add_suffix("_std")).reset_index()

    # ---- Choice of k and intrinsic dimension
    kk = {f"k{int(t * 100)}": int(min(np.searchsorted(cum, t) + 1, len(cum))) for t in (0.90, 0.95, 0.99)}
    k_star = a.k or kk["k95"]
    sub = Xtr[rng.choice(len(Xtr), min(a.id_n, len(Xtr)), replace=False)]
    top = pca.transform(sub)[:, :kk["k99"]]
    idim = dict(kk, k_star=k_star, twonn=twonn(sub), twonn_mle=twonn_mle(sub),
                twonn_on_pca_k99=twonn(top), n_features=d, n_train=len(Xtr), n_test=len(Xte), temporal_cut=D["cut"])
    json.dump(idim, open(out / "intrinsic_dim.json", "w"), indent=2)
    print("intrinsic dimension:", idim)

    # ---- Cluster validity in the reduced space (Unit V)
    Zs = (Xte - pca.mean_) @ pca.components_[:k_star].T
    sidx = rng.choice(len(Zs), min(4000, len(Zs)), replace=False)
    cv = []
    for c in range(2, 9):
        lab = KMeans(c, n_init=10, random_state=C.SEED).fit_predict(Zs[sidx])
        cv.append(dict(clusters=c, silhouette=silhouette_score(Zs[sidx], lab), davies_bouldin=davies_bouldin_score(Zs[sidx], lab)))
    pd.DataFrame(cv).to_csv(out / "cluster_validity.csv", index=False)

    # ---- Per-depth RMSE (original units) at k_star
    sc = D["scaler"].scale_
    W = pca.components_[:k_star]
    e_pca = M.per_feature_rmse(Xte, ((Xte - pca.mean_) @ W.T) @ W + pca.mean_) * sc
    e_rp = np.mean([M.per_feature_rmse(Xte, M.rp_reconstruct(GaussianRandomProjection(k_star, random_state=s).fit(Xtr).components_, Xte)[1])
                    for s in range(a.seeds)], axis=0) * sc
    pdepth = pd.DataFrame(dict(feature=D["names"], pca=e_pca, rp=e_rp))
    pdepth.to_csv(out / "per_depth_rmse.csv", index=False)

    # ---- Optional manifold comparison (advanced extension)
    mani, est2 = (None, {})
    if a.manifold:
        print("manifold methods...")
        mani, est2 = manifold_block(a, D, pca, k_star, rng)
        mani.to_csv(out / "metrics_manifold.csv", index=False)
        mani[mani.k == k_star].to_csv(out / "summary_manifold.csv", index=False)

    # ---- Tables
    pca_df.to_csv(out / "metrics_pca.csv", index=False)
    rp_runs.to_csv(out / "metrics_rp_runs.csv", index=False)
    rp_df.to_csv(out / "metrics_rp.csv", index=False)
    r = rp_df[rp_df.k == k_star].iloc[0]
    p = pca_df[pca_df.k == k_star].iloc[0]
    summ = pd.DataFrame([dict(model="Baseline: Gaussian random projection", k=k_star, **{m: r[m] for m in METR},
                              **{m + "_std": r[m + "_std"] for m in METR}),
                         dict(model="PCA", k=k_star, **{m: p[m] for m in METR})])
    summ.to_csv(out / "summary_table.csv", index=False)
    print(summ.round(4).to_string(index=False))

    # ---- Figures + explorer data
    plots.scree(pca.explained_variance_ratio_, figs / "scree.png")
    plots.metrics_vs_k(pca_df, rp_df, mani, figs / "metrics_vs_k.png")
    plots.per_depth(pdepth, D["depth"], nd, k_star, figs / "per_depth_rmse.png")
    plots.pc_profiles(pca.components_, D["depth"], nd, figs / "pc_profiles.png")
    plots.pc_maps(pca.transform(Xtr)[:, :3], D["meta_tr"], figs / "pc_maps.png")
    save_explorer(a, D, pca, est2, out)
    print("done ->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(C.RAW_FILE))
    ap.add_argument("--kmax", type=int, default=20)
    ap.add_argument("--k", type=int, default=None, help="dimension for the headline comparison (default: 95%% variance)")
    ap.add_argument("--seeds", type=int, default=10, help="random-projection repeats")
    ap.add_argument("--nb_n", type=int, default=2000, help="test points for neighbourhood metrics")
    ap.add_argument("--id_n", type=int, default=10000, help="points for intrinsic dimension")
    ap.add_argument("--manifold", action="store_true", help="also run Isomap/UMAP (slower)")
    ap.add_argument("--man_fit", type=int, default=3000)
    ap.add_argument("--man_test", type=int, default=1500)
    ap.add_argument("--iso_nn", type=int, default=10)
    ap.add_argument("--clusters", type=int, default=4)
    ap.add_argument("--exp_n", type=int, default=3000, help="explorer points per split")
    main(ap.parse_args())

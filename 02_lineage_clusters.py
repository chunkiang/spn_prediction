#!/usr/bin/env python3
"""
02_lineage_clusters.py
SPN k-mer 项目 第二步：用 k-mer 矩阵做谱系聚类，得到用于分组交叉验证的 lineage group

思路：
  - 取"可变" k-mer（在 1%–99% 菌株中存在）的存在/缺失谱，计算株间 Jaccard 距离
  - 平均连锁层次聚类，在一系列距离阈值下切树
  - 对每个阈值报告：簇数、最大簇占比、被拆散的 ST 数（理想为 0，即同一 ST 不跨簇）、
    ST271 与 ST320（CC271）是否同簇
  - 由用户根据报告选定阈值，输出 lineage_groups.csv 供第三步建模使用

输入：01_matrix/ 下的 kmer_matrix.npy、samples.txt、phenotype_clean.csv、pca_coords.csv
输出（02_lineage/）：
  - threshold_scan.csv       各阈值的汇总
  - lineage_groups.csv       每株在各阈值下的簇号 + ST
  - cluster_vs_ST_<t>.csv    推荐阈值下每簇包含的 ST 及耐药构成
  - dist_hist.png            距离分布
  - pca_by_lineage.png       PCA 图按推荐阈值的簇着色
  - lineage_report.txt
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

BASE = "/Users/chunjiang/Documents/20260930_SPN_ML"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--indir", default=f"{BASE}/01_matrix")
    p.add_argument("--outdir", default=f"{BASE}/02_lineage")
    p.add_argument("--prev-min", type=float, default=0.01)
    p.add_argument("--prev-max", type=float, default=0.99)
    p.add_argument("--cc271", nargs="+", default=["271", "320"],
                   help="用来校准阈值的同一克隆群内的 ST")
    return p.parse_args()


def jaccard_distance(P):
    """P: n x m 的 0/1 矩阵（float32）。返回 n x n Jaccard 距离。"""
    inter = P @ P.T
    a = np.diag(inter)
    union = a[:, None] + a[None, :] - inter
    D = 1.0 - inter / np.maximum(union, 1)
    np.fill_diagonal(D, 0.0)
    return np.clip(D, 0, 1)


def main():
    a = parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    rep = open(os.path.join(a.outdir, "lineage_report.txt"), "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        rep.write(str(msg) + "\n")

    M = np.load(os.path.join(a.indir, "kmer_matrix.npy"), mmap_mode="r")
    samples = open(os.path.join(a.indir, "samples.txt")).read().split()
    ph = pd.read_csv(os.path.join(a.indir, "phenotype_clean.csv"), dtype={"ID": str, "ST": str})
    assert ph["ID"].tolist() == samples, "表型表顺序与矩阵行顺序不一致"

    # ---------- 可变 k-mer 的存在/缺失谱 ----------
    n = M.shape[0]
    prev = np.zeros(M.shape[1])
    for j in range(0, M.shape[1], 100000):
        prev[j:j+100000] = (M[:, j:j+100000] > 0).mean(axis=0)
    var_idx = np.flatnonzero((prev > a.prev_min) & (prev < a.prev_max))
    log(f"用于聚类的可变 k-mer：{len(var_idx):,}（存在率 {a.prev_min:.0%}–{a.prev_max:.0%}）")
    P = (np.asarray(M[:, var_idx]) > 0).astype(np.float32)

    D = jaccard_distance(P)
    del P
    dv = squareform(D, checks=False)
    log(f"株间 Jaccard 距离：中位数 {np.median(dv):.3f}，"
        f"5%/25%/75%/95% 分位 {np.round(np.quantile(dv, [.05, .25, .75, .95]), 3).tolist()}")

    # 同一 ST 内部的距离 vs 不同 ST 间的距离（帮助理解阈值尺度）
    st = ph["ST"].astype(str).to_numpy(dtype=object)
    same = st[:, None] == st[None, :]
    iu = np.triu_indices(n, 1)
    within, between = D[iu][same[iu]], D[iu][~same[iu]]
    log(f"同 ST 株间距离：中位数 {np.median(within):.3f}，95% 分位 {np.quantile(within, .95):.3f}")
    log(f"不同 ST 株间距离：中位数 {np.median(between):.3f}，5% 分位 {np.quantile(between, .05):.3f}")
    cc = np.isin(st, a.cc271)
    if cc.sum() > 1:
        sub = D[np.ix_(cc, cc)][np.triu_indices(cc.sum(), 1)]
        log(f"{'/'.join(a.cc271)} 株间距离：中位数 {np.median(sub):.3f}，最大 {sub.max():.3f}")

    Z = linkage(dv, method="average")

    # ---------- 阈值扫描 ----------
    st_multi = ph["ST"].value_counts()
    st_multi = st_multi[(st_multi >= 2) & st_multi.index.str.fullmatch(r"\d+")].index
    # 阈值按距离分布的分位数取，自动适应实际距离尺度
    qs = [.002, .005, .01, .015, .02, .03, .04, .05, .06, .08, .10, .12, .15, .20, .25, .30, .40, .50]
    thresholds = np.unique(np.round(np.quantile(dv, qs), 4))
    rows, groups = [], pd.DataFrame({"ID": samples, "ST": st, "Origin": ph["Origin"]})
    for t in thresholds:
        cl = fcluster(Z, t=t, criterion="distance")
        groups[f"lin_{t:.4f}"] = cl
        tab = pd.crosstab(ph["ST"], cl)
        split = int((tab.loc[st_multi] > 0).sum(axis=1).gt(1).sum())
        cc_same = (len(set(cl[cc])) == 1) if cc.sum() > 1 else np.nan
        sizes = pd.Series(cl).value_counts()
        rows.append({
            "threshold": t, "n_clusters": len(sizes),
            "n_singleton_clusters": int((sizes == 1).sum()),
            "largest_cluster_n": int(sizes.max()),
            "largest_cluster_frac": round(sizes.max() / n, 3),
            "ST_split_across_clusters": split,
            "CC271_in_one_cluster": cc_same,
            "clusters_mixing_origin": int((pd.crosstab(cl, ph["Origin"]) > 0).all(axis=1).sum()),
        })
    scan = pd.DataFrame(rows)
    scan.to_csv(os.path.join(a.outdir, "threshold_scan.csv"), index=False)
    groups.to_csv(os.path.join(a.outdir, "lineage_groups.csv"), index=False)
    log("\n阈值扫描：")
    log(scan.to_string(index=False))

    # ---------- 推荐阈值：CC271 同簇、拆散 ST 最少、最大簇 ≤ 35% 中取最小阈值 ----------
    ok = scan[(scan.CC271_in_one_cluster == True) & (scan.largest_cluster_frac <= 0.35)]
    if len(ok):
        ok = ok[ok.ST_split_across_clusters == ok.ST_split_across_clusters.min()]
        t_best = float(ok.threshold.min())
    else:
        t_best = float(scan.threshold.iloc[len(scan) // 2])
        log("警告：没有阈值同时满足 CC271 同簇且最大簇 ≤35%，暂取中间阈值，请人工查看")
    col = f"lin_{t_best:.4f}"
    log(f"\n推荐阈值：{t_best}（列 {col}）")

    cl = groups[col].values
    summ = []
    for c in pd.Series(cl).value_counts().index:
        m = cl == c
        sts = ph.loc[m, "ST"].value_counts()
        summ.append({
            "cluster": c, "n": int(m.sum()),
            "n_domestic": int((ph.Origin[m] == "Domestic").sum()),
            "n_GPS": int((ph.Origin[m] == "GPS").sum()),
            "top_ST": "; ".join(f"{k}({v})" for k, v in sts.head(6).items()),
            "top_serotype": "; ".join(f"{k}({v})" for k, v in
                                      ph.loc[m, "SEROTYPE"].value_counts().head(4).items()),
            **{f"{d}_NS%": round(100 * ph.loc[m, f"{d}_NS"].mean(), 1)
               for d in ["PEN", "CRO", "ERY", "SXT"]},
        })
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(a.outdir, f"cluster_vs_ST_{t_best:.4f}.csv"), index=False)
    log("\n推荐阈值下最大的 15 个簇：")
    log(summ.head(15).to_string(index=False))

    # ---------- 图 ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(within, bins=100, alpha=.6, density=True, label="Same ST")
    ax.hist(between, bins=100, alpha=.6, density=True, label="Different ST")
    ax.axvline(t_best, color="k", ls="--", lw=1, label=f"Threshold {t_best}")
    ax.set_xlabel("Jaccard distance (variable k-mers)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(a.outdir, "dist_hist.png"), dpi=200)

    pca_path = os.path.join(a.indir, "pca_coords.csv")
    if os.path.exists(pca_path):
        pc = pd.read_csv(pca_path, dtype={"ID": str}).merge(groups[["ID", col]], on="ID")
        top = pd.Series(cl).value_counts().index[:10]
        fig, ax = plt.subplots(figsize=(7, 5.5))
        other = ~pc[col].isin(top)
        ax.scatter(pc.PC1[other], pc.PC2[other], s=6, c="lightgrey", label="Other lineages")
        cmap = plt.get_cmap("tab10")
        for i, c in enumerate(top):
            m = pc[col] == c
            lab = summ.loc[summ.cluster == c, "top_ST"].iloc[0].split(";")[0]
            ax.scatter(pc.PC1[m], pc.PC2[m], s=8, color=cmap(i), label=f"L{c}: {lab}, n={m.sum()}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"Lineage clusters (threshold {t_best})")
        ax.legend(frameon=False, fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(a.outdir, "pca_by_lineage.png"), dpi=200)

    log(f"\n完成。输出目录：{a.outdir}")
    rep.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
01_build_kmer_matrix.py
SPN k-mer 项目 第一步：长表 -> 菌株 x k-mer 矩阵，表型对齐，数据质控

输入：
  - 两个 KMC 长表（空格分隔，无表头）：kmer  count  sample
  - 表型表 927_combined.xlsx
输出（--outdir）：
  - kmer_matrix.npy        uint16，行 = 菌株（顺序见 samples.txt），列 = 非零 k-mer
  - kmer_index.npy         int32，每列对应的 k-mer 2-bit 编码（A0 C1 G2 T3）
  - samples.txt            行顺序
  - phenotype_clean.csv    对齐后的表型：log2 MIC、S/I/R、ST、血清型、来源
  - qc_per_strain.csv      每株 distinct k-mer 数、总计数、最小/最大计数、离群标记
  - qc_report.txt          汇总报告（特征数、稀疏度、canonical 检查、ID 匹配等）
  - pca_batch.png / pca_coords.csv   批次效应 PCA

用法：
  python 01_build_kmer_matrix.py            # 使用下面的默认路径
  python 01_build_kmer_matrix.py --help
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import polars as pl

K = 10
NCOL_FULL = 4 ** K

# CLSI M100 折点（S 上限, R 下限）；PEN 用口服折点，CRO/AMC 用非脑膜炎折点，与原稿一致
BREAKPOINTS = {
    "PEN": (0.06, 2), "AMC": (2, 8), "CRO": (1, 4), "ERY": (0.25, 1),
    "CLI": (0.25, 1), "LVX": (2, 8), "MFX": (1, 4), "SXT": (0.5, 4),
}

BASE = "/Users/chunjiang/Documents"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--kmer", nargs="+", default=[
        f"{BASE}/20260930_633菌株/spn607_clean/k10_output/607_long-merge-kmer-file.tsv",
        f"{BASE}/20260930_GPS世界菌株/k10_output/320_long-merge-kmer-file.tsv",
    ])
    p.add_argument("--pheno", default=f"{BASE}/927_combined.xlsx")
    p.add_argument("--outdir", default=f"{BASE}/20260930_SPN_ML/01_matrix")
    p.add_argument("--n-top-var", type=int, default=20000,
                   help="PCA 使用方差最大的 k-mer 数（仅用于批次效应可视化）")
    return p.parse_args()


def log(msg, fh=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh:
        fh.write(msg + "\n")


def revcomp_index(k=K):
    """每个 2-bit 编码 k-mer 的反向互补编码。"""
    idx = np.arange(4 ** k, dtype=np.int64)
    rc = np.zeros_like(idx)
    for i in range(k):
        d = (idx >> (2 * i)) & 3          # 第 i 位（从 3' 端数）
        rc |= (3 - d) << (2 * (k - 1 - i))
    return rc


def scan_long(path):
    """按首行自动识别分隔符（制表符 / 空格 / 混合），返回 kmer, count, sample 三列的 LazyFrame。"""
    with open(path, "r") as fh:
        first = fh.readline().rstrip("\n\r")
    has_tab, has_space = "\t" in first, " " in first
    schema = {"kmer": pl.Utf8, "count": pl.Int64, "sample": pl.Utf8}
    if has_tab != has_space:                       # 单一分隔符：直接按列读取（最快）
        sep = "\t" if has_tab else " "
        return pl.scan_csv(path, separator=sep, has_header=False, quote_char=None,
                           new_columns=list(schema), schema_overrides=schema)
    # 混合分隔符：整行读入后统一替换再拆分
    parts = (pl.col("line").str.replace_all("\t", " ").str.strip_chars()
               .str.split(" ").list)
    return (pl.scan_csv(path, separator="\x01", has_header=False, quote_char=None,
                        new_columns=["line"], schema_overrides={"line": pl.Utf8})
              .select(parts.get(0).alias("kmer"),
                      parts.get(1).cast(pl.Int64).alias("count"),
                      parts.get(-1).alias("sample")))


def read_long(path, row_of):
    """读取长表 -> (row, idx, count)。菌株名在流式读取时即映射为行号（不在表型表中的记为 -1），
    k-mer 直接编码为 0..4^k-1 的整数，避免把数亿个字符串留在内存里。"""
    lf = scan_long(path).select(
        pl.col("sample").replace_strict(row_of, default=-1, return_dtype=pl.Int32).alias("row"),
        pl.col("kmer").str.replace_many(["A", "C", "G", "T"], ["0", "1", "2", "3"])
          .str.to_integer(base=4, strict=False).cast(pl.Int32).alias("idx"),
        pl.col("count").clip(0, 65535).cast(pl.UInt16),
    )
    return lf.collect(engine="streaming")


def sample_names(path):
    """单独扫描一次，拿到长表中的全部菌株名（用于 ID 匹配报告）。"""
    return set(scan_long(path).select(pl.col("sample").unique())
               .collect(engine="streaming")["sample"].to_list())


def load_pheno(path):
    df = pd.read_excel(path, dtype={"ID": str, "ST": str})
    df["ID"] = df["ID"].astype(str).str.strip()
    keep = ["ID", "SEROTYPE", "ST", "Origin", "Country", "COMMENT"] + \
           [f"{d}_MIC" for d in BREAKPOINTS]
    df = df[keep].copy()
    for d, (s, r) in BREAKPOINTS.items():
        # 取 log2 后四舍五入到整数稀释档：统一 0.064/0.0625、0.032/0.03125 等写法
        l2 = np.round(np.log2(df[f"{d}_MIC"].astype(float)))
        df[f"{d}_log2"] = l2
        mic = np.power(2.0, l2)
        df[f"{d}_SIR"] = np.select([mic <= s * 1.001, mic >= r * 0.999], ["S", "R"], "I")
        df[f"{d}_NS"] = (df[f"{d}_SIR"] != "S").astype(int)
    # ST：'320*'（最近似 ST）、'-'（无法分型）单独标记，分组时各自成组
    df["ST_raw"] = df["ST"]
    df["ST_nonstd"] = ~df["ST"].str.fullmatch(r"\d+")
    return df


def main():
    a = parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    rep = open(os.path.join(a.outdir, "qc_report.txt"), "w", encoding="utf-8")

    # ---------- 表型 ----------
    ph = load_pheno(a.pheno)
    log(f"表型表：{len(ph)} 株；来源 {ph.Origin.value_counts().to_dict()}", rep)

    # ---------- ID 匹配 ----------
    km_ids = set()
    for f in a.kmer:
        km_ids |= sample_names(f)
    ph_ids = set(ph["ID"])
    only_km, only_ph = sorted(km_ids - ph_ids), sorted(ph_ids - km_ids)
    log(f"ID 匹配：两者共有 {len(km_ids & ph_ids)}；仅 k-mer 有 {len(only_km)}；"
        f"仅表型有 {len(only_ph)}", rep)
    if only_km:
        log(f"  仅 k-mer 有（前 20）：{only_km[:20]}", rep)
    if only_ph:
        log(f"  仅表型有（前 20）：{only_ph[:20]}", rep)
    ph = ph[ph["ID"].isin(km_ids)].reset_index(drop=True)
    samples = ph["ID"].tolist()
    row_of = {s: i for i, s in enumerate(samples)}

    # ---------- 逐个文件读入并填充矩阵 ----------
    M = np.zeros((len(samples), NCOL_FULL), dtype=np.uint16)
    cmin, cmax, n_bad = np.inf, 0, 0
    for f in a.kmer:
        t0 = time.time()
        d = read_long(f, row_of)
        n_bad += d["idx"].null_count()
        d = d.filter((pl.col("row") >= 0) & pl.col("idx").is_not_null())
        cmin, cmax = min(cmin, d["count"].min()), max(cmax, d["count"].max())
        M[d["row"].to_numpy(), d["idx"].to_numpy()] = d["count"].to_numpy()
        log(f"读取 {os.path.basename(f)}：{d.height:,} 行，{time.time()-t0:.0f}s", rep)
        del d
    if n_bad:
        log(f"警告：{n_bad} 行 k-mer 含非 ACGT 字符，已丢弃", rep)

    # ---------- 计数范围（用于推断 KMC -ci / -cs 设置） ----------
    msg = f"k-mer 计数范围：{cmin} – {cmax}"
    if cmin == 2:
        msg += "；最小值为 2，提示 KMC 使用了默认 -ci2（计数为 1 的 k-mer 被丢弃）"
    if cmax == 255:
        msg += "；最大值恰为 255，提示被默认 -cs255 截断"
    log(msg, rep)

    present = (M > 0)
    col_nz = present.any(axis=0)

    # canonical 检查：同一 k-mer 与其反向互补是否同时出现
    rc = revcomp_index()
    idx_all = np.arange(NCOL_FULL)
    both = col_nz & col_nz[rc] & (idx_all < rc)
    n_pal = int((idx_all == rc).sum())
    log(f"canonical 检查：k-mer 与其反向互补同时出现的对数 = {int(both.sum()):,}"
        f"（0 表示为 canonical 计数；回文 k-mer 共 {n_pal}）", rep)

    keep_idx = np.flatnonzero(col_nz).astype(np.int32)
    M = M[:, keep_idx]
    present = present[:, keep_idx]
    n_feat = M.shape[1]
    prev = present.mean(axis=0)
    log(f"非零 k-mer（特征）数：{n_feat:,} / 理论 {NCOL_FULL:,}", rep)
    log(f"矩阵稀疏度（0 的比例）：{1 - present.mean():.4f}", rep)
    log(f"所有菌株都存在的 k-mer：{(prev == 1).sum():,}（{(prev == 1).mean():.1%}）；"
        f"存在率 1%–99% 的 k-mer：{((prev > 0.01) & (prev < 0.99)).sum():,}；"
        f"≤1% 菌株存在：{(prev <= 0.01).sum():,}", rep)

    # ---------- 每株 QC ----------
    n_distinct = present.sum(axis=1)
    total = M.sum(axis=1, dtype=np.int64)
    Mmask = np.where(present, M, np.iinfo(np.uint16).max)
    qc = pd.DataFrame({
        "ID": samples, "Origin": ph["Origin"], "COMMENT": ph["COMMENT"],
        "n_distinct_kmer": n_distinct, "total_count": total.astype(np.int64),
        "min_count": Mmask.min(axis=1), "max_count": M.max(axis=1),
    })
    del Mmask
    for c in ["n_distinct_kmer", "total_count"]:
        z = (qc[c] - qc[c].median()) / (1.4826 * (qc[c] - qc[c].median()).abs().median())
        qc[f"{c}_robust_z"] = z.round(2)
    qc["outlier"] = (qc[["n_distinct_kmer_robust_z", "total_count_robust_z"]].abs() > 3.5).any(axis=1)
    qc.to_csv(os.path.join(a.outdir, "qc_per_strain.csv"), index=False)
    log("total_count 约等于组装长度（canonical 计数下每个位置计一次）", rep)
    log(qc.groupby("Origin")[["n_distinct_kmer", "total_count"]]
          .describe().T.round(0).to_string(), rep)
    log(f"离群株（|robust z|>3.5）：{int(qc.outlier.sum())}；"
        f"其中标注 contamination 的：{int((qc.outlier & qc.COMMENT.eq('contamination')).sum())}"
        f" / {int(qc.COMMENT.eq('contamination').sum())}", rep)
    if qc.outlier.any():
        log(qc[qc.outlier][["ID", "Origin", "COMMENT", "n_distinct_kmer",
                             "total_count"]].to_string(index=False), rep)

    # ---------- 保存矩阵 ----------
    np.save(os.path.join(a.outdir, "kmer_matrix.npy"), M)
    np.save(os.path.join(a.outdir, "kmer_index.npy"), keep_idx)
    with open(os.path.join(a.outdir, "samples.txt"), "w") as fh:
        fh.write("\n".join(samples) + "\n")
    ph.to_csv(os.path.join(a.outdir, "phenotype_clean.csv"), index=False)

    # ---------- 表型汇总 ----------
    log("\nS/I/R 分布（按来源）：", rep)
    for d in BREAKPOINTS:
        log(f"  {d}: {pd.crosstab(ph.Origin, ph[f'{d}_SIR']).to_dict('index')}", rep)
    log(f"ST 数：{ph.ST.nunique()}；单株 ST：{(ph.ST.value_counts() == 1).sum()}；"
        f"非标准 ST：{ph.loc[ph.ST_nonstd, 'ST'].tolist()}", rep)

    # ---------- 批次效应 PCA ----------
    from sklearn.decomposition import PCA
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    del present
    # 分块计算 log1p 后的方差，避免整体转 float
    var = np.empty(n_feat, dtype=np.float64)
    for j in range(0, n_feat, 50000):
        var[j:j+50000] = np.log1p(M[:, j:j+50000].astype(np.float32)).var(axis=0)
    top = np.argsort(var)[::-1][:min(a.n_top_var, n_feat)]
    X = np.log1p(M[:, top].astype(np.float32))
    pca = PCA(n_components=10, random_state=0)
    Z = pca.fit_transform(X - X.mean(axis=0))
    ev = pca.explained_variance_ratio_
    pc = pd.DataFrame(Z[:, :5], columns=[f"PC{i+1}" for i in range(5)])
    pc.insert(0, "ID", samples)
    pc["Origin"] = ph["Origin"].values
    pc["PEN_SIR"] = ph["PEN_SIR"].values
    pc.to_csv(os.path.join(a.outdir, "pca_coords.csv"), index=False)
    log(f"\nPCA（top {len(top)} 方差 k-mer，log1p）解释方差 PC1–5：{[round(float(x), 3) for x in ev[:5]]}", rep)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for lab, c in [("Domestic", "#1f77b4"), ("GPS", "#d62728")]:
        m = pc.Origin == lab
        ax[0].scatter(pc.PC1[m], pc.PC2[m], s=8, alpha=.6, c=c, label=f"{lab} (n={m.sum()})")
    for lab, c in [("S", "#2ca02c"), ("I", "#ff7f0e"), ("R", "#9467bd")]:
        m = pc.PEN_SIR == lab
        ax[1].scatter(pc.PC1[m], pc.PC2[m], s=8, alpha=.6, c=c, label=f"PEN {lab} (n={m.sum()})")
    for i, t in enumerate(["By origin", "By penicillin category (oral breakpoint)"]):
        ax[i].set_xlabel(f"PC1 ({ev[0]:.1%})")
        ax[i].set_ylabel(f"PC2 ({ev[1]:.1%})")
        ax[i].set_title(t)
        ax[i].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.outdir, "pca_batch.png"), dpi=200)

    log(f"\n完成。输出目录：{a.outdir}", rep)
    rep.close()


if __name__ == "__main__":
    sys.exit(main())

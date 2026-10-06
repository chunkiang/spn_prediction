#!/usr/bin/env python3
"""
05_validation_kmers.py —— 临床验证菌株 contigs → 10-mer 计数矩阵（与训练矩阵列完全对齐）

做了什么
  1. 每株 contigs 先做组装 QC：contig 数、总长、N50、GC%、N 碱基数
  2. 用与训练集相同的 KMC 参数（默认 -k10 -ci1 -fm，canonical）计数并导出
  3. 2-bit 编码（A0 C1 G2 T3，首碱基为最高位，与 01_build_kmer_matrix.py 一致），
     按训练矩阵的 kmer_index.npy 对齐：训练集没有的 k-mer 丢弃，缺失的填 0
  4. 编码/参数一致性检查：验证株 k-mer 在训练索引中的命中率、计数范围、每株 distinct k-mer 数
     与训练集分布比较（异常提示污染或组装问题）
  5. 克隆重叠评估（回应 R3-2）：在训练集可变 k-mer 上计算每株验证菌与 927 株训练菌的
     Jaccard 距离，给出最近邻训练株、其 ST/来源/谱系簇，以及是否落在训练集已有谱系之外

输出（--outdir）
  val_kmer_matrix.npy   n_val × 503,875 uint16，列顺序 = 训练 kmer_index.npy
  val_samples.txt       行顺序
  val_qc.csv            每株组装 QC + k-mer QC + 最近邻训练株
  val_train_jaccard.npy n_val × 927 Jaccard 距离（列顺序 = 训练 samples.txt）
  val_report.txt        汇总
  long/<ID>.tsv         （--keep-long 时）与训练长表同格式的 kmer/count/ID 文本

用法
  conda activate ai
  python 05_validation_kmers.py                      # 默认路径
  python 05_validation_kmers.py --jobs 6 --keep-long
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

K = 10
BASE = "/Users/chunjiang/Documents"
LUT = np.full(256, 255, dtype=np.uint8)
for _i, _c in enumerate("ACGT"):
    LUT[ord(_c)] = _i
    LUT[ord(_c.lower())] = _i
POW = (4 ** np.arange(K - 1, -1, -1)).astype(np.int64)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--contigs", default=f"{BASE}/20260930_临床验证菌株50株/Assem_临床验证菌株序列拼接")
    p.add_argument("--pattern", default="*.fasta")
    p.add_argument("--matrix-dir", default=f"{BASE}/20260930_SPN_ML/01_matrix",
                   help="01_build_kmer_matrix.py 的输出目录")
    p.add_argument("--lineage", default=f"{BASE}/20260930_SPN_ML/02_lineage/lineage_groups.csv",
                   help="02_lineage_clusters.py 的输出；找不到则只报告 ST")
    p.add_argument("--lineage-col", default="lin_0.2363")
    p.add_argument("--outdir", default=f"{BASE}/20260930_SPN_ML/05_validation/kmers")
    # KMC 参数：必须与训练集批处理脚本完全一致
    p.add_argument("--ci", type=int, default=1, help="KMC -ci（训练集为 -ci1）")
    p.add_argument("--cs", type=int, default=None, help="KMC -cs；训练集未设置则保持默认")
    p.add_argument("--jobs", type=int, default=4, help="同时运行的 KMC 进程数")
    p.add_argument("--threads", type=int, default=2, help="每个 KMC 进程的线程数")
    p.add_argument("--mem", type=int, default=4, help="每个 KMC 进程内存上限 GB（不影响结果）")
    p.add_argument("--keep-long", action="store_true", help="保存与训练长表同格式的文本")
    p.add_argument("--prev-min", type=float, default=0.01)
    p.add_argument("--prev-max", type=float, default=0.99)
    return p.parse_args()


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def fasta_qc(path):
    lens, gc, nn, seq_len = [], 0, 0, 0
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if seq_len:
                    lens.append(seq_len)
                seq_len = 0
                continue
            s = line.strip().upper()
            seq_len += len(s)
            gc += s.count("G") + s.count("C")
            nn += s.count("N")
    if seq_len:
        lens.append(seq_len)
    lens = np.sort(np.array(lens))[::-1]
    tot = int(lens.sum())
    n50 = int(lens[np.searchsorted(np.cumsum(lens), tot / 2)]) if tot else 0
    return dict(n_contigs=len(lens), total_bp=tot, N50=n50,
                GC_pct=round(100 * gc / max(tot - nn, 1), 2), N_bases=nn)


def encode_canonical(kmers, counts):
    """ACGT 字符串 → canonical 2-bit 整数，返回 (唯一编码, 计数, 非 canonical 条数)。"""
    b = np.frombuffer("".join(kmers).encode("ascii"), dtype=np.uint8).reshape(-1, K)
    d = LUT[b]
    if (d > 3).any():
        raise ValueError("出现非 ACGT 字符")
    fwd = d.astype(np.int64) @ POW
    rc = (3 - d[:, ::-1]).astype(np.int64) @ POW
    can = np.minimum(fwd, rc)
    n_noncanon = int((can != fwd).sum())
    u, inv = np.unique(can, return_inverse=True)
    c = np.bincount(inv, weights=counts).astype(np.int64)
    return u, c, n_noncanon


def run_kmc(fasta, sid, work, a):
    pre = os.path.join(work, sid)
    tmp = os.path.join(work, sid + "_tmp")
    os.makedirs(tmp, exist_ok=True)
    cmd = ["kmc", f"-k{K}", f"-ci{a.ci}", "-fm", f"-t{a.threads}", f"-m{a.mem}"]
    if a.cs:
        cmd.append(f"-cs{a.cs}")
    cmd += [fasta, pre, tmp]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{sid} KMC 失败：{r.stderr[-500:]}")
    out = pre + ".txt"
    if shutil.which("kmc_dump"):
        dcmd = ["kmc_dump", pre, out]
    else:
        dcmd = ["kmc_tools", "transform", pre, "dump", out]
    r = subprocess.run(dcmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{sid} dump 失败：{r.stderr[-500:]}")
    df = pd.read_csv(out, sep=r"\s+", header=None, names=["kmer", "count"],
                     dtype={"kmer": str, "count": np.int64}, engine="c")
    if a.keep_long:
        os.makedirs(os.path.join(a.outdir, "long"), exist_ok=True)
        df.assign(ID=sid).to_csv(os.path.join(a.outdir, "long", f"{sid}.tsv"),
                                 sep="\t", header=False, index=False)
    for f in glob.glob(pre + ".*"):
        os.remove(f)
    shutil.rmtree(tmp, ignore_errors=True)
    return df["kmer"].values, df["count"].values


def main():
    a = parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    rep = open(os.path.join(a.outdir, "val_report.txt"), "w", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        rep.write(msg + "\n"); rep.flush()

    for tool in ["kmc"]:
        if not shutil.which(tool):
            raise SystemExit(f"找不到 {tool}，请先 conda activate 含 KMC 的环境")

    files = sorted(glob.glob(os.path.join(a.contigs, a.pattern)), key=natural_key)
    if not files:
        raise SystemExit(f"{a.contigs} 下没有 {a.pattern}")
    ids = [re.sub(r"\.(fasta|fa|fna)(\.gz)?$", "", os.path.basename(f)) for f in files]
    log(f"验证菌株 {len(ids)} 株：{ids[0]} … {ids[-1]}")
    log(f"KMC 参数：-k{K} -ci{a.ci} -fm" + (f" -cs{a.cs}" if a.cs else "（-cs 默认）"))

    # ---------- 训练集 ----------
    kidx = np.load(os.path.join(a.matrix_dir, "kmer_index.npy")).astype(np.int64)
    assert np.all(np.diff(kidx) > 0), "kmer_index.npy 应为升序"
    M = np.load(os.path.join(a.matrix_dir, "kmer_matrix.npy"), mmap_mode="r")
    tr_ids = open(os.path.join(a.matrix_dir, "samples.txt")).read().split()
    ph = pd.read_csv(os.path.join(a.matrix_dir, "phenotype_clean.csv"),
                     dtype={"ID": str, "ST": str, "SEROTYPE": str})
    assert ph["ID"].tolist() == tr_ids
    ncol = len(kidx)
    log(f"训练矩阵：{M.shape[0]} × {ncol:,}")

    log("扫描训练矩阵：每株 distinct k-mer 数、最大计数、k-mer 存在率")
    tr_distinct = np.zeros(M.shape[0], np.int64)
    tr_max = 0
    prev = np.zeros(ncol)
    for j in range(0, ncol, 50000):
        ch = np.asarray(M[:, j:j + 50000])
        pres = ch > 0
        tr_distinct += pres.sum(axis=1)
        prev[j:j + 50000] = pres.mean(axis=0)
        tr_max = max(tr_max, int(ch.max()))
    log(f"训练集每株 distinct 10-mer：中位数 {int(np.median(tr_distinct)):,}"
        f"（范围 {tr_distinct.min():,}–{tr_distinct.max():,}）；最大计数 {tr_max}")

    # ---------- KMC 并行 ----------
    X = np.zeros((len(ids), ncol), dtype=np.uint16)
    qc = []
    work = tempfile.mkdtemp(prefix="kmc_val_")
    t0 = time.time()

    def one(i):
        km, cnt = run_kmc(files[i], ids[i], work, a)
        u, c, n_nc = encode_canonical(km, cnt)
        pos = np.searchsorted(kidx, u)
        ok = pos < ncol
        hit = np.zeros(len(u), bool)
        hit[ok] = kidx[pos[ok]] == u[ok]
        return i, u, c, n_nc, pos, hit

    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(one, i) for i in range(len(ids))]
        for done, fu in enumerate(as_completed(futs), 1):
            i, u, c, n_nc, pos, hit = fu.result()
            X[i, pos[hit]] = np.clip(c[hit], 0, 65535).astype(np.uint16)
            row = dict(ID=ids[i], **fasta_qc(files[i]),
                       n_distinct_kmer=len(u), total_count=int(c.sum()), max_count=int(c.max()),
                       n_noncanonical_in_dump=n_nc,
                       frac_distinct_in_train_index=round(hit.mean(), 5),
                       frac_count_in_train_index=round(c[hit].sum() / c.sum(), 5))
            qc.append(row)
            print(f"  [{done}/{len(ids)}] {ids[i]}  distinct={len(u):,}  "
                  f"命中训练索引 {hit.mean():.2%}  ({time.time() - t0:.0f}s)", flush=True)
    shutil.rmtree(work, ignore_errors=True)
    qc = pd.DataFrame(qc).set_index("ID").loc[ids].reset_index()

    # ---------- 一致性检查 ----------
    log(f"dump 中非 canonical k-mer 总条数：{qc['n_noncanonical_in_dump'].sum()}"
        "（0 = KMC canonical 输出，与训练集一致）")
    hit_min = qc["frac_distinct_in_train_index"].min()
    log(f"验证株 k-mer 命中训练索引比例：最低 {hit_min:.4%}")
    if hit_min < 0.95:
        log("警告：命中率偏低，可能是编码顺序或 k 值与训练集不一致，请先排查再进行预测！")
    log(f"验证株最大计数 {qc['max_count'].max()}（训练集 {tr_max}）"
        "——两者若一个恰为 255、另一个远大于 255，说明 -cs 设置不一致")
    mu, sd = tr_distinct.mean(), tr_distinct.std()
    qc["distinct_z_vs_train"] = ((qc["n_distinct_kmer"] - mu) / sd).round(2)
    qc["flag"] = ""
    qc.loc[(qc["total_bp"] < 1.9e6) | (qc["total_bp"] > 2.4e6), "flag"] += "基因组长度异常;"
    qc.loc[qc["n_contigs"] > 300, "flag"] += "contig 过多;"
    qc.loc[qc["distinct_z_vs_train"].abs() > 3, "flag"] += "distinct k-mer 偏离训练集 >3SD;"
    qc.loc[(qc["GC_pct"] < 38.5) | (qc["GC_pct"] > 41.0), "flag"] += "GC 异常;"
    flagged = qc[qc["flag"] != ""]
    log(f"组装/k-mer QC 提示 {len(flagged)} 株：" +
        ("; ".join(f"{r.ID}({r.flag})" for r in flagged.itertuples()) if len(flagged) else "无"))

    # ---------- 与训练集的克隆重叠（可变 k-mer 存在/缺失 Jaccard） ----------
    var = np.flatnonzero((prev > a.prev_min) & (prev < a.prev_max))
    log(f"Jaccard 使用训练集可变 k-mer {len(var):,} 个（存在率 {a.prev_min:.0%}–{a.prev_max:.0%}）")
    Pva = (X[:, var] > 0).astype(np.float32)
    inter = np.zeros((len(ids), M.shape[0]), np.float64)
    ntr = np.zeros(M.shape[0], np.float64)
    for j in range(0, len(var), 50000):
        cols = var[j:j + 50000]
        Ptr = (np.asarray(M[:, cols.min():cols.max() + 1])[:, cols - cols.min()] > 0).astype(np.float32)
        inter += Pva[:, j:j + 50000] @ Ptr.T
        ntr += Ptr.sum(axis=1)
    nva = Pva.sum(axis=1).astype(np.float64)
    D = 1 - inter / (nva[:, None] + ntr[None, :] - inter)
    np.save(os.path.join(a.outdir, "val_train_jaccard.npy"), D.astype(np.float32))

    lin = None
    thr = None
    if os.path.exists(a.lineage):
        lg = pd.read_csv(a.lineage, dtype={"ID": str})
        col = a.lineage_col
        if col not in lg.columns:
            lc = [c for c in lg.columns if c.startswith("lin_")]
            col = min(lc, key=lambda c: abs(float(c[4:]) - float(a.lineage_col[4:])))
        thr = float(col[4:])
        lin = ph[["ID"]].merge(lg[["ID", col]], on="ID", how="left")[col].astype(str).values
        log(f"谱系分组列：{col}（阈值 {thr}）")
    nn = D.argmin(axis=1)
    qc["nn_train_ID"] = [tr_ids[k] for k in nn]
    qc["nn_dist"] = D[np.arange(len(ids)), nn].round(4)
    qc["nn_ST"] = ph["ST"].values[nn]
    qc["nn_SEROTYPE"] = ph["SEROTYPE"].values[nn]
    qc["nn_Origin"] = ph["Origin"].values[nn]
    if lin is not None:
        qc["nn_lineage"] = lin[nn]
        qc["n_train_within_thr"] = (D <= thr).sum(axis=1)
        qc["outside_train_lineages"] = qc["nn_dist"] > thr
        n_out = int(qc["outside_train_lineages"].sum())
        log(f"最近邻距离 > {thr}（训练集未见谱系）：{n_out}/{len(ids)} 株；"
            f"其余 {len(ids) - n_out} 株与训练集至少一株处于同一谱系尺度内")
        top = qc.loc[~qc["outside_train_lineages"], "nn_lineage"].value_counts().head(5)
        log("验证株最近邻所在谱系（前 5）：" + ", ".join(f"{k}:{v}" for k, v in top.items()))
        log("注：平均连锁聚类的阈值针对簇间平均距离，这里用最近邻距离近似，仅作克隆重叠描述。")
    log(f"最近邻 Jaccard 距离：中位数 {qc['nn_dist'].median():.3f}，范围 "
        f"{qc['nn_dist'].min():.3f}–{qc['nn_dist'].max():.3f}")
    log("最近邻训练株 ST（前 5）：" +
        ", ".join(f"ST{k}:{v}" for k, v in qc["nn_ST"].value_counts().head(5).items()))

    np.save(os.path.join(a.outdir, "val_kmer_matrix.npy"), X)
    with open(os.path.join(a.outdir, "val_samples.txt"), "w") as f:
        f.write("\n".join(ids) + "\n")
    qc.to_csv(os.path.join(a.outdir, "val_qc.csv"), index=False)
    log(f"完成，用时 {time.time() - t0:.0f}s。输出：{a.outdir}")
    rep.close()


if __name__ == "__main__":
    main()

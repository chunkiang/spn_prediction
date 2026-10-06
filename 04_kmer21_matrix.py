#!/usr/bin/env python3
"""
04_kmer21_matrix.py
SPN k-mer 项目 第四步：k=21 的存在/缺失矩阵（回应 R1-5：k 值比较；R3-4：特征维度）

不需要 KMC：直接从组装 contig 计算 canonical 21-mer（2-bit 编码为 uint64），
只记录存在/缺失。流程分三遍扫描 927 个基因组（每遍约 5–15 分钟，取决于 CPU 核数）：
  第 1 遍：统计每个 21-mer 在多少株中出现（泛 k-mer 频数）
  第 2 遍：对"可变" k-mer（在 ≥min-strains 且 ≤n-min-strains 株中出现）计算存在模式哈希，
          存在模式完全相同的 k-mer 合并为一个特征（一个"模式"）
  第 3 遍：为每个模式的代表 k-mer 生成 927 × P 的 0/1 矩阵

输出（--outdir，格式与 01_matrix 相同，可直接用于第三步脚本）：
  kmer_matrix.npy        uint8，927 × P（P = 唯一存在模式数）
  kmer_index.npy         uint64，每个模式的代表 21-mer（第三步解码为序列）
  pattern_size.npy       每个模式包含的 k-mer 数
  variable_kmers.npz     全部可变 k-mer 及其所属模式编号（后续把重要模式映射回基因用）
  samples.txt / phenotype_clean.csv   从 01_matrix 复制，行顺序一致
  manifest.tsv           ID 与 contig 文件路径的对应关系
  per_strain_kmers.csv   每株 distinct 21-mer 数（质控）
  k21_report.txt

用法：
  python 04_kmer21_matrix.py --dry-run     # 先只生成 manifest.tsv，检查 ID 与文件是否一一对应
  python 04_kmer21_matrix.py               # 正式运行
"""
import argparse
import gzip
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

DOC = "/Users/chunjiang/Documents"
K = 21
FASTA_EXT = (".fa", ".fasta", ".fna", ".fas", ".fa.gz", ".fasta.gz", ".fna.gz")
SKIP_DIRS = {"k10_output", "tmp", "__pycache__"}

LUT = np.full(256, 4, dtype=np.uint8)
for ch, v in zip(b"ACGTacgt", [0, 1, 2, 3, 0, 1, 2, 3]):
    LUT[ch] = v


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--roots", nargs="+", default=[f"{DOC}/20260930_633菌株",
                                                   f"{DOC}/20260930_GPS世界菌株"],
                   help="递归搜索 contig 文件的根目录")
    p.add_argument("--manifest", default=None,
                   help="直接指定 ID<TAB>路径 的清单文件（不自动搜索）")
    p.add_argument("--matrix-dir", default=f"{DOC}/20260930_SPN_ML/01_matrix")
    p.add_argument("--outdir", default=f"{DOC}/20260930_SPN_ML/04_k21")
    p.add_argument("--k", type=int, default=K)
    p.add_argument("--min-strains", type=int, default=10,
                   help="可变 k-mer 的最低出现株数（≈1%%），上限为 n - min-strains")
    p.add_argument("--batch", type=int, default=64, help="第 1 遍合并频数的批大小")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


# ---------------------------------------------------------------- 找文件
def find_fastas(roots, ids):
    ids = set(ids)
    hits = {}
    for root in roots:
        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns if d not in SKIP_DIRS]
            parent = os.path.basename(dp)
            for f in fns:
                if not f.lower().endswith(FASTA_EXT):
                    continue
                stem = f.split(".")[0]
                sid = stem if stem in ids else (parent if parent in ids else None)
                if sid:
                    hits.setdefault(sid, []).append(os.path.join(dp, f))
    chosen, ambiguous = {}, {}
    for sid, paths in hits.items():
        if len(paths) > 1:
            pref = [p for p in paths if "contig" in os.path.basename(p).lower()
                    and "scaffold" not in os.path.basename(p).lower()]
            ambiguous[sid] = paths
            paths = sorted(pref or paths)
        chosen[sid] = paths[0]
    return chosen, ambiguous


# ---------------------------------------------------------------- k-mer 计算（子进程）
def read_fasta(path):
    op = gzip.open if path.endswith(".gz") else open
    seqs, cur = [], []
    with op(path, "rb") as fh:
        for line in fh:
            if line.startswith(b">"):
                if cur:
                    seqs.append(b"".join(cur)); cur = []
            else:
                cur.append(line.strip())
    if cur:
        seqs.append(b"".join(cur))
    return seqs


def canonical_kmers(path, k=K):
    """返回该基因组排序去重后的 canonical k-mer（uint64）。不跨 contig、不跨 N。"""
    out = []
    mask = np.uint64((1 << (2 * k)) - 1)
    for s in read_fasta(path):
        c = LUT[np.frombuffer(s, dtype=np.uint8)]
        bad = np.flatnonzero(c == 4)
        starts = np.concatenate([[0], bad + 1])
        ends = np.concatenate([bad, [len(c)]])
        for a, b in zip(starts, ends):
            seg = c[a:b].astype(np.uint64)
            L = len(seg)
            if L < k:
                continue
            m = L - k + 1
            fwd = np.zeros(m, dtype=np.uint64)
            rc = np.zeros(m, dtype=np.uint64)
            comp = np.uint64(3) - seg
            for j in range(k):
                fwd = (fwd << np.uint64(2)) | seg[j:j + m]
                rc |= comp[j:j + m] << np.uint64(2 * j)
            out.append(np.minimum(fwd & mask, rc))
    if not out:
        return np.empty(0, dtype=np.uint64)
    return np.unique(np.concatenate(out))


def _worker(args):
    path, k = args
    return canonical_kmers(path, k)


# ---------------------------------------------------------------- 主流程
def main():
    a = parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    rep = open(os.path.join(a.outdir, "k21_report.txt"), "a", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        rep.write(line + "\n"); rep.flush()

    samples = open(os.path.join(a.matrix_dir, "samples.txt")).read().split()
    n = len(samples)

    # ---------- manifest ----------
    if a.manifest:
        man = dict(l.rstrip("\n").split("\t")[:2] for l in open(a.manifest) if l.strip())
        ambiguous = {}
    else:
        man, ambiguous = find_fastas(a.roots, samples)
    missing = [s for s in samples if s not in man]
    with open(os.path.join(a.outdir, "manifest.tsv"), "w") as fh:
        for s in samples:
            fh.write(f"{s}\t{man.get(s, 'NOT_FOUND')}\n")
    log(f"manifest：{n - len(missing)}/{n} 株找到 contig 文件")
    if ambiguous:
        log(f"  {len(ambiguous)} 株有多个候选文件，已优先选文件名含 contig 的，请在 manifest.tsv 中核对，例如：")
        for sid, ps in list(ambiguous.items())[:5]:
            log(f"    {sid}: {ps}")
    if missing:
        log(f"  缺失（前 20）：{missing[:20]}")
        log("  请补全 manifest.tsv（ID<TAB>路径）后用 --manifest 指定")
        sys.exit(1)
    if a.dry_run:
        log("dry-run 结束：请检查 manifest.tsv")
        return
    paths = [man[s] for s in samples]
    jobs = [(p, a.k) for p in paths]

    def genomes(tag):
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for i, km in enumerate(ex.map(_worker, jobs, chunksize=4)):
                if (i + 1) % 50 == 0 or i + 1 == n:
                    log(f"  {tag}：{i+1}/{n} 株（{time.time()-t0:.0f}s）")
                yield i, km

    # ---------- 第 1 遍：泛 k-mer 频数 ----------
    log(f"第 1 遍：统计 {a.k}-mer 出现株数（{a.workers} 个进程）")
    per_strain = np.zeros(n, dtype=np.int64)
    batch, batch_res = [], []

    def flush(batch):
        u, c = np.unique(np.concatenate(batch), return_counts=True)
        return u, c.astype(np.int32)

    for i, km in genomes("第 1 遍"):
        per_strain[i] = len(km)
        batch.append(km)
        if len(batch) == a.batch:
            batch_res.append(flush(batch)); batch = []
    if batch:
        batch_res.append(flush(batch))
    del batch
    allu = np.concatenate([u for u, _ in batch_res])
    allc = np.concatenate([c for _, c in batch_res])
    del batch_res
    pan, inv = np.unique(allu, return_inverse=True)
    prev = np.bincount(inv, weights=allc, minlength=len(pan)).astype(np.int32)
    del allu, allc, inv
    import pandas as pd
    pd.DataFrame({"ID": samples, "n_distinct_kmer": per_strain}).to_csv(
        os.path.join(a.outdir, "per_strain_kmers.csv"), index=False)
    lo, hi = a.min_strains, n - a.min_strains
    var_mask = (prev >= lo) & (prev <= hi)
    V = pan[var_mask]
    log(f"每株 distinct {a.k}-mer：中位数 {int(np.median(per_strain)):,}"
        f"（范围 {per_strain.min():,}–{per_strain.max():,}）")
    log(f"泛 {a.k}-mer 总数：{len(pan):,}；所有株共有：{int((prev == n).sum()):,}；"
        f"仅 1 株：{int((prev == 1).sum()):,}；可变（{lo}–{hi} 株）：{len(V):,}")
    del pan, prev

    # ---------- 第 2 遍：存在模式哈希（随机权重 XOR，碰撞概率约 2^-64） ----------
    log("第 2 遍：计算可变 k-mer 的存在模式")
    rng = np.random.default_rng(a.seed)
    R = rng.integers(0, np.iinfo(np.uint64).max, size=n, dtype=np.uint64, endpoint=True)
    h = np.zeros(len(V), dtype=np.uint64)
    for i, km in genomes("第 2 遍"):
        pos = np.searchsorted(V, km)
        ok = pos < len(V)
        pos, kk = pos[ok], km[ok]
        pos = pos[V[pos] == kk]
        h[pos] ^= R[i]
    uh, first, pid, psize = np.unique(h, return_index=True, return_inverse=True, return_counts=True)
    reps = V[first]                          # 每个模式的代表 k-mer
    order = np.argsort(reps)
    reps_sorted = reps[order]
    remap = np.empty_like(order); remap[order] = np.arange(len(order))
    pid = remap[pid]; psize = psize[order]
    log(f"唯一存在模式数（最终特征数）P = {len(reps_sorted):,}；"
        f"每个模式平均含 {len(V)/len(reps_sorted):.1f} 个 k-mer")
    np.savez_compressed(os.path.join(a.outdir, "variable_kmers.npz"),
                        kmer=V, pattern=pid.astype(np.int32))
    np.save(os.path.join(a.outdir, "pattern_size.npy"), psize.astype(np.int32))
    del h, V, pid

    # ---------- 第 3 遍：代表 k-mer 的 0/1 矩阵 ----------
    log("第 3 遍：生成 0/1 矩阵")
    X = np.zeros((n, len(reps_sorted)), dtype=np.uint8)
    for i, km in genomes("第 3 遍"):
        pos = np.searchsorted(reps_sorted, km)
        ok = pos < len(reps_sorted)
        pos, kk = pos[ok], km[ok]
        X[i, pos[reps_sorted[pos] == kk]] = 1
    np.save(os.path.join(a.outdir, "kmer_matrix.npy"), X)
    np.save(os.path.join(a.outdir, "kmer_index.npy"), reps_sorted)
    for f in ["samples.txt", "phenotype_clean.csv"]:
        shutil.copy(os.path.join(a.matrix_dir, f), os.path.join(a.outdir, f))
    pr = X.mean(axis=0)
    log(f"矩阵 {X.shape[0]} × {X.shape[1]:,}；1 的比例 {X.mean():.3f}；"
        f"模式存在率中位数 {np.median(pr):.3f}")
    log(f"完成。输出目录：{a.outdir}")
    rep.close()


if __name__ == "__main__":
    main()

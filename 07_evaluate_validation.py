#!/usr/bin/env python3
"""
07_evaluate_validation.py —— 临床验证株：预测 MIC vs 表型 MIC 评估

指标（CLSI M52 / ISO 20776-2 口径）
  EA   预测与实测 log2 MIC 相差 ≤1 个稀释度
       同时给出"截尾 EA"：实测值超出训练集 MIC 范围时先截到训练范围（模型无法预测训练集未出现的档位）
  CA   S/I/R 三分类一致
  VME  实测 R、预测 S（分母 = 实测 R 株数）
  ME   实测 S、预测 R（分母 = 实测 S 株数）
  mE   minor error：一方为 I（分母 = 全部）
  S/NS 二分类：敏感性（检出 NS）、特异性、PPV、NPV —— 回归判读与分类模型分别计算
  Bland–Altman：log2(预测) − log2(实测) 的偏倚与 95% 一致性界限（R3-7）
  所有比例给 Wilson 95% CI
分层
  置信度：高 = 回归与分类判读一致 且 bootstrap 区间不跨 S/R 折点；其余为低
  谱系：最近邻训练株距离是否超过谱系阈值（需 predictions_wide.csv 或 val_qc.csv）

用法:
	cd ~/Documents/20260930_SPN_ML/05_validation/predict_2026
	python 07_evaluate_validation.py \
  		--pred predictions_long.csv \
  		--pheno val_phenotype_template.csv \
  		--wide predictions_wide.csv \
  		--model-info model_info.csv \
  		--outdir eval --include-all
  
  python 07_evaluate_validation.py --pred predictions_long.csv --pheno val_phenotype.csv \
         --wide predictions_wide.csv --outdir eval
  默认剔除 V-36、V-48（混合培养/污染）；--include-all 作为敏感性分析
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BREAKPOINTS = {"PEN": (0.06, 2), "AMC": (2, 8), "CRO": (1, 4), "ERY": (0.25, 1),
               "CLI": (0.25, 1), "LVX": (2, 8), "MFX": (1, 4), "SXT": (0.5, 4)}
TRAIN_RANGE = {"PEN": (-8, 3), "CRO": (-7, 3), "ERY": (-5, 5), "CLI": (-5, 4),
               "LVX": (-1, 4), "SXT": (-4, 4)}   # 来自 model_info.csv
NAMES = {"PEN": "Penicillin", "CRO": "Ceftriaxone", "ERY": "Erythromycin",
         "CLI": "Clindamycin", "LVX": "Levofloxacin", "SXT": "SXT"}
MIC_LABEL = {-9: "0.002", -8: "0.004", -7: "0.008", -6: "0.015", -5: "0.03", -4: "0.06",
             -3: "0.12", -2: "0.25", -1: "0.5"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", default="predictions_long.csv")
    p.add_argument("--pheno", default="val_phenotype_template.csv")
    p.add_argument("--wide", default="predictions_wide.csv", help="含最近邻谱系信息，可缺省")
    p.add_argument("--model-info", default="model_info.csv", help="读取训练集 MIC 范围，可缺省")
    p.add_argument("--outdir", default="eval")
    p.add_argument("--exclude", nargs="*", default=["V-36", "V-48"])
    p.add_argument("--include-all", action="store_true")
    return p.parse_args()


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), min(1, c + h)


def fmt(k, n):
    p, lo, hi = wilson(k, n)
    return "—" if n == 0 else f"{k}/{n} ({100*p:.1f}%, {100*lo:.1f}–{100*hi:.1f})"


def sir(x, drug):
    s, r = BREAKPOINTS[drug]
    ls, lr = np.round(np.log2(s)), np.round(np.log2(r))
    x = np.asarray(x, float)
    return np.select([x <= ls + 1e-6, x >= lr - 1e-6], ["S", "R"], "I")


def lab(e):
    e = int(e)
    return MIC_LABEL.get(e, f"{2.0 ** e:g}")


def evaluate(df, drug):
    o, p = df["obs_log2"].values, df["pred_log2"].values.astype(float)
    lo, hi = TRAIN_RANGE[drug]
    oc = np.clip(o, lo, hi)
    so, sp = sir(o, drug), df["pred_SIR"].values
    n = len(df)
    nR, nS = (so == "R").sum(), (so == "S").sum()
    vme = ((so == "R") & (sp == "S")).sum()
    me = ((so == "S") & (sp == "R")).sum()
    mi = ((so != sp) & ((so == "I") | (sp == "I"))).sum()
    ns_o = so != "S"
    ns_r = sp != "S"
    row = dict(drug=drug, n=n, obs_S=int(nS), obs_I=int((so == "I").sum()), obs_R=int(nR),
               EA=fmt((np.abs(p - o) <= 1).sum(), n),
               EA_truncated=fmt((np.abs(p - oc) <= 1).sum(), n),
               CA=fmt((so == sp).sum(), n),
               VME=fmt(vme, nR), ME=fmt(me, nS), minor=fmt(mi, n),
               reg_sens_NS=fmt((ns_r & ns_o).sum(), ns_o.sum()),
               reg_spec=fmt((~ns_r & ~ns_o).sum(), (~ns_o).sum()),
               reg_PPV=fmt((ns_r & ns_o).sum(), ns_r.sum()),
               reg_NPV=fmt((~ns_r & ~ns_o).sum(), (~ns_r).sum()))
    if "clf_NS" in df and df["clf_NS"].notna().any():
        c = df["clf_NS"].values == 1
        row.update(clf_sens_NS=fmt((c & ns_o).sum(), ns_o.sum()),
                   clf_spec=fmt((~c & ~ns_o).sum(), (~ns_o).sum()))
    d = p - oc
    row.update(BA_bias=round(d.mean(), 2),
               BA_LoA=f"{d.mean() - 1.96 * d.std(ddof=1):.2f} to {d.mean() + 1.96 * d.std(ddof=1):.2f}",
               within_exact=fmt((d == 0).sum(), n))
    return row


def main():
    a = parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    L = pd.read_csv(a.pred, dtype={"ID": str})
    P = pd.read_csv(a.pheno, dtype={"ID": str})
    drugs = [d for d in TRAIN_RANGE if d in L["drug"].unique() and f"{d}_MIC" in P]
    if os.path.exists(a.model_info):
        mi = pd.read_csv(a.model_info)
        for r in mi.itertuples():
            lo, hi = str(r.train_log2_range).split("..")
            TRAIN_RANGE[r.drug] = (float(lo), float(hi))

    ph = P.melt(id_vars="ID", value_vars=[f"{d}_MIC" for d in drugs],
                var_name="drug", value_name="obs_MIC")
    ph["drug"] = ph["drug"].str.replace("_MIC", "", regex=False)
    ph["obs_MIC"] = pd.to_numeric(ph["obs_MIC"].astype(str).str.replace(r"[≤<>=]", "", regex=True),
                                  errors="coerce")
    ph["obs_log2"] = np.round(np.log2(ph["obs_MIC"]))
    df = L.merge(ph, on=["ID", "drug"], how="inner").dropna(subset=["obs_log2"])

    # 置信度
    df["cross_bp"] = False
    for d in drugs:
        s, r = (np.round(np.log2(x)) for x in BREAKPOINTS[d])
        m = df["drug"] == d
        lo, hi = df.loc[m, "bag_log2_lo"], df.loc[m, "bag_log2_hi"]
        df.loc[m, "cross_bp"] = ((lo <= s + .5) & (hi >= s + .5)) | ((lo <= r - .5) & (hi >= r - .5))
    agree = df["reg_clf_agree"].fillna(1) == 1
    df["confidence"] = np.where(agree & ~df["cross_bp"], "high", "low")
    if os.path.exists(a.wide):
        w = pd.read_csv(a.wide, dtype={"ID": str})
        if "outside_train_lineages" in w:
            df = df.merge(w[["ID", "outside_train_lineages", "nn_dist"]], on="ID", how="left")

    df["obs_SIR"] = [sir([x], d)[0] for x, d in zip(df["obs_log2"], df["drug"])]
    df["abs_diff"] = (df["pred_log2"] - df["obs_log2"]).abs()
    df.to_csv(os.path.join(a.outdir, "per_strain_comparison.csv"), index=False)

    sets = {"main": df[~df["ID"].isin(a.exclude)]}
    if a.include_all:
        sets["all50"] = df
    out = []
    for name, sub in sets.items():
        for d in drugs:
            g = sub[sub["drug"] == d]
            out.append(dict(set=name, stratum="all", **evaluate(g, d)))
            for c in ("high", "low"):
                gg = g[g["confidence"] == c]
                if len(gg):
                    out.append(dict(set=name, stratum=f"confidence_{c}", **evaluate(gg, d)))
            if "outside_train_lineages" in g:
                gg = g[g["outside_train_lineages"] == True]  # noqa: E712
                if len(gg):
                    out.append(dict(set=name, stratum="novel_lineage", **evaluate(gg, d)))
    res = pd.DataFrame(out)
    res.to_csv(os.path.join(a.outdir, "metrics.csv"), index=False)

    main_set = sets["main"]
    errs = main_set[(main_set["obs_SIR"] != main_set["pred_SIR"]) | (main_set["abs_diff"] > 1)]
    errs[["ID", "drug", "obs_MIC", "pred_MIC", "obs_SIR", "pred_SIR", "prob_NS",
          "bag_log2_lo", "bag_log2_hi", "confidence"]].sort_values(["drug", "ID"]).to_csv(
        os.path.join(a.outdir, "discordant_strains.csv"), index=False)

    # ---------- 图：预测 vs 实测网格 ----------
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    for ax, d in zip(axes.ravel(), drugs):
        g = main_set[main_set["drug"] == d]
        lo = int(min(g["obs_log2"].min(), g["pred_log2"].min()))
        hi = int(max(g["obs_log2"].max(), g["pred_log2"].max()))
        rng = np.arange(lo, hi + 1)
        C = np.zeros((len(rng), len(rng)), int)
        for o, p in zip(g["obs_log2"].astype(int), g["pred_log2"].astype(int)):
            C[p - lo, o - lo] += 1
        ax.imshow(np.where(C > 0, C, np.nan), origin="lower", cmap="Blues", vmin=0)
        for i in range(len(rng)):
            for j in range(len(rng)):
                ok = abs(i - j) <= 1
                if ok:
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=True,
                                               color="#2e7d32", alpha=.08, lw=0))
                if C[i, j]:
                    ax.text(j, i, C[i, j], ha="center", va="center", fontsize=9,
                            color="black" if ok else "#c62828", fontweight="bold")
        s, r = (np.round(np.log2(x)) - lo for x in BREAKPOINTS[d])
        for v in (s + .5, r - .5):
            ax.axvline(v, color="grey", ls="--", lw=.8)
            ax.axhline(v, color="grey", ls="--", lw=.8)
        ax.set_xlim(-.5, len(rng) - .5)
        ax.set_ylim(-.5, len(rng) - .5)
        ax.set_xticks(range(len(rng)))
        ax.set_xticklabels([lab(e) for e in rng], rotation=90, fontsize=8)
        ax.set_yticks(range(len(rng)))
        ax.set_yticklabels([lab(e) for e in rng], fontsize=8)
        r0 = res[(res.set == "main") & (res.stratum == "all") & (res.drug == d)].iloc[0]
        ax.set_title(f"{NAMES[d]}\nEA {r0.EA.split('(')[1].split(',')[0]}, "
                     f"CA {r0.CA.split('(')[1].split(',')[0]}", fontsize=10)
        ax.set_xlabel("Reference MIC (μg/mL)", fontsize=9)
        ax.set_ylabel("Predicted MIC (μg/mL)", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(a.outdir, "Fig_pred_vs_obs.png"), dpi=300)
    plt.savefig(os.path.join(a.outdir, "Fig_pred_vs_obs.pdf"))

    # ---------- 图：Bland–Altman ----------
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharey=True)
    rs = np.random.default_rng(1)
    for ax, d in zip(axes.ravel(), drugs):
        g = main_set[main_set["drug"] == d]
        lo_, hi_ = TRAIN_RANGE[d]
        oc = np.clip(g["obs_log2"], lo_, hi_)
        diff = g["pred_log2_cont"] - oc
        mean = (g["pred_log2_cont"] + oc) / 2
        ax.scatter(mean + rs.uniform(-.08, .08, len(g)), diff, s=18, alpha=.7, color="#1565c0")
        m, sd = diff.mean(), diff.std(ddof=1)
        for v, ls in ((m, "-"), (m - 1.96 * sd, "--"), (m + 1.96 * sd, "--")):
            ax.axhline(v, color="#c62828", ls=ls, lw=1)
        ax.axhspan(-1, 1, color="#2e7d32", alpha=.07)
        ax.set_title(f"{NAMES[d]}  bias {m:.2f}, LoA {m-1.96*sd:.2f} to {m+1.96*sd:.2f}", fontsize=9)
        ax.set_xlabel("Mean log2 MIC", fontsize=9)
        ax.set_ylabel("Predicted − reference (log2)", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(a.outdir, "Fig_bland_altman.png"), dpi=300)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    show = res[res.stratum == "all"][["set", "drug", "n", "obs_S", "obs_I", "obs_R",
                                      "EA", "EA_truncated", "CA", "VME", "ME", "minor", "BA_bias"]]
    print(show.to_string(index=False))
    print(f"\n输出：{a.outdir}")


if __name__ == "__main__":
    main()

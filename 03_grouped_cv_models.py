#!/usr/bin/env python3
"""
03_grouped_cv_models.py
SPN k-mer 项目 第三步：按谱系分组的（嵌套）交叉验证建模

对应审稿意见：
  R3-1  随机拆分 -> 外层 StratifiedGroupKFold，group = k-mer 谱系簇（或 ST，敏感性分析）
  R3-2  50 次重复/中位数 -> 嵌套 CV：内层分组 CV 调参，外层只做评估
  R3-3  克隆共线性 -> 三套特征：kmer / typing(血清型+ST) / combined
  R3-4  p>>n -> 每个外层训练折内做无监督方差筛选（不看标签），报告筛选前后特征数
  R3-5  不平衡 -> 混淆矩阵、Sens/Spec/PPV/NPV/F1/AUROC/AUPRC + bootstrap 95% CI
  R3-7  MIC 取整 -> 四舍五入到最近稀释档；报告 Bland–Altman bias 与 LoA
  R1-4  各药物单独调参

用法：
  python 03_grouped_cv_models.py --quick            # 固定超参数，先快速看结果（约数十分钟）
  python 03_grouped_cv_models.py                    # 完整嵌套调参（耗时较长）
  python 03_grouped_cv_models.py --group-col ST --outdir .../03_models_ST   # ST 分组敏感性分析
  python 03_grouped_cv_models.py --group-col none --outdir .../03_models_random  # 随机拆分（乐观上界）
  python 03_grouped_cv_models.py --quick --matrix-dir .../04_k21 --k 21 --outdir .../03_models_k21   # k=21
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, r2_score,
                             roc_auc_score)
from sklearn.model_selection import (GroupKFold, RandomizedSearchCV,
                                     StratifiedGroupKFold)
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)

BASE = "/Users/chunjiang/Documents/20260930_SPN_ML"
DRUGS = ["PEN", "AMC", "CRO", "ERY", "CLI", "LVX", "MFX", "SXT"]
BREAKPOINTS = {  # 与第一步一致：PEN 口服折点；CRO/AMC 非脑膜炎折点
    "PEN": (0.06, 2), "AMC": (2, 8), "CRO": (1, 4), "ERY": (0.25, 1),
    "CLI": (0.25, 1), "LVX": (2, 8), "MFX": (1, 4), "SXT": (0.5, 4),
}
MIN_MINORITY_CLF = 10   # 少数类少于该数的药物不做分类模型（如 LVX、MFX）
FIXED = dict(n_estimators=300, max_depth=4, learning_rate=0.1,
             subsample=0.8, colsample_bytree=0.5, min_child_weight=1)
SEARCH = dict(n_estimators=[100, 200, 400], max_depth=[2, 3, 4, 6],
              learning_rate=[0.03, 0.1, 0.2], subsample=[0.7, 0.9],
              colsample_bytree=[0.3, 0.5, 0.8], min_child_weight=[1, 3])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix-dir", default=f"{BASE}/01_matrix")
    p.add_argument("--lineage", default=f"{BASE}/02_lineage/lineage_groups.csv")
    p.add_argument("--group-col", default="lin_0.2363",
                   help="lineage_groups.csv 中的簇列名；或 ST；或 none（普通随机分层拆分）")
    p.add_argument("--outdir", default=f"{BASE}/03_models")
    p.add_argument("--drugs", nargs="+", default=DRUGS)
    p.add_argument("--featsets", nargs="+", default=["kmer", "typing", "combined"])
    p.add_argument("--n-outer", type=int, default=5)
    p.add_argument("--n-inner", type=int, default=3)
    p.add_argument("--n-iter", type=int, default=8, help="内层随机搜索次数")
    p.add_argument("--top-k", type=int, default=20000,
                   help="每个训练折内按方差保留的 k-mer 数（无监督，不看标签）")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--quick", action="store_true", help="不做内层调参，用固定超参数")
    p.add_argument("--no-threshold-tuning", action="store_true",
                   help="分类阈值固定为 0.5（默认在训练折内部用 Youden 指数选阈值）")
    p.add_argument("--k", type=int, default=10, help="k-mer 长度（用于把特征解码为序列）")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--n-jobs", type=int, default=-1)
    return p.parse_args()


# ---------------------------------------------------------------- 工具函数
def decode_kmer(i, k=10):
    return "".join("ACGT"[(int(i) >> (2 * (k - 1 - j))) & 3] for j in range(k))


def sir(log2mic, drug):
    s, r = BREAKPOINTS[drug]
    mic = np.power(2.0, log2mic)
    return np.select([mic <= s * 1.001, mic >= r * 0.999], ["S", "R"], "I")


def boot_ci(fn, idx_arrays, n_boot, rng):
    """对若干等长数组做株水平 bootstrap，返回 (估计值, 2.5%, 97.5%)。"""
    n = len(idx_arrays[0])
    est = fn(*idx_arrays)
    vals = []
    for _ in range(n_boot):
        b = rng.integers(0, n, n)
        try:
            v = fn(*[x[b] for x in idx_arrays])
        except ValueError:
            continue
        if v is not None and np.isfinite(v):
            vals.append(v)
    if len(vals) < n_boot * 0.5:
        return est, np.nan, np.nan
    return est, np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def clf_metrics(y, prob, n_boot, rng, thr=0.5):
    """thr 可为标量或与 y 等长的数组（每株所在外层折在训练集内选出的阈值）。"""
    y = np.asarray(y).astype(int)
    t = np.broadcast_to(np.asarray(thr, float), y.shape).copy()
    pred = (prob >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    out = dict(n=len(y), n_NS=int(y.sum()), n_S=int((1 - y).sum()), TP=tp, TN=tn, FP=fp, FN=fn)
    two = len(np.unique(y)) == 2
    defs = {
        "accuracy": lambda a, p, t: np.mean((p >= t) == a),
        "sensitivity": lambda a, p, t: np.nan if a.sum() == 0 else np.mean(p[a == 1] >= t[a == 1]),
        "specificity": lambda a, p, t: np.nan if (a == 0).sum() == 0 else np.mean(p[a == 0] < t[a == 0]),
        "PPV": lambda a, p, t: np.nan if (p >= t).sum() == 0 else np.mean(a[p >= t] == 1),
        "NPV": lambda a, p, t: np.nan if (p < t).sum() == 0 else np.mean(a[p < t] == 0),
        "F1": lambda a, p, t: f1_score(a, (p >= t).astype(int), zero_division=0),
        "AUROC": lambda a, p, t: roc_auc_score(a, p) if len(np.unique(a)) == 2 else np.nan,
        "AUPRC": lambda a, p, t: average_precision_score(a, p) if len(np.unique(a)) == 2 else np.nan,
    }
    for k, f in defs.items():
        if k in ("AUROC", "AUPRC") and not two:
            out[k], out[k + "_lo"], out[k + "_hi"] = np.nan, np.nan, np.nan
            continue
        e, lo, hi = boot_ci(f, [y, prob, t], n_boot, rng)
        out[k], out[k + "_lo"], out[k + "_hi"] = e, lo, hi
    if not two:
        out["note"] = "单一类别：AUROC/AUPRC 无法计算"
    return out


def youden_threshold(y, p):
    """在训练集内部的 OOF 概率上取 Youden 指数最大的阈值。"""
    from sklearn.metrics import roc_curve
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, p)
    j = np.argmax(tpr - fpr)
    return float(np.clip(thr[j], 1e-6, 1 - 1e-6))


def reg_metrics(y, pred_cont, drug, n_boot, rng):
    y = np.asarray(y, float)
    pred_int = np.round(pred_cont)            # 四舍五入到最近稀释档（不再向上取整）
    ct, cp = sir(y, drug), sir(pred_int, drug)
    diff = pred_cont - y
    out = dict(n=len(y))
    for name, f in {
        "EA": lambda a, p: np.mean(np.abs(np.round(p) - a) <= 1),
        "exact": lambda a, p: np.mean(np.round(p) == a),
        "CA": lambda a, p: np.mean(sir(a, drug) == sir(np.round(p), drug)),
    }.items():
        e, lo, hi = boot_ci(f, [y, pred_cont], n_boot, rng)
        out[name], out[name + "_lo"], out[name + "_hi"] = e, lo, hi
    nR, nS = (ct == "R").sum(), (ct == "S").sum()
    out.update(
        n_S=int(nS), n_I=int((ct == "I").sum()), n_R=int(nR),
        VME_n=int(((ct == "R") & (cp == "S")).sum()),
        ME_n=int(((ct == "S") & (cp == "R")).sum()),
        minor_n=int(((ct != cp) & ((ct == "I") | (cp == "I"))).sum()),
    )
    out["VME"] = out["VME_n"] / nR if nR else np.nan   # 以真实 R 株为分母
    out["ME"] = out["ME_n"] / nS if nS else np.nan     # 以真实 S 株为分母
    out["minor"] = out["minor_n"] / len(y)
    out["bias_log2"] = diff.mean()
    out["LoA_low"] = diff.mean() - 1.96 * diff.std(ddof=1)
    out["LoA_high"] = diff.mean() + 1.96 * diff.std(ddof=1)
    out["R2"] = r2_score(y, pred_cont) if np.var(y) > 0 else np.nan
    return out


def var_topk(M, rows, top_k, chunk=50000):
    """训练折内按 log1p 计数方差选前 top_k 个 k-mer 列（无监督）。"""
    ncol = M.shape[1]
    v = np.empty(ncol, dtype=np.float64)
    for j in range(0, ncol, chunk):
        v[j:j+chunk] = np.log1p(np.asarray(M[rows, j:j+chunk], dtype=np.float32)).var(axis=0)
    n_nonconst = int((v > 0).sum())
    keep = np.sort(np.argsort(v)[::-1][:min(top_k, n_nonconst)])
    return keep, n_nonconst


# ---------------------------------------------------------------- 主流程
def main():
    a = parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    logf = open(os.path.join(a.outdir, "run_log.txt"), "w", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    log(f"参数：{json.dumps(vars(a), ensure_ascii=False)}")

    M = np.load(os.path.join(a.matrix_dir, "kmer_matrix.npy"), mmap_mode="r")
    kidx = np.load(os.path.join(a.matrix_dir, "kmer_index.npy"))
    ph = pd.read_csv(os.path.join(a.matrix_dir, "phenotype_clean.csv"),
                     dtype={"ID": str, "ST": str, "SEROTYPE": str})
    samples = open(os.path.join(a.matrix_dir, "samples.txt")).read().split()
    assert ph["ID"].tolist() == samples

    if a.group_col == "ST":
        groups = ph["ST"].astype(str).values
    elif a.group_col == "none":            # 普通分层随机拆分（每株自成一组），作为"见过克隆"的乐观上界
        groups = ph["ID"].astype(str).values
    else:
        lg = pd.read_csv(a.lineage, dtype={"ID": str})
        groups = ph[["ID"]].merge(lg[["ID", a.group_col]], on="ID", how="left")[a.group_col].values
        assert not pd.isna(groups).any(), "有菌株缺少谱系分组"
    groups = pd.Series(groups).astype(str).values
    origin = ph["Origin"].values
    n = len(ph)
    log(f"{n} 株；分组列 {a.group_col}：{len(set(groups))} 组")

    # ---------- 外层折：所有药物共用，按 PEN S/I/R × CRO NS × ERY NS 分层 ----------
    strata = (ph["PEN_SIR"] + ph["CRO_NS"].astype(str) + ph["ERY_NS"].astype(str)).values
    outer = list(StratifiedGroupKFold(n_splits=a.n_outer, shuffle=True, random_state=a.seed)
                 .split(np.zeros(n), strata, groups))
    fold_of = np.empty(n, int)
    for f, (_, te) in enumerate(outer):
        fold_of[te] = f
    log("外层各折测试集株数：" + str([len(te) for _, te in outer]))
    pd.DataFrame({"ID": samples, "group": groups, "outer_fold": fold_of,
                  "Origin": origin, "ST": ph["ST"].values, "SEROTYPE": ph["SEROTYPE"].values}).to_csv(
        os.path.join(a.outdir, "outer_folds.csv"), index=False)
    # 每个外层折：训练/测试株数、谱系数、来源构成、各药物 NS 株数、测试集包含哪些谱系
    summ = []
    for f, (tr, te) in enumerate(outer):
        row = dict(outer_fold=f, n_train=len(tr), n_test=len(te),
                   n_groups_train=len(set(groups[tr])), n_groups_test=len(set(groups[te])),
                   groups_shared=len(set(groups[tr]) & set(groups[te])),
                   test_domestic=int((origin[te] == "Domestic").sum()),
                   test_GPS=int((origin[te] == "GPS").sum()))
        for d in DRUGS:
            row[f"{d}_NS_train"] = int(ph[f"{d}_NS"].values[tr].sum())
            row[f"{d}_NS_test"] = int(ph[f"{d}_NS"].values[te].sum())
        big = pd.Series(groups[te]).value_counts()
        row["largest_test_groups"] = "; ".join(f"{g}({c})" for g, c in big.head(5).items())
        summ.append(row)
    pd.DataFrame(summ).to_csv(os.path.join(a.outdir, "outer_fold_summary.csv"), index=False)
    inner_rows = []

    # ---------- 每折的 k-mer 筛选（与药物无关，缓存） ----------
    fold_kcols, feat_log = [], []
    for f, (tr, te) in enumerate(outer):
        t0 = time.time()
        keep, n_nonconst = var_topk(M, np.sort(tr), a.top_k)
        fold_kcols.append(keep)
        feat_log.append(dict(fold=f, n_train=len(tr), n_test=len(te), kmer_total=M.shape[1],
                             kmer_nonconstant_in_train=n_nonconst, kmer_used=len(keep)))
        log(f"折 {f}：训练集非恒定 k-mer {n_nonconst:,}，保留方差前 {len(keep):,}（{time.time()-t0:.0f}s）")
    pd.DataFrame(feat_log).to_csv(os.path.join(a.outdir, "feature_counts.csv"), index=False)

    typing_raw = ph[["SEROTYPE", "ST"]].astype(str).values

    def build_X(featset, f, rows_tr, rows_te):
        blocks_tr, blocks_te, names = [], [], []
        if featset in ("kmer", "combined"):
            kc = fold_kcols[f]
            blocks_tr.append(np.asarray(M[np.ix_(rows_tr, kc)], dtype=np.float32))
            blocks_te.append(np.asarray(M[np.ix_(rows_te, kc)], dtype=np.float32))
            names += [f"k_{decode_kmer(kidx[c], a.k)}" for c in kc]
        if featset in ("typing", "combined"):
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
            blocks_tr.append(enc.fit_transform(typing_raw[rows_tr]))
            blocks_te.append(enc.transform(typing_raw[rows_te]))
            names += [f"t_{n_}" for n_ in enc.get_feature_names_out(["serotype", "ST"])]
        return np.hstack(blocks_tr), np.hstack(blocks_te), names

    recorded = set()

    def make_est(kind, Xtr, ytr, gtr, strat_tr):
        common = dict(tree_method="hist", n_jobs=a.n_jobs, random_state=a.seed, verbosity=0)
        if kind == "clf":
            pos = ytr.sum()
            common["scale_pos_weight"] = (len(ytr) - pos) / max(pos, 1)
            est = XGBClassifier(eval_metric="logloss", **common)
            scoring = "roc_auc"
            inner = StratifiedGroupKFold(n_splits=a.n_inner, shuffle=True, random_state=a.seed)
            splits = list(inner.split(Xtr, ytr, gtr))
        else:
            est = XGBRegressor(**common)
            scoring = "neg_mean_absolute_error"
            inner = StratifiedGroupKFold(n_splits=a.n_inner, shuffle=True, random_state=a.seed)
            splits = list(inner.split(Xtr, strat_tr, gtr))
        return est, scoring, splits

    def fit_model(kind, Xtr, ytr, gtr, strat_tr, tr_ids=None, tag=None):
        est, scoring, splits = make_est(kind, Xtr, ytr, gtr, strat_tr)
        if tag is not None and tag not in recorded:
            recorded.add(tag)
            for k_, (_, ite) in enumerate(splits):
                for i_ in ite:
                    inner_rows.append(dict(drug=tag[0], task=tag[1], outer_fold=tag[2],
                                           ID=tr_ids[i_], inner_fold=k_))
        if a.quick:
            est.set_params(**FIXED)
            est.fit(Xtr, ytr)
            return est, FIXED
        rs = RandomizedSearchCV(est, SEARCH, n_iter=a.n_iter, scoring=scoring, cv=splits,
                                refit=True, random_state=a.seed, n_jobs=1, error_score=np.nan)
        rs.fit(Xtr, ytr)
        return rs.best_estimator_, rs.best_params_

    oof_rows, metrics_rows, params_rows, imp_rows = [], [], [], []
    for drug in a.drugs:
        y_log2 = ph[f"{drug}_log2"].values.astype(float)
        y_ns = ph[f"{drug}_NS"].values.astype(int)
        y_sir = ph[f"{drug}_SIR"].values
        do_clf = min(y_ns.sum(), n - y_ns.sum()) >= MIN_MINORITY_CLF
        if not do_clf:
            log(f"{drug}：NS 仅 {y_ns.sum()} 株，跳过分类模型，只做 MIC 回归")
        for fs in a.featsets:
            t0 = time.time()
            p_clf = np.full(n, np.nan)
            thr_clf = np.full(n, 0.5)
            p_reg = np.full(n, np.nan)
            imp_acc = {}
            for f, (tr, te) in enumerate(outer):
                tr, te = np.sort(tr), np.sort(te)
                Xtr, Xte, names = build_X(fs, f, tr, te)
                for kind in (["clf"] if do_clf else []) + ["reg"]:
                    ytr = y_ns[tr] if kind == "clf" else y_log2[tr]
                    model, prm = fit_model(kind, Xtr, ytr, groups[tr], y_sir[tr],
                                           tr_ids=[samples[i] for i in tr], tag=(drug, kind, f))
                    if kind == "clf":
                        p_clf[te] = model.predict_proba(Xte)[:, 1]
                        if not a.no_threshold_tuning:
                            # 用选定超参数在训练集内部做分组 CV，得到 OOF 概率，再定阈值（不接触测试折）
                            inner = StratifiedGroupKFold(n_splits=a.n_inner, shuffle=True,
                                                         random_state=a.seed)
                            p_in = np.full(len(tr), np.nan)
                            for itr, ite in inner.split(Xtr, ytr, groups[tr]):
                                if len(np.unique(ytr[itr])) < 2:
                                    continue
                                mi = XGBClassifier(**model.get_params())
                                pos = ytr[itr].sum()
                                mi.set_params(scale_pos_weight=(len(itr) - pos) / max(pos, 1))
                                mi.fit(Xtr[itr], ytr[itr])
                                p_in[ite] = mi.predict_proba(Xtr[ite])[:, 1]
                            ok = ~np.isnan(p_in)
                            thr_clf[te] = youden_threshold(ytr[ok], p_in[ok]) if ok.sum() else 0.5
                    else:
                        p_reg[te] = model.predict(Xte)
                    params_rows.append(dict(drug=drug, featset=fs, task=kind, fold=f,
                                            **{k: v for k, v in prm.items()}))
                    if fs != "typing":
                        gain = model.get_booster().get_score(importance_type="gain")
                        for fk, g in gain.items():
                            nm = names[int(fk[1:])] if fk.startswith("f") and fk[1:].isdigit() else fk
                            imp_acc[(kind, nm)] = imp_acc.get((kind, nm), 0) + g / len(outer)
            log(f"{drug} / {fs}：完成（{time.time()-t0:.0f}s）")

            for i in range(n):
                oof_rows.append(dict(ID=samples[i], drug=drug, featset=fs, outer_fold=fold_of[i],
                                     origin=origin[i], true_log2=y_log2[i], true_NS=y_ns[i],
                                     pred_log2=p_reg[i], pred_log2_rounded=np.round(p_reg[i]),
                                     prob_NS=p_clf[i], threshold=thr_clf[i]))
            for sub in ["All", "Domestic", "GPS"]:
                m = np.ones(n, bool) if sub == "All" else origin == sub
                row = dict(drug=drug, featset=fs, subset=sub)
                r = reg_metrics(y_log2[m], p_reg[m], drug, a.n_boot, rng)
                metrics_rows.append({**row, "task": "MIC_regression", **r})
                if do_clf:
                    c = clf_metrics(y_ns[m], p_clf[m], a.n_boot, rng, thr=thr_clf[m])
                    metrics_rows.append({**row, "task": "S_vs_NS", **c})
            for (kind, nm), g in sorted(imp_acc.items(), key=lambda x: -x[1])[:200]:
                imp_rows.append(dict(drug=drug, featset=fs, task=kind, feature=nm, mean_gain=g))

            # 每完成一个药物/特征集就落盘，避免中途中断丢结果
            pd.DataFrame(metrics_rows).to_csv(os.path.join(a.outdir, "metrics_summary.csv"), index=False)
            pd.DataFrame(oof_rows).to_csv(os.path.join(a.outdir, "oof_predictions.csv"), index=False)
            pd.DataFrame(params_rows).to_csv(os.path.join(a.outdir, "best_params.csv"), index=False)
            pd.DataFrame(imp_rows).to_csv(os.path.join(a.outdir, "feature_importance.csv"), index=False)
            pd.DataFrame(inner_rows).to_csv(os.path.join(a.outdir, "inner_folds.csv"), index=False)

    # ---------- 简表 ----------
    met = pd.DataFrame(metrics_rows)
    fmt = lambda e, lo, hi: "" if pd.isna(e) else (f"{100*e:.1f} ({100*lo:.1f}–{100*hi:.1f})"
                                                   if pd.notna(lo) else f"{100*e:.1f}")
    reg = met[met.task == "MIC_regression"].copy()
    reg["EA% (95%CI)"] = [fmt(*r) for r in reg[["EA", "EA_lo", "EA_hi"]].values]
    reg["CA% (95%CI)"] = [fmt(*r) for r in reg[["CA", "CA_lo", "CA_hi"]].values]
    reg["VME%"] = (100 * reg.VME).round(1); reg["ME%"] = (100 * reg.ME).round(1)
    reg["minor%"] = (100 * reg.minor).round(1)
    reg["bias (LoA)"] = [f"{b:.2f} ({l:.2f}, {h:.2f})" for b, l, h in
                         reg[["bias_log2", "LoA_low", "LoA_high"]].values]
    reg[["drug", "featset", "subset", "n", "n_S", "n_I", "n_R", "EA% (95%CI)", "CA% (95%CI)",
         "VME%", "ME%", "minor%", "bias (LoA)"]].to_csv(
        os.path.join(a.outdir, "table_MIC_regression.csv"), index=False)
    clf = met[met.task == "S_vs_NS"].copy()
    if len(clf):
        for k in ["sensitivity", "specificity", "PPV", "NPV", "F1", "AUROC", "AUPRC"]:
            clf[k + " (95%CI)"] = [fmt(*r) for r in clf[[k, k + "_lo", k + "_hi"]].values]
        clf[["drug", "featset", "subset", "n", "n_S", "n_NS", "TP", "FN", "FP", "TN"] +
            [c for c in clf.columns if c.endswith("(95%CI)")]].to_csv(
            os.path.join(a.outdir, "table_S_vs_NS.csv"), index=False)
    log(f"全部完成。结果在 {a.outdir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
06_predict_validation.py —— 固定训练集（927 株）建最终模型，预测验证株 MIC 与 S/NS

设计（对应 R3-2：固定训练集，不再用 50 次随机 80% 子集取中位数）
  * 训练集：01_matrix 中全部 927 株，一次性训练每个药物的最终模型
  * 特征：仅 k-mer（R3-3 的 k-mer-only 模型；验证株不需要 Quellung/SeroBA/MLST）
    k-mer 筛选与 03 一致：在训练集上按 log1p 计数方差取前 --top-k（无监督，不看标签）
  * 超参数：在 927 株上做谱系分组的 StratifiedGroupKFold 随机搜索（搜索空间从 03 导入）
  * 分类阈值：同一分组 CV 的 OOF 概率上取 Youden 指数
  * MIC：回归预测连续 log2 MIC，四舍五入到最近稀释档（不再向上取整，R3-7），
    再按 CLSI 折点判 S/I/R；PEN 口服折点、CRO 非脑膜炎折点，与原稿一致
  * 不确定性（可选，--n-bags）：按谱系簇做 bootstrap 重抽样训练 B 个模型，
    给出每株预测 log2 MIC 的 95% 区间和判为 NS 的比例。最终预测值仍来自全量模型
  * 默认药物：PEN CRO ERY CLI LVX SXT（AMC、MFX 不在修回稿中呈现，可用 --drugs 加回）
    LVX 的 NS 株过少，只做 MIC 回归

输出（--outdir）
  predictions_long.csv   每株 × 每药：连续/取整 log2 MIC、预测 MIC、S/I/R、NS 概率、阈值、区间
  predictions_wide.csv   每株一行，便于和表型并排核对
  model_info.csv         每药最终超参数、阈值、训练集分组 CV 的 OOF AUROC/EA/CA（自检用）
  models/*.json          XGBoost 模型文件（可复现）
  selected_kmers.txt     使用的 k-mer 序列
  val_phenotype_template.csv   拿到表型后填 MIC 用的模板

用法
  python 06_predict_validation.py                 # 完整调参
  python 06_predict_validation.py --quick         # 固定超参数，几分钟出结果
  python 06_predict_validation.py --n-bags 0      # 不做 bootstrap 区间
"""
import argparse
import importlib.util
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold, cross_val_predict
from xgboost import XGBClassifier, XGBRegressor

BASE = "/Users/chunjiang/Documents"
K = 10
BREAKPOINTS = {  # CLSI M100（S 上限, R 下限），与 01 一致
    "PEN": (0.06, 2), "AMC": (2, 8), "CRO": (1, 4), "ERY": (0.25, 1),
    "CLI": (0.25, 1), "LVX": (2, 8), "MFX": (1, 4), "SXT": (0.5, 4),
}
# 稀释档的 CLSI 习惯写法
MIC_LABEL = {-9: "0.002", -8: "0.004", -7: "0.008", -6: "0.015", -5: "0.03", -4: "0.06",
             -3: "0.12", -2: "0.25", -1: "0.5"}
FALLBACK_SEARCH = {
    "n_estimators": [100, 200, 400, 800], "max_depth": [2, 3, 4, 6],
    "learning_rate": [0.03, 0.05, 0.1, 0.2], "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.1, 0.3, 0.5], "min_child_weight": [1, 3, 5],
}
FALLBACK_FIXED = dict(n_estimators=300, max_depth=4, learning_rate=0.1,
                      subsample=0.85, colsample_bytree=0.3)


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument("--matrix-dir", default=f"{BASE}/20260930_SPN_ML/01_matrix")
    p.add_argument("--val-dir", default=f"{BASE}/20260930_SPN_ML/05_validation/kmers")
    p.add_argument("--lineage", default=f"{BASE}/20260930_SPN_ML/02_lineage/lineage_groups.csv")
    p.add_argument("--group-col", default="lin_0.2363", help="谱系列名；或 ST")
    p.add_argument("--cv-script", default=os.path.join(here, "03_grouped_cv_models.py"),
                   help="从 03 导入 SEARCH/FIXED/MIN_MINORITY_CLF，保证与 CV 分析一致")
    p.add_argument("--outdir", default=f"{BASE}/20260930_SPN_ML/05_validation/predict")
    p.add_argument("--drugs", nargs="+", default=["PEN", "CRO", "ERY", "CLI", "LVX", "SXT"])
    p.add_argument("--top-k", type=int, default=20000, help="与 03 的 --top-k 保持一致")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--n-iter", type=int, default=8, help="随机搜索次数（03 默认 8）")
    p.add_argument("--n-bags", type=int, default=20, help="谱系 bootstrap 模型数；0 关闭")
    p.add_argument("--quick", action="store_true", help="不调参，用固定超参数")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--n-jobs", type=int, default=-1)
    return p.parse_args()


def load_cv_module(path):
    if not os.path.exists(path):
        return None
    try:
        spec = importlib.util.spec_from_file_location("cv03", path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m
    except Exception as e:  # noqa: BLE001
        print(f"导入 03 失败（{e}），使用脚本内置的搜索空间")
        return None


def decode_kmer(i, k=K):
    return "".join("ACGT"[(int(i) >> (2 * (k - 1 - j))) & 3] for j in range(k))


def sir_log2(x, drug):
    """按整数 log2 折点判读：0.06 视为 2^-4（0.0625），避免 0.0625 被误判为 I。"""
    s, r = BREAKPOINTS[drug]
    ls, lr = np.round(np.log2(s)), np.round(np.log2(r))
    x = np.asarray(x, float)
    return np.select([x <= ls + 1e-6, x >= lr - 1e-6], ["S", "R"], "I")


def mic_label(e):
    e = int(e)
    return MIC_LABEL.get(e, f"{2.0 ** e:g}")


def var_topk(M, top_k, chunk=50000):
    ncol = M.shape[1]
    v = np.empty(ncol)
    for j in range(0, ncol, chunk):
        v[j:j + chunk] = np.log1p(np.asarray(M[:, j:j + chunk], dtype=np.float32)).var(axis=0)
    n_nonconst = int((v > 0).sum())
    return np.sort(np.argsort(v)[::-1][:min(top_k, n_nonconst)]), n_nonconst


def youden(y, p):
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, p)
    return float(np.clip(thr[np.argmax(tpr - fpr)], 1e-6, 1 - 1e-6))


def main():
    a = parse_args()
    os.makedirs(os.path.join(a.outdir, "models"), exist_ok=True)
    logf = open(os.path.join(a.outdir, "run_log.txt"), "w", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    log(f"参数：{json.dumps(vars(a), ensure_ascii=False)}")
    cv = load_cv_module(a.cv_script)
    SEARCH = getattr(cv, "SEARCH", FALLBACK_SEARCH)
    FIXED = getattr(cv, "FIXED", FALLBACK_FIXED)
    MIN_MIN = getattr(cv, "MIN_MINORITY_CLF", 10)
    log(f"搜索空间来源：{'03 脚本' if cv is not None and hasattr(cv, 'SEARCH') else '内置'}；"
        f"分类所需少数类最少 {MIN_MIN} 株")

    # ---------- 数据 ----------
    M = np.load(os.path.join(a.matrix_dir, "kmer_matrix.npy"), mmap_mode="r")
    kidx = np.load(os.path.join(a.matrix_dir, "kmer_index.npy"))
    ph = pd.read_csv(os.path.join(a.matrix_dir, "phenotype_clean.csv"),
                     dtype={"ID": str, "ST": str, "SEROTYPE": str})
    tr_ids = open(os.path.join(a.matrix_dir, "samples.txt")).read().split()
    assert ph["ID"].tolist() == tr_ids
    V = np.load(os.path.join(a.val_dir, "val_kmer_matrix.npy"))
    va_ids = open(os.path.join(a.val_dir, "val_samples.txt")).read().split()
    assert V.shape[1] == M.shape[1], "验证矩阵列数与训练矩阵不一致，请用同一个 01_matrix 重跑 05"

    if a.group_col == "ST" or not os.path.exists(a.lineage):
        groups = ph["ST"].astype(str).values
        gname = "ST"
    else:
        lg = pd.read_csv(a.lineage, dtype={"ID": str})
        col = a.group_col if a.group_col in lg.columns else min(
            [c for c in lg.columns if c.startswith("lin_")],
            key=lambda c: abs(float(c[4:]) - float(a.group_col[4:])))
        groups = ph[["ID"]].merge(lg[["ID", col]], on="ID", how="left")[col].astype(str).values
        gname = col
    log(f"训练 {len(tr_ids)} 株，验证 {len(va_ids)} 株；分组列 {gname}（{len(set(groups))} 组）")

    keep, n_nc = var_topk(M, a.top_k)
    log(f"非恒定 k-mer {n_nc:,}，按方差保留 {len(keep):,}")
    Xtr = np.asarray(M[:, keep], dtype=np.float32)
    Xva = V[:, keep].astype(np.float32)
    with open(os.path.join(a.outdir, "selected_kmers.txt"), "w") as f:
        f.write("\n".join(decode_kmer(kidx[c]) for c in keep) + "\n")
    np.save(os.path.join(a.outdir, "selected_cols.npy"), keep)

    rng = np.random.default_rng(a.seed)
    common = dict(tree_method="hist", n_jobs=a.n_jobs, random_state=a.seed, verbosity=0)
    long_rows, info_rows = [], []

    for drug in a.drugs:
        t0 = time.time()
        y_all = ph[f"{drug}_log2"].astype(float).values
        idx = np.flatnonzero(~np.isnan(y_all))
        X, y, g = Xtr[idx], y_all[idx], groups[idx]
        sir_tr = sir_log2(y, drug)
        yns = (sir_tr != "S").astype(int)
        lo, hi = np.round(y.min()), np.round(y.max())
        do_clf = min(yns.sum(), len(yns) - yns.sum()) >= MIN_MIN
        info = dict(drug=drug, n_train=len(idx), n_NS=int(yns.sum()),
                    train_log2_range=f"{lo:g}..{hi:g}")
        log(f"{drug}：训练 {len(idx)} 株（NS {yns.sum()}）"
            + ("" if do_clf else "；NS 过少，只做 MIC 回归"))

        # ----- 回归 -----
        splits_r = list(StratifiedGroupKFold(a.n_folds, shuffle=True, random_state=a.seed)
                        .split(X, sir_tr, g))
        reg = XGBRegressor(**common)
        if a.quick:
            reg.set_params(**FIXED); prm_r = FIXED
        else:
            rs = RandomizedSearchCV(reg, SEARCH, n_iter=a.n_iter, scoring="neg_mean_absolute_error",
                                    cv=splits_r, refit=False, random_state=a.seed, n_jobs=1,
                                    error_score=np.nan)
            rs.fit(X, y); prm_r = rs.best_params_; reg.set_params(**prm_r)
        oof_r = cross_val_predict(reg, X, y, cv=splits_r)
        info.update(reg_params=json.dumps(prm_r),
                    oof_EA=np.mean(np.abs(np.round(oof_r) - np.round(y)) <= 1),
                    oof_CA=np.mean(sir_log2(np.round(oof_r), drug) == sir_tr))
        reg.fit(X, y)
        reg.save_model(os.path.join(a.outdir, "models", f"{drug}_reg.json"))
        pr = reg.predict(Xva)

        # ----- 分类 -----
        pc, thr = np.full(len(va_ids), np.nan), np.nan
        if do_clf:
            spw = (len(yns) - yns.sum()) / max(yns.sum(), 1)
            splits_c = list(StratifiedGroupKFold(a.n_folds, shuffle=True, random_state=a.seed)
                            .split(X, yns, g))
            clf = XGBClassifier(eval_metric="logloss", scale_pos_weight=spw, **common)
            if a.quick:
                clf.set_params(**FIXED); prm_c = FIXED
            else:
                rs = RandomizedSearchCV(clf, SEARCH, n_iter=a.n_iter, scoring="roc_auc",
                                        cv=splits_c, refit=False, random_state=a.seed, n_jobs=1,
                                        error_score=np.nan)
                rs.fit(X, yns); prm_c = rs.best_params_; clf.set_params(**prm_c)
            oof_c = cross_val_predict(clf, X, yns, cv=splits_c, method="predict_proba")[:, 1]
            thr = youden(yns, oof_c)
            info.update(clf_params=json.dumps(prm_c), threshold=thr,
                        oof_AUROC=roc_auc_score(yns, oof_c),
                        oof_sens=np.mean(oof_c[yns == 1] >= thr),
                        oof_spec=np.mean(oof_c[yns == 0] < thr))
            clf.fit(X, yns)
            clf.save_model(os.path.join(a.outdir, "models", f"{drug}_clf.json"))
            pc = clf.predict_proba(Xva)[:, 1]

        # ----- 谱系 bootstrap 区间 -----
        bag_r = np.full((a.n_bags, len(va_ids)), np.nan)
        bag_c = np.full((a.n_bags, len(va_ids)), np.nan)
        if a.n_bags > 0:
            ug = np.unique(g)
            rows_of = {k: np.flatnonzero(g == k) for k in ug}
            for b in range(a.n_bags):
                rows = np.concatenate([rows_of[k] for k in rng.choice(ug, len(ug), replace=True)])
                mb = XGBRegressor(**{**common, "random_state": a.seed + b}).set_params(**prm_r)
                bag_r[b] = mb.fit(X[rows], y[rows]).predict(Xva)
                if do_clf and 0 < yns[rows].sum() < len(rows):
                    mc = XGBClassifier(eval_metric="logloss", scale_pos_weight=spw,
                                       **{**common, "random_state": a.seed + b}).set_params(**prm_c)
                    bag_c[b] = mc.fit(X[rows], yns[rows]).predict_proba(Xva)[:, 1] >= thr

        pr_int = np.clip(np.round(pr), lo, hi)
        sir_pred = sir_log2(pr_int, drug)
        for i, sid in enumerate(va_ids):
            r = dict(ID=sid, drug=drug, pred_log2_cont=round(float(pr[i]), 3),
                     pred_log2=int(pr_int[i]), pred_MIC=mic_label(pr_int[i]),
                     pred_SIR=sir_pred[i],
                     at_range_edge=bool(pr_int[i] in (lo, hi)))
            if do_clf:
                r.update(prob_NS=round(float(pc[i]), 4), threshold=round(thr, 4),
                         clf_NS=int(pc[i] >= thr),
                         reg_clf_agree=int((sir_pred[i] != "S") == (pc[i] >= thr)))
            if a.n_bags > 0:
                r.update(bag_log2_lo=round(float(np.nanpercentile(bag_r[:, i], 2.5)), 2),
                         bag_log2_hi=round(float(np.nanpercentile(bag_r[:, i], 97.5)), 2))
                if do_clf:
                    r["bag_frac_NS"] = round(float(np.nanmean(bag_c[:, i])), 3)
            long_rows.append(r)
        info["seconds"] = round(time.time() - t0)
        info_rows.append(info)
        log(f"  {drug} 完成（{info['seconds']}s）；训练集分组 CV 自检：EA {info['oof_EA']:.1%}，"
            f"CA {info['oof_CA']:.1%}" + (f"，AUROC {info['oof_AUROC']:.3f}，阈值 {thr:.3f}"
                                         if do_clf else ""))
        log(f"  验证株预测：S {np.sum(sir_pred == 'S')} / I {np.sum(sir_pred == 'I')} / "
            f"R {np.sum(sir_pred == 'R')}；分类与回归判读不一致 "
            f"{int(sum(1 - r.get('reg_clf_agree', 1) for r in long_rows if r['drug'] == drug))} 株")

    # ---------- 输出 ----------
    lt = pd.DataFrame(long_rows)
    lt.to_csv(os.path.join(a.outdir, "predictions_long.csv"), index=False)
    wide = lt.pivot(index="ID", columns="drug", values=["pred_MIC", "pred_SIR"])
    wide.columns = [f"{d}_{v.replace('pred_', 'pred')}" for v, d in wide.columns]
    wide = wide[[f"{d}_{s}" for d in a.drugs for s in ("predMIC", "predSIR")]].loc[va_ids]
    qc_path = os.path.join(a.val_dir, "val_qc.csv")
    if os.path.exists(qc_path):
        qc = pd.read_csv(qc_path, dtype={"ID": str, "nn_ST": str})
        keepc = [c for c in ["ID", "flag", "nn_train_ID", "nn_dist", "nn_ST", "nn_lineage",
                             "outside_train_lineages"] if c in qc.columns]
        wide = wide.reset_index().merge(qc[keepc], on="ID", how="left").set_index("ID")
    wide.to_csv(os.path.join(a.outdir, "predictions_wide.csv"))
    pd.DataFrame(info_rows).to_csv(os.path.join(a.outdir, "model_info.csv"), index=False)
    pd.DataFrame({"ID": va_ids, **{f"{d}_MIC": "" for d in a.drugs}}).to_csv(
        os.path.join(a.outdir, "val_phenotype_template.csv"), index=False)
    log(f"完成。输出：{a.outdir}")
    logf.close()


if __name__ == "__main__":
    main()

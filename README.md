# SPN k-mer AMR Prediction — Revision Pipeline (iLABMED)

Prediction of antimicrobial susceptibility of *Streptococcus pneumoniae* from whole-genome k-mer features with XGBoost.

This repository contains the re-analysis performed for the revised manuscript *"Predicting Antimicrobial Resistance Phenotypes in Streptococcus pneumoniae Using Machine Learning Analysis of Genomic Data: A Practical Approach"*, submitted to **iLABMED**.

---

## 1. Background

The first version of this study used a random 80/20 split (`train_test_split`) of 953 isolates, 10-mer counts from raw reads, serotype and MLST as features, and a validation step that repeated random training 50 times and took the median. In the first round of review, Reviewer 3 raised several basic methodological problems:

- Random splitting lets closely related isolates of the same clone enter both training and test sets. The model may learn clonal lineage rather than resistance mechanism, so EA/CA are overestimated.
- Serotype and MLST are collinear with resistance and reflect lineage.
- The number of features, sparsity and feature handling were not reported.
- Accuracy alone is not suitable for imbalanced data.
- MIC values were rounded **up**, which biases predictions toward higher MICs.
- The 50 repeated trainings were not independent, and the 33 validation isolates were not a true external set.

To answer these comments, the whole analysis was rebuilt from the assembled contigs. Many of these steps were difficult for us in 2022, for example lineage-grouped nested cross-validation, k = 21 presence/absence matrices with tens of millions of k-mers, and bootstrap confidence intervals. With AI-assisted programming, these steps could be written, tested on synthetic data and run on a normal laptop within a short time. The scripts in this repository are the result.

The main changes compared with the original analysis are:

| Item | Original manuscript | Revision |
|---|---|---|
| k-mer source | Raw reads, KMC `-ci2`, count ceiling 255 | Assembled contigs, KMC `-k10 -ci1 -fm -cs65535` (copy number, independent of depth) |
| Training set | 633 domestic + 320 GPS (random 762/191 split) | 607 domestic + 320 GPS = **927**, cross-validation only |
| Data split | Random | **Lineage-grouped** (k-mer Jaccard clusters), with random split as an upper-bound reference |
| Feature sets | k-mer + serotype + MLST together | k-mer only / typing only / combined, reported separately |
| Hyperparameters | Random search on whole data | Nested CV, tuned separately for each drug |
| MIC rounding | Rounded up | Rounded to the **nearest** doubling dilution |
| Metrics | Accuracy, EA, CA, VME, ME | + sensitivity, specificity, PPV, NPV, F1, AUROC, AUPRC, Bland–Altman, 95% CI |
| k value | 10 only | 10 (main) and 21 (comparison) |
| Clinical validation | 33 isolates, 50 random re-trainings, median | 50 isolates (48 analysable), **one fixed model** trained on all 927, nearest-neighbour clonal overlap reported |

---

## 2. Repository contents

```
01_build_kmer_matrix.py     Build the 927 × 10-mer count matrix with QC
02_lineage_clusters.py      k-mer based lineage clustering for grouped CV
03_grouped_cv_models.py     Nested grouped CV, XGBoost classifier + regressor
04_kmer21_matrix.py         k = 21 presence/absence matrix from contigs
05_validation_kmers.py      10-mer features of the validation isolates, QC, clonal overlap
06_predict_validation.py    Final models on 927 isolates, predict validation isolates
07_evaluate_validation.py   Compare predictions with phenotypic MICs

927_combined.xlsx           Phenotype and typing table of the 927 training isolates
607qc.txt                   Assembly QC of the 607 domestic training isolates
val_metadata_template.csv   Metadata of the clinical validation isolates
val_phenotype_template.csv  Phenotypic MICs (Etest) of the clinical validation isolates
```

### 2.1 Auxiliary files

**`927_combined.xlsx`** — one row per training isolate. Important columns:

| Column | Content |
|---|---|
| `ID` | Isolate ID; must match the sample names in the k-mer files |
| `Origin` | `Domestic` (n = 607) or `GPS` (n = 320) |
| `ST` | MLST sequence type (MLST 2.0 for domestic isolates, GPS metadata for GPS isolates) |
| `SEROTYPE` | Serotype used in the analysis. Domestic: Quellung reaction; GPS: in silico serotype from the GPS project |
| `SEROTYPE_AI2` | SeroBA result (final, correct run). For comparison only |
| `SEROTYPE_AI` | Earlier SeroBA run with wrong parameters. **Not used** |
| `PEN`, `AMC`, `CRO`, `ERY`, `CLI`, `LVX`, `MFX`, `SXT` | MIC (μg/mL). Values such as `≤0.03`, `>16`, `0.064` are accepted and normalised to log2 dilution steps |

Note: for the 320 GPS isolates, the AMC, MFX and SXT values are in silico values filled to match the domestic drug panel. AMC and MFX are therefore not presented in the revised manuscript, and SXT results of GPS isolates should be read with this in mind. PEN, CRO, ERY, CLI and LVX are measured values in both sources.

**`607qc.txt`** — whitespace-separated assembly QC of the 607 domestic training isolates (total length, number of contigs, N50, GC%, coverage depth and low-coverage contig length), produced by the server script `02_assembly_qc.sh`. It is used as the reference distribution when the validation assemblies are checked. 74 of the 607 isolates show abnormal values (genome size, GC% or depth). They were kept in the training set in this revision; a stricter clean-up is left for future work.

**`val_metadata_template.csv`** — basic information of the 50 clinical validation isolates (V-01 to V-50): isolate ID, collection year, specimen type, patient age group, and ST and serotype from the server MLST/SeroBA runs. It is used for the descriptive table of the validation set. The models do not use it, because the final models use k-mer features only.

**`val_phenotype_template.csv`** — template written by `06_predict_validation.py`. Fill in the Etest MIC (bioMérieux) of each validation isolate as raw values (`0.06`, `≤0.03`, `>16` are all accepted). It is read by `07_evaluate_validation.py`.

---

## 3. Requirements

Analysis was run on a MacBook Pro (Apple M3) in a conda environment named `ai`. The assemblies were produced on the laboratory Linux server (SPAdes 3.13.0, `--careful`, automatic k values).

- Python ≥ 3.10
- numpy, pandas, polars, scipy, scikit-learn, xgboost, matplotlib, openpyxl
- KMC 3.2.4 (`kmc`, `kmc_tools`) in `PATH` (needed by `05_validation_kmers.py`)

Installation with the Tsinghua mirrors:

```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda
conda config --set show_channel_urls yes

conda create -n ai python=3.11 -y
conda activate ai
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    numpy pandas polars scipy scikit-learn xgboost matplotlib openpyxl

conda install -y kmc        # or copy the KMC binaries to ~/.local/bin
# macOS only, if the binaries are blocked by Gatekeeper:
xattr -dr com.apple.quarantine ~/.local/bin/kmc*
```

---

## 4. Directory layout

All scripts have default paths under `/Users/chunjiang/Documents`. On another computer, pass the paths explicitly with the command-line options (every script supports `--help`).

```
~/Documents/
├── 927_combined.xlsx
├── 20260930_633菌株/spn607_clean/k10_output/607_long-merge-kmer-file.tsv
├── 20260930_GPS世界菌株/k10_output/320_long-merge-kmer-file.tsv
├── 20260930_临床验证菌株50株/Assem_2026/*.fasta      # validation contigs
└── 20260930_SPN_ML/
    ├── 01_matrix/          # step 1
    ├── 02_lineage/         # step 2
    ├── 03_models/          # step 3, lineage-grouped
    ├── 03_models_random/   # step 3, random split
    ├── 03_models_k21/      # step 3 on k = 21
    ├── 04_k21/             # step 4
    ├── 05_validation/kmers/
    ├── 06_predict/
    └── 07_eval/
```

The input k-mer files are KMC dumps in long format, one line per k-mer per isolate (`kmer <TAB> count <TAB> ID`), produced from the cleaned contigs with `kmc -k10 -fm -ci1 -cs65535`.

---

## 5. Pipeline

```
contigs ─► KMC (k=10) ─► 01 matrix ─► 02 lineage ─► 03 grouped CV ───────┐
contigs ─────────────────────────────► 04 k=21 matrix ─► 03 (--k 21)     │
validation contigs ─► 05 validation k-mers ─► 06 predict ─► 07 evaluate ◄┘ (search space, lineage)
```

### Step 1 — `01_build_kmer_matrix.py`

**Function.** Reads the two long-format 10-mer files and the phenotype table, and builds a strain × k-mer count matrix.

- Encodes each 10-mer as a 2-bit integer (A0 C1 G2 T3) and checks that the files contain canonical k-mers only.
- Detects the separator automatically (tab, space or mixed) and skips broken lines (for example, two records joined after an interrupted write). Skipped lines are counted in the report.
- Matches isolate IDs between the k-mer files and `927_combined.xlsx` and reports unmatched IDs.
- Converts MICs to log2 dilution steps and S/I/R categories with CLSI M100 breakpoints (PEN oral; CRO and AMC non-meningitis).
- Flags outlier isolates by number of distinct k-mers and total count.
- Runs PCA on the most variable k-mers to check for batch effects between domestic and GPS isolates.

**Example.**

```bash
conda activate ai
python 01_build_kmer_matrix.py \
    --kmer ~/Documents/20260930_633菌株/spn607_clean/k10_output/607_long-merge-kmer-file.tsv \
           ~/Documents/20260930_GPS世界菌株/k10_output/320_long-merge-kmer-file.tsv \
    --pheno ~/Documents/927_combined.xlsx \
    --outdir ~/Documents/20260930_SPN_ML/01_matrix
```

**Output.**

| File | Content |
|---|---|
| `kmer_matrix.npy` | 927 × 503,875 uint16 count matrix |
| `kmer_index.npy` | 2-bit code of each column |
| `samples.txt` | Row order |
| `phenotype_clean.csv` | Aligned phenotype: log2 MIC, S/I/R, ST, serotype, origin |
| `qc_per_strain.csv` | Distinct k-mers, total count, min/max count, outlier flag |
| `qc_report.txt` | Number of features, sparsity, canonical check, ID matching |
| `pca_batch.png`, `pca_coords.csv` | PCA coloured by origin |

**How to read.** Each isolate contains about 420,000 distinct 10-mers, about 80% of the 524,800 possible canonical 10-mers, so the 10-mer space is close to saturation. The matrix has 503,875 non-empty columns, which is far more than 927 isolates (p ≫ n). This number and the sparsity are reported in `qc_report.txt` (R3-4). The PCA shows that the main structure is clonal (CC271 forms a separate cloud) rather than a simple domestic/GPS batch effect.

---

### Step 2 — `02_lineage_clusters.py`

**Function.** Defines lineage groups for grouped cross-validation, without external tools such as PopPUNK.

- Uses only variable k-mers (present in 1%–99% of isolates) as presence/absence profiles.
- Calculates pairwise Jaccard distances and performs average-linkage hierarchical clustering.
- Cuts the tree at a series of distance thresholds and reports, for each threshold: number of clusters, singleton clusters, size of the largest cluster, number of STs split across clusters, whether ST271 and ST320 (CC271) fall in the same cluster, and number of clusters that contain both domestic and GPS isolates.
- Recommends the smallest threshold that keeps CC271 together, splits the fewest STs, and keeps the largest cluster below 35%.

**Example.**

```bash
python 02_lineage_clusters.py \
    --indir  ~/Documents/20260930_SPN_ML/01_matrix \
    --outdir ~/Documents/20260930_SPN_ML/02_lineage
# optional: change the STs used for calibration
python 02_lineage_clusters.py --cc271 271 320 236 1464
```

**Output.** `threshold_scan.csv`, `lineage_groups.csv` (cluster label of each isolate at every threshold, columns `lin_<threshold>`), `cluster_vs_ST_<t>.csv`, `dist_hist.png`, `pca_by_lineage.png`, `lineage_report.txt`.

**How to read.** The threshold used in the main analysis is **`lin_0.2363`**:

| Threshold | Clusters | Singletons | Largest cluster | CC271 in n clusters | Mixed-origin clusters |
|---|---|---|---|---|---|
| 0.0033 | 732 | 641 | 2.0% | 129 | 0 |
| 0.1117 | 183 | 87 | 24.1% | 1 | 6 |
| **0.2363** | **123** | **45** | **25.1%** | **1** | **11** |
| 0.4049 | 4 | 0 | 61.9% | 1 | 4 |

At 0.2363 no ST is split (the only "split ST" is the four GPS isolates with unresolved ST `-`), CC271 (ST271, ST320, ST236, ST1464; 233 isolates) forms one cluster, and the resolution is close to GPSC. Only 11 of 123 clusters contain both domestic and GPS isolates, so lineage grouping largely also groups by origin. Cluster numbers are labels only and have no biological meaning.

---

### Step 3 — `03_grouped_cv_models.py`

**Function.** Main model evaluation (R3-1, R3-3, R3-4, R3-5, R3-7; R1-4).

- For each drug separately: nested cross-validation with `StratifiedGroupKFold`. Outer folds (default 5) for evaluation; inner folds for random hyperparameter search.
- Grouping variable: lineage cluster (`--group-col lin_0.2363`, default), `ST`, or `none` (random split, upper-bound reference).
- Three feature sets: k-mer only, typing only (serotype + ST, one-hot), and combined.
- Unsupervised feature filtering (top 20,000 k-mers by variance) is done **inside each training fold** to avoid information leakage.
- XGBoost classifier for S vs NS. The classification threshold is chosen inside the training fold by the Youden index on inner out-of-fold probabilities; the test fold is never used.
- XGBoost regressor for log2 MIC. Predictions are rounded to the **nearest** doubling dilution, then interpreted by CLSI breakpoints.
- Bootstrap 95% CIs for all metrics; results also stratified by origin (All / Domestic / GPS).
- LVX has very few NS isolates, so it is evaluated by MIC regression only.

**Example.**

```bash
# quick mode (fixed hyperparameters) – first look, lineage-grouped
python 03_grouped_cv_models.py --quick \
    --outdir ~/Documents/20260930_SPN_ML/03_models

# random split as reference ("clone already seen")
python 03_grouped_cv_models.py --quick --group-col none \
    --outdir ~/Documents/20260930_SPN_ML/03_models_random

# ST-grouped sensitivity analysis
python 03_grouped_cv_models.py --quick --group-col ST \
    --outdir ~/Documents/20260930_SPN_ML/03_models_ST

# full nested tuning, selected drugs
python 03_grouped_cv_models.py --drugs PEN CRO ERY CLI LVX SXT \
    --outdir ~/Documents/20260930_SPN_ML/03_models_full

# k = 21 matrix from step 4
python 03_grouped_cv_models.py --quick --k 21 \
    --matrix-dir ~/Documents/20260930_SPN_ML/04_k21 \
    --outdir ~/Documents/20260930_SPN_ML/03_models_k21
```

Other options: `--n-outer`, `--n-inner`, `--top-k`, `--n-boot`, `--seed` (default 2026).

**Output.**

| File | Content |
|---|---|
| `table_S_vs_NS.csv` | Per drug × feature set × subset: n_S, n_NS, confusion matrix (TP/FN/FP/TN), sensitivity, specificity, PPV, NPV, F1, AUROC, AUPRC with 95% CI |
| `table_MIC_regression.csv` | EA, CA, VME, ME, minor error, Bland–Altman bias and limits of agreement, with 95% CI |
| `feature_importance.csv` | Top k-mers (decoded to sequence) per drug |

**How to read.** Lineage-grouped CV is the lower bound ("new clone"); random split is the upper bound ("clone already seen in training"). The gap between them measures how much of the performance comes from clonal memory. Example results (quick mode, combined features, all 927 isolates):

| Drug | EA: lineage → random | CA: lineage → random | VME: lineage → random | AUROC: lineage → random |
|---|---|---|---|---|
| PEN | 85.2 → 90.5 | 77.8 → 86.0 | 0.3 → 0.3 | 99.2 → 99.9 |
| CRO | 78.5 → 89.5 | 79.4 → 83.6 | 92.2 → 18.0 | 88.5 → 91.7 |
| ERY | 82.3 → 89.6 | 93.7 → 97.4 | 2.5 → 0.6 | 98.4 → 98.9 |
| CLI | 82.1 → 88.7 | 92.8 → 95.0 | 0.3 → 0.2 | 94.7 → 96.0 |
| SXT | 64.4 → 84.8 | 72.2 → 85.0 | 3.4 → 4.5 | 93.5 → 95.9 |

- **PEN, ERY, CLI**: small gap, and the k-mer model is much better than the typing-only model. The signal transfers across lineages (for ERY/CLI consistent with presence/absence of *erm*(B)).
- **CRO (and AMC)**: performance depends almost completely on recognising CC271, which contains about 80% of the NS isolates. When CC271 is held out as a whole, VME is very high. 10-mers do not capture the PBP mosaic signal well.
- Pooled AUROC for PEN and ERY is partly confounded by origin, because all domestic isolates are PEN-NS. The GPS-subset AUROC is a less confounded estimate.
- These numbers are from the quick mode. Final numbers in the manuscript are from the full nested run.

---

### Step 4 — `04_kmer21_matrix.py`

**Function.** Tests whether a longer k (R1-5, R3-4) captures allele-level signals better, for example PBP mosaic alleles, *folA*/*folP* and QRDR mutations.

- Enumerates canonical 21-mers directly from contigs (no KMC; a text dump at k = 21 would be tens of GB), records presence/absence only.
- Pass 1: counts in how many isolates each 21-mer occurs. Pass 2: keeps variable 21-mers (present in 10–917 isolates) and merges 21-mers with identical presence patterns across all isolates into one feature. Pass 3: writes the 927 × P binary matrix.
- Output layout is compatible with step 3 (`--matrix-dir`, `--k 21`).

**Example.**

```bash
# 1. locate contig files only, write 04_k21/manifest.tsv, no calculation
python 04_kmer21_matrix.py --dry-run
# 2. check manifest.tsv (927 isolates, same contigs as used for k=10), edit if needed
python 04_kmer21_matrix.py --manifest ~/Documents/20260930_SPN_ML/04_k21/manifest.tsv
```

Each pass takes about 5–15 min depending on CPU cores; peak memory about 4–6 GB.

**How to read.** The report lists the pan-21-mer total, core 21-mers, singletons, variable 21-mers and the final number of patterns (about 344,000). In our data k = 21 did not perform better than k = 10 overall and mainly helped SXT. This supports keeping k = 10 as the main analysis, and the comparison is reported in the supplement.

---

### Step 5 — `05_validation_kmers.py`

**Function.** Prepares the clinical validation isolates so that their features are exactly comparable with the training matrix (R3-2).

1. Assembly QC of each contig file: number of contigs, total length, N50, GC%, N bases.
2. KMC with the same parameters as the training set (`-k10 -ci1 -fm`, canonical).
3. Alignment to the training `kmer_index.npy`: k-mers absent from training are dropped, missing k-mers are filled with 0. Columns are identical to the 927-isolate matrix.
4. Consistency checks: proportion of validation k-mers found in the training index (should be close to 100%; < 95% triggers a warning), maximum count, and distinct k-mers compared with the training distribution (|z| > 3 flagged).
5. Clonal overlap: Jaccard distance (variable k-mers) between each validation isolate and all 927 training isolates; nearest training isolate with its ST, origin and lineage cluster. Isolates with nearest distance > 0.2363 are flagged as "lineage not seen in training".

**Example.**

```bash
python 05_validation_kmers.py \
    --contigs ~/Documents/20260930_临床验证菌株50株/Assem_2026 \
    --pattern "*.fasta" \
    --outdir  ~/Documents/20260930_SPN_ML/05_validation/kmers \
    --jobs 4
# add --keep-long to also write long-format k-mer text files
# add --cs <value> only if the training KMC run used a count ceiling
```

**Output.** `val_kmer_matrix.npy` (n_val × 503,875), `val_samples.txt`, `val_qc.csv`, `val_train_jaccard.npy` (n_val × 927), `val_report.txt`.

**How to read.** Compare `val_qc.csv` with `607qc.txt`. In our run, 48 isolates were normal (total length 2.03–2.28 Mb, GC 39.6–40.6%). Two isolates failed and were excluded:

- **V-36**: 4.29 Mb, normal GC, low depth — two different pneumococci in one culture.
- **V-48**: 5.48 Mb, GC 43.7%, > 4,000 contigs, N50 about 2 kb — heavy contamination or failed library.

The validation isolates were sequenced at higher depth (244–603×) than the training isolates. Short low-coverage contigs can add several thousand extra distinct 10-mers, so a mildly raised z value of distinct k-mers is expected and is not by itself a reason for exclusion. The nearest-neighbour table should be reported in the manuscript to describe honestly how much the validation set overlaps with the training clones.

---

### Step 6 — `06_predict_validation.py`

**Function.** Trains one final model per drug on all 927 isolates and predicts the validation isolates.

- Fixed training set (all 927 isolates) instead of 50 random 80% subsets with median (R3-2).
- **k-mer features only.** Validation isolates therefore do not need Quellung, SeroBA or MLST (R3-3).
- Hyperparameter search space and settings are imported from `03_grouped_cv_models.py`, so the final model is consistent with the CV analysis. Tuning uses lineage-grouped CV on the 927 isolates; the classification threshold is the Youden index on the same out-of-fold probabilities.
- Regression output is rounded to the nearest dilution (R3-7), written in CLSI style (0.06, 0.12, …) and interpreted as S/I/R. The classifier gives the NS probability.
- Uncertainty: lineage-stratified bootstrap (default 20 models, `--n-bags`) gives a 95% interval of log2 MIC and the proportion of bootstrap models calling NS for each isolate.
- Default drugs: PEN, CRO, ERY, CLI, LVX, SXT. AMC and MFX can be added with `--drugs`. LVX: regression only.

**Example.**

```bash
python 06_predict_validation.py \
    --val-dir ~/Documents/20260930_SPN_ML/05_validation/kmers \
    --outdir  ~/Documents/20260930_SPN_ML/06_predict
# fast check with fixed hyperparameters
python 06_predict_validation.py --quick --outdir ~/Documents/20260930_SPN_ML/06_predict_quick
# no bootstrap intervals
python 06_predict_validation.py --n-bags 0
```

**Output.**

| File | Content |
|---|---|
| `predictions_long.csv` | Isolate × drug: continuous and rounded log2 MIC, predicted MIC, S/I/R, NS probability, threshold, bootstrap interval |
| `predictions_wide.csv` | One row per isolate, with nearest-neighbour information |
| `model_info.csv` | Final hyperparameters, threshold, and the OOF AUROC/EA/CA of the final model (self-check) |
| `models/*.json` | XGBoost model files (one per drug and task) |
| `selected_kmers.txt` | k-mer sequences used by the final models |
| `val_phenotype_template.csv` | Template for the phenotypic MICs |

**How to read.** The regression MIC and its interpretation are the primary result, because this is how routine AST is reported. The classifier is supportive. An isolate is **low confidence** when the regression and classifier calls disagree, or when the bootstrap interval crosses the S/R breakpoint. In practice, low-confidence results should go back to phenotypic testing (this is the clinical pathway asked for by R1-2).

---

### Step 7 — `07_evaluate_validation.py`

**Function.** Compares predicted and phenotypic MICs of the validation isolates (CLSI M52 / ISO 20776-2).

- EA: predicted and reference log2 MIC within ±1 dilution. Off-scale values outside the training MIC range are truncated to the training range before comparison ("truncated EA"), because the model cannot predict dilutions it has never seen.
- CA (S/I/R), VME (denominator: reference R), ME (denominator: reference S), minor error (denominator: all).
- S vs NS sensitivity, specificity, PPV and NPV for the regression call and the classifier call separately.
- Bland–Altman bias and 95% limits of agreement of log2(predicted) − log2(reference).
- Wilson 95% CI for all proportions.
- Stratification by confidence (high/low) and by lineage (seen/not seen in training).
- V-36 and V-48 are excluded by default; `--include-all` runs a sensitivity analysis with all isolates.

**Example.**

```bash
cd ~/Documents/20260930_SPN_ML/06_predict
python 07_evaluate_validation.py \
    --pred  predictions_long.csv \
    --pheno val_phenotype_template.csv \
    --wide  predictions_wide.csv \
    --model-info model_info.csv \
    --outdir ~/Documents/20260930_SPN_ML/07_eval
```

**Output.** `metrics.csv`, `discordant_strains.csv`, `Fig_pred_vs_obs.png` (predicted vs reference MIC grid), `Fig_bland_altman.png`.

**How to read.** Results of the 48 analysable isolates (Wilson 95% CI):

| Drug | Reference S/I/R | EA | CA | VME | ME | Minor error |
|---|---|---|---|---|---|---|
| PEN | 1/28/19 | 85.4% (72.8–92.8) | 81.2% (68.1–89.8) | 0/19 | 0/1 | 9 (18.8%) |
| CRO | 31/3/14 | 83.3% (70.4–91.3) | 79.2% (65.7–88.3) | **2/14** | 2/31 | 6 (12.5%) |
| ERY | 0/1/47 | 81.2% (68.1–89.8) | 97.9% (89.1–99.6) | 0/47 | — | 1 |
| CLI | 2/1/45 | 68.8% (54.7–80.1) | 93.8% (83.2–97.9) | 0/45 | 2/2 | 1 |
| LVX | 48/0/0 | 100% | 100% | — | 0/48 | 0 |
| SXT | 7/11/30 | 72.9% (59.0–83.4) | 72.9% (59.0–83.4) | 0/30 | 0/7 | 13 (27.1%) |

- The validation results are close to the lineage-grouped CV estimates (PEN EA 85%, CRO 83%, CLI 69%, SXT 74%), and much lower than the 94–98% obtained with random splitting in the original manuscript. Grouped CV is therefore a realistic estimate.
- **CRO VME (V-29, V-30)**: both MIC 8 μg/mL by Etest, predicted 1 μg/mL with high confidence. Their nearest training neighbours belong to poorly represented lineages; uncommon PBP mosaic types are a blind spot of the 10-mer model. *pbp1a/2b/2x* typing of these isolates is recommended.
- **CLI ME (V-23, V-40)**: ERY 1–2 μg/mL, CLI 0.25 μg/mL, consistent with the M phenotype (*mef*(A/E)). The training set is dominated by *erm*(B)-mediated MLS_B resistance, so the model tends to predict "ERY-R = CLI-R".
- **PEN**: all errors are minor errors at the 1 ↔ 2 μg/mL I/R boundary; S vs NS sensitivity is 100% (47/47).
- **SXT**: minor errors concentrate at 1–2 μg/mL, in line with the known difficulty of reading the 80% inhibition end point.

---

## 6. Mapping to reviewer comments

| Comment | Script(s) | Output used in the response |
|---|---|---|
| R3-1 random split, clonal leakage | 02, 03 | Lineage-grouped vs random vs ST-grouped tables |
| R3-2 validation design, 50 repeated trainings | 05, 06, 07 | Fixed model, nearest-neighbour overlap, bootstrap intervals, Wilson CI |
| R3-3 serotype/MLST collinearity | 03, 06 | k-mer only / typing only / combined |
| R3-4 p ≫ n, feature handling | 01, 03, 04 | Feature number and sparsity; in-fold filtering; k = 21 dedup |
| R3-5 metrics for imbalanced data | 03, 07 | Confusion matrix, Se, Sp, PPV, NPV, F1, AUROC, AUPRC, 95% CI |
| R3-6 missing isolates for AMC/MFX/SXT | 01 | AMC and MFX removed from the main results; GPS in silico values annotated |
| R3-7 rounding up | 03, 06, 07 | Nearest dilution; Bland–Altman |
| R1-3 serotype/MLST source | 01 | `SEROTYPE` column only (Quellung / GPS in silico) |
| R1-4 drug-specific tuning | 03, 06 | Per-drug nested search |
| R1-5 choice of k | 04 | k = 10 vs k = 21 |

---

## 7. Notes and cautions

1. **Run the steps in order.** 05 depends on `kmer_index.npy` from 01; 06 imports the search space from 03 and needs `lineage_groups.csv` from 02. Keep the scripts in the same folder.
2. **Use one set of KMC parameters.** Training and validation k-mers must come from contigs with the same KMC settings (`-k10 -ci1 -fm`, high count ceiling). Do not mix read-based and contig-based k-mers; their counts differ by orders of magnitude.
3. **Use the same assembler settings.** Validation isolates were re-assembled with SPAdes 3.13.0 `--careful` to match the training isolates. A different assembler or version can change the k-mer profile.
4. **Do not overwrite earlier runs.** Each 03 run writes to `03_models` by default. Use a different `--outdir` for each design (lineage, random, ST, k = 21, full).
5. **Quick vs full mode.** `--quick` uses fixed hyperparameters and is for checking. Numbers in the manuscript should come from the full nested run. Changing only the threshold option does not change regression results or AUROC/AUPRC; it changes the confusion matrix and derived metrics.
6. **Lineage threshold.** `lin_0.2363` was chosen for this dataset. If isolates are added or removed, re-run 02 and check that CC271 is still one cluster and no ST is split.
7. **Isolates with unresolved ST** (`-`) are treated as separate groups in the ST-grouped analysis. X88 (`320*`) is a single-locus variant of ST320 and falls in the CC271 cluster.
8. **PEN breakpoint and origin.** The oral breakpoint (S ≤ 0.06 μg/mL) is used. The domestic isolates were tested down to 0.016 μg/mL and none was susceptible, so all PSSP in the training set come from GPS. PEN S vs NS results are partly confounded by origin; use the GPS-subset results and the domestic I vs R analysis as supportive evidence.
9. **Small validation set.** 48 isolates give wide confidence intervals, and the isolates come from one hospital. The validation should be described as single-centre clinical validation, not as independent multi-centre external validation.
10. **Training data quality.** 74 of the 607 domestic isolates have abnormal assemblies in `607qc.txt` and were kept. Results may improve after a stricter clean-up; this is planned as future work.
11. **Not for clinical reporting.** The models are research tools. Predictions, especially for β-lactams in lineages not seen in training, should be confirmed by phenotypic AST.
12. **Memory and time** (MacBook Pro M3): step 1 a few minutes; step 2 a few minutes, peak 1–2 GB; step 3 quick mode tens of minutes, full nested several hours; step 4 about 15–45 min, peak 4–6 GB.

---

## 8. Citation and contact

Zhao C, Wang S, Yang S, Zhang F, Wang H. Predicting antimicrobial resistance phenotypes in *Streptococcus pneumoniae* using machine learning analysis of genomic data: a practical approach. *iLABMED* (under revision).

Department of Clinical Laboratory, Peking University People's Hospital, Beijing 100044, China.
Corresponding author: Hui Wang (wanghui@pkuph.edu.cn).
Code: https://github.com/chunkiang/spn_prediction

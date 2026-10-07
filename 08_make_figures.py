"""Revised Figures 1-4 for the SPN k-mer manuscript (vector PDF + 600-dpi TIFF + PNG preview)."""
import os, itertools
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D

U = "/home/claude/fig/up/upload_results/"
V = pd.read_csv("/mnt/user-data/outputs/eval_val/val_per_strain.csv")
OUT = "/home/claude/fig/out"; os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 7.5, "axes.titlesize": 8.5,
    "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "pdf.fonttype": 42, "svg.fonttype": "none",
    "axes.spines.top": False, "axes.spines.right": False,
})
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
C_RAND, C_LIN = "#2a78d6", "#eb6834"          # categorical slots 1-2 (validated pair)
BLUE_RAMP = ["#ffffff", "#e3eefb", "#bcd6f4", "#8db8ea", "#5a97dd", "#2a78d6", "#1d5aa6", "#123d72"]
DRUGS = ["PEN", "CRO", "ERY", "CLI", "LVX", "SXT"]
NAME = {"PEN": "Penicillin", "CRO": "Ceftriaxone", "ERY": "Erythromycin", "CLI": "Clindamycin",
        "LVX": "Levofloxacin", "SXT": "Trimethoprim-\nsulfamethoxazole", "AMC": "Amoxicillin-\nclavulanate"}
BP = {"PEN": (0.06, 2), "CRO": (1, 4), "ERY": (0.25, 1), "CLI": (0.25, 1), "LVX": (2, 8), "SXT": (0.5, 4)}
MM = 1 / 25.4


def save(fig, name):
    for ext, kw in [("pdf", {}), ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}), ("png", {"dpi": 200})]:
        fig.savefig(f"{OUT}/{name}.{ext}", bbox_inches="tight", facecolor="white", **kw)
    plt.close(fig)


def mic_lab(l):
    v = 2.0 ** l
    return {0.015625: "0.016", 0.03125: "0.03", 0.0625: "0.06", 0.125: "0.12"}.get(v, f"{v:g}")


# ---------------------------------------------------------------- Figure 1: study flowchart
def figure1():
    fig, ax = plt.subplots(figsize=(174 * MM, 150 * MM))
    ax.set_xlim(0, 174); ax.set_ylim(0, 150); ax.axis("off")

    def box(x, y, w, h, title, body="", fc="#f4f3f0", ec="#8a8984", tc=INK, lw=0.7):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6",
                                    fc=fc, ec=ec, lw=lw))
        if body:
            ax.text(x + w / 2, y + h - 3.2, title, ha="center", va="top", fontsize=7.5, weight="bold", color=tc)
            ax.text(x + w / 2, y + h - 7.6, body, ha="center", va="top", fontsize=6.3, color=INK2, linespacing=1.35)
        else:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=7.5, weight="bold", color=tc)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=0.7, color=INK2, mutation_scale=7, shrinkA=0, shrinkB=0))

    # sources
    box(8, 124, 72, 22, "Domestic isolates (n = 607)",
        "China, 2010–2018\nMIC: broth / agar dilution (CLSI)\nSerotype: Quellung; ST: MLST 2.0")
    box(94, 124, 72, 22, "GPS isolates (n = 320)",
        "UK, Norway, South Africa and 5 other countries\nPublished assemblies; MIC from GPS metadata\n(AMC, MFX, SXT: in silico values)")
    arrow(44, 124, 70, 116); arrow(130, 124, 104, 116)
    box(36, 98, 102, 18, "Assembly and quality control (n = 927)",
        "SPAdes v3.13.0 (--careful); contigs ≥500 bp and coverage ≥0.3× main peak\nAccepted: genome 1.9–2.3 Mb, GC 39%–40%")
    arrow(87, 98, 87, 92)
    box(36, 76, 94, 16, "10-mer counting (KMC 3)",
        "503,875 canonical 10-mers (counts)\nTop 20,000 by variance, selected within each training fold")
    # lineage side box
    box(136, 73, 34, 22, "Lineage clusters",
        "Jaccard distance\nUPGMA, cut at 0.2363\n123 clusters", fc="#ffffff")
    arrow(130, 84, 136, 84)
    arrow(87, 76, 87, 70)
    # nested CV
    ax.add_patch(FancyBboxPatch((8, 30), 158, 40, boxstyle="round,pad=0,rounding_size=1.6",
                                fc="#ffffff", ec="#8a8984", lw=0.7, ls=(0, (3, 2))))
    ax.text(87, 67, "Nested cross-validation (5 outer × 3 inner folds), XGBoost", ha="center", va="top",
            fontsize=7.5, weight="bold", color=INK)
    ax.text(87, 62.6, "Regressor (log₂ MIC) and classifier (S vs NS) for each agent; inner-fold tuning and Youden threshold",
            ha="center", va="top", fontsize=6.6, color=INK2)
    box(14, 34, 46, 22, "Random splitting", "Related isolates may be in\ntraining and test folds\n→ upper bound",
        fc="#eaf1fb", ec=C_RAND, tc=C_RAND, lw=0.9)
    box(64, 34, 46, 22, "Lineage-grouped splitting", "Whole clusters held out;\nno cluster shared\n→ lower bound",
        fc="#fdeee7", ec=C_LIN, tc="#b3461c", lw=0.9)
    box(114, 34, 46, 22, "Sensitivity analyses", "Feature sets: k-mer, serotype + ST,\nor combined (Table 3)\nk = 10 vs k = 21 (Table S4)")
    arrow(153, 73, 153, 70)  # cluster definition feeds CV
    arrow(87, 30, 87, 24)
    box(8, 4, 76, 20, "Final model (n = 927)",
        "k-mer features only; trained once on all isolates\n20 lineage-bootstrap models → 95% prediction interval")
    box(94, 4, 72, 20, "Clinical validation (n = 48)",
        "50 isolates (PKUPH, 2021); 2 excluded\nReference: Etest (bioMérieux)\nNearest-neighbour lineage check (Table S1)")
    arrow(84, 14, 94, 14)
    save(fig, "Figure1_flowchart")


# ---------------------------------------------------------------- Figure 2: random vs lineage
def figure2():
    R = pd.read_csv(U + "03_models_full_random__metrics_summary.csv")
    L = pd.read_csv(U + "03_models_full_lineage__metrics_summary.csv")
    rows = [(d, "All") for d in DRUGS[:-1]] + [("SXT", "Domestic")]
    labels = [NAME[d].replace("\n", "") for d, _ in rows[:-1]] + ["Trimethoprim-sulfamethoxazole\n(domestic, measured MIC)"]
    panels = [("EA", "MIC_regression", "Essential agreement (%)"),
              ("CA", "MIC_regression", "Categorical agreement (%)"),
              ("sensitivity", "S_vs_NS", "Sensitivity for NS isolates (%)")]
    fig, axes = plt.subplots(1, 3, figsize=(174 * MM, 68 * MM), sharey=True)
    y = np.arange(len(rows))[::-1]
    for k, (ax, (m, task, xl)) in enumerate(zip(axes, panels)):
        for yi, (d, s) in zip(y, rows):
            vals = []
            for M, c, mk, off in [(R, C_RAND, "o", 0.13), (L, C_LIN, "s", -0.13)]:
                x = M[(M.drug == d) & (M.featset == "kmer") & (M.subset == s) & (M.task == task)]
                if x.empty or pd.isna(x.iloc[0][m]):
                    vals.append(None); continue
                r = x.iloc[0]
                ax.plot([100 * r[m + "_lo"], 100 * r[m + "_hi"]], [yi + off] * 2, color=c, lw=1.2, solid_capstyle="round")
                ax.plot(100 * r[m], yi + off, mk, ms=4.2, color=c, mec="white", mew=0.6, zorder=3)
                vals.append(100 * r[m])
            if vals[0] is None:
                ax.text(41, yi, "not fitted (<10 NS)", va="center", fontsize=6.5, color=MUTED)
            else:
                ax.plot([vals[0], vals[1]], [yi + 0.13, yi - 0.13], color=GRID, lw=0.8, zorder=1)
        ax.axvline(90, color=MUTED, lw=0.6, ls=(0, (2, 2)), zorder=0)
        ax.set_xlim(40, 101); ax.set_xticks([40, 60, 80, 90, 100])
        ax.set_xlabel(xl); ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
        ax.text(-0.02 if k else -0.62, 1.04, "ABC"[k], transform=ax.transAxes, fontsize=10, weight="bold", va="bottom")
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y); axes[0].set_yticklabels(labels)
    handles = [Line2D([], [], color=C_RAND, marker="o", lw=1.2, ms=4.5, mec="white", label="Random splitting (upper bound)"),
               Line2D([], [], color=C_LIN, marker="s", lw=1.2, ms=4.5, mec="white", label="Lineage-grouped splitting (lower bound)"),
               Line2D([], [], color=MUTED, ls=(0, (2, 2)), lw=0.8, label="90% reference")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.55, 1.08))
    fig.tight_layout(w_pad=1.2)
    save(fig, "Figure2_random_vs_lineage")


# ---------------------------------------------------------------- Figure 3: clinical validation
def figure3():
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(BLUE_RAMP); norm = BoundaryNorm([0, 1, 2, 3, 5, 8, 12, 17, 40], cmap.N)
    fig, axes = plt.subplots(2, 3, figsize=(174 * MM, 118 * MM))
    for k, (ax, d) in enumerate(zip(axes.flat, DRUGS)):
        g = V[V.drug == d]
        lo = int(min(g.ref_log2.min(), g.pred_log2.min())) - 1
        hi = int(max(g.ref_log2.max(), g.pred_log2.max())) + 1
        lv = np.arange(lo, hi + 1); n = len(lv)
        M = np.zeros((n, n), int)
        for r, p in zip(g.ref_log2.astype(int), g.pred_log2.astype(int)):
            M[r - lo, p - lo] += 1
        for i in range(n):          # EA band underlay
            for j in range(n):
                if abs(i - j) <= 1:
                    ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fc="#f4f3f0", ec="none", zorder=0))
        Mm = np.ma.masked_where(M == 0, M)
        ax.imshow(Mm, cmap=cmap, norm=norm, origin="lower", zorder=1, aspect="equal")
        for i in range(n):
            for j in range(n):
                if M[i, j]:
                    ax.text(j, i, M[i, j], ha="center", va="center", fontsize=6.5,
                            color="white" if M[i, j] >= 8 else INK, zorder=3)
        s, r_ = BP[d]
        for v, ls in [(np.log2(s), (0, (3, 2))), (np.log2(r_), "-")]:
            pos = v - lo + 0.5 if v != np.log2(r_) else v - lo - 0.5
            ax.axvline(pos, color=MUTED, lw=0.6, ls=ls, zorder=2); ax.axhline(pos, color=MUTED, lw=0.6, ls=ls, zorder=2)
        step = 1 if n <= 9 else 2
        ax.set_xticks(range(0, n, step)); ax.set_xticklabels([mic_lab(l) for l in lv[::step]], rotation=90)
        ax.set_yticks(range(0, n, step)); ax.set_yticklabels([mic_lab(l) for l in lv[::step]])
        ax.set_xlim(-.5, n - .5); ax.set_ylim(-.5, n - .5)
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        ea = 100 * g.EA.mean(); ca = 100 * g.CA.mean()
        nv = (g.err == "VME").sum(); nm = (g.err == "ME").sum()
        ax.set_title(f"{NAME[d].replace(chr(10), '')}", loc="left", fontsize=8, weight="bold", pad=10)
        ax.text(0, 1.015, f"EA {ea:.1f}%  CA {ca:.1f}%  VME {nv}  ME {nm}", transform=ax.transAxes,
                fontsize=6.6, color=INK2, va="bottom")
        ax.text(-0.22, 1.12, "ABCDEF"[k], transform=ax.transAxes, fontsize=10, weight="bold")
        if k % 3 == 0: ax.set_ylabel("Etest MIC (mg/L)")
        if k >= 3: ax.set_xlabel("Predicted MIC (mg/L)")
    handles = [Rectangle((0, 0), 1, 1, fc="#f4f3f0", ec="none", label="Within ±1 dilution (EA)"),
               Line2D([], [], color=MUTED, ls=(0, (3, 2)), lw=0.8, label="Susceptible breakpoint"),
               Line2D([], [], color=MUTED, lw=0.8, label="Resistant breakpoint")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.035))
    fig.tight_layout(h_pad=1.6, w_pad=1.0)
    save(fig, "Figure3_clinical_validation")


# ---------------------------------------------------------------- Figure 4: important k-mers
def figure4():
    F = pd.read_csv(U + "03_models_full_lineage__feature_importance.csv")
    F = F[(F.featset == "kmer") & (F.task == "reg")].copy()
    F["kmer"] = F.feature.str.replace("k_", "", regex=False)
    F["share"] = F.groupby("drug").mean_gain.transform(lambda s: 100 * s / s.sum())
    order = ["PEN", "AMC", "CRO", "ERY", "CLI", "SXT"]
    sel = []
    for d in order:
        for km in F[F.drug == d].nlargest(6, "mean_gain").kmer:
            if km not in sel: sel.append(km)
    H = F.pivot_table(index="kmer", columns="drug", values="share").reindex(index=sel, columns=order).fillna(0)
    top = {d: set(g.nlargest(50, "mean_gain").kmer) for d, g in F.groupby("drug")}
    O = pd.DataFrame([[len(top[a] & top[b]) for b in order] for a in order], index=order, columns=order)

    fig = plt.figure(figsize=(174 * MM, 120 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], wspace=0.45)
    ax = fig.add_subplot(gs[0])
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("b", BLUE_RAMP)
    vmax = np.percentile(H.values[H.values > 0], 95)
    im = ax.imshow(np.ma.masked_where(H.values == 0, H.values), cmap=cm, vmin=0, vmax=vmax, aspect="auto")
    ax.set_facecolor("#ffffff")
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order)
    ax.set_yticks(range(len(sel))); ax.set_yticklabels(sel, family="Liberation Mono", fontsize=6.2)
    ax.xaxis.tick_top(); ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(order)), minor=True); ax.set_yticks(np.arange(-.5, len(sel)), minor=True)
    ax.grid(which="minor", color=GRID, lw=0.5); ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, extend="max")
    cb.set_label("Share of total gain (%)"); cb.outline.set_linewidth(0.4)
    ax.text(-0.42, 1.06, "A", transform=ax.transAxes, fontsize=10, weight="bold")

    ax2 = fig.add_subplot(gs[1])
    mask = np.triu(np.ones_like(O, bool), 1)
    Ov = O.values.astype(float); Ov[mask] = np.nan
    ax2.imshow(np.ma.masked_invalid(np.where(np.eye(len(order), dtype=bool), np.nan, Ov)), cmap=cm, vmin=0, vmax=25)
    for i in range(len(order)):
        for j in range(len(order)):
            if j < i:
                v = O.iat[i, j]; ax2.text(j, i, v, ha="center", va="center", fontsize=7, color="white" if v >= 15 else INK)
            elif j == i:
                ax2.text(j, i, "50", ha="center", va="center", fontsize=7, color=MUTED)
    ax2.set_xticks(range(len(order))); ax2.set_xticklabels(order)
    ax2.set_yticks(range(len(order))); ax2.set_yticklabels(order)
    ax2.tick_params(length=0)
    for sp in ax2.spines.values(): sp.set_visible(False)
    ax2.set_title("Shared k-mers among the top 50 of each agent", fontsize=8, pad=8)
    ax2.text(-0.2, 1.06, "B", transform=ax2.transAxes, fontsize=10, weight="bold")
    save(fig, "Figure4_kmer_importance")


if __name__ == "__main__":
    figure1(); figure2(); figure3(); figure4()
    print(sorted(os.listdir(OUT)))

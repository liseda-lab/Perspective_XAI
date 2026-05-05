import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------------
# 1) CONFIG – edit only the paths if your filenames live elsewhere
# ------------------------------------------------------------------------
TASK_FILES = {
    "Drug Repurposing": {
        "persona": Path("personas_DR.tsv"),
        "real"   : Path("real_users_DR_Task.tsv"),
    },
    "Drug-Target Interaction": {
        "persona": Path("personas_DTI.tsv"),
        "real"   : Path("real_users_DTI_Task.tsv"),
    },
}
SCORE_COLS = ["Relevance", "Completeness", "Validity"]
KEY_COLS   = ["Explanation", "System"]  # columns that uniquely tie persona ↔ real rows
CMAP       = "RdBu_r"  # colour map for heat-maps
VMIN, VMAX = -1, 1                     # consistent colour range for both plots

plt.rcParams.update({
    "axes.titlesize":   20,   # subplot titles
    "axes.labelsize":   20,   # usually x / y labels
    "xtick.labelsize":  16,   # x tick labels
    "ytick.labelsize":  16,   # y tick labels
})

# ------------------------------------------------------------------------
# 2) Function to compute correlation matrix (persona × score) for one task
# ------------------------------------------------------------------------
def persona_corr_matrix(persona_path: Path, real_path: Path) -> pd.DataFrame:
    df_p = pd.read_csv(persona_path, sep="\t")
    df_u = pd.read_csv(real_path,  sep="\t")

    # build composite key that exists in BOTH tables
    df_p["key"] = df_p[KEY_COLS].astype(str).agg("_".join, axis=1)
    df_u["key"] = df_u[KEY_COLS].astype(str).agg("_".join, axis=1)

    # numeric safety
    df_p[SCORE_COLS] = df_p[SCORE_COLS].apply(pd.to_numeric, errors="coerce")
    df_u[SCORE_COLS] = df_u[SCORE_COLS].apply(pd.to_numeric, errors="coerce")

    # average real-user scores per key
    df_u_grp = df_u.groupby("key")[SCORE_COLS].mean()

    # container
    rows = []
    for persona in df_p["User"].unique():
        df_p_sub = df_p[df_p["User"] == persona].set_index("key")[SCORE_COLS]

        # align keys
        common = df_p_sub.index.intersection(df_u_grp.index)
        if len(common) < 2:
            rows.extend(
                {"Persona": persona, "Score": sc, "Pearson_r": np.nan}
                for sc in SCORE_COLS
            )
            continue

        for sc in SCORE_COLS:
            x, y = df_p_sub.loc[common, sc], df_u_grp.loc[common, sc]
            r = np.nan
            if x.std() != 0 and y.std() != 0:
                r = pearsonr(x, y)[0]
            rows.append({"Persona": persona, "Score": sc, "Pearson_r": r})

    # pivot to heat-map friendly shape
    return (
        pd.DataFrame(rows)
        .pivot(index="Persona", columns="Score", values="Pearson_r")
        .reindex(columns=SCORE_COLS)          # keep column order fixed
        .sort_index()                         # tidy row order
    )

# ------------------------------------------------------------------------
# 3) Compute matrices for all tasks
# ------------------------------------------------------------------------
corr_mats = {
    task: persona_corr_matrix(paths["persona"], paths["real"])
    for task, paths in TASK_FILES.items()
}

# ------------------------------------------------------------------------
# 4) Plot side-by-side heat-maps with uniform sizes
# ------------------------------------------------------------------------
n_tasks = len(corr_mats)
fig, axes = plt.subplots(1, n_tasks, figsize=(6 * n_tasks, 4), sharey=True)

# If there's only one task, axes isn't a list; wrap it
if n_tasks == 1:
    axes = [axes]

# Store the mappable from the first heatmap for the colorbar
mappable = None

for ax, (task, mat) in zip(axes, corr_mats.items()):
    im = sns.heatmap(
        mat,
        ax=ax,
        cmap=CMAP,
        vmin=VMIN, vmax=VMAX,
        annot=True,
        annot_kws={"size": 16},   # font size of the numbers inside the cells
        cbar=False,  # Don't add individual colorbars
    )
    
    # Store the mappable from the first heatmap
    if mappable is None:
        mappable = im.collections[0]
    
    ax.set_title(task, pad=12, weight="bold")
    ax.set_xlabel("")  # cleaner look
    ax.set_ylabel("")


# Add a single colorbar to the right of the last subplot
cbar = fig.colorbar(mappable, ax=axes[-1], shrink=0.9, pad=0.02)
cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

fig.tight_layout()

plt.savefig("persona_correlationsv.png", dpi=300, bbox_inches="tight")
plt.show()
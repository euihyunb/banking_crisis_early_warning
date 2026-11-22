import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent
file_dir = project_dir/"data"/"processed"

# load out-of-sample prediction from 07
oos = pd.read_csv(file_dir/"oos_predictions.csv")
print(oos.shape, oos["target"].sum())

# sample without GFC test years (2006-07 = 2008-09 crises)
no_gfc = oos[~oos["year"].isin([2006, 2007])]
print(no_gfc.shape, no_gfc["target"].sum())

# AUC, Gini, PR AUC for each model
rows = []
for model in ["p_logit", "p_xgb", "p_xgb_all"]:
    auc_all = roc_auc_score(oos["target"], oos[model])
    pr_all = average_precision_score(oos["target"], oos[model])
    auc_ex = roc_auc_score(no_gfc["target"], no_gfc[model])
    pr_ex = average_precision_score(no_gfc["target"], no_gfc[model])
    rows.append({"model": model,
                 "auc_all": auc_all, "gini_all": 2 * auc_all -1, "pr_all": pr_all,
                 "auc_ex": auc_ex, "pr_ex": pr_ex})

metrics = pd.DataFrame(rows)
print(metrics.round(3))

# check 1: average predicted risk by year vs actual positives
by_year = oos.groupby("year").agg(pos=("target", "sum"),
                                  mean_logit=("p_logit", "mean"),
                                  mean_xgb=("p_xgb", "mean"))
print(by_year.round(3))

# check 2: rank countries within each year, then AUC on those ranks
for model in ["p_logit", "p_xgb", "p_xgb_all"]:
    oos[model + "_pct"] = oos.groupby("year")[model].rank(pct=True)
    no_gfc = oos[~oos["year"].isin([2006, 2007])]
    auc_w = roc_auc_score(oos["target"], oos[model + "_pct"])
    auc_w_ex = roc_auc_score(no_gfc["target"], no_gfc[model + "_pct"])
    print(model, round(auc_w, 3), round(auc_w_ex, 3))

## Boostrap Confidence Interval
# block boostrap by year: resample the 24 test years with replacement
rng = np.random.default_rng(42)
years = oos["year"].unique()
boot = []

for b in range(1000):
    pick = rng.choice(years, size=len(years), replace=True)

    # stack the rows of the picked years
    parts = []
    for y in pick:
        parts.append(oos[oos["year"] == y])
    sample = pd.concat(parts)

    # within-year AUC for each model, and the difference
    auc_l = roc_auc_score(sample["target"], sample["p_logit_pct"])
    auc_x = roc_auc_score(sample["target"], sample["p_xgb_pct"])
    boot.append({"auc_logit": auc_l, "auc_xgb": auc_x, "diff": auc_x - auc_l})

boot = pd.DataFrame(boot)
print(boot.quantile([0.025, 0.5, 0.975]).round(3))
print((boot["diff"] > 0).mean())
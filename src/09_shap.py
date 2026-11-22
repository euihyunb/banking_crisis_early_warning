import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from xgboost.sklearn import XGBClassifier

project_dir = Path(__file__).resolve().parent.parent
file_dir = project_dir/"data"/"processed"

# rebuild complete-case training sample (same as 07)
panel = pd.read_csv(file_dir/"panel.csv")
train = panel[panel["sample"]=="train"]
x_cols = ["ngdp_rpch", "log_pcpipch", "bca_ngdpd", "log_ngdpdpc",
          "pvd_chg3", "res_gdp", "world_growth"]
complete = train.dropna(subset=x_cols)
y = complete["target"]

#final xgboost on all labeled years (1980-2023)
spw = (y == 0).sum() / (y==1).sum()
print(spw)
xgb_final = XGBClassifier(n_estimators=300, max_depth=2, learning_rate=0.05,
                          scale_pos_weight=spw, random_state=42)
xgb_final.fit(complete[x_cols], y)

#### shap values for every training row
explainer = shap.TreeExplainer(xgb_final)
sv = explainer(complete[x_cols])
print(sv.values.shape)

# check: base value + sum of shap = model output (log-odds)
margin = xgb_final.predict(complete[x_cols], output_margin=True)
check = sv.base_values + sv.values.sum(axis=1)
print(np.abs(check - margin).max())

# average size of each variable's contribution
mean_abs = pd.Series(np.abs(sv.values).mean(axis=0), index=x_cols)
print(mean_abs.sort_values(ascending=False).round(3))

# summary (beeswarm) plot
shap.plots.beeswarm(sv, show=False)
plt.tight_layout()
plt.savefig(project_dir/"figures"/"shap_explained.png", dpi=150)
plt.close()

# check: world_growth by year vs positives
wg = complete.groupby("year").agg(world_growth=("world_growth", "first"),
                                  pos=("target", "sum"),
                                  n=("target", "size"))
print(wg.sort_values("world_growth", ascending=False).head(10))
print(wg.sort_values("world_growth").head(5))

# dependence plot: pvd_chg3 (color = strongest interacting variable)
shap.plots.scatter(sv[:, "pvd_chg3"], color=sv, show=False)
plt.tight_layout()
plt.savefig(project_dir/"figures"/"shap_dependence_pvd_chg3.png", dpi=150)
plt.close()

# average shap of pvd_chg3 by credit-growth bins
dep = pd.DataFrame({"pvd_chg3": sv[:, "pvd_chg3"].data,
                    "shap": sv[:, "pvd_chg3"].values})
dep["bin"] = pd.cut(dep["pvd_chg3"], bins=[-300, -10, 0, 5, 10, 15, 20, 30, 300])
print(dep.groupby("bin", observed=True)["shap"].agg(["mean", "count"]).round(3))

#### 2007 model: same as 07 step 3 (train 1980-2005, test 2007)
tr07 = complete[complete["year"] < 2006]
te07 = complete[complete["year"] == 2007]
y07 = tr07["target"]
spw07 = (y07 == 0).sum() / (y07 == 1).sum()
xgb07 = XGBClassifier(n_estimators=300, max_depth=2, learning_rate=0.05,
                      scale_pos_weight=spw07, random_state=42)
xgb07.fit(tr07[x_cols], y07)

# shap for 2007 rows, one row per country
sv07 = shap.TreeExplainer(xgb07)(te07[x_cols])
tab07 = pd.DataFrame(sv07.values, columns=x_cols, index=te07["iso3"])
tab07["base"] = sv07.base_values
tab07["margin"] = tab07["base"] + tab07[x_cols].sum(axis=1)
tab07["p"] = 1 / (1 + np.exp(-tab07["margin"]))

# selected countries
cases = ["CAN", "FRA", "ITA", "ISL", "IRL", "DEU", "NLD", "RUS"]
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(tab07.loc[cases].round(2))

# save 2007 case table for README
tab07.loc[cases].round(2).to_csv(project_dir/"data"/"processed"/"shap_2007_cases.csv")

# rank of each country in 2007 (for README table)
tab07["rank"] = tab07["p"].rank(ascending=False)
print(tab07.loc[cases, ["p", "rank"]])
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

project_dir = Path(__file__).resolve().parent.parent
file_dir = project_dir / "data"/"processed"

panel = pd.read_csv(file_dir / "panel.csv")
print(panel.shape)
print(panel.head())

# load main panel data
train = panel[panel["sample"] == "train"]
print(train.shape)
print(train["target"].sum())

# main model variables
x_cols = ["ngdp_rpch", "log_pcpipch", "bca_ngdpd", "log_ngdpdpc", "pvd_chg3", "res_gdp", "world_growth"]

# check missing values
print(train[x_cols].isna().sum())
print(train[x_cols].describe().T)

# complete case: loses 125 of 303 positives, mostly 1980-1090s due to missing values in pvd_chg3
# same complete sample for logit and xgboost so the comparison is fair, see README for more details.
complete = train.dropna(subset=x_cols)
print(complete.shape)
print(complete["target"].sum())
print(complete["year"].min(), complete["year"].max())
print(complete["iso3"].nunique())

# sample run - sample year == 2007
#### Logistic regression
test_year = 2007

train_07 = complete[complete["year"]< test_year -1]
test_07 = complete[complete["year"] == test_year]

print(train_07["year"].min(), train_07["year"].max())
print(train_07.shape, train_07["target"].sum())
print(test_07.shape, test_07["target"].sum())
print(test_07.loc[test_07["target"] == 1, "iso3"].tolist())

# winsorize at 1/99 pct with training cutoffs (for logit only)
lower = train_07[x_cols].quantile(0.01)
upper = train_07[x_cols].quantile(0.99)
X_train_logit = train_07[x_cols].clip(lower=lower, upper=upper, axis=1)
X_test_logit = test_07[x_cols].clip(lower=lower, upper=upper, axis=1)

# standardize with training mean and sd
mean = X_train_logit.mean()
sd = X_train_logit.std()
X_train_logit = (X_train_logit - mean) / sd
X_test_logit = (X_test_logit - mean) / sd

print(lower)
print(upper)
print(X_train_logit.describe().T[["mean", "std", "min", "max"]])
print(X_test_logit.describe().T[["mean", "std", "min", "max"]])

# outcome
y_train = train_07["target"]
y_test = test_07["target"]

# Logit, no penalty
logit = LogisticRegression(C=np.inf, max_iter=1000, )
logit.fit(X_train_logit, y_train)

# coefficients: effect of a 1 sd changes on log-odds
coef = pd.Series(logit.coef_[0], index=x_cols)
print(coef.round(3))
print(logit.intercept_)
print(logit.predict_proba(X_train_logit)[:,1].mean())

# predict crisis probability for 2007 rows
result_07 = test_07[["iso3", "year", "target"]]
result_07["p_logit"] = logit.predict_proba(X_test_logit)[:,1]
print(result_07.sort_values("p_logit", ascending=False).head(15))

# where do the actual crisis countries rank?
result_07["rank_logit"] = result_07["p_logit"].rank(ascending=False)
print(result_07[result_07["target"] == 1].sort_values("rank_logit"))

#### XGBOOST
# xgboost uses raw values (no winsorizing or standardizing)
X_train_xgb = train_07[x_cols]
X_test_xgb = test_07[x_cols]

# weight on positives: negatives / positives
spw = (y_train == 0).sum()  / (y_train == 1).sum()
print(spw)

#shallow trees, small learning rate
xgb = XGBClassifier(n_estimators=300, max_depth=2, learning_rate=0.05,
                    scale_pos_weight=spw, random_state=42)
xgb.fit(X_train_xgb, y_train)

#predict 2007 and compare ranks with logit
result_07["p_xgb"] = xgb.predict_proba(X_test_xgb)[:,1]
result_07["rank_xgb"] = result_07["p_xgb"].rank(ascending=False)
print(result_07[result_07["target"] == 1].sort_values("rank_xgb")[["iso3", "rank_logit", "rank_xgb"]])
print(result_07.sort_values("p_xgb", ascending=False).head(15))

## 2000-2023 repeat

#expanding-window basktest: test years 2000-2023
test_years = range(2000,2024)
results = []
diag = []

for test_year in test_years:

    #split: train up to t-2 test =t
    tr = complete[complete["year"] < test_year -1]
    te = complete[complete["year"] == test_year]
    y_tr = tr["target"]

    # logit: winsorize and standardize with training stats
    lower = tr[x_cols].quantile(0.01)
    upper = tr[x_cols].quantile(0.99)
    X_tr = tr[x_cols].clip(lower=lower, upper=upper, axis=1)
    X_te = te[x_cols].clip(lower=lower, upper=upper, axis=1)
    mean = X_tr.mean()
    sd = X_tr.std()
    X_tr = (X_tr - mean) / sd
    X_te = (X_te - mean) / sd

    logit = LogisticRegression(C=np.inf, max_iter=1000)
    logit.fit(X_tr, y_tr)
    p_logit = logit.predict_proba(X_te)[:,1]

    # xgboost: raw value
    spw = (y_tr==0).sum() / (y_tr==1).sum()
    xgb = XGBClassifier(n_estimators=300, max_depth=2, learning_rate=0.05,
                        scale_pos_weight=spw, random_state=42)
    xgb.fit(tr[x_cols], y_tr)
    p_xgb = xgb.predict_proba(te[x_cols])[:,1]

    # robustness check: xgboost trained on all rows (NaN kept), same test rows
    tr_all = train[train["year"] < test_year -1]
    y_tr_all = tr_all["target"]
    spw_all = (y_tr_all ==0).sum() / (y_tr_all==1).sum()
    xgb_all = XGBClassifier(n_estimators=300, max_depth=2, learning_rate=0.05,
                            scale_pos_weight=spw_all, random_state=42)
    xgb_all.fit(tr_all[x_cols], y_tr_all)
    p_xgb_all = xgb_all.predict_proba(te[x_cols])[:,1]

    #save predictions for this year
    res = pd.DataFrame({"iso3": te["iso3"], "year": te["year"],
                        "target": te["target"],
                        "p_logit": p_logit, "p_xgb": p_xgb, "p_xgb_all": p_xgb_all})
    results.append(res)

    #save sample sizes for this year
    diag.append({"test_year": test_year, "n_train": len(tr),
                 "pos_train": y_tr.sum(), "n_test": len(te),
                 "pos_test": te["target"].sum(), "n_trail_all": len(tr_all), "pos_train_all": y_tr_all.sum()})

    print(test_year,"done")

# stack all years
oos = pd.concat(results)
diag = pd.DataFrame(diag)
print(oos.shape)
print(oos["target"].sum())
print(diag)

# consistency check: loop result for FRA 2007 should match step 3
print(oos[(oos["iso3"] == "FRA") & (oos["year"] == 2007)])

# save out-of-sample predictions and sample sizes
oos.to_csv(file_dir/"oos_predictions.csv", index=False)
diag.to_csv(file_dir/"backtest_diag.csv", index=False)

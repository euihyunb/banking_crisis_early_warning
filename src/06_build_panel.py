import pandas as pd
import numpy as np
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent
processed = project_dir / "data" / "processed"

## load all four cleaned datasets
crises = pd.read_csv(processed / "crises_clean.csv")
weo = pd.read_csv(processed / "weo_clean.csv")
debt = pd.read_csv(processed / "debt_clean.csv")
res = pd.read_csv(processed / "reserves_clean.csv")

for name, d in [("crises", crises), ("weo", weo), ("debt", debt), ("res", res)]:
    print(name, d.shape)

print(weo.columns.tolist())

for name, d in [("weo", weo), ("debt", debt), ("res", res)]:
    print(name, d["year"].min(), d["year"].max(), d["iso3"].nunique())

## transform variables as needed
weo["log_ngdpdpc"] = np.log(weo["ngdpdpc"])
weo["log_pcpipch"] = np.log1p(weo["pcpipch"]/100)

debt = debt.sort_values(["iso3", "year"])
debt["pvd_chg3"] = debt.groupby("iso3")["f3t4_ps_pogdp_pt"].diff(3)

print(weo[["ngdpdpc", "log_ngdpdpc", "pcpipch", "log_pcpipch"]].describe().round(2))
print(debt["pvd_chg3"].describe().round(2))
print(debt.sort_values("pvd_chg3", ascending=False).head(10))
print(debt.sort_values("pvd_chg3").head(10))


## merge datasets
panel = weo.merge(debt, on=["iso3", "year"], how= "left")
print(panel.shape)

panel = panel.merge(res, on=["iso3", "year"], how="left")
print(panel.shape)

#reserve in percent of ngdpd
panel["res_gdp"] = panel["rxf11_revs_usd"] / (panel["ngdpd"] * 1000) * 100

##create the crisis indicator
panel["crisis_start"] = 0
panel["in_crisis"] = 0
panel["borderline_window"] = 0

for i, row in crises.iterrows():
    iso = row["iso3"]
    start = row["start"]
    end = row["end"]
    if pd.isna(end):
        end = 2025
    same_country = panel["iso3"] == iso

    if row["borderline"] == 1:
        panel.loc[same_country & panel["year"].between(start - 2, end), "borderline_window"] = 1
    else:
        panel.loc[same_country & (panel["year"] == start), "crisis_start"] = 1
        panel.loc[same_country & panel["year"].between(start, end), "in_crisis"] = 1

g = panel.groupby("iso3")["crisis_start"]
panel["target"] = ((g.shift(-1) == 1) | (g.shift(-2) == 1)).astype(int)

print(panel["crisis_start"].sum())
print(panel["in_crisis"].sum())
print(panel["borderline_window"].sum())
print(panel["target"].sum())

## create train and predict variables
excluded = (panel["in_crisis"] == 1) | (panel["borderline_window"] == 1)

panel["sample"] = "none"
panel.loc[panel["year"].between(1980, 2023) & ~excluded, "sample"] = "train"
panel.loc[panel["year"].between(2024, 2025) & ~excluded, "sample"] = "predict"

print(panel["sample"].value_counts())
print(panel.groupby("sample")["target"].sum())
print(panel[panel["sample"] == "train"]["target"].mean().round(3))

## save the file
panel = panel.sort_values(["iso3", "year"]).reset_index(drop=True)

assert len(panel) == 197 * 52
assert panel.duplicated(["iso3", "year"]).sum() == 0
assert panel["crisis_start"].sum() == 158

panel.to_csv(processed / "panel.csv", index=False)
print(panel.shape)
print(panel.columns.tolist())
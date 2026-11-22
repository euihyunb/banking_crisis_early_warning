import pandas as pd
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent
file_dir = project_dir/"data"/"raw"/"dataset_2026-11-17T18_54_26.722416321Z_DEFAULT_INTEGRATION_IMF.FAD_GDD_2.0.0.csv"

df = pd.read_csv(file_dir)

print(df.shape)
print(df.head())

## Separating iso3, freq, code
df[["iso3", "code", "freq"]] = df["SERIES_CODE"].str.split(".", expand=True)

print(df["freq"].unique())
print(df["iso3"].nunique())

## Keep only 1980 - 2024
years = [str(y) for y in range(1980, 2025)]

for c in ["F3T4_PS_POGDP_PT", "FL_PS_POGDP_PT"]:
    sub = df[df["code"] == c]
    print(c)
    print(sub[years].notna().sum(axis=1).describe().round(1))
    print(sub[years].notna().mean().groupby(lambda y: int(y) // 10* 10).mean().round(2))

## keep only F3T4_PS_POGDP_PT
debt = df[df["code"] == "F3T4_PS_POGDP_PT"]

print(debt[debt["iso3"].str.len() !=3]["iso3"].unique())
debt = debt[debt["iso3"].str.len() == 3]

year_cols = [c for c in debt.columns if c.isdigit()]
debt = debt[["iso3"] + year_cols]

## reshape the data from wide to long
debt = debt.melt(id_vars="iso3", var_name="year", value_name="f3t4_ps_pogdp_pt")
debt["year"] = debt["year"].astype(int)
debt = debt.sort_values(["iso3", "year"]).reset_index(drop=True)

print(debt.shape)

#save the file
debt.to_csv(project_dir / "data" / "processed" / "debt_clean.csv", index=False)


# REER (IMF EER, CPI-based, annual)
# Not used in the main model: 54 of the 120 crisis countries have no REER series
# (incl. KOR, THA, IDN, ARG, TUR), so missingness would act as a country-group signal.
# Output is kept for a robustness check on the 94-country REER sample.

import pandas as pd
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent
file_dir = project_dir / "data"/"raw"/"dataset_2026-11-18T02_57_55.966259565Z_DEFAULT_INTEGRATION_IMF.STA_EER_6.0.0.csv"

df = pd.read_csv(file_dir)

print(df.shape)
print(df.head())

df[["iso3", "code", "freq"]] = df["SERIES_CODE"].str.split(".", expand=True)

print(df["freq"].value_counts())
print(df["iso3"].nunique())  ## only 100 countries

pd.set_option("display.max_columns", None)
print(df[["code", "INDICATOR"]].drop_duplicates())

print(df.columns.tolist()[:40])
print(df.columns.tolist()[-10:])

## checking coverage of REER
reer = df[(df["code"]=="REER_IX_RY2010_ACW_RCPI") & (df["freq"]=="A")]

year_cols = [c for c in reer.columns if c.isdigit()]
print(reer.shape)
print(year_cols[0], year_cols[-1])

print(reer[year_cols].notna().mean().groupby(lambda y: int(y) //10*10).mean().round(2))

crises = pd.read_csv(project_dir/"data"/"processed"/"crises_clean.csv")
missing = set(crises["iso3"]) - set(reer["iso3"])
print(len(missing))
print(sorted(missing))

## saving only for robustness check
reer = reer[["iso3"] + year_cols]

reer = reer.melt(id_vars="iso3", var_name="year", value_name="reer_ix_ry2010_acw_rcpi")
reer["year"] = reer["year"].astype(int)
reer = reer.sort_values(["iso3", "year"]).reset_index(drop=True)

print(reer.shape)

reer.to_csv(project_dir / "data" / "processed" / "reer_clean.csv", index=False)
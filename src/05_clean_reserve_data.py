import pandas as pd
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent
file_dir = project_dir / "data"/"raw"/"dataset_2026-11-18T03_00_16.086332408Z_DEFAULT_INTEGRATION_IMF.STA_IL_13.0.1.csv"

df = pd.read_csv(file_dir)

print(df.shape)
print(df.head())

df[["iso3", "code", "unit", "freq"]] = df["SERIES_CODE"].str.split(".", expand=True)

print(df["freq"].value_counts())
print(df["unit"].value_counts())

pd.set_option("display.max_colwidth", None)
codes = df[df["freq"] == "A"][["code", "unit", "INDICATOR"]].drop_duplicates()
print(len(codes))
print(codes.sort_values(["code", "unit"]))

## checking RXF11_REVS data and coverage

res = df[(df["code"]=="RXF11_REVS") & (df["freq"]=="A") & (df["unit"]=="USD")]

year_cols = [c for c in res.columns if c.isdigit()]
print(res.shape)
print(year_cols[0], year_cols[-1])
print(res["SCALE"].unique())

print(res[year_cols].notna().mean().groupby(lambda y: int(y) // 10*10).mean().round(2))

crises = pd.read_csv(project_dir / "data" / "processed" / "crises_clean.csv")
missing = set(crises["iso3"])- set(res["iso3"])
print(len(missing))
print(sorted(missing))

## save the file
res = res[["iso3"] + year_cols]

res = res.melt(id_vars="iso3", var_name="year", value_name="rxf11_revs_usd")
res["year"] = res["year"].astype(int)
res = res.sort_values(["iso3", "year"]).reset_index(drop=True)

print(res.shape)

assert res.duplicated(["iso3", "year"]).sum() == 0

res.to_csv(project_dir / "data" / "processed" / "reserves_clean.csv", index=False)
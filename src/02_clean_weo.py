import pandas as pd
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent
file_dir = project_dir/"data"/"raw"/"dataset_2026-11-17T01_22_43.123391318Z_DEFAULT_INTEGRATION_IMF.RES_WEO_9.0.0.csv"

df = pd.read_csv(file_dir)

print(df.shape)
print(df.columns.tolist())
print(df.head())

## Separating iso3, freq, code
df[["iso3", "code", "freq"]] = df["SERIES_CODE"].str.split(".", expand=True)

indicators = ["NGDP_RPCH", "PCPIPCH", "BCA_NGDPD", "GGXWDG_NGDP", "NGDPD", "NGDPDPC"]

df = df[df["code"].isin(indicators)].copy()

world = df[(df["iso3"]== "G001")&(df["code"] == "NGDP_RPCH")].copy()

df = df[df["iso3"].str.len() == 3].copy()

print(df.shape)
print(world.shape)

## Keep only 1980 - 2031
years = [str(year) for year in range(1980, 2032)]

df = df[["iso3", "code"] + years]
world = world[["iso3", "code"] + years]

print(df.shape)
print(world.shape)

## reshape wide to long
long = df.melt(id_vars=["iso3", "code"], var_name="year", value_name="value")
long["year"] = long["year"].astype(int)

print(long.shape)
print(long.head())

weo = long.pivot(index=["iso3", "year"], columns="code", values="value").reset_index()
weo.columns.name = None

print(weo.shape)
print(weo.head())

## Reshape World Data & Merge with main data
world_long = world.melt(id_vars=["iso3", "code"], var_name="year", value_name="world_growth")
world_long["year"] = world_long["year"].astype(int)
world_long = world_long[["year", "world_growth"]]

print(world_long.head())

weo = weo.merge(world_long, on="year", how="left")
print(weo.shape)
print(weo.head())

variables = ["NGDP_RPCH", "PCPIPCH", "BCA_NGDPD", "GGXWDG_NGDP", "NGDPD", "NGDPDPC", "world_growth"]

## File save
weo.columns = weo.columns.str.lower()

weo = weo[["iso3", "year", "ngdp_rpch", "pcpipch", "bca_ngdpd", "ggxwdg_ngdp", "ngdpd", "ngdpdpc", "world_growth"]]
weo = weo.sort_values(["iso3", "year"]).reset_index(drop=True)

print(weo.columns.tolist())
print(weo.shape)

weo.to_csv(project_dir / "data" / "processed" / "weo_clean.csv", index=False)
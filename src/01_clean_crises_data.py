import pandas as pd
from pathlib import Path

print(Path.cwd())

project_dir = Path(__file__).resolve().parent.parent
print(project_dir)

file_path = project_dir/"data"/"raw"/"SYSTEMIC_BANKING_CRISES_DATABASE_2026.xlsx"
print(file_path.exists())

df = pd.read_excel(file_path, sheet_name= "Crisis Resolution and Outcomes")

print(df.shape)
print(df.columns.tolist())
print(df.head(10))
print(df.tail(15))

## Clean up data
# keep necessary rows
print(df["Start"].isna().sum())
df = df[df["Start"].isna() == False]

print(df.shape)
print(df.tail(3))

# keep necessary variables
df = df[["Country", "Start", "End"]].copy()

df = df.rename(columns={"Country": "country_raw",
                        "Start": "start",
                        "End": "end_raw"})

print(df.head())
print(df.dtypes)

# Clean note numbers from country names
df["country"] = df["country_raw"].str.replace(r"\s*\d/\s*$","", regex=True)
df["country"] = df["country"].str.strip()

changed = df[df["country"] != df["country_raw"]]
print(len(changed))
print(changed[["country_raw", "country"]])

print(df["country_raw"].nunique())
print(df["country"].nunique())

# Clean start and end year
df["start"] = df["start"].astype(int)

df["end"] = df["end_raw"].astype(str).str.extract(r"(\d{4})")[0]
df["end"] = pd.to_numeric(df["end"].astype("Int64"))

df["truncated"] = df["end_raw"].astype(str).str.contains("5/").astype(int)

print(df[["country", "start", "end_raw", "end", "truncated"]].head())
print(df.dtypes)
print(df[df["end"].isna()])
print((df["end"] < df["start"]).sum())
print((df["end"] - df["start"]).max())

## Mapping ISO3 code onto country names
iso3_map = {
    "Albania": "ALB", "Algeria": "DZA", "Angola": "AGO", "Argentina": "ARG",
    "Armenia": "ARM", "Austria": "AUT", "Azerbaijan": "AZE", "Bangladesh": "BGD",
    "Belarus": "BLR", "Belgium": "BEL", "Benin": "BEN", "Bolivia": "BOL",
    "Bosnia and Herzegovina": "BIH", "Brazil": "BRA", "Bulgaria": "BGR",
    "Burkina Faso": "BFA", "Burundi": "BDI", "Cameroon": "CMR", "Cape Verde": "CPV",
    "Central African Republic": "CAF", "Chad": "TCD", "Chile": "CHL", "China": "CHN",
    "Colombia": "COL", "Costa Rica": "CRI", "Cote d'Ivoire": "CIV", "Croatia": "HRV",
    "Cyprus": "CYP", "Czech Republic": "CZE", "Democratic Republic of Congo": "COD",
    "Denmark": "DNK", "Djibouti": "DJI", "Dominican Republic": "DOM", "Ecuador": "ECU",
    "Egypt": "EGY", "El Salvador": "SLV", "Equatorial Guinea": "GNQ", "Eritrea": "ERI",
    "Estonia": "EST", "Eswatini": "SWZ", "Finland": "FIN", "France": "FRA",
    "Georgia": "GEO", "Germany": "DEU", "Ghana": "GHA", "Greece": "GRC",
    "Guinea": "GIN", "Guinea-Bissau": "GNB", "Guyana": "GUY", "Haiti": "HTI",
    "Hungary": "HUN", "Iceland": "ISL", "India": "IND", "Indonesia": "IDN",
    "Ireland": "IRL", "Israel": "ISR", "Italy": "ITA", "Jamaica": "JAM",
    "Japan": "JPN", "Jordan": "JOR", "Kazakhstan": "KAZ", "Kenya": "KEN",
    "Korea": "KOR", "Kuwait": "KWT", "Kyrgyz Republic": "KGZ", "Latvia": "LVA",
    "Lebanon": "LBN", "Liberia": "LBR", "Lithuania": "LTU", "Luxembourg": "LUX",
    "Madagascar": "MDG", "Malaysia": "MYS", "Mali": "MLI", "Mauritania": "MRT",
    "Mexico": "MEX", "Moldova": "MDA", "Mongolia": "MNG", "Morocco": "MAR",
    "Mozambique": "MOZ", "Nepal": "NPL", "Netherlands": "NLD", "Nicaragua": "NIC",
    "Niger": "NER", "Nigeria": "NGA", "North Macedonia": "MKD", "Norway": "NOR",
    "Panama": "PAN", "Paraguay": "PRY", "Peru": "PER", "Philippines": "PHL",
    "Poland": "POL", "Portugal": "PRT", "Republic of Congo": "COG", "Romania": "ROU",
    "Russia": "RUS", "Senegal": "SEN", "Sierra Leone": "SLE", "Slovak Republic": "SVK",
    "Slovenia": "SVN", "Spain": "ESP", "Sri Lanka": "LKA", "Sweden": "SWE",
    "Switzerland": "CHE", "São Tomé and Príncipe": "STP", "Tajikistan": "TJK",
    "Tanzania": "TZA", "Thailand": "THA", "Togo": "TGO", "Tunisia": "TUN",
    "Türkiye": "TUR", "Uganda": "UGA", "Ukraine": "UKR", "United Kingdom": "GBR",
    "United States": "USA", "Uruguay": "URY", "Venezuela": "VEN", "Vietnam": "VNM",
    "Yemen": "YEM", "Zambia": "ZMB", "Zimbabwe": "ZWE",
}

df["iso3"] = df["country"].map(iso3_map)

print(len(iso3_map))
print(df["iso3"].isna().sum())
print(df[df["iso3"].isna()]["country"].unique())

# Marking borderline cases
borderline_list = [("NIC", 2018), ("VNM", 2022), ("LKA", 2023)]

df["borderline"] = 0

for iso, year in borderline_list:
    df.loc[(df["iso3"] == iso) & (df["start"] == year), "borderline"] = 1

print(df["borderline"].sum())
print(df[df["borderline"] == 1][["country", "start", "end"]])

## Save processed file
df = df[["country","iso3", "start", "end", "truncated", "borderline"]]

df = df.sort_values(["iso3", "start"]).reset_index(drop=True)

print(df.shape)
print(df["iso3"].nunique())
print(df.duplicated(["iso3","start"]).sum())
print(df.head())

out_dir = project_dir/"data"/"processed"
out_dir.mkdir(parents=True, exist_ok=True)

df.to_csv(out_dir/"crises_clean.csv", index=False)
"""
Step 1 - Preprocessing (pandas)
Input : World Bank GEM 'Official exchange rate (LCU per USD, period average)' xlsx, sheet 'monthly'
Output: data/exchange_rate_long.csv   (tidy: country, iso3, date, rate, ...)
        data/preprocess_report.csv    (what was kept/dropped and why)
"""
import numpy as np
import pandas as pd
import pycountry
from pathlib import Path

XLSX = "Official_exchange_rate__LCU_per_USD__period_average.xlsx"   # <- path to your file
START = "2015-01-01"
MAX_MISSING = 12          # drop a country if more than 12 months are missing since START
OUT = Path("data"); OUT.mkdir(exist_ok=True)

# World Bank names that pycountry cannot resolve on its own
NAME_FIX = {
    "Bahamas, The": "BHS", "Congo, Dem. Rep.": "COD", "Congo, Rep.": "COG", "Cote d'Ivoire": "CIV",
    "Egypt, Arab Rep.": "EGY", "Gambia, The": "GMB", "Hong Kong SAR, China": "HKG",
    "Iran, Islamic Rep.": "IRN", "Korea, Rep.": "KOR", "Kyrgyz Republic": "KGZ", "Lao, PDR": "LAO",
    "Macao SAR, China": "MAC", "Micronesia, Fed. Sts.": "FSM", "Moldova, Rep.": "MDA",
    "Netherlands Antilles": "ANT", "Russian Federation": "RUS", "Syrian Arab Republic": "SYR",
    "Taiwan, China": "TWN", "Tanzania, United Rep.": "TZA", "Venezuela, RB": "VEN",
    "Viet Nam": "VNM", "Yemen, Rep.": "YEM", "Virgin Islands, U.S.": "VIR", "West Bank and Gaza": "PSE",
    "St. Kitts and Nevis": "KNA", "St. Lucia": "LCA", "St. Vincent and the Grenadines": "VCT",
    "Cabo Verde": "CPV", "Czech Republic": "CZE", "Kosovo": "XKX", "Turkey": "TUR",
    "Eswatini": "SWZ", "Brunei Darussalam": "BRN", "Slovakia": "SVK",
}

def to_iso3(name):
    if name in NAME_FIX:
        return NAME_FIX[name]
    try:
        return pycountry.countries.lookup(name).alpha_3
    except LookupError:
        return None

# ---------- 1. read + reshape (wide -> long) ----------
raw = pd.read_excel(XLSX, sheet_name="monthly", header=0)
raw = raw.rename(columns={raw.columns[0]: "period"})
raw = raw[raw["period"].astype(str).str.match(r"^\d{4}M\d{2}$")].copy()   # drops the blank 2nd row
raw["date"] = pd.to_datetime(raw["period"].str.replace("M", "-") + "-01")  # '2015M03' -> 2015-03-01
raw = raw.drop(columns="period").set_index("date")
raw = raw.apply(pd.to_numeric, errors="coerce")
raw = raw.loc[START:]

# ---------- 2. missing-data rule ----------
report = []
clean = {}
for country in raw.columns:
    s = raw[country]
    n_miss = int(s.isna().sum())
    if s.isna().all() or n_miss > MAX_MISSING:
        report.append((country, n_miss, "dropped: too many missing months"))
        continue
    s = s.interpolate(method="linear", limit_area="inside")   # small interior gaps
    s = s.ffill().bfill()                                      # at most a few edge months
    clean[country] = s
    report.append((country, n_miss, "kept" if n_miss == 0 else f"kept: {n_miss} month(s) filled"))
wide = pd.DataFrame(clean)

# ---------- 3. ISO3 codes ----------
iso = {c: to_iso3(c) for c in wide.columns}
unmapped = [c for c, i in iso.items() if i is None]
if unmapped:
    print("WARNING - no ISO3 for:", unmapped, "-> add them to NAME_FIX")
wide = wide[[c for c in wide.columns if iso[c]]]

# ---------- 4. comparable indicators ----------
# raw rates (80 vs 15,000 LCU/USD) can't be compared across countries, so derive:
#   pct_change_mom : month-on-month % change
#   yoy_pct        : 12-month % change   (good choropleth variable; + = currency weakened vs USD)
#   index_100      : rate / rate in START month * 100
long = (wide.rename_axis("date").reset_index()
            .melt(id_vars="date", var_name="country", value_name="rate")
            .sort_values(["country", "date"]))
g = long.groupby("country")["rate"]
long["pct_change_mom"] = g.pct_change() * 100
long["yoy_pct"] = g.pct_change(12) * 100
long["index_100"] = long["rate"] / g.transform("first") * 100
long["iso3"] = long["country"].map(iso)
long = long[["country", "iso3", "date", "rate", "pct_change_mom", "yoy_pct", "index_100"]]

long.to_csv(OUT / "exchange_rate_long.csv", index=False)
pd.DataFrame(report, columns=["country", "missing_months", "decision"]).to_csv(OUT / "preprocess_report.csv", index=False)

print(f"Countries kept: {long['country'].nunique()}  |  months: {long['date'].nunique()}  |  rows: {len(long)}")
print(f"Range: {long['date'].min():%Y-%m} to {long['date'].max():%Y-%m}")
print("Dropped:", [r[0] for r in report if r[2].startswith("dropped")])

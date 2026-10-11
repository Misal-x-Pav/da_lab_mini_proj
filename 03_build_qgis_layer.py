"""
Step 3 - Build the QGIS layer (GeoPandas)
Input : data/forecasts.csv  +  Natural Earth countries (GeoJSON or Shapefile)
Output: output/exchange_rates_temporal.gpkg   layer 'exchange_rates'
        -> ONE row per country per month, with a real DateTime field for the Temporal Controller.

Download boundaries (Natural Earth 'Admin 0 - Countries', 1:110m or 1:50m) and set BOUNDARIES below, e.g.
https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson
"""
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import geopandas as gpd

BOUNDARIES = "ne_50m_admin_0_countries.geojson"
Path("output").mkdir(exist_ok=True)

# ---------- 1. attribute table ----------
fc = pd.read_csv("data/forecasts.csv", parse_dates=["date"]).sort_values(["country", "date"])
g = fc.groupby("country")["rate"]
fc["index_100"] = fc["rate"] / g.transform("first") * 100      # base = Jan 2015
fc["yoy_pct"] = g.pct_change(12) * 100                          # + = currency weakened vs USD
fc["pct_change_mom"] = g.pct_change() * 100
fc["status"] = np.where(fc["is_forecast"], "Forecast", "Historical")
fc["is_forecast"] = fc["is_forecast"].astype(int)
fc = fc.rename(columns={"rate": "lcu_per_usd", "lower": "lcu_lower95", "upper": "lcu_upper95"})
# winsorise for MAPPING ONLY so hyper-inflating currencies (e.g. Argentina) don't flatten the colour scale
fc["yoy_pct_clip"] = fc["yoy_pct"].clip(-50, 100)

# ---------- 2. geometries ----------
world = gpd.read_file(BOUNDARIES)
world["iso3"] = np.where(world["ISO_A3_EH"].isin(["-99", None]), world["ADM0_A3"], world["ISO_A3_EH"])
world["iso3"] = world["iso3"].replace({"KOS": "XKX"})      # Natural Earth codes Kosovo as KOS
# every country polygon is repeated once per month, so simplify (~5 km) to keep the file small and QGIS fast
world["geometry"] = world.geometry.simplify(0.05, preserve_topology=True)
# Some sovereign states have extra shapes with the SAME code (Australia = mainland + Indian Ocean Territories +
# Ashmore and Cartier Is.). Left alone, the join would repeat those countries (and their labels) several times.
# Keep only the largest shape per code (area is used only for ranking, so degrees are fine here).
n_before = len(world)
world = (world.assign(_area=world.geometry.area).sort_values("_area", ascending=False)
              .drop_duplicates("iso3").drop(columns="_area"))
if n_before != len(world):
    print(f"Ignored {n_before - len(world)} extra polygon(s) that share a country code (e.g. AUS dependencies)")
world = world[["iso3", "NAME", "geometry"]].rename(columns={"NAME": "map_name"})

# Eurozone members share the euro: GEM usually lists them individually, so most will match directly.
# Territories that have no polygon in Natural Earth 110m (small islands) are simply not drawn.
unmatched = sorted(set(fc["iso3"]) - set(world["iso3"]))
print(f"{fc['iso3'].nunique() - len(unmatched)} / {fc['iso3'].nunique()} countries matched to a polygon")
print("No polygon for:", unmatched)

# ---------- 2b. label text for the map ----------
def fmt_rate(x):
    """94.71 -> '94.71', 152.4 -> '152.4', 16234.5 -> '16,234' (keeps labels short and readable)"""
    return f"{x:,.0f}" if x >= 1000 else f"{x:,.1f}" if x >= 100 else f"{x:,.2f}"

# ---------- 3. join (every month repeats the country polygon) ----------
layer = world.merge(fc, on="iso3", how="inner")
layer["date"] = pd.to_datetime(layer["date"])
layer["month"] = layer["date"].dt.strftime("%Y-%m")          # readable month label, e.g. 2016-07
# QGIS compares these timestamps with UTC animation frames. Stamping a month at 00:00 on the 1st lets a
# time-zone offset (e.g. India UTC+5:30) push it into the PREVIOUS month's frame, so every colour/label looked
# one month late. Stamping mid-month (15th, 12:00) keeps each record inside its own month for every time zone.
layer["date"] = layer["date"] + pd.Timedelta(days=14, hours=12)
# ready-made label fields for QGIS (Layer Properties > Labels > Single Labels > Value = label)
rate_txt = layer["lcu_per_usd"].map(fmt_rate)
layer["label"] = layer["map_name"] + "\n" + rate_txt + np.where(layer["is_forecast"] == 1, "\nforecast", "")
layer["label_short"] = layer["iso3"] + " " + rate_txt        # saved as DateTime -> usable by Temporal Controller
layer = gpd.GeoDataFrame(layer, geometry="geometry", crs="EPSG:4326")
layer.to_file("output/exchange_rates_temporal.gpkg", layer="exchange_rates", driver="GPKG")

print(f"Saved {len(layer)} features ({layer['date'].nunique()} months) -> output/exchange_rates_temporal.gpkg")
print(layer.dtypes[["date", "yoy_pct", "status"]])

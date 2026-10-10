"""
Step 3 - Build the QGIS layer (GeoPandas)
Input : data/forecasts.csv  +  Natural Earth countries (GeoJSON or Shapefile)
Output: output/exchange_rates_temporal.gpkg   layer 'exchange_rates'
        -> ONE row per country per month, with a real DateTime field for the Temporal Controller.

Download boundaries (Natural Earth 'Admin 0 - Countries', 1:110m or 1:50m) and set BOUNDARIES below, e.g.
https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson
"""
from pathlib import Path
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
world = world[["iso3", "NAME", "geometry"]].rename(columns={"NAME": "map_name"})

# Eurozone members share the euro: GEM usually lists them individually, so most will match directly.
# Territories that have no polygon in Natural Earth 110m (small islands) are simply not drawn.
unmatched = sorted(set(fc["iso3"]) - set(world["iso3"]))
print(f"{fc['iso3'].nunique() - len(unmatched)} / {fc['iso3'].nunique()} countries matched to a polygon")
print("No polygon for:", unmatched)

# ---------- 3. join (every month repeats the country polygon) ----------
layer = world.merge(fc, on="iso3", how="inner")
layer["date"] = pd.to_datetime(layer["date"])        # saved as DateTime -> usable by Temporal Controller
layer = gpd.GeoDataFrame(layer, geometry="geometry", crs="EPSG:4326")
layer.to_file("output/exchange_rates_temporal.gpkg", layer="exchange_rates", driver="GPKG")

print(f"Saved {len(layer)} features ({layer['date'].nunique()} months) -> output/exchange_rates_temporal.gpkg")
print(layer.dtypes[["date", "yoy_pct", "status"]])

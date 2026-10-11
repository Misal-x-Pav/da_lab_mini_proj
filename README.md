# Exchange-rate spatio-temporal analysis (pandas + ARIMA/SARIMA + QGIS)

## Run order
```
pip install pandas openpyxl pycountry statsmodels joblib geopandas pyogrio
python 01_preprocess.py        # xlsx -> data/exchange_rate_long.csv        (seconds)
python 02_forecast.py          # -> data/forecasts.csv, model_metrics.csv   (~3 min)
python 03_build_qgis_layer.py  # -> output/exchange_rates_temporal.gpkg     (seconds, ~85 MB)
```

## QGIS steps
1. Layer > Add Layer > Add Vector Layer > `output/exchange_rates_temporal.gpkg` (layer `exchange_rates`).
2. Layer Properties > **Temporal** > tick *Dynamic Temporal Control* > Configuration: *Single Field with Date/Time* > Field: `date` > Event duration 1 month (or limit = Fixed time span).
3. Layer Properties > **Symbology** > *Graduated* > Value `yoy_pct_clip` > Mode *Quantile* (or Equal Interval with -50..100) > diverging ramp (e.g. RdYlGn inverted: red = currency weakened).
4. Open View > Panels > **Temporal Controller** > Animated > Set to Full Range > Step 1 month > Play.
5. To separate forecast months: Rule-based style or Layer Properties > Temporal > expression on `status`, or add a second layer filtered with `"status" = 'Forecast'`.
6. Optional: Layout > Add Map > Atlas / animation export, or Project > Import/Export > export animation frames.

## Map labels (country name + currency value)
Layer Properties > Labels > Single Labels > Value = `label` (or `label_short`). Add a white text buffer. Labels animate with the Temporal Controller; forecast months show a third line "forecast".

## Fields in the layer
`iso3, map_name, country, date, lcu_per_usd, lcu_lower95, lcu_upper95, is_forecast, status, model, index_100, yoy_pct, pct_change_mom, yoy_pct_clip, label, label_short`
`yoy_pct_clip` is `yoy_pct` clipped to [-50, 100] for colouring only.

## Script 1 also prints (set VERBOSE = False to hide)
EDA, a gap-fill back-test (mean vs linear vs polynomial), duplicate check, Z-score/IQR outlier flags (flag only), and group-by/pivot summaries saved to data/analysis/.

## Method notes
- Missing rule: drop a country with more than 12 missing months since Jan 2015, linearly interpolate smaller gaps. See `data/preprocess_report.csv`.
- Models fit on log(rate); ADF picks d; (p,q) from AIC grid; SARIMA adds seasonal (0,1,1,12).
- Validation: last 12 observed months held out; RMSE and MAPE compared against a naive (last value) baseline.

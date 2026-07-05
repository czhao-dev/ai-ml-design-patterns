# Reports

This folder contains generated model artifacts from:

```bash
python3 scripts/train_model.py
```

- `metrics.json`: train/test metadata and regression metrics for each target.
- `figures/prcp_actual_vs_predicted.svg`: 2016 precipitation forecast chart.
- `figures/snow_actual_vs_predicted.svg`: 2016 snowfall forecast chart.
- `figures/tobs_actual_vs_predicted.svg`: 2016 observed-temperature forecast chart.

These files are committed intentionally so the GitHub project has visible,
reviewable results without requiring a local run first.

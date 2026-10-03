# Freight rate prediction

Predict the posted dollar rate of a truck load, then price two files that have no rates: 12,000 November–December loads, and one fixed lane for each day of December 2025.

## Where to look

| If you want | Notebook | Script |
| --- | --- | --- |
| Every model, its training and October charts, and the submitted files | `model_training.ipynb` | `score.py` |
| Charts of what affects the rate, the fit check, calibration, and the report | `freight_rate_analysis.ipynb` | `analysis.py` |

The notebook and the script in each row are the same code.

## Run

From this directory, with Python 3.11:

```bash
python -m pip install -r requirements.txt
python score.py
python analysis.py
```

Or open `model_training.ipynb` and run all cells, then open `freight_rate_analysis.ipynb` and run all cells. The working directory has to be this folder. Run the model file before the analysis file. The report reads the scores the model file writes.

`score.py` and `model_training.ipynb` write:

- `validation_predictions.csv` — `load_id,predicted_rate` for all 12,000 validation loads
- `december_predictions.csv` — the December chart inputs with `predicted_rate` filled in
- `outputs/metrics.json` and `outputs/model_comparison.json`
- `outputs/figures/models/` — one chart per model
- `scorer_results/candidate_december.png`

The assessment checker is the `run_scorer` cell in that notebook. It checks both files and writes the December chart. It does not score accuracy. Spotter does that after submission.

`analysis.py` and `freight_rate_analysis.ipynb` write the correlation and driver charts, `outputs/fit_diagnostics.json`, the calibrated prediction files, `scorer_results_calibrated/candidate_december.png`, and `freight-rate-report.docx`.

## Model

Labeled data is 1 January 2025 through 31 October 2025. The unscored loads are November and December, so model choice uses an October holdout trained only on January–September. The submitted rates are a refit on all 48,000 labeled loads.

The model predicts rate per mile with histogram gradient boosting, then multiplies by distance. On an October holdout trained only on January–September, that model is the best of the set below. Submitted rates are a refit on all 48,000 labeled loads.

October holdout, 4,853 loads:

| Model | MAE | MAPE | R² |
| --- | ---: | ---: | ---: |
| Gradient boosting | $115.50 | 5.31% | 0.819 |
| Random forest | $143.55 | 6.64% | 0.815 |
| Per-equipment mileage regression | $147.43 | 8.40% | 0.817 |
| Prophet | $259.66 | 11.26% | 0.790 |
| ARIMA | $265.17 | 11.37% | 0.788 |
| SARIMAX | $279.25 | 11.63% | 0.782 |

Prophet, ARIMA, and SARIMAX forecast the daily median rate per mile. Each load is then priced as that forecast times an equipment factor times distance. They cannot see the difference between a short load and a long one on the same day, so they lose to the load-level models. R² is similar across models because distance already explains most of the dollar variance. MAE and MAPE are the scores that separate them.

The analysis notebook compares training error with later months. Training MAE is $80.65 and October MAE is $115.50. A deeper unrestricted tree does not close that gap, and October sits in the middle of the month-to-month range, so the model is not memorizing and it is not missing a deeper tree.

The same notebook adds the lag and calibration case. The calibrated prediction files are `validation_predictions_calibrated.csv` and `december_predictions_calibrated.csv`. The base submission files are unchanged.

# Group Experiment Pipeline

This folder contains the latest end-to-end experiment pipeline for the group project. It fetches price and news data, routes English news through FinBERT and Chinese news through a Chinese financial sentiment dictionary, injects the resulting daily sentiment signal into a DQN trading state, and compares `With_NLP`, `Without_NLP`, and `Buy-and-Hold`.

## Environment

The repo includes `environment.yml` for a clean Conda setup:

```powershell
conda env create -p .\group\.conda -f group\environment.yml
```

On this machine, the executed run used workspace-local fallback packages under `group/vendor_site` because `conda env create` hit local permission issues. Re-run from the `group` directory with:

```powershell
Set-Location .\group
python -m src.run_experiment --episodes 220 --walk-forward-folds 3 --lookback-years 2 --seed 42 --min-train-size 160 --gamma 0.95 --learning-rate 0.0003 --batch-size 64 --hidden-dim 128 --window-size 96 --reward-risk-penalty 0.04 --tau 0.02
```

## Outputs

All committed deliverables are written into `group/results`:

- `nlp_metrics.csv`
- `daily_sentiment_scores.csv`
- `feature_dataset.csv`
- `walkforward_fold_metrics.csv`
- `ablation_summary.csv`
- `equity_curve_comparison.png`
- `training_curve.png`

Reproducibility snapshots are kept in `group/data/raw`, including:

- `prices_raw.csv`
- `news_raw.csv`
- `news_bootstrap_real.csv`
- `phrasebank_allagree.csv`
- `FinancialPhraseBank-v1.0.zip`
- `Chinese_financial_sentiment_dictionary_Jiang_2020.xlsx`

The project does not require a real news article on every calendar day. Instead:

- `news_raw.csv` stores real collected news only
- `daily_sentiment_scores.csv` stores the daily aggregated sentiment signal
- missing-news dates are handled in feature engineering by sentiment decay / forward carry, not by fabricating news items

## Demo Dashboard

Run the presentation dashboard from the `group` directory:

```powershell
Set-Location .\group
streamlit run .\streamlit_dashboard.py
```

The dashboard reads directly from `group/results` and is intended for live presentation. It shows:

- overall ablation metrics
- per-fold strategy metrics
- price and normalized equity curves
- daily sentiment signals and news counts
- recent headlines scored by FinBERT or the Chinese financial sentiment dictionary

## Presentation File

For the report presentation deck, use `open_presentation.bat` or open `report_presentation_standalone.html` directly. The standalone file embeds the charts so it can be submitted or opened on another computer without the `results` folder.

If VS Code keeps showing `Failed to open ... (0x2)`, do not use the editor's external-open button. Use:

1. `Terminal > Run Task > Open Standalone Presentation`
2. Or run `Terminal > Run Task > Start Local Presentation Server`, then `Terminal > Run Task > Open Presentation URL`

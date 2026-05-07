from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import random
import sys

base_dir = Path(__file__).resolve().parents[1]
for extra_path in (
    base_dir / "vendor_site",
    base_dir / ".venv" / "Lib" / "site-packages",
):
    if extra_path.exists():
        sys.path.append(str(extra_path))

cache_root = base_dir / "cache_runtime"
(cache_root / "huggingface").mkdir(parents=True, exist_ok=True)
(cache_root / "datasets").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(cache_root / "datasets"))

import numpy as np
import pandas as pd
import torch
from dateutil.relativedelta import relativedelta

from src.data_fetch import DEFAULT_STOCKS, fetch_news_data, fetch_phrasebank_data, fetch_price_data
from src.dqn_agent import DQNConfig, evaluate_buy_and_hold, evaluate_dqn, train_dqn
from src.evaluation import compute_performance_metrics, plot_equity_curves, plot_training_curves, prepare_curve_export, summarize_metrics
from src.feature_engineering import build_feature_dataset, get_feature_columns, make_walk_forward_splits
from src.nlp_pipeline import run_nlp_pipeline


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the group experiment pipeline.")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--walk-forward-folds", type=int, default=3)
    parser.add_argument("--lookback-years", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-train-size", type=int, default=160)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=96)
    parser.add_argument("--reward-risk-penalty", type=float, default=0.04)
    parser.add_argument("--tau", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    raw_dir = base_dir / "data" / "raw"
    results_dir = base_dir / "results"
    raw_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    end_date = datetime.today()
    start_date = end_date - relativedelta(years=args.lookback_years)

    print(f"Fetching prices from {start_date.date()} to {end_date.date()} ...")
    price_df = fetch_price_data(DEFAULT_STOCKS, start=start_date, end=end_date, raw_dir=raw_dir)
    print(f"Fetched {len(price_df)} price rows.")

    print("Fetching stock-related news from RSS sources ...")
    news_df = fetch_news_data(DEFAULT_STOCKS, start=start_date, end=end_date, raw_dir=raw_dir)
    print(f"Fetched {len(news_df)} news rows.")

    print("Downloading Financial PhraseBank ...")
    phrasebank_df = fetch_phrasebank_data(raw_dir=raw_dir)
    print(f"Loaded {len(phrasebank_df)} labeled phrasebank rows.")

    print("Running NLP evaluation and sentiment scoring ...")
    daily_sentiment_df, _ = run_nlp_pipeline(phrasebank_df, news_df, results_dir=results_dir, random_state=args.seed)
    print(f"Generated {len(daily_sentiment_df)} daily sentiment rows.")

    feature_df = build_feature_dataset(price_df, daily_sentiment_df)
    feature_df.to_csv(results_dir / "feature_dataset.csv", index=False, encoding="utf-8-sig")
    print(f"Feature dataset has {len(feature_df)} rows after technical-indicator warmup.")

    metrics_rows: list[dict] = []
    training_rows: list[dict] = []
    all_curves: list[pd.DataFrame] = []

    dqn_config = DQNConfig(
        episodes=args.episodes,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        window_size=args.window_size,
        reward_risk_penalty=args.reward_risk_penalty,
        tau=args.tau,
    )

    stock_splits: dict[str, list[tuple[int, pd.DataFrame, pd.DataFrame]]] = {}
    for stock in DEFAULT_STOCKS:
        stock_df = feature_df[feature_df["symbol"] == stock.symbol].sort_values("date").reset_index(drop=True)
        splits = make_walk_forward_splits(
            stock_df,
            folds=args.walk_forward_folds,
            min_train_size=args.min_train_size,
        )
        if splits:
            stock_splits[stock.symbol] = splits

    if not stock_splits:
        raise RuntimeError("No stocks have enough rows for walk-forward validation.")

    fold_count = min(len(splits) for splits in stock_splits.values())

    for fold_idx in range(fold_count):
        fold_id = fold_idx + 1
        fold_train_sets = {
            symbol: stock_splits[symbol][fold_idx][1]
            for symbol in stock_splits
        }
        fold_test_sets = {
            symbol: stock_splits[symbol][fold_idx][2]
            for symbol in stock_splits
        }

        for strategy_name, include_sentiment in (("With_NLP", True), ("Without_NLP", False)):
            feature_columns = get_feature_columns(include_sentiment)
            trained = train_dqn(
                train_df=list(fold_train_sets.values()),
                feature_columns=feature_columns,
                seed=args.seed + fold_id,
                config=dqn_config,
            )
            for episode_idx, reward in enumerate(trained.training_rewards, start=1):
                training_rows.append(
                    {
                        "stock": "ALL",
                        "fold": fold_id,
                        "strategy": strategy_name,
                        "episode": episode_idx,
                        "reward": reward,
                    }
                )

            for symbol, test_df in fold_test_sets.items():
                train_df = fold_train_sets[symbol]
                curve_df = evaluate_dqn(trained, test_df)
                curve_df["stock"] = symbol
                curve_df["fold"] = fold_id
                curve_df["strategy"] = strategy_name
                all_curves.append(curve_df)

                row = compute_performance_metrics(curve_df)
                row.update(
                    {
                        "stock": symbol,
                        "fold": fold_id,
                        "strategy": strategy_name,
                        "train_rows": len(train_df),
                        "test_rows": len(test_df),
                    }
                )
                metrics_rows.append(row)

        for symbol, test_df in fold_test_sets.items():
            buy_hold_curve = evaluate_buy_and_hold(test_df, transaction_cost=dqn_config.transaction_cost)
            buy_hold_curve["stock"] = symbol
            buy_hold_curve["fold"] = fold_id
            buy_hold_curve["strategy"] = "Buy_and_Hold"
            all_curves.append(buy_hold_curve)

            row = compute_performance_metrics(buy_hold_curve)
            row.update(
                {
                    "stock": symbol,
                    "fold": fold_id,
                    "strategy": "Buy_and_Hold",
                    "train_rows": len(fold_train_sets[symbol]),
                    "test_rows": len(test_df),
                }
            )
            metrics_rows.append(row)

    if not metrics_rows:
        raise RuntimeError("No experiment results were generated. Check data availability and split sizes.")

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["stock", "fold", "strategy"])
    metrics_df.to_csv(results_dir / "walkforward_fold_metrics.csv", index=False, encoding="utf-8-sig")

    training_df = pd.DataFrame(training_rows)
    training_df.to_csv(results_dir / "training_rewards.csv", index=False, encoding="utf-8-sig")

    curves_df = pd.concat(all_curves, ignore_index=True)
    curve_export_df = prepare_curve_export(curves_df)
    curve_export_df.to_csv(results_dir / "equity_curves_detailed.csv", index=False, encoding="utf-8-sig")

    summary_df = summarize_metrics(metrics_df)
    summary_df.to_csv(results_dir / "ablation_summary.csv", index=False, encoding="utf-8-sig")

    plot_equity_curves(curves_df, results_dir / "equity_curve_comparison.png")
    plot_training_curves(training_df, results_dir / "training_curve.png")

    print("Experiment completed.")
    print(f"Results written to: {results_dir}")


if __name__ == "__main__":
    main()

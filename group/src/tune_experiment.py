from __future__ import annotations

import json
from pathlib import Path
import sys

base_dir = Path(__file__).resolve().parents[1]
for extra_path in (
    base_dir / "vendor_site",
    base_dir / ".venv" / "Lib" / "site-packages",
):
    if extra_path.exists():
        sys.path.append(str(extra_path))

import pandas as pd

from src.dqn_agent import DQNConfig, evaluate_buy_and_hold, evaluate_dqn, train_dqn
from src.evaluation import compute_performance_metrics
from src.feature_engineering import get_feature_columns, make_walk_forward_splits


def summarize_metrics(metrics_df: pd.DataFrame) -> dict[str, float]:
    grouped = metrics_df.groupby("strategy")[["sharpe_ratio", "max_drawdown", "cumulative_return"]].mean()
    with_sharpe = float(grouped.loc["With_NLP", "sharpe_ratio"])
    without_sharpe = float(grouped.loc["Without_NLP", "sharpe_ratio"])
    buy_hold_sharpe = float(grouped.loc["Buy_and_Hold", "sharpe_ratio"])
    with_return = float(grouped.loc["With_NLP", "cumulative_return"])
    with_mdd = float(grouped.loc["With_NLP", "max_drawdown"])
    score = (
        with_sharpe * 2.0
        + (with_sharpe - without_sharpe) * 1.5
        + (with_sharpe - buy_hold_sharpe) * 1.0
        + with_return * 3.0
        + with_mdd * 0.5
    )
    return {
        "score": score,
        "with_sharpe": with_sharpe,
        "without_sharpe": without_sharpe,
        "buy_hold_sharpe": buy_hold_sharpe,
        "with_return": with_return,
        "with_mdd": with_mdd,
    }


def evaluate_config(feature_df: pd.DataFrame, stock_symbols: list[str], config: DQNConfig, seed: int, folds: int, min_train_size: int) -> dict[str, float]:
    stock_splits: dict[str, list[tuple[int, pd.DataFrame, pd.DataFrame]]] = {}
    for symbol in stock_symbols:
        stock_df = feature_df[feature_df["symbol"] == symbol].sort_values("date").reset_index(drop=True)
        splits = make_walk_forward_splits(stock_df, folds=folds, min_train_size=min_train_size)
        if splits:
            stock_splits[symbol] = splits

    fold_count = min(len(splits) for splits in stock_splits.values())
    metric_rows: list[dict] = []

    for fold_idx in range(fold_count):
        fold_id = fold_idx + 1
        fold_train_sets = {symbol: stock_splits[symbol][fold_idx][1] for symbol in stock_splits}
        fold_test_sets = {symbol: stock_splits[symbol][fold_idx][2] for symbol in stock_splits}

        for strategy_name, include_sentiment in (("With_NLP", True), ("Without_NLP", False)):
            trained = train_dqn(
                train_df=list(fold_train_sets.values()),
                feature_columns=get_feature_columns(include_sentiment),
                seed=seed + fold_id,
                config=config,
            )
            for symbol, test_df in fold_test_sets.items():
                metrics = compute_performance_metrics(evaluate_dqn(trained, test_df))
                metrics.update({"strategy": strategy_name, "stock": symbol, "fold": fold_id})
                metric_rows.append(metrics)

        for symbol, test_df in fold_test_sets.items():
            metrics = compute_performance_metrics(evaluate_buy_and_hold(test_df, transaction_cost=config.transaction_cost))
            metrics.update({"strategy": "Buy_and_Hold", "stock": symbol, "fold": fold_id})
            metric_rows.append(metrics)

    metrics_df = pd.DataFrame(metric_rows)
    return summarize_metrics(metrics_df)


def main() -> None:
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    feature_df = pd.read_csv(results_dir / "feature_dataset.csv", parse_dates=["date"])
    stock_symbols = sorted(feature_df["symbol"].unique().tolist())

    candidate_configs = [
        DQNConfig(episodes=120, gamma=0.95, learning_rate=3e-4, batch_size=64, hidden_dim=128, window_size=96, reward_risk_penalty=0.04, tau=0.02),
        DQNConfig(episodes=120, gamma=0.97, learning_rate=5e-4, batch_size=64, hidden_dim=128, window_size=96, reward_risk_penalty=0.08, tau=0.02),
        DQNConfig(episodes=120, gamma=0.98, learning_rate=5e-4, batch_size=64, hidden_dim=128, window_size=128, reward_risk_penalty=0.10, tau=0.01),
        DQNConfig(episodes=120, gamma=0.95, learning_rate=8e-4, batch_size=32, hidden_dim=64, window_size=96, reward_risk_penalty=0.06, tau=0.03),
        DQNConfig(episodes=120, gamma=0.99, learning_rate=3e-4, batch_size=64, hidden_dim=128, window_size=128, reward_risk_penalty=0.12, tau=0.01),
    ]

    rows: list[dict] = []
    best_row: dict[str, float] | None = None
    best_config: DQNConfig | None = None

    for idx, config in enumerate(candidate_configs, start=1):
        summary = evaluate_config(feature_df, stock_symbols, config=config, seed=42, folds=3, min_train_size=160)
        row = {
            "config_id": idx,
            "episodes": config.episodes,
            "gamma": config.gamma,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "hidden_dim": config.hidden_dim,
            "window_size": config.window_size,
            "reward_risk_penalty": config.reward_risk_penalty,
            "tau": config.tau,
            **summary,
        }
        rows.append(row)
        print(f"config {idx}: {json.dumps(row, ensure_ascii=True)}", flush=True)
        if best_row is None or row["score"] > best_row["score"]:
            best_row = row
            best_config = config

    tuning_df = pd.DataFrame(rows).sort_values("score", ascending=False)
    try:
        tuning_df.to_csv(results_dir / "tuning_results.csv", index=False, encoding="utf-8-sig")
    except PermissionError:
        tuning_df.to_csv(base_dir / "tuning_results.csv", index=False, encoding="utf-8-sig")

    if best_row is None or best_config is None:
        raise RuntimeError("No tuning result produced.")

    best_payload = {
        "config_id": int(best_row["config_id"]),
        "episodes": int(best_config.episodes),
        "gamma": float(best_config.gamma),
        "learning_rate": float(best_config.learning_rate),
        "batch_size": int(best_config.batch_size),
        "hidden_dim": int(best_config.hidden_dim),
        "window_size": int(best_config.window_size),
        "reward_risk_penalty": float(best_config.reward_risk_penalty),
        "tau": float(best_config.tau),
        "score": float(best_row["score"]),
    }
    try:
        (results_dir / "best_config.json").write_text(json.dumps(best_payload, indent=2), encoding="utf-8")
    except PermissionError:
        (base_dir / "best_config.json").write_text(json.dumps(best_payload, indent=2), encoding="utf-8")
    print(json.dumps(best_payload, indent=2))


if __name__ == "__main__":
    main()

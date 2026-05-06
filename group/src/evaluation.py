from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_performance_metrics(curve_df: pd.DataFrame) -> dict[str, float]:
    equity = curve_df["equity"].astype(float).reset_index(drop=True)
    daily_returns = equity.pct_change().fillna(0.0)
    mean_return = float(daily_returns.mean())
    std_return = float(daily_returns.std(ddof=0))
    sharpe = 0.0 if std_return < 1e-12 else float(np.sqrt(252.0) * mean_return / std_return)
    running_max = equity.cummax()
    mdd = float((equity / running_max - 1.0).min())
    cumulative_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    return {
        "sharpe_ratio": sharpe,
        "max_drawdown": mdd,
        "cumulative_return": cumulative_return,
    }


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        metrics_df.groupby("strategy", as_index=False)
        .agg(
            runs=("strategy", "size"),
            sharpe_ratio_mean=("sharpe_ratio", "mean"),
            sharpe_ratio_std=("sharpe_ratio", "std"),
            max_drawdown_mean=("max_drawdown", "mean"),
            cumulative_return_mean=("cumulative_return", "mean"),
        )
        .sort_values("strategy")
    )
    return summary


def prepare_curve_export(curves_df: pd.DataFrame) -> pd.DataFrame:
    export_df = curves_df.copy()
    export_df["step"] = export_df.groupby(["strategy", "stock", "fold"]).cumcount()
    export_df["normalized_equity"] = export_df.groupby(["strategy", "stock", "fold"])["equity"].transform(
        lambda series: series / max(float(series.iloc[0]), 1e-9)
    )
    return export_df


def plot_equity_curves(curves_df: pd.DataFrame, out_path: Path) -> None:
    export_df = prepare_curve_export(curves_df)
    avg_curves = (
        export_df.groupby(["strategy", "step"], as_index=False)["normalized_equity"].mean().sort_values(["strategy", "step"])
    )

    plt.figure(figsize=(10, 6))
    for strategy, group_df in avg_curves.groupby("strategy"):
        plt.plot(group_df["step"], group_df["normalized_equity"], label=strategy)
    plt.xlabel("Test Step")
    plt.ylabel("Average Normalized Equity")
    plt.title("Average Equity Curve Across Stocks and Walk-Forward Folds")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_training_curves(training_df: pd.DataFrame, out_path: Path) -> None:
    avg_rewards = (
        training_df.groupby(["strategy", "episode"], as_index=False)["reward"].mean().sort_values(["strategy", "episode"])
    )

    plt.figure(figsize=(10, 6))
    for strategy, group_df in avg_rewards.groupby("strategy"):
        plt.plot(group_df["episode"], group_df["reward"], label=strategy)
    plt.xlabel("Episode")
    plt.ylabel("Average Episode Reward")
    plt.title("DQN Training Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


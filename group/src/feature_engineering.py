from __future__ import annotations

import numpy as np
import pandas as pd


BASE_FEATURE_COLUMNS = ["close", "ma50", "ma200", "rsi", "macd"]


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def build_feature_dataset(price_df: pd.DataFrame, daily_sentiment_df: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for symbol, stock_df in price_df.groupby("symbol"):
        stock_df = stock_df.sort_values("date").copy()
        stock_df["return"] = stock_df["close"].pct_change().fillna(0.0)
        stock_df["ma50"] = stock_df["close"].rolling(window=50).mean()
        stock_df["ma200"] = stock_df["close"].rolling(window=200).mean()
        ema12 = stock_df["close"].ewm(span=12, adjust=False).mean()
        ema26 = stock_df["close"].ewm(span=26, adjust=False).mean()
        stock_df["macd"] = ema12 - ema26
        stock_df["rsi"] = compute_rsi(stock_df["close"], window=14)
        stock_df["symbol"] = symbol
        frames.append(stock_df)

    feature_df = pd.concat(frames, ignore_index=True)
    merged = feature_df.merge(
        daily_sentiment_df[["symbol", "date", "sentiment_score", "news_count"]],
        on=["symbol", "date"],
        how="left",
    )
    merged["sentiment_score"] = merged["sentiment_score"].fillna(0.0)
    merged["news_count"] = merged["news_count"].fillna(0).astype(int)
    merged = merged.dropna(subset=["ma50", "ma200", "rsi", "macd"]).reset_index(drop=True)
    return merged


def get_feature_columns(include_sentiment: bool) -> list[str]:
    return BASE_FEATURE_COLUMNS + (["sentiment_score"] if include_sentiment else [])


def make_walk_forward_splits(
    stock_df: pd.DataFrame,
    folds: int = 3,
    min_train_size: int = 160,
    min_test_size: int = 20,
) -> list[tuple[int, pd.DataFrame, pd.DataFrame]]:
    n_rows = len(stock_df)
    if n_rows < min_train_size + min_test_size:
        return []

    usable_train_size = max(min_train_size, int(n_rows * 0.55))
    usable_train_size = min(usable_train_size, n_rows - min_test_size)
    remaining = n_rows - usable_train_size
    test_size = max(min_test_size, remaining // folds)
    splits: list[tuple[int, pd.DataFrame, pd.DataFrame]] = []

    for fold_idx in range(folds):
        train_end = usable_train_size + fold_idx * test_size
        test_end = train_end + test_size if fold_idx < folds - 1 else n_rows
        if test_end - train_end < min_test_size:
            continue
        train_df = stock_df.iloc[:train_end].copy()
        test_df = stock_df.iloc[train_end:test_end].copy()
        splits.append((fold_idx + 1, train_df, test_df))

    return splits

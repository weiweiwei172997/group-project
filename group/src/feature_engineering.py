from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_TECHNICAL_COLUMNS = ["close", "ma50", "ma200", "rsi", "macd"]
BASE_FEATURE_COLUMNS = [
    "close",
    "ma50",
    "ma200",
    "rsi",
    "macd",
    "close_to_ma50",
    "ma50_to_ma200",
    "return_1d",
    "return_5d",
    "volatility_20",
    "macd_signal",
    "macd_hist",
    "volume_ratio_20",
    "stock_id_0",
    "stock_id_1",
    "stock_id_2",
]
NLP_FEATURE_COLUMNS = [
    "sentiment_score",
    "sentiment_ema_3",
    "sentiment_ema_7",
    "sentiment_change",
    "sentiment_abs",
    "news_count_log",
    "days_since_news_log",
]


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def build_feature_dataset(price_df: pd.DataFrame, daily_sentiment_df: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    symbols = sorted(price_df["symbol"].unique().tolist())
    symbol_to_idx = {symbol: idx for idx, symbol in enumerate(symbols)}

    for symbol, stock_df in price_df.groupby("symbol"):
        stock_df = stock_df.sort_values("date").copy()
        stock_df["return_1d"] = stock_df["close"].pct_change().fillna(0.0)
        stock_df["return_5d"] = stock_df["close"].pct_change(periods=5)
        stock_df["ma50"] = stock_df["close"].rolling(window=50).mean()
        stock_df["ma200"] = stock_df["close"].rolling(window=200).mean()
        ema12 = stock_df["close"].ewm(span=12, adjust=False).mean()
        ema26 = stock_df["close"].ewm(span=26, adjust=False).mean()
        stock_df["macd"] = ema12 - ema26
        stock_df["macd_signal"] = stock_df["macd"].ewm(span=9, adjust=False).mean()
        stock_df["macd_hist"] = stock_df["macd"] - stock_df["macd_signal"]
        stock_df["rsi"] = compute_rsi(stock_df["close"], window=14)
        stock_df["volatility_20"] = stock_df["return_1d"].rolling(window=20).std()
        stock_df["volume_ma20"] = stock_df["volume"].rolling(window=20).mean()
        stock_df["volume_ratio_20"] = stock_df["volume"] / stock_df["volume_ma20"]
        stock_df["close_to_ma50"] = stock_df["close"] / stock_df["ma50"] - 1.0
        stock_df["ma50_to_ma200"] = stock_df["ma50"] / stock_df["ma200"] - 1.0
        for idx in range(len(symbols)):
            stock_df[f"stock_id_{idx}"] = 1.0 if idx == symbol_to_idx[symbol] else 0.0
        stock_df["symbol"] = symbol
        frames.append(stock_df)

    feature_df = pd.concat(frames, ignore_index=True)
    merged = feature_df.merge(
        daily_sentiment_df[["symbol", "date", "sentiment_score", "news_count"]],
        on=["symbol", "date"],
        how="left",
    )
    merged["sentiment_score_raw"] = merged["sentiment_score"]
    merged["news_count"] = merged["news_count"].fillna(0).astype(int)
    merged = merged.sort_values(["symbol", "date"]).reset_index(drop=True)
    merged["sentiment_score"] = 0.0
    merged["sentiment_ema_3"] = 0.0
    merged["sentiment_ema_7"] = 0.0
    merged["sentiment_change"] = 0.0
    merged["sentiment_abs"] = 0.0
    merged["days_since_news"] = 0.0

    for symbol, stock_df in merged.groupby("symbol"):
        idx = stock_df.index
        decayed_sentiment: list[float] = []
        days_since_news: list[int] = []
        last_signal = 0.0
        last_gap = 0
        for raw_signal, news_count in zip(stock_df["sentiment_score_raw"], stock_df["news_count"]):
            if pd.notna(raw_signal) and news_count > 0:
                last_signal = float(raw_signal)
                last_gap = 0
            else:
                last_gap += 1
                last_signal *= 0.85
            decayed_sentiment.append(float(np.clip(last_signal, -1.0, 1.0)))
            days_since_news.append(last_gap)

        sentiment_series = pd.Series(decayed_sentiment, index=idx, dtype=float)
        merged.loc[idx, "sentiment_score"] = sentiment_series.to_numpy(dtype=float)
        merged.loc[idx, "sentiment_ema_3"] = sentiment_series.ewm(span=3, adjust=False).mean().to_numpy(dtype=float)
        merged.loc[idx, "sentiment_ema_7"] = sentiment_series.ewm(span=7, adjust=False).mean().to_numpy(dtype=float)
        merged.loc[idx, "sentiment_change"] = sentiment_series.diff().fillna(0.0).to_numpy(dtype=float)
        merged.loc[idx, "sentiment_abs"] = sentiment_series.abs().to_numpy(dtype=float)
        merged.loc[idx, "days_since_news"] = np.asarray(days_since_news, dtype=float)

    merged["news_count_log"] = np.log1p(merged["news_count"])
    merged["days_since_news_log"] = np.log1p(merged["days_since_news"])
    merged = merged.dropna(
        subset=[
            "ma50",
            "ma200",
            "rsi",
            "macd",
            "return_5d",
            "volatility_20",
            "volume_ratio_20",
        ]
    ).reset_index(drop=True)
    return merged


def get_feature_columns(include_sentiment: bool) -> list[str]:
    return BASE_FEATURE_COLUMNS + (NLP_FEATURE_COLUMNS if include_sentiment else [])


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

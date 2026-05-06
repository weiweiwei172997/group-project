from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import io
import time
import zipfile
from typing import Iterable
from urllib.parse import quote_plus

import feedparser
import pandas as pd
import requests
from dateutil import parser as dt_parser


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


@dataclass(frozen=True)
class StockSpec:
    symbol: str
    name: str
    keywords: tuple[str, ...]


DEFAULT_STOCKS: tuple[StockSpec, ...] = (
    StockSpec("000001.SZ", "Ping An Bank", ("Ping An Bank", "平安银行", "000001")),
    StockSpec("600036.SH", "China Merchants Bank", ("China Merchants Bank", "招商银行", "600036")),
    StockSpec("600519.SH", "Kweichow Moutai", ("Kweichow Moutai", "贵州茅台", "600519")),
)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _to_yf_symbol(symbol: str) -> str:
    if symbol.endswith(".SH"):
        return symbol.replace(".SH", ".SS")
    return symbol


def _to_cn_symbol(symbol: str) -> str:
    market = "sh" if symbol.endswith(".SH") else "sz"
    return f"{market}{symbol.split('.')[0]}"


def _fetch_price_sina(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    url = "https://quotes.sina.cn/cn/api/openapi.php/CN_MarketDataService.getKLineData"
    params = {
        "symbol": _to_cn_symbol(symbol),
        "scale": "240",
        "ma": "no",
        "datalen": "1023",
    }
    resp = requests.get(url, params=params, timeout=30, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    payload = resp.json()
    rows = payload.get("result", {}).get("data", [])
    if not rows:
        raise ValueError(f"sina returned empty price data for {symbol}")

    out = pd.DataFrame(rows).rename(columns={"day": "date"})
    out = out[["date", "open", "high", "low", "close", "volume"]].copy()
    out["date"] = pd.to_datetime(out["date"])
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna()
    out = out[(out["date"] >= pd.Timestamp(start.date())) & (out["date"] <= pd.Timestamp(end.date()))]
    out["symbol"] = symbol
    if out.empty:
        raise ValueError(f"sina data filtered to empty range for {symbol}")
    return out[["date", "symbol", "open", "high", "low", "close", "volume"]].sort_values("date")


def _fetch_price_tencent(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    url = "http://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    params = {
        "param": f"{_to_cn_symbol(symbol)},day,,,{1023},qfq",
    }
    resp = requests.get(url, params=params, timeout=30, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    payload = resp.json()
    symbol_payload = payload.get("data", {}).get(_to_cn_symbol(symbol), {})
    rows = symbol_payload.get("qfqday") or symbol_payload.get("day") or []
    if not rows:
        raise ValueError(f"tencent returned empty price data for {symbol}")

    out = pd.DataFrame(rows, columns=["date", "open", "close", "high", "low", "volume"])
    out["date"] = pd.to_datetime(out["date"])
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna()
    out = out[(out["date"] >= pd.Timestamp(start.date())) & (out["date"] <= pd.Timestamp(end.date()))]
    out["symbol"] = symbol
    if out.empty:
        raise ValueError(f"tencent data filtered to empty range for {symbol}")
    return out[["date", "symbol", "open", "high", "low", "close", "volume"]].sort_values("date")


def _fetch_price_akshare(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    import akshare as ak

    base = symbol.split(".")[0]
    start_s = start.strftime("%Y%m%d")
    end_s = end.strftime("%Y%m%d")
    df = ak.stock_zh_a_hist(
        symbol=base,
        period="daily",
        start_date=start_s,
        end_date=end_s,
        adjust="qfq",
    )
    if df is None or df.empty:
        raise ValueError(f"akshare returned empty price data for {symbol}")

    rename_map = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
    }
    if not set(rename_map).issubset(df.columns):
        raise ValueError(f"akshare schema changed for {symbol}: {df.columns.tolist()}")

    out = df.rename(columns=rename_map)[list(rename_map.values())].copy()
    out["date"] = pd.to_datetime(out["date"])
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna().sort_values("date")
    out["symbol"] = symbol
    return out[["date", "symbol", "open", "high", "low", "close", "volume"]]


def _fetch_price_yfinance(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    import yfinance as yf

    yf_symbol = _to_yf_symbol(symbol)
    df = yf.download(
        yf_symbol,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )
    if df is None or df.empty:
        raise ValueError(f"yfinance returned empty price data for {symbol}")

    out = df.reset_index().rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    out = out[["date", "open", "high", "low", "close", "volume"]].copy()
    out["date"] = pd.to_datetime(out["date"])
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna().sort_values("date")
    out["symbol"] = symbol
    return out[["date", "symbol", "open", "high", "low", "close", "volume"]]


def fetch_price_data(stocks: Iterable[StockSpec], start: datetime, end: datetime, raw_dir: Path) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for spec in stocks:
        try:
            price_df = _fetch_price_sina(spec.symbol, start, end)
        except Exception:
            try:
                price_df = _fetch_price_tencent(spec.symbol, start, end)
            except Exception:
                try:
                    price_df = _fetch_price_akshare(spec.symbol, start, end)
                except Exception:
                    price_df = _fetch_price_yfinance(spec.symbol, start, end)
        records.append(price_df)

    out = pd.concat(records, ignore_index=True)
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    _ensure_dir(raw_dir)
    out.to_csv(raw_dir / "prices_raw.csv", index=False, encoding="utf-8-sig")
    return out


def _safe_parse_datetime(entry: feedparser.FeedParserDict) -> datetime | None:
    for key in ("published", "updated"):
        value = entry.get(key)
        if value:
            try:
                return dt_parser.parse(value)
            except Exception:
                pass
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed:
        return datetime(*parsed[:6])
    return None


def _rss_fetch(url: str) -> feedparser.FeedParserDict:
    resp = requests.get(url, timeout=20, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    return feedparser.parse(resp.content)


def fetch_news_data(stocks: Iterable[StockSpec], start: datetime, end: datetime, raw_dir: Path) -> pd.DataFrame:
    rss_templates = (
        "https://www.bing.com/news/search?q={query}&format=rss",
        "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q={query}&hl=zh-CN&gl=CN&ceid=CN:zh-Hans",
    )

    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for spec in stocks:
        for kw in spec.keywords:
            query = quote_plus(kw)
            for template in rss_templates:
                url = template.format(query=query)
                try:
                    feed = _rss_fetch(url)
                except Exception:
                    continue

                for entry in feed.entries:
                    published_dt = _safe_parse_datetime(entry)
                    if published_dt is None:
                        continue
                    if not (start.date() <= published_dt.date() <= end.date()):
                        continue

                    title = (entry.get("title") or "").strip()
                    summary = (entry.get("summary") or "").strip()
                    link = (entry.get("link") or "").strip()
                    if not title or not link:
                        continue

                    dedup_key = (spec.symbol, link)
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)

                    rows.append(
                        {
                            "date": published_dt.date().isoformat(),
                            "published_at": published_dt.isoformat(),
                            "symbol": spec.symbol,
                            "source": feed.feed.get("title", "rss"),
                            "title": title,
                            "summary": summary,
                            "url": link,
                            "query_keyword": kw,
                            "text": f"{title}. {summary}".strip(),
                        }
                    )
                time.sleep(0.2)

    news_df = pd.DataFrame(rows)
    if news_df.empty:
        raise RuntimeError("No news items collected from RSS sources.")

    news_df["date"] = pd.to_datetime(news_df["date"])
    news_df = news_df.sort_values(["symbol", "date", "published_at"]).reset_index(drop=True)
    _ensure_dir(raw_dir)
    news_df.to_csv(raw_dir / "news_raw.csv", index=False, encoding="utf-8-sig")
    return news_df


def fetch_phrasebank_data(raw_dir: Path) -> pd.DataFrame:
    _ensure_dir(raw_dir)
    zip_path = raw_dir / "FinancialPhraseBank-v1.0.zip"
    data_url = "https://huggingface.co/datasets/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip"

    if not zip_path.exists():
        resp = requests.get(data_url, timeout=60, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        zip_path.write_bytes(resp.content)

    with zipfile.ZipFile(io.BytesIO(zip_path.read_bytes())) as zf:
        target = "FinancialPhraseBank-v1.0/Sentences_AllAgree.txt"
        if target not in zf.namelist():
            raise RuntimeError("Sentences_AllAgree.txt not found in PhraseBank zip")
        lines = zf.read(target).decode("latin-1", errors="ignore").splitlines()

    samples: list[dict] = []
    for line in lines:
        line = line.strip()
        if not line or "@" not in line:
            continue
        sentence, label = line.rsplit("@", 1)
        label = label.strip().lower()
        if label not in {"negative", "neutral", "positive"}:
            continue
        samples.append({"text": sentence.strip(), "label": label})

    df = pd.DataFrame(samples)
    if df.empty:
        raise RuntimeError("Parsed PhraseBank is empty")

    df.to_csv(raw_dir / "phrasebank_allagree.csv", index=False, encoding="utf-8-sig")
    return df

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import io
import re
import time
import zipfile
from typing import Iterable
from urllib.parse import quote_plus

from bs4 import BeautifulSoup
import feedparser
import pandas as pd
import requests
from dateutil import parser as dt_parser


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

ENGLISH_RSS_TEMPLATES = (
    "https://www.bing.com/news/search?q={query}&format=rss",
    "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en",
)
CHINESE_RSS_TEMPLATES = (
    "https://www.bing.com/news/search?q={query}&format=rss",
    "https://news.google.com/rss/search?q={query}&hl=zh-CN&gl=CN&ceid=CN:zh-Hans",
)

GDELT_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
NEWS_PIPELINE_VERSION = "mixed_real_news_v6"
BOOTSTRAP_NEWS_FILENAME = "news_bootstrap_real.csv"


@dataclass(frozen=True)
class StockSpec:
    symbol: str
    name: str
    keywords: tuple[str, ...]


DEFAULT_STOCKS: tuple[StockSpec, ...] = (
    StockSpec("000001.SZ", "Ping An Bank", ("Ping An Bank", "平安银行")),
    StockSpec("600036.SH", "China Merchants Bank", ("China Merchants Bank", "招商银行")),
    StockSpec("600519.SH", "Kweichow Moutai", ("Kweichow Moutai", "贵州茅台", "Moutai")),
)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_cached_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    return None


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
    cached_path = raw_dir / "prices_raw.csv"
    cached_df = _load_cached_csv(cached_path)
    if cached_df is not None and not cached_df.empty:
        min_date = cached_df["date"].min()
        max_date = cached_df["date"].max()
        symbols = set(cached_df["symbol"].unique().tolist())
        expected_symbols = {spec.symbol for spec in stocks}
        if min_date <= pd.Timestamp(start.date()) and max_date >= pd.Timestamp(end.date()) and expected_symbols.issubset(symbols):
            return cached_df.sort_values(["symbol", "date"]).reset_index(drop=True)

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
    out.to_csv(cached_path, index=False, encoding="utf-8-sig")
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


def _clean_html_text(text: str) -> str:
    if not text:
        return ""
    if "<" in text or "&" in text:
        cleaned = BeautifulSoup(text, "html.parser").get_text(" ", strip=True)
    else:
        cleaned = text
    return " ".join(cleaned.split())


def _looks_relevant(text: str, spec: StockSpec) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in spec.keywords)


def _english_query_keywords(spec: StockSpec) -> tuple[str, ...]:
    english_keywords = tuple(keyword for keyword in spec.keywords if re.search(r"[A-Za-z]", keyword))
    return english_keywords or (spec.name,)


def _is_english_news_text(text: str) -> bool:
    text = str(text)
    english_letters = len(re.findall(r"[A-Za-z]", text))
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    return english_letters >= 12 and chinese_chars == 0


def _news_language_bucket(text: str) -> str:
    text = str(text)
    english_letters = len(re.findall(r"[A-Za-z]", text))
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    if chinese_chars > 0 and english_letters == 0:
        return "zh"
    if english_letters > 0 and chinese_chars == 0:
        return "en"
    if chinese_chars > english_letters / 2:
        return "zh"
    if english_letters >= 8:
        return "en"
    return "mixed"


def _fetch_gdelt_articles(query: str, start: datetime, end: datetime, tag: str) -> pd.DataFrame:
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": 250,
        "sort": "datedesc",
        "startdatetime": start.strftime("%Y%m%d000000"),
        "enddatetime": end.strftime("%Y%m%d235959"),
    }
    resp = requests.get(GDELT_ENDPOINT, params=params, timeout=45, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    payload = resp.json()
    articles = payload.get("articles", []) or payload.get("data", []) or []

    rows: list[dict] = []
    for art in articles:
        title = _clean_html_text((art.get("title") or "").strip())
        summary = _clean_html_text((art.get("summary") or "").strip())
        url = (art.get("url") or art.get("sourceurl") or "").strip()
        seen_raw = str(art.get("seendate") or art.get("datetime") or "").strip()
        if not title or not url or not seen_raw:
            continue
        try:
            published_dt = dt_parser.parse(seen_raw)
        except Exception:
            continue
        if not (start.date() <= published_dt.date() <= end.date()):
            continue

        source = (art.get("domain") or art.get("sourceCountry") or "gdelt").strip()
        text = " ".join(part for part in [title, summary] if part).strip()
        if not _is_english_news_text(text):
            continue

        rows.append(
            {
                "date": published_dt.date().isoformat(),
                "published_at": published_dt.isoformat(),
                "source": source,
                "title": title,
                "summary": summary,
                "url": url,
                "query_keyword": tag,
                "text": text,
                "source_type": "gdelt",
            }
        )

    return pd.DataFrame(rows)


def fetch_news_data(stocks: Iterable[StockSpec], start: datetime, end: datetime, raw_dir: Path) -> pd.DataFrame:
    cached_path = raw_dir / "news_raw.csv"
    cached_df = _load_cached_csv(cached_path)
    if cached_df is not None and not cached_df.empty and "crawl_version" in cached_df.columns:
        cached_df["date"] = pd.to_datetime(cached_df["date"])
        cached_df = cached_df[cached_df["date"].between(pd.Timestamp(start.date()), pd.Timestamp(end.date()))]
        cached_versions = set(cached_df["crawl_version"].dropna().astype(str).unique().tolist())
        cached_symbols = set(cached_df["symbol"].unique().tolist())
        expected_symbols = {spec.symbol for spec in stocks}
        if NEWS_PIPELINE_VERSION in cached_versions and expected_symbols.issubset(cached_symbols):
            return cached_df.sort_values(["symbol", "date", "published_at"]).reset_index(drop=True)

    article_frames: list[pd.DataFrame] = []
    seen: set[tuple[str, str]] = set()

    for spec in stocks:
        english_keywords = _english_query_keywords(spec)
        chinese_keywords = tuple(keyword for keyword in spec.keywords if re.search(r"[\u4e00-\u9fff]", keyword))
        if not english_keywords and not chinese_keywords:
            continue

        for language_code, keywords, templates in (
            ("en", english_keywords, ENGLISH_RSS_TEMPLATES),
            ("zh", chinese_keywords, CHINESE_RSS_TEMPLATES),
        ):
            if not keywords:
                continue
            for kw in keywords:
                query = quote_plus(kw)
                for template in templates:
                    try:
                        feed = _rss_fetch(template.format(query=query))
                    except Exception:
                        continue

                    feed_rows: list[dict] = []
                    for entry in feed.entries:
                        published_dt = _safe_parse_datetime(entry)
                        if published_dt is None:
                            continue
                        if not (start.date() <= published_dt.date() <= end.date()):
                            continue

                        title = _clean_html_text((entry.get("title") or "").strip())
                        summary = _clean_html_text((entry.get("summary") or "").strip())
                        link = (entry.get("link") or "").strip()
                        if not title or not link:
                            continue

                        text = " ".join(part for part in [title, summary] if part).strip()
                        if not _looks_relevant(text, spec):
                            continue

                        language_bucket = _news_language_bucket(text)
                        if language_code == "en" and language_bucket == "zh":
                            continue
                        if language_code == "zh" and language_bucket == "en":
                            continue

                        dedup_key = (spec.symbol, link)
                        if dedup_key in seen:
                            continue
                        seen.add(dedup_key)

                        feed_rows.append(
                            {
                                "date": published_dt.date().isoformat(),
                                "published_at": published_dt.isoformat(),
                                "symbol": spec.symbol,
                                "source": feed.feed.get("title", "rss"),
                                "title": title,
                                "summary": summary,
                                "url": link,
                                "query_keyword": kw,
                                "text": text,
                                "language": language_bucket,
                                "source_type": "direct",
                                "crawl_version": NEWS_PIPELINE_VERSION,
                            }
                        )

                    if feed_rows:
                        article_frames.append(pd.DataFrame(feed_rows))
                    time.sleep(0.2)

    if article_frames:
        news_df = pd.concat(article_frames, ignore_index=True)
    else:
        bootstrap_path = raw_dir / BOOTSTRAP_NEWS_FILENAME
        if not bootstrap_path.exists():
            raise RuntimeError("No news items collected from news sources and no bootstrap news snapshot is available.")
        news_df = pd.read_csv(bootstrap_path)
        if "language" not in news_df.columns:
            news_df["language"] = news_df["text"].fillna("").astype(str).map(_news_language_bucket)
        news_df["source_type"] = news_df.get("source_type", "bootstrap")
        news_df["crawl_version"] = NEWS_PIPELINE_VERSION
        news_df["date"] = pd.to_datetime(news_df["date"])
        news_df = news_df[news_df["date"].between(pd.Timestamp(start.date()), pd.Timestamp(end.date()))]
        expected_symbols = {spec.symbol for spec in stocks}
        news_df = news_df[news_df["symbol"].isin(expected_symbols)].copy()

    news_df["date"] = pd.to_datetime(news_df["date"])
    news_df = news_df.drop_duplicates(subset=["symbol", "url", "title"], keep="first").sort_values(
        ["symbol", "date", "published_at"]
    ).reset_index(drop=True)
    _ensure_dir(raw_dir)
    news_df.to_csv(cached_path, index=False, encoding="utf-8-sig")
    return news_df


def fetch_phrasebank_data(raw_dir: Path) -> pd.DataFrame:
    _ensure_dir(raw_dir)
    cached_csv = raw_dir / "phrasebank_allagree.csv"
    if cached_csv.exists():
        return pd.read_csv(cached_csv)

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

    df.to_csv(cached_csv, index=False, encoding="utf-8-sig")
    return df

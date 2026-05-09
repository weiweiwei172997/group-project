from __future__ import annotations

from pathlib import Path
import re
import tempfile
import os
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CHINESE_FIN_DICT_URL = (
    "https://raw.githubusercontent.com/MengLingchao/Chinese_financial_sentiment_dictionary/"
    "master/中文金融情感词典_姜富伟等(2020).xlsx"
)
CHINESE_FIN_DICT_FILENAME = "Chinese_financial_sentiment_dictionary_Jiang_2020.xlsx"


def _metrics_row(method: str, y_true: list[str], y_pred: list[str]) -> dict[str, float | str]:
    return {
        "method": method,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _dictionary_cache_path() -> Path:
    return _project_root() / "data" / "raw" / CHINESE_FIN_DICT_FILENAME


def _download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        out_path.write_bytes(response.content)
        return
    except Exception:
        pass

    fallback_url = (
        "https://api.github.com/repos/MengLingchao/Chinese_financial_sentiment_dictionary/contents/"
        + quote("中文金融情感词典_姜富伟等(2020).xlsx")
    )
    meta = requests.get(fallback_url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    meta.raise_for_status()
    download_url = meta.json()["download_url"]
    payload = requests.get(download_url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    payload.raise_for_status()
    out_path.write_bytes(payload.content)


def load_chinese_financial_dictionary() -> tuple[set[str], set[str]]:
    cache_path = _dictionary_cache_path()
    if not cache_path.exists():
        _download_file(CHINESE_FIN_DICT_URL, cache_path)

    positive_df = pd.read_excel(cache_path, sheet_name="positive")
    negative_df = pd.read_excel(cache_path, sheet_name="negative")

    positive_terms = {
        str(value).strip()
        for value in positive_df.iloc[:, 0].dropna().tolist()
        if str(value).strip() and str(value).strip().lower() != "positive word"
    }
    negative_terms = {
        str(value).strip()
        for value in negative_df.iloc[:, 0].dropna().tolist()
        if str(value).strip() and str(value).strip().lower() != "negative word"
    }
    return positive_terms, negative_terms


def _news_language_bucket(text: str) -> str:
    text = str(text)
    zh_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    en_chars = len(re.findall(r"[A-Za-z]", text))
    if zh_chars > 0 and en_chars == 0:
        return "zh"
    if en_chars > 0 and zh_chars == 0:
        return "en"
    if zh_chars >= en_chars / 2 and zh_chars > 0:
        return "zh"
    if en_chars >= 8:
        return "en"
    return "mixed"


def chinese_finance_sentiment_score(text: str, positive_terms: set[str], negative_terms: set[str]) -> tuple[str, float]:
    text = str(text)
    pos_count = sum(text.count(term) for term in positive_terms)
    neg_count = sum(text.count(term) for term in negative_terms)
    if pos_count == 0 and neg_count == 0:
        return "neutral", 0.0
    score = float((pos_count - neg_count) / max(pos_count + neg_count, 1))
    if score > 0.1:
        return "positive", score
    if score < -0.1:
        return "negative", score
    return "neutral", score


class FinBertSentiment:
    def __init__(self, model_name: str = "ProsusAI/finbert") -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.id2label = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        self.negative_idx = self._label_index("negative")
        self.positive_idx = self._label_index("positive")

    def _label_index(self, keyword: str) -> int:
        for idx, label in self.id2label.items():
            if keyword in label:
                return idx
        raise KeyError(f"Unable to find label index for {keyword}: {self.id2label}")

    def predict(self, texts: list[str], batch_size: int = 16) -> tuple[list[str], np.ndarray]:
        if not texts:
            return [], np.empty((0, len(self.id2label)))

        probs_chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**encoded).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                probs_chunks.append(probs)

        prob_array = np.vstack(probs_chunks)
        pred_indices = prob_array.argmax(axis=1)
        labels = [self.id2label[int(idx)] for idx in pred_indices]
        return labels, prob_array

    def score(self, texts: list[str], batch_size: int = 16) -> tuple[list[str], np.ndarray]:
        labels, probs = self.predict(texts, batch_size=batch_size)
        if probs.size == 0:
            return labels, np.array([], dtype=float)
        scores = probs[:, self.positive_idx] - probs[:, self.negative_idx]
        return labels, scores


def run_nlp_pipeline(
    phrasebank_df: pd.DataFrame,
    news_df: pd.DataFrame,
    results_dir: Path,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results_dir.mkdir(parents=True, exist_ok=True)
    zh_positive_terms, zh_negative_terms = load_chinese_financial_dictionary()

    train_df, test_df = train_test_split(
        phrasebank_df,
        test_size=0.2,
        stratify=phrasebank_df["label"],
        random_state=random_state,
    )

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)
    x_train = vectorizer.fit_transform(train_df["text"])
    x_test = vectorizer.transform(test_df["text"])

    lr_model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
    lr_model.fit(x_train, train_df["label"])
    lr_pred = lr_model.predict(x_test)

    finbert = FinBertSentiment()
    finbert_test_pred, _ = finbert.predict(test_df["text"].tolist(), batch_size=16)

    metrics_df = pd.DataFrame(
        [
            _metrics_row("TF-IDF + Logistic Regression", test_df["label"].tolist(), lr_pred.tolist()),
            _metrics_row("FinBERT", test_df["label"].tolist(), finbert_test_pred),
        ]
    )
    metrics_df.to_csv(results_dir / "nlp_metrics.csv", index=False, encoding="utf-8-sig")

    scored_news = news_df.copy()
    scored_news["text"] = scored_news["text"].fillna("").astype(str)
    scored_news["language"] = scored_news["text"].map(_news_language_bucket)
    scored_news["finbert_label"] = "neutral"
    scored_news["sentiment_score"] = 0.0
    scored_news["sentiment_method"] = ""

    english_mask = scored_news["language"].isin(["en", "mixed"])
    if english_mask.any():
        finbert_news_labels, finbert_news_scores = finbert.score(
            scored_news.loc[english_mask, "text"].tolist(),
            batch_size=16,
        )
        scored_news.loc[english_mask, "finbert_label"] = finbert_news_labels
        scored_news.loc[english_mask, "sentiment_score"] = finbert_news_scores
        scored_news.loc[english_mask, "sentiment_method"] = "finbert"

    chinese_mask = scored_news["language"] == "zh"
    if chinese_mask.any():
        zh_outputs = scored_news.loc[chinese_mask, "text"].map(
            lambda text: chinese_finance_sentiment_score(text, zh_positive_terms, zh_negative_terms)
        )
        scored_news.loc[chinese_mask, "finbert_label"] = zh_outputs.map(lambda x: x[0])
        scored_news.loc[chinese_mask, "sentiment_score"] = zh_outputs.map(lambda x: x[1]).astype(float)
        scored_news.loc[chinese_mask, "sentiment_method"] = "chinese_financial_dictionary"

    if "news_count" in scored_news.columns:
        daily_sentiment = (
            scored_news.groupby(["symbol", "date"], as_index=False)
            .agg(
                sentiment_score=("sentiment_score", "mean"),
                news_count=("news_count", "sum"),
                english_news=("sentiment_method", lambda x: int((pd.Series(x) == "finbert").sum())),
                chinese_news=("sentiment_method", lambda x: int((pd.Series(x) == "zh_lexicon").sum())),
            )
            .sort_values(["symbol", "date"])
        )
    else:
        daily_sentiment = (
            scored_news.groupby(["symbol", "date"], as_index=False)
            .agg(
                sentiment_score=("sentiment_score", "mean"),
                news_count=("text", "size"),
                english_news=("sentiment_method", lambda x: int((pd.Series(x) == "finbert").sum())),
                chinese_news=("sentiment_method", lambda x: int((pd.Series(x) == "zh_lexicon").sum())),
            )
            .sort_values(["symbol", "date"])
        )

    scored_news.to_csv(results_dir / "news_with_finbert_scores.csv", index=False, encoding="utf-8-sig")
    daily_sentiment.to_csv(results_dir / "daily_sentiment_scores.csv", index=False, encoding="utf-8-sig")
    return daily_sentiment, metrics_df

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _metrics_row(method: str, y_true: list[str], y_pred: list[str]) -> dict[str, float | str]:
    return {
        "method": method,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


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
    finbert_news_labels, finbert_news_scores = finbert.score(scored_news["text"].tolist(), batch_size=16)
    scored_news["finbert_label"] = finbert_news_labels
    scored_news["sentiment_score"] = finbert_news_scores

    daily_sentiment = (
        scored_news.groupby(["symbol", "date"], as_index=False)
        .agg(sentiment_score=("sentiment_score", "mean"), news_count=("text", "size"))
        .sort_values(["symbol", "date"])
    )

    scored_news.to_csv(results_dir / "news_with_finbert_scores.csv", index=False, encoding="utf-8-sig")
    daily_sentiment.to_csv(results_dir / "daily_sentiment_scores.csv", index=False, encoding="utf-8-sig")
    return daily_sentiment, metrics_df


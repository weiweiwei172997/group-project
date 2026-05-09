from __future__ import annotations

import html
import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

STRATEGY_COLORS = {
    "With_NLP": "#ff6b35",
    "Without_NLP": "#1f7a8c",
    "Buy_and_Hold": "#2a9d8f",
}

LABEL_COLORS = {
    "positive": "#0f9d58",
    "neutral": "#b08900",
    "negative": "#d93025",
}

STOCK_LABELS = {
    "000001.SZ": "Ping An Bank",
    "600036.SH": "China Merchants Bank",
    "600519.SH": "Kweichow Moutai",
}


def stock_name(symbol: str) -> str:
    return f"{STOCK_LABELS.get(symbol, symbol)} ({symbol})"


def clean_text(value: str) -> str:
    if not isinstance(value, str):
        return ""
    text = html.unescape(value)
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(text.split())


@st.cache_data(show_spinner=False)
def load_data() -> dict[str, pd.DataFrame]:
    data = {
        "summary": pd.read_csv(RESULTS_DIR / "ablation_summary.csv"),
        "fold_metrics": pd.read_csv(RESULTS_DIR / "walkforward_fold_metrics.csv"),
        "equity": pd.read_csv(RESULTS_DIR / "equity_curves_detailed.csv", parse_dates=["date"]),
        "sentiment_daily": pd.read_csv(RESULTS_DIR / "daily_sentiment_scores.csv", parse_dates=["date"]),
        "news": pd.read_csv(RESULTS_DIR / "news_with_finbert_scores.csv", parse_dates=["date"]),
        "features": pd.read_csv(RESULTS_DIR / "feature_dataset.csv", parse_dates=["date"]),
        "nlp_metrics": pd.read_csv(RESULTS_DIR / "nlp_metrics.csv"),
    }
    data["news"]["title"] = data["news"]["title"].map(clean_text)
    data["news"]["summary"] = data["news"]["summary"].map(clean_text)
    data["news"]["text"] = data["news"]["text"].map(clean_text)
    return data


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

        :root {
            --bg: #f5efe1;
            --panel: rgba(255, 251, 243, 0.88);
            --ink: #14213d;
            --muted: #6b705c;
            --accent: #ff6b35;
            --accent-2: #1f7a8c;
            --accent-3: #2a9d8f;
            --line: rgba(20, 33, 61, 0.10);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255, 107, 53, 0.16), transparent 28%),
                radial-gradient(circle at top right, rgba(31, 122, 140, 0.18), transparent 22%),
                linear-gradient(135deg, #f7f1e3 0%, #f3ede2 52%, #ece4d0 100%);
            color: var(--ink);
            font-family: "Space Grotesk", sans-serif;
        }

        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 2rem;
        }

        h1, h2, h3 {
            font-family: "Space Grotesk", sans-serif;
            color: var(--ink);
            letter-spacing: -0.02em;
        }

        .eyebrow {
            display: inline-block;
            padding: 0.3rem 0.7rem;
            border-radius: 999px;
            background: rgba(20, 33, 61, 0.08);
            color: var(--ink);
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .hero {
            padding: 1.4rem 1.5rem;
            border: 1px solid var(--line);
            border-radius: 24px;
            background:
                linear-gradient(145deg, rgba(255, 255, 255, 0.84), rgba(255, 247, 235, 0.82)),
                repeating-linear-gradient(
                    90deg,
                    rgba(20, 33, 61, 0.02) 0px,
                    rgba(20, 33, 61, 0.02) 1px,
                    transparent 1px,
                    transparent 22px
                );
            box-shadow: 0 18px 48px rgba(20, 33, 61, 0.10);
            margin-bottom: 1rem;
        }

        .hero-grid {
            display: grid;
            grid-template-columns: 1.7fr 1fr;
            gap: 1rem;
            align-items: start;
        }

        .hero-title {
            font-size: 2.35rem;
            line-height: 1.02;
            margin: 0.6rem 0 0.8rem 0;
            max-width: 10ch;
        }

        .hero-copy {
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.6;
            max-width: 64ch;
        }

        .hero-side {
            display: grid;
            gap: 0.8rem;
        }

        .mini-card {
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.65);
            border: 1px solid rgba(20, 33, 61, 0.08);
            padding: 0.9rem 1rem;
        }

        .mini-label, .stat-label {
            color: var(--muted);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
        }

        .mini-value {
            font-family: "IBM Plex Mono", monospace;
            font-size: 1.2rem;
            margin-top: 0.35rem;
            font-weight: 600;
        }

        .stat-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.9rem;
            margin: 1rem 0 1.4rem 0;
        }

        .stat-card {
            border-radius: 18px;
            padding: 1rem;
            border: 1px solid rgba(20, 33, 61, 0.10);
            background: rgba(255, 255, 255, 0.78);
            box-shadow: 0 12px 28px rgba(20, 33, 61, 0.06);
        }

        .stat-value {
            font-family: "IBM Plex Mono", monospace;
            font-size: 1.5rem;
            font-weight: 600;
            margin: 0.35rem 0 0.2rem 0;
        }

        .stat-delta {
            font-size: 0.88rem;
            color: var(--muted);
        }

        .section-title {
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 0.4rem;
        }

        .section-copy {
            color: var(--muted);
            margin-bottom: 0.75rem;
        }

        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(20, 33, 61, 0.08);
            padding: 0.8rem 0.9rem;
            border-radius: 16px;
        }

        div[data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(255, 250, 242, 0.96), rgba(244, 235, 221, 0.96));
            border-right: 1px solid rgba(20, 33, 61, 0.08);
        }

        .note-box {
            border-left: 4px solid var(--accent);
            background: rgba(255, 255, 255, 0.72);
            padding: 0.9rem 1rem;
            border-radius: 0 14px 14px 0;
            color: var(--ink);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(summary_df: pd.DataFrame, nlp_df: pd.DataFrame) -> None:
    with_nlp = summary_df.loc[summary_df["strategy"] == "With_NLP"].iloc[0]
    without_nlp = summary_df.loc[summary_df["strategy"] == "Without_NLP"].iloc[0]
    buy_hold = summary_df.loc[summary_df["strategy"] == "Buy_and_Hold"].iloc[0]
    finbert_f1 = float(nlp_df.loc[nlp_df["method"] == "FinBERT", "f1_weighted"].iloc[0])
    sharpe_gap = with_nlp["sharpe_ratio_mean"] - without_nlp["sharpe_ratio_mean"]
    buy_hold_gap = with_nlp["sharpe_ratio_mean"] - buy_hold["sharpe_ratio_mean"]

    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-grid">
            <div>
              <span class="eyebrow">Live Demo Layer</span>
              <h1 class="hero-title">NLP + RL Trading Monitor</h1>
              <p class="hero-copy">
                This dashboard turns the assignment into a demo narrative: English news is scored by FinBERT,
                Chinese news is scored by a Chinese financial sentiment dictionary, both are turned into a
                daily sentiment signal, and the DQN policy uses that signal in the state vector.
              </p>
            </div>
            <div class="hero-side">
              <div class="mini-card">
                <div class="mini-label">Sharpe Lift vs Without NLP</div>
                <div class="mini-value">{sharpe_gap:+.3f}</div>
              </div>
              <div class="mini-card">
                <div class="mini-label">Sharpe Lift vs Buy-and-Hold</div>
                <div class="mini-value">{buy_hold_gap:+.3f}</div>
              </div>
              <div class="mini-card">
                <div class="mini-label">FinBERT Weighted F1</div>
                <div class="mini-value">{finbert_f1:.3f}</div>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stat_cards(selected_metrics: pd.Series, benchmark_metrics: pd.Series, selected_strategy: str) -> None:
    delta_sharpe = selected_metrics["sharpe_ratio"] - benchmark_metrics["sharpe_ratio"]
    delta_return = selected_metrics["cumulative_return"] - benchmark_metrics["cumulative_return"]
    delta_mdd = selected_metrics["max_drawdown"] - benchmark_metrics["max_drawdown"]

    cards = [
        ("Selected Strategy", selected_strategy.replace("_", " "), f"{selected_metrics['stock']} | Fold {int(selected_metrics['fold'])}"),
        ("Sharpe Ratio", f"{selected_metrics['sharpe_ratio']:.3f}", f"vs Buy-and-Hold {delta_sharpe:+.3f}"),
        ("Cumulative Return", f"{selected_metrics['cumulative_return'] * 100:.2f}%", f"vs Buy-and-Hold {delta_return * 100:+.2f}pp"),
        ("Max Drawdown", f"{selected_metrics['max_drawdown'] * 100:.2f}%", f"vs Buy-and-Hold {delta_mdd * 100:+.2f}pp"),
    ]

    html_cards = "".join(
        f"""
        <div class="stat-card">
          <div class="stat-label">{label}</div>
          <div class="stat-value">{value}</div>
          <div class="stat-delta">{delta}</div>
        </div>
        """
        for label, value, delta in cards
    )
    st.markdown(f'<div class="stat-grid">{html_cards}</div>', unsafe_allow_html=True)


def build_equity_figure(
    equity_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    daily_sentiment_df: pd.DataFrame,
    stock: str,
    fold: int,
    selected_strategy: str,
) -> go.Figure:
    stock_prices = feature_df[feature_df["symbol"] == stock][["date", "close"]].drop_duplicates().sort_values("date")
    fold_curves = equity_df[(equity_df["stock"] == stock) & (equity_df["fold"] == fold)].copy()
    fold_sent = daily_sentiment_df[daily_sentiment_df["symbol"] == stock].copy()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.68, 0.32],
        specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
    )

    fig.add_trace(
        go.Scatter(
            x=stock_prices["date"],
            y=stock_prices["close"],
            name="Close Price",
            line=dict(color="#14213d", width=2),
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    for strategy in ["With_NLP", "Without_NLP", "Buy_and_Hold"]:
        curve = fold_curves[fold_curves["strategy"] == strategy]
        if curve.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=curve["date"],
                y=curve["normalized_equity"],
                name=strategy.replace("_", " "),
                mode="lines",
                line=dict(color=STRATEGY_COLORS[strategy], width=3 if strategy == selected_strategy else 2),
                opacity=1.0 if strategy == selected_strategy else 0.45,
            ),
            row=1,
            col=1,
            secondary_y=True,
        )

    selected_curve = fold_curves[fold_curves["strategy"] == selected_strategy]
    for action, symbol_shape, color in [("BUY", "triangle-up", "#0f9d58"), ("SELL", "triangle-down", "#d93025")]:
        action_rows = selected_curve[selected_curve["action"] == action]
        if action_rows.empty:
            continue
        action_points = action_rows.merge(stock_prices, on="date", how="left")
        fig.add_trace(
            go.Scatter(
                x=action_points["date"],
                y=action_points["close"],
                mode="markers",
                name=action.title(),
                marker=dict(symbol=symbol_shape, size=10, color=color, line=dict(color="white", width=1.2)),
            ),
            row=1,
            col=1,
            secondary_y=False,
        )

    if not fold_sent.empty:
        fig.add_trace(
            go.Bar(
                x=fold_sent["date"],
                y=fold_sent["news_count"],
                name="News Count",
                marker_color="rgba(31, 122, 140, 0.35)",
            ),
            row=2,
            col=1,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=fold_sent["date"],
                y=fold_sent["sentiment_score"],
                name="Daily Sentiment",
                line=dict(color="#ff6b35", width=2.6),
                fill="tozeroy",
                fillcolor="rgba(255, 107, 53, 0.10)",
            ),
            row=2,
            col=1,
            secondary_y=True,
        )

    fig.update_layout(
        height=700,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.42)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(title_text="Price", row=1, col=1, secondary_y=False, showgrid=True, gridcolor="rgba(20,33,61,0.08)")
    fig.update_yaxes(title_text="Normalized Equity", row=1, col=1, secondary_y=True, showgrid=False)
    fig.update_yaxes(title_text="News Count", row=2, col=1, secondary_y=False, showgrid=True, gridcolor="rgba(20,33,61,0.08)")
    fig.update_yaxes(title_text="Sentiment", row=2, col=1, secondary_y=True, showgrid=False)
    return fig


def build_summary_bar(summary_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for metric, axis, colors in [
        ("sharpe_ratio_mean", "Sharpe", "#ff6b35"),
        ("cumulative_return_mean", "Cumulative Return", "#1f7a8c"),
        ("max_drawdown_mean", "Max Drawdown", "#2a9d8f"),
    ]:
        fig.add_trace(
            go.Bar(
                x=summary_df["strategy"].str.replace("_", " "),
                y=summary_df[metric],
                name=axis,
                marker_color=colors,
                opacity=0.88,
                visible=True if metric == "sharpe_ratio_mean" else "legendonly",
            )
        )

    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.42)",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_yaxes(showgrid=True, gridcolor="rgba(20,33,61,0.08)")
    return fig


def build_nlp_radar(nlp_df: pd.DataFrame) -> go.Figure:
    metrics = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    fig = go.Figure()
    for _, row in nlp_df.iterrows():
        fig.add_trace(
            go.Scatterpolar(
                r=[row[m] for m in metrics] + [row[metrics[0]]],
                theta=labels + [labels[0]],
                fill="toself",
                name=row["method"],
            )
        )
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(255,255,255,0.42)",
            radialaxis=dict(range=[0.7, 1.0], showline=False, gridcolor="rgba(20,33,61,0.10)")
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def render_headlines(news_df: pd.DataFrame, stock: str, limit: int = 8) -> None:
    st.markdown('<div class="section-title">Recent Headlines</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">Use this live during the demo to connect a concrete article to the sentiment score and trading state.</div>', unsafe_allow_html=True)
    stock_news = (
        news_df[news_df["symbol"] == stock]
        .sort_values(["date", "sentiment_score"], ascending=[False, False])
        .head(limit)
        .copy()
    )
    if stock_news.empty:
        st.info("No news rows available for the selected stock.")
        return

    for _, row in stock_news.iterrows():
        label = str(row["finbert_label"]).lower()
        color = LABEL_COLORS.get(label, "#6b705c")
        st.markdown(
            f"""
            <div class="mini-card" style="margin-bottom:0.7rem;">
              <div style="display:flex;justify-content:space-between;gap:1rem;align-items:center;">
                <div style="font-weight:700;">{row['title']}</div>
                <div style="font-family:'IBM Plex Mono', monospace;color:{color};font-weight:700;">{label.upper()} {row['sentiment_score']:+.3f}</div>
              </div>
              <div style="margin-top:0.45rem;color:#6b705c;font-size:0.92rem;">{row['date'].date()} · {clean_text(str(row['source']))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(
        page_title="NLP + RL Trading Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()
    data = load_data()

    st.sidebar.markdown("## Demo Controls")
    stock = st.sidebar.selectbox(
        "Stock",
        options=sorted(data["fold_metrics"]["stock"].unique().tolist()),
        format_func=stock_name,
    )
    stock_folds = sorted(data["fold_metrics"].loc[data["fold_metrics"]["stock"] == stock, "fold"].unique().tolist())
    fold = st.sidebar.selectbox("Walk-Forward Fold", options=stock_folds)
    selected_strategy = st.sidebar.radio("Primary Strategy", options=["With_NLP", "Without_NLP", "Buy_and_Hold"], horizontal=False)

    st.sidebar.markdown("---")
    st.sidebar.markdown("## Live Storyline")
    st.sidebar.markdown(
        """
        1. Choose a stock and fold.  
        2. Show the sentiment panel and recent headlines.  
        3. Point at the trade markers on the price chart.  
        4. Finish with the ablation summary bar chart.
        """
    )

    render_hero(data["summary"], data["nlp_metrics"])

    selected_metrics = data["fold_metrics"][
        (data["fold_metrics"]["stock"] == stock)
        & (data["fold_metrics"]["fold"] == fold)
        & (data["fold_metrics"]["strategy"] == selected_strategy)
    ].iloc[0]
    benchmark_metrics = data["fold_metrics"][
        (data["fold_metrics"]["stock"] == stock)
        & (data["fold_metrics"]["fold"] == fold)
        & (data["fold_metrics"]["strategy"] == "Buy_and_Hold")
    ].iloc[0]
    render_stat_cards(selected_metrics, benchmark_metrics, selected_strategy)

    left, right = st.columns([1.7, 1.0], gap="large")
    with left:
        st.markdown('<div class="section-title">Price, Equity and NLP Signal</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-copy">Top panel shows market price plus normalized strategy equity. Bottom panel shows the daily sentiment signal and how often relevant news appeared.</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            build_equity_figure(
                data["equity"],
                data["features"],
                data["sentiment_daily"],
                stock=stock,
                fold=fold,
                selected_strategy=selected_strategy,
            ),
            use_container_width=True,
        )

    with right:
        st.markdown('<div class="section-title">Ablation Snapshot</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-copy">This is the headline comparison to use when the teacher asks whether NLP helped.</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(build_summary_bar(data["summary"]), use_container_width=True)
        st.markdown(
            '<div class="note-box">Interpretation: for the current run, <strong>With NLP</strong> improves Sharpe and loss control relative to <strong>Without NLP</strong>, even though absolute returns remain modest in a weak market window.</div>',
            unsafe_allow_html=True,
        )

    bottom_left, bottom_right = st.columns([1.05, 0.95], gap="large")
    with bottom_left:
        st.markdown('<div class="section-title">NLP Model Quality</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-copy">Use this panel to justify why we combine FinBERT with a Chinese financial sentiment dictionary instead of forcing one model onto both languages.</div>', unsafe_allow_html=True)
        st.plotly_chart(build_nlp_radar(data["nlp_metrics"]), use_container_width=True)

        stock_metrics = data["fold_metrics"][data["fold_metrics"]["stock"] == stock].copy()
        stock_metrics["strategy"] = stock_metrics["strategy"].str.replace("_", " ")
        st.dataframe(
            stock_metrics[["fold", "strategy", "sharpe_ratio", "max_drawdown", "cumulative_return", "train_rows", "test_rows"]],
            use_container_width=True,
            hide_index=True,
        )

    with bottom_right:
        render_headlines(data["news"], stock=stock, limit=7)


if __name__ == "__main__":
    main()

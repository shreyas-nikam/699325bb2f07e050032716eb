import os
import io
import tempfile
from contextlib import redirect_stdout

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import yfinance as yf

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score


# ----------------------------
# Streamlit config
# ----------------------------
st.set_page_config(
    page_title="Market Sentinel: Regimes → Risk Policy (Concept XYZ)",
    page_icon="📈",
    layout="wide",
)
st.title("QuLab: Lab 13 - Market Sentinel: Regimes → Risk Policy (Concept XYZ)")
st.divider()
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
# Quieter TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# ----------------------------
# Core model / data functions (from the provided notebook)
# ----------------------------
@st.cache_data(show_spinner=False)
def acquire_and_engineer_features(start_date: str = "2005-01-01", end_date: str = "2024-12-31"):
    """
    Acquires raw financial data and engineers features and regime labels.
    """
    sp500 = yf.download("^GSPC", start=start_date, end=end_date, progress=False)
    vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)
    tnx = yf.download("^TNX", start=start_date, end=end_date, progress=False)
    gld = yf.download("GLD", start=start_date, end=end_date, progress=False)

    df = pd.DataFrame(index=sp500.index)

    # Engineer features
    df["sp500_ret"] = sp500["Close"].pct_change()
    df["sp500_ret_5d"] = sp500["Close"].pct_change(5)
    df["sp500_ret_21d"] = sp500["Close"].pct_change(21)
    df["realized_vol_21d"] = (df["sp500_ret"].rolling(21).std() * np.sqrt(252))

    df["vix"] = vix["Close"]
    df["vix_change_5d"] = df["vix"].pct_change(5)

    df["yield_10y"] = tnx["Close"]
    df["yield_spread"] = df["yield_10y"] - df["yield_10y"].shift(5)  # rough proxy

    df["gold_ret"] = gld["Close"].pct_change()

    # Regime definition (full-sample median VIX per spec)
    vix_median = df["vix"].median()
    df["regime"] = np.where(
        df["sp500_ret_21d"] <= 0,
        "Bear",
        np.where(df["vix"] > vix_median, "Bull-HighVol", "Bull-LowVol"),
    )

    regime_map = {"Bull-LowVol": 0, "Bull-HighVol": 1, "Bear": 2}
    df["regime_encoded"] = df["regime"].map(regime_map)

    df = df.dropna()

    feature_cols = [
        "sp500_ret", "sp500_ret_5d", "sp500_ret_21d",
        "realized_vol_21d", "vix", "vix_change_5d",
        "yield_10y", "yield_spread", "gold_ret",
    ]

    return df, feature_cols, regime_map


def create_sequences(features: np.ndarray, targets: np.ndarray, lookback: int = 60):
    """
    Creates sliding-window sequences for LSTM input.
    """
    X, y = [], []
    for i in range(lookback, len(features)):
        X.append(features[i - lookback:i])
        y.append(targets[i])
    return np.array(X), np.array(y)


def build_lstm_model(
    input_shape,
    n_classes: int,
    learning_rate: float = 0.001,
    dropout_rate_lstm: float = 0.3,
    dropout_rate_dense: float = 0.2,
):
    """
    Builds a sequential LSTM model for market regime classification.
    """
    model = Sequential([
        LSTM(units=64, input_shape=input_shape, return_sequences=True),
        Dropout(dropout_rate_lstm),
        BatchNormalization(),

        LSTM(units=32, return_sequences=False),
        Dropout(dropout_rate_lstm),
        BatchNormalization(),

        Dense(units=16, activation="relu"),
        Dropout(dropout_rate_dense),
        Dense(units=n_classes, activation="softmax"),
    ])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train_model(
    model,
    X_train: np.ndarray,
    y_train_oh: np.ndarray,
    epochs: int = 100,
    batch_size: int = 64,
    validation_split: float = 0.2,
    model_filepath: str = "best_lstm_regime.keras",
):
    """
    Trains the LSTM model with callbacks.
    """
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        ModelCheckpoint(model_filepath, save_best_only=True, monitor="val_loss", verbose=1),
    ]

    history = model.fit(
        X_train, y_train_oh,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def fig_training_history(history):
    """
    Returns a matplotlib Figure plotting train/val loss and accuracy curves.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history["loss"], label="Train Loss")
    ax1.plot(history.history["val_loss"], label="Val Loss")
    ax1.set_title("Model Fit: Loss Curves")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(history.history["accuracy"], label="Train Accuracy")
    ax2.plot(history.history["val_accuracy"], label="Validation Accuracy")
    ax2.set_title("Model Fit: Accuracy Curves")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    fig.suptitle("Overfitting Diagnostic (Finance Reality Check)", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def evaluate_models(y_true, y_pred_lstm, y_pred_lr, y_pred_persist, regime_label_map):
    """
    Evaluates LSTM and baselines and returns a metrics dataframe (incl. Transition Recall).
    """
    # Accuracy & Macro-F1
    lstm_accuracy = accuracy_score(y_true, y_pred_lstm)
    lstm_f1_macro = f1_score(y_true, y_pred_lstm, average="macro", zero_division=0)

    lr_accuracy = accuracy_score(y_true, y_pred_lr)
    lr_f1_macro = f1_score(y_true, y_pred_lr, average="macro", zero_division=0)

    persist_accuracy = accuracy_score(y_true, y_pred_persist)
    persist_f1_macro = f1_score(y_true, y_pred_persist, average="macro", zero_division=0)

    # Bear Recall (defensive detection)
    lstm_bear_recall = recall_score(
        y_true, y_pred_lstm, labels=[regime_label_map["Bear"]], average="macro", zero_division=0
    )
    lr_bear_recall = recall_score(
        y_true, y_pred_lr, labels=[regime_label_map["Bear"]], average="macro", zero_division=0
    )
    persist_bear_recall = recall_score(
        y_true, y_pred_persist, labels=[regime_label_map["Bear"]], average="macro", zero_division=0
    )

    # Transition Recall (turning-point detection)
    def calculate_transition_recall(y_true_seq, y_pred_seq):
        true_transitions = 0
        correct_transitions = 0
        for i in range(1, len(y_true_seq)):
            if y_true_seq[i] != y_true_seq[i - 1]:
                true_transitions += 1
                if (y_pred_seq[i] == y_true_seq[i]) and (y_pred_seq[i - 1] == y_true_seq[i - 1]):
                    correct_transitions += 1
        return correct_transitions / true_transitions if true_transitions > 0 else 0.0

    lstm_transition_recall = calculate_transition_recall(y_true, y_pred_lstm)
    lr_transition_recall = calculate_transition_recall(y_true, y_pred_lr)
    persist_transition_recall = calculate_transition_recall(y_true, y_pred_persist)

    metrics_df = pd.DataFrame({
        "Model": ["Probability Model (LSTM)", "Linear Baseline (Logistic Regression)", "Common-Sense Baseline (Persistence)"],
        "Accuracy": [lstm_accuracy, lr_accuracy, persist_accuracy],
        "Macro-F1": [lstm_f1_macro, lr_f1_macro, persist_f1_macro],
        "Bear Recall": [lstm_bear_recall, lr_bear_recall, persist_bear_recall],
        "Transition Recall": [lstm_transition_recall, lr_transition_recall, persist_transition_recall],
    })
    return metrics_df


def fig_regime_probabilities_and_signal(prob_df: pd.DataFrame, sp500_prices: pd.Series):
    """
    Returns a matplotlib Figure with:
      - S&P 500 reconstructed price w/ dominant regime shading
      - stacked regime probabilities
      - conceptual equity allocation signal
    """
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(16, 12), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]}
    )

    ax1.plot(sp500_prices.index, sp500_prices.values, "k-", linewidth=0.8, label="S&P 500 (Rebased)")
    ax1.set_ylabel("S&P 500 (Base=100)")
    ax1.set_title("Regime Probabilities → Risk Posture (Exposure Mapping)", fontsize=14)
    ax1.legend(loc="upper left")

    dominant = prob_df[[c for c in prob_df.columns if c.startswith("P(")]].idxmax(axis=1)
    color_map = {
        "P(Bull-LowVol)": "lightgreen",
        "P(Bull-HighVol)": "lightsalmon",
        "P(Bear)": "lightcoral",
    }
    for i in range(len(dominant) - 1):
        ax1.axvspan(
            prob_df.index[i],
            prob_df.index[i + 1],
            alpha=0.2,
            color=color_map.get(dominant.iloc[i], "grey"),
            lw=0,
        )

    ax2.stackplot(
        prob_df.index,
        prob_df["P(Bull-LowVol)"],
        prob_df["P(Bull-HighVol)"],
        prob_df["P(Bear)"],
        labels=["Bull-LowVol", "Bull-HighVol", "Bear"],
        alpha=0.7,
    )
    ax2.set_ylabel("Probability")
    ax2.legend(loc="upper right")
    ax2.set_ylim(0, 1)

    ax3.plot(prob_df.index, prob_df["equity_weight"], label="Equity Weight (Policy Output)", linewidth=1.5)
    ax3.fill_between(prob_df.index, 0, prob_df["equity_weight"], alpha=0.6)
    ax3.set_ylabel("Equity Weight")
    ax3.set_xlabel("Date")
    ax3.set_ylim(0, 1)
    ax3.legend(loc="upper right")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


# ----------------------------
# Session state initialization
# ----------------------------
if "page" not in st.session_state:
    st.session_state.page = "Introduction"

# Data Acquisition & Regime Definition outputs
for k, default in [
    ("df_data", None),
    ("feature_columns", None),
    ("regime_label_map", None),
    ("start_date", "2005-01-01"),
    ("end_date", "2024-12-31"),
]:
    if k not in st.session_state:
        st.session_state[k] = default

# Data Preparation for LSTM outputs
for k, default in [
    ("scaler", None),
    ("X_train", None),
    ("y_train", None),
    ("X_test", None),
    ("y_test", None),
    ("y_train_oh", None),
    ("y_test_oh", None),
    ("n_classes", None),
    ("lookback", 60),
]:
    if k not in st.session_state:
        st.session_state[k] = default

# LSTM Model / Training outputs
for k, default in [
    ("model", None),
    ("history", None),
]:
    if k not in st.session_state:
        st.session_state[k] = default

# Evaluation & Baselines outputs
for k in ["y_prob_lstm", "y_pred_lstm", "y_pred_lr", "y_pred_persist", "comparison_metrics_df"]:
    if k not in st.session_state:
        st.session_state[k] = None

# Portfolio Signal & Visualization outputs
for k in ["prob_df", "sp500_test_prices"]:
    if k not in st.session_state:
        st.session_state[k] = None


# ----------------------------
# Sidebar navigation
# ----------------------------
st.sidebar.title("Learning Path")
page_options = [
    "Introduction",
    "1. Define Regimes (Labels)",
    "2. Build the Decision Information Set",
    "3. Probability Engine (What it Outputs)",
    "4. Fit + Overfitting Check",
    "5. Trust Check: Baselines + Turning Points",
    "6. Risk Policy: Probabilities → Exposure",
]
st.session_state.page = st.sidebar.selectbox(
    "Go to a section",
    page_options,
    index=page_options.index(st.session_state.page) if st.session_state.page in page_options else 0,
)

st.sidebar.markdown("---")
with st.sidebar.expander("Defaults (Recommended First Pass)", expanded=True):
    st.session_state.start_date = st.text_input(
        "Start date (YYYY-MM-DD)",
        st.session_state.start_date,
        help="Recommended first pass: keep defaults to maximize sample size while staying in modern market structure.",
    )
    st.session_state.end_date = st.text_input(
        "End date (YYYY-MM-DD)",
        st.session_state.end_date,
        help="End date is exclusive (data up to but not including this date). Keep default unless you have a reason.",
    )
    st.session_state.lookback = st.number_input(
        "Evidence window (days of history used)",
        min_value=20,
        max_value=252,
        value=int(st.session_state.lookback),
        step=5,
        help="Longer windows = slower, more stable decisions; shorter windows = faster, more reactive decisions.",
    )

st.sidebar.markdown("---")
st.sidebar.caption("**Non-technical framing:** This app is about building a **regime-based risk policy**. The model is a measurement tool; the policy is the decision lever.")


# ----------------------------
# Page helpers
# ----------------------------
def assumptions_box(lines):
    with st.expander("Assumptions & Watch-outs (read before interpreting outputs)", expanded=False):
        for line in lines:
            st.markdown(f"- {line}")


def checkpoint_question(prompt, options, correct_index, explanation_if_correct, explanation_if_incorrect):
    st.markdown("#### Quick checkpoint (30 seconds)")
    choice = st.radio(prompt, options, index=None)
    if choice is None:
        st.info("Pick an answer to see the explanation.")
        return
    if options.index(choice) == correct_index:
        st.success(f"Correct. {explanation_if_correct}")
    else:
        st.warning(f"Not quite. {explanation_if_incorrect}")


# ----------------------------
# Pages
# ----------------------------
if st.session_state.page == "Introduction":
    st.title("Market Sentinel: Regimes → Risk Policy (Concept XYZ)")
    st.subheader("A decision-focused walkthrough for CFA charterholders and investment professionals")

    st.markdown(
        "This application is designed to help you **identify market regimes** and translate them into a **disciplined risk posture**.\n\n"
        "The emphasis is not on coding. The emphasis is on:\n"
        "- how regimes are defined (the labeling rule),\n"
        "- how we test whether the approach adds value vs common-sense baselines, and\n"
        "- how probabilities translate into exposure choices."
    )

    st.info(
        "**What this is:** a learning tool for regime-based decision-making.\n\n"
        "**What this is not:** a fully-specified trading strategy or a backtested performance claim."
    )

    st.markdown("### What you will learn (decision-relevant)")
    st.markdown(
        "- **Regime labels:** what counts as Bull-LowVol vs Bull-HighVol vs Bear (and why definitions matter).\n"
        "- **Turning points:** why transition detection is often more P&L-critical than average-day accuracy.\n"
        "- **Risk policy mapping:** how to translate regime probabilities into exposure in a transparent way."
    )

    st.markdown("### Micro-example (finance-native)")
    st.markdown(
        "“In March 2020, the key question wasn’t whether volatility was elevated—it was whether the regime had shifted fast enough "
        "to justify cutting equity risk before drawdowns accelerated.”"
    )

    assumptions_box([
        "**No performance claims** are made. We are evaluating classification and decision mapping, not Sharpe ratio.",
        "Regimes are **labels we define** (a convention). Results depend on the definition.",
        "All outputs require documentation: no number should be interpreted without understanding how it was constructed.",
    ])

    checkpoint_question(
        prompt="Why do probabilities matter more than a single regime label for investment decisions?",
        options=[
            "Because probabilities allow position sizing and risk budgeting, not just yes/no calls.",
            "Because probabilities always guarantee higher returns.",
            "Because probabilities remove all model risk.",
        ],
        correct_index=0,
        explanation_if_correct="A probability surface can be mapped into exposure in a controlled, policy-driven way.",
        explanation_if_incorrect="In portfolio work, probabilities are useful because they support graded exposure decisions under uncertainty—not because they guarantee performance.",
    )

elif st.session_state.page == "1. Define Regimes (Labels)":
    st.title("1) Define Regimes: What Counts as ‘Bear’ vs ‘Bull’ Here?")
    st.markdown(
        "You are about to define the **labeling rule** that serves as the training target. "
        "This is your operational definition of market regimes for tactical risk posture."
    )

    st.markdown("#### Why this matters ")
    st.markdown(
        "- If your definition is too coarse, your regime tool will be too coarse.\n"
        "- If your definition embeds hindsight, your conclusions may not survive real-time use.\n"
        "- The label definition determines which errors are most costly (e.g., missing Bear conditions)."
    )

    with st.expander("Show the regime definition (math)", expanded=True):
        st.markdown("Regimes are defined using trailing 21-day S&P 500 returns and VIX relative to its full-sample median:")
        # IMPORTANT: Do not remove formulae
        st.latex(r"""
            \text{Regime}_t =
            \begin{cases}
                \text{Bull-LowVol} & \text{if } r_{21d,t} > 0 \land \text{VIX}_t \le \text{VIX}_{\text{median}} \\
                \text{Bull-HighVol} & \text{if } r_{21d,t} > 0 \land \text{VIX}_t > \text{VIX}_{\text{median}} \\
                \text{Bear} & \text{if } r_{21d,t} \le 0
            \end{cases}
        """)
        st.caption(
            "Interpretation: ‘Bear’ here is a **tactical proxy** (recent return ≤ 0), not an NBER recession label. "
            "VIX median is a stable long-run volatility reference; it has implications for real-time use."
        )

    assumptions_box([
        "VIX threshold uses **full-sample median** (stable benchmark, but not a rolling ex-ante estimate). Interpret accordingly.",
        "‘Yield spread’ here is a short-difference proxy, **not** the classic 10y–2y curve slope. Treat it as a rates-change signal.",
        "The regime label is a convention; alternative definitions may be equally defensible and should be stress-tested.",
    ])

    st.markdown("---")
    st.subheader("Load market data + compute regimes")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(
            "**Data proxies (intuition):**\n"
            "- S&P 500: market trend and realized volatility\n"
            "- VIX: risk sentiment / implied volatility\n"
            "- 10Y yield: rates regime / discount-rate pressure\n"
            "- Gold: flight-to-quality / alternative risk proxy"
        )
    with c2:
        st.markdown(
            "**What you should sanity-check:**\n"
            "- Regime counts (are Bear labels rare?)\n"
            "- Any missing-data gaps\n"
            "- Whether ‘Bull-HighVol’ appears during fragile rallies"
        )

    if st.button("Load data + label regimes"):
        with st.spinner("Acquiring data and engineering features..."):
            df_data, feature_columns, regime_label_map = acquire_and_engineer_features(
                start_date=st.session_state.start_date,
                end_date=st.session_state.end_date,
            )
            st.session_state.df_data = df_data
            st.session_state.feature_columns = feature_columns
            st.session_state.regime_label_map = regime_label_map

        st.success("Complete. You have a labeled regime dataset.")

        st.markdown("### What you’re looking at")
        st.write(f"**Dataset shape:** `{st.session_state.df_data.shape}`")
        st.write("**Regime prevalence (class balance):**")
        st.dataframe(st.session_state.df_data["regime"].value_counts().rename("Days").to_frame(), use_container_width=True)

        st.info(
            "Guardrail: When regimes persist (few transitions), **accuracy can look good even if the model misses turning points**. "
            "We will explicitly test for that later."
        )

        st.markdown("**Preview (features + regime label):**")
        st.dataframe(
            st.session_state.df_data[st.session_state.feature_columns + ["regime", "regime_encoded"]].head(),
            use_container_width=True,
        )
    else:
        st.info("Click **Load data + label regimes** to proceed.")

    checkpoint_question(
        prompt="If S&P 500 21-day return is positive, but VIX is above its long-run median, what regime label applies?",
        options=["Bull-LowVol", "Bull-HighVol", "Bear"],
        correct_index=1,
        explanation_if_correct="Positive trend with elevated volatility is treated as a fragile risk-on state.",
        explanation_if_incorrect="Under this definition, elevated VIX with positive trend implies Bull-HighVol (risk-on, but fragile).",
    )

elif st.session_state.page == "2. Build the Decision Information Set":
    st.title("2) Build the Decision Information Set (What you ‘knew’ as of each day)")
    st.markdown(
        "For each decision date *t*, the model sees an **evidence window** of the prior *T* days of signals and outputs regime probabilities for date *t*."
    )

    st.markdown("#### Why this matters ")
    st.markdown(
        "- This step enforces a real-time discipline: use only information that would have been observable before or at the decision date.\n"
        "- Evidence window length drives responsiveness vs stability.\n"
        "- A time-based split tests whether the approach generalizes to a later market environment."
    )

    assumptions_box([
        "Train/Test split uses a fixed cutoff: **2020-01-01**. This is meant to stress-test modern transitions and crisis behavior.",
        "Evidence windows use features up to *t-1* to avoid leakage into the prediction of *t*.",
        "Scaling is fit on the training set and applied to the test set to mimic forward use.",
    ])

    with st.expander("Show the concept", expanded=False):
        st.markdown(
            "Think of this as a rolling dossier:\n"
            "- At time *t*, you have the last *T* days of market evidence.\n"
            "- You ask: “Given this evidence, what is the probability we are in each regime today?”"
        )
        st.latex(r"X_t = \{ \text{features}_{t-T}, \ldots, \text{features}_{t-1} \}")
        st.latex(r"\hat{p}_t = \left[P(\text{Bull-LowVol}),\; P(\text{Bull-HighVol}),\; P(\text{Bear})\right]_t")

    st.markdown("---")
    st.subheader("Create evidence windows for learning + testing")

    if st.session_state.df_data is None:
        st.error("Please complete **1) Define Regimes** first.")
    else:
        st.caption(
            "Interpretation of the evidence window: "
            "Longer = slower, more stable policy decisions; shorter = faster, more reactive decisions."
        )

        if st.button("Build evidence windows"):
            LOOKBACK = int(st.session_state.lookback)

            train_mask = st.session_state.df_data.index < "2020-01-01"
            test_mask = st.session_state.df_data.index >= "2020-01-01"

            train_features_raw = st.session_state.df_data.loc[train_mask, st.session_state.feature_columns].values
            train_targets_raw = st.session_state.df_data.loc[train_mask, "regime_encoded"].values
            test_features_raw = st.session_state.df_data.loc[test_mask, st.session_state.feature_columns].values
            test_targets_raw = st.session_state.df_data.loc[test_mask, "regime_encoded"].values

            scaler = StandardScaler()
            train_features_scaled = scaler.fit_transform(train_features_raw)
            test_features_scaled = scaler.transform(test_features_raw)

            X_train, y_train = create_sequences(train_features_scaled, train_targets_raw, LOOKBACK)
            X_test, y_test = create_sequences(test_features_scaled, test_targets_raw, LOOKBACK)

            n_classes = len(st.session_state.regime_label_map)
            y_train_oh = to_categorical(y_train, num_classes=n_classes)
            y_test_oh = to_categorical(y_test, num_classes=n_classes)

            st.session_state.scaler = scaler
            st.session_state.X_train = X_train
            st.session_state.y_train = y_train
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.n_classes = n_classes
            st.session_state.y_train_oh = y_train_oh
            st.session_state.y_test_oh = y_test_oh

            st.success("Complete. The decision information set is now structured for sequential learning.")

            st.markdown("### Sanity checks (what these numbers mean)")
            st.write(f"**Training days (raw):** `{train_features_raw.shape[0]}`  |  **Testing days (raw):** `{test_features_raw.shape[0]}`")
            st.write(f"**Evidence window (T):** `{LOOKBACK}` days")
            st.write(f"**X_train shape:** `{X_train.shape}`  → `(decisions, evidence_window, signals)`")
            st.write(f"**X_test shape:** `{X_test.shape}`")
            st.write(f"**Number of regimes:** `{n_classes}`")
        else:
            st.info("Click **Build evidence windows** to proceed.")

    checkpoint_question(
        prompt="If you increase the evidence window from 60 to 120 days, what is the most likely impact on the decision tool?",
        options=[
            "More stable but slower-to-react regime probabilities (less sensitive to short shocks).",
            "Always higher returns regardless of market.",
            "Eliminates overfitting entirely.",
        ],
        correct_index=0,
        explanation_if_correct="A longer window typically smooths the decision context and reduces responsiveness to sudden changes.",
        explanation_if_incorrect="In regime tools, longer evidence tends to stabilize the signal but can delay reaction to rapid transitions.",
    )

elif st.session_state.page == "3. Probability Engine (What it Outputs)":
    st.title("3) Probability Engine: What It Outputs (in Investment Terms)")
    st.markdown(
        "This step initializes the model that turns evidence windows into **regime probabilities**. "
        "For investment work, the key output is **a probability vector**, not a hard label."
    )

    st.markdown("#### Why this matters ")
    st.markdown(
        "- Probabilities support **position sizing** and **risk budgeting**.\n"
        "- A hard label hides uncertainty (often the most important information).\n"
        "- We can map probabilities into exposure rules transparently (next section)."
    )

    assumptions_box([
        "The internal mechanics are not the decision lever. The decision lever is **how you map probabilities to exposure**.",
        "Transparency comes from (1) explicit labels, (2) explicit evaluation vs baselines, (3) explicit policy mapping, and (4) documented failure modes.",
    ])

    with st.expander("Show the output target (math)", expanded=False):
        st.latex(r"\hat{p}_t = \left[P(\text{Bull-LowVol}),\; P(\text{Bull-HighVol}),\; P(\text{Bear})\right]_t")
        st.caption("Interpretation: these are **belief weights** implied by the evidence window at date *t*.")

    st.markdown("---")
    st.subheader("Initialize probability engine")

    if st.session_state.X_train is None or st.session_state.n_classes is None:
        st.error("Please complete **2) Build the Decision Information Set** first.")
    else:
        if st.button("Initialize model"):
            input_shape = (st.session_state.X_train.shape[1], st.session_state.X_train.shape[2])
            model = build_lstm_model(input_shape=input_shape, n_classes=int(st.session_state.n_classes))
            st.session_state.model = model

            st.success("Initialized. You can now fit the model and evaluate it against baselines.")

            with st.expander("For the skeptical reader: model summary (transparency)", expanded=False):
                f = io.StringIO()
                with redirect_stdout(f):
                    model.summary()
                st.code(f.getvalue())
        else:
            st.info("Click **Initialize model** to proceed.")

    checkpoint_question(
        prompt="What does a result like P(Bear)=35% mean in decision terms?",
        options=[
            "There is meaningful uncertainty; a risk policy might tighten exposure depending on thresholds.",
            "We are definitely in a bear market.",
            "The model guarantees a 35% drawdown.",
        ],
        correct_index=0,
        explanation_if_correct="Probabilities are uncertainty-aware inputs to a policy rule (thresholds, sizing, hedges).",
        explanation_if_incorrect="Probabilities do not assert certainty, nor do they directly forecast returns/drawdowns; they represent belief given the evidence window.",
    )

elif st.session_state.page == "4. Fit + Overfitting Check":
    st.title("4) Fit the Model + Check for Overfitting (Finance Reality Check)")
    st.markdown(
        "Financial signals are noisy and non-stationary. This step fits the model **while monitoring generalization** using a validation split."
    )

    st.markdown("#### What you should look for while training")
    st.markdown(
        "- **Healthy**: validation improves and stabilizes alongside training.\n"
        "- **Overfitting**: training improves while validation stalls or deteriorates.\n"
        "- **Unstable**: validation is extremely erratic → treat the downstream policy signal with skepticism."
    )

    assumptions_box([
        "A ‘best’ model is chosen based on validation loss to reduce hindsight bias in selecting an attractive-looking fit.",
        "Training curves are diagnostics; they do not prove investability or robustness across regimes.",
    ])

    st.markdown("---")
    st.subheader("Fit model (with safeguards)")

    if st.session_state.X_train is None or st.session_state.y_train_oh is None or st.session_state.n_classes is None:
        st.error("Please complete **2) Build the Decision Information Set** first.")
    else:
        if st.button("Fit model now"):
            with st.spinner("Fitting the probability engine..."):
                input_shape = (st.session_state.X_train.shape[1], st.session_state.X_train.shape[2])
                model = build_lstm_model(input_shape=input_shape, n_classes=int(st.session_state.n_classes))

                tmp = tempfile.NamedTemporaryFile(suffix=".keras", delete=False)
                tmp_path = tmp.name
                tmp.close()

                history = train_model(
                    model,
                    st.session_state.X_train,
                    st.session_state.y_train_oh,
                    model_filepath=tmp_path,
                )

                model.load_weights(tmp_path)
                st.session_state.model = model
                st.session_state.history = history

            st.success("Complete. Now review the diagnostic curves before trusting anything downstream.")
            st.pyplot(fig_training_history(st.session_state.history), clear_figure=True)

            st.info(
                "Guardrail: If the validation curve deteriorates while training improves, "
                "treat the model as likely overfit and expect weaker transition detection out-of-sample."
            )
        else:
            st.info("Click **Fit model now** to train.")

    checkpoint_question(
        prompt="If training improves but validation worsens, what is the right interpretation for portfolio decision use?",
        options=[
            "The signal is likely fit to the past; treat it as unreliable out-of-sample without stronger guardrails.",
            "The model is proven investable.",
            "This guarantees higher transition recall.",
        ],
        correct_index=0,
        explanation_if_correct="This is the classic finance failure mode: the model matches historical noise patterns that don’t repeat.",
        explanation_if_incorrect="Strong in-sample fit is not evidence of decision reliability; validation deterioration is a red flag.",
    )

elif st.session_state.page == "5. Trust Check: Baselines + Turning Points":
    st.title("5) Trust Check: Does It Beat Common-Sense Baselines (and Catch Turning Points)?")
    st.markdown(
        "We compare the probability model against baselines that investment professionals recognize:\n"
        "- **Persistence**: tomorrow looks like today (regimes persist).\n"
        "- **Linear baseline**: transparent classifier on the same evidence, without sequence structure.\n\n"
        "We prioritize **Transition Recall** because turning points are often where drawdowns concentrate."
    )

    assumptions_box([
        "Accuracy can be inflated when regimes persist (few transitions). Do not treat accuracy as sufficient evidence of usefulness.",
        "Bear Recall aligns with defensive detection; Transition Recall aligns with turning-point detection.",
        "Metric differences may not be statistically significant; interpret as diagnostic evidence, not proof.",
    ])

    with st.expander("Metric definitions (investment translations)", expanded=False):
        st.markdown(
            "- **Accuracy:** fraction of correct days. Can be misleading if regimes are persistent.\n"
            "- **Macro-F1:** balances performance across regimes (treats each regime as important).\n"
            "- **Bear Recall:** ability to correctly identify Bear-labeled days (defensive detection).\n"
            "- **Transition Recall:** ability to correctly capture regime changes (turning-point detection)."
        )
        st.markdown(
            "**Why Transition Recall matters:** A model can be ‘accurate’ most days and still miss the few days that drive risk outcomes."
        )

    st.markdown("---")
    st.subheader("Compare vs benchmarks")

    if st.session_state.model is None or st.session_state.X_test is None or st.session_state.y_test is None:
        st.error("Please complete **4) Fit + Overfitting Check** and **2) Build the Decision Information Set** first.")
    else:
        if st.button("Run trust check"):
            with st.spinner("Evaluating probability model and baselines..."):
                # Probability model
                y_prob_lstm = st.session_state.model.predict(st.session_state.X_test, verbose=0)
                y_pred_lstm = np.argmax(y_prob_lstm, axis=1)

                # Linear baseline on flattened sequences
                X_train_flat = st.session_state.X_train.reshape(st.session_state.X_train.shape[0], -1)
                X_test_flat = st.session_state.X_test.reshape(st.session_state.X_test.shape[0], -1)

                lr_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
                lr_model.fit(X_train_flat, st.session_state.y_train)
                y_pred_lr = lr_model.predict(X_test_flat)

                # Persistence baseline
                y_pred_persist = np.roll(st.session_state.y_test, 1)
                y_pred_persist[0] = st.session_state.y_test[0]

                metrics_df = evaluate_models(
                    st.session_state.y_test,
                    y_pred_lstm,
                    y_pred_lr,
                    y_pred_persist,
                    st.session_state.regime_label_map,
                )

                st.session_state.y_prob_lstm = y_prob_lstm
                st.session_state.y_pred_lstm = y_pred_lstm
                st.session_state.y_pred_lr = y_pred_lr
                st.session_state.y_pred_persist = y_pred_persist
                st.session_state.comparison_metrics_df = metrics_df

            st.success("Complete. Now interpret results through a decision lens (not a leaderboard lens).")

            # Highlight decision-critical metrics
            st.markdown("### Decision-critical metrics (what to focus on)")
            dfm = st.session_state.comparison_metrics_df.copy()
            # Pull the probability model row
            prob_row = dfm[dfm["Model"].str.contains("Probability Model")].iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Transition Recall (turning points)", f"{prob_row['Transition Recall']:.3f}")
            c2.metric("Bear Recall (defensive detection)", f"{prob_row['Bear Recall']:.3f}")
            c3.metric("Macro-F1 (balanced)", f"{prob_row['Macro-F1']:.3f}")

            st.info(
                "Decision translation: If **Transition Recall** is not meaningfully better than Persistence, "
                "a regime-based risk overlay may not tighten exposure early enough during regime shifts."
            )

            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("### Confusion matrix (probability model)")
                st.caption(
                    "How to read: diagonal = correct. Pay special attention to **Bear misclassified as Bull** (costly risk)."
                )
                fig_cm = plt.figure(figsize=(7, 5))
                cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred_lstm)
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=list(st.session_state.regime_label_map.keys()),
                    yticklabels=list(st.session_state.regime_label_map.keys()),
                )
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                plt.title("Confusion Matrix: Where the Model Makes Mistakes")
                st.pyplot(fig_cm, clear_figure=True)

                st.warning(
                    "Guardrail: False negatives on **Bear** are typically more costly than false positives, "
                    "because they can leave portfolios overexposed into drawdowns."
                )

            with col2:
                st.markdown("### Metrics table (rounded)")
                st.dataframe(dfm.set_index("Model").round(3), use_container_width=True)

                with st.expander("Common misconceptions (watch-outs)", expanded=False):
                    st.markdown(
                        "- “High accuracy means investable.” **No**—accuracy can be inflated when regimes persist.\n"
                        "- “Probabilities predict returns.” **No**—they represent regime belief given evidence.\n"
                        "- “One good period proves robustness.” **No**—financial dynamics shift; stress tests matter."
                    )

            st.markdown("### Comparison by metric (diagnostic)")
            metrics_to_plot = ["Transition Recall", "Bear Recall", "Macro-F1", "Accuracy"]
            fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(18, 4), sharey=True)
            for i, metric in enumerate(metrics_to_plot):
                sns.barplot(x="Model", y=metric, data=dfm, ax=axes[i])
                axes[i].set_title(metric)
                axes[i].set_ylim(0, 1)
                axes[i].set_ylabel("")
                axes[i].tick_params(axis="x", rotation=20)
            axes[0].set_ylabel("Score")
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)

        else:
            st.info("Click **Run trust check** to evaluate vs baselines.")

    checkpoint_question(
        prompt="Why can a model have high accuracy yet still be poor for risk management?",
        options=[
            "Because regimes persist; the model can be right on easy days but miss regime transitions that drive drawdowns.",
            "Because accuracy is mathematically undefined.",
            "Because higher accuracy always implies lower Bear Recall.",
        ],
        correct_index=0,
        explanation_if_correct="Accuracy can be dominated by the majority regime; turning points may remain undetected.",
        explanation_if_incorrect="In regime problems, you can be ‘right most days’ and still fail at transitions—the decision-critical moments.",
    )

elif st.session_state.page == "6. Risk Policy: Probabilities → Exposure":
    st.title("6) Risk Policy: Translate Probabilities into Exposure (Transparent Mapping)")
    st.markdown(
        "This section turns regime probabilities into an **exposure policy**. "
        "This is where investment judgment enters explicitly: you choose how cautious to be when volatility is high."
    )

    st.warning(
        "Guardrail: This is an **exposure mapping**, not a performance-optimized trading strategy. "
        "No transaction costs, slippage, taxes, turnover limits, or mandate constraints are modeled here."
    )

    st.markdown("#### Why this matters ")
    st.markdown(
        "- A probability model is only useful if it is mapped into an implementable policy.\n"
        "- The mapping should reflect risk appetite, drawdown tolerance, and constraints.\n"
        "- Transparent mapping is how you reduce ‘black box’ discomfort."
    )

    assumptions_box([
        "The policy below is intentionally simple for pedagogy. In practice you would add constraints: max turnover, smoothing, drawdown triggers, hedging overlays, etc.",
        "If the model’s probabilities oscillate rapidly, the implied policy may generate high turnover. Consider smoothing as a risk control.",
    ])

    # Risk appetite calibration
    st.markdown("---")
    st.subheader("Set risk appetite (policy parameter)")
    bull_highvol_weight = st.slider(
        "Equity exposure in Bull-HighVol (risk-on but fragile)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="0.2 = highly defensive, 0.5 = balanced, 0.8 = aggressive. This is a policy choice, not a model output.",
    )
    st.caption(
        "Interpretation: when markets are ‘risk-on but fragile,’ this parameter controls how much equity exposure you keep."
    )

    with st.expander("Show the exposure mapping formula (math)", expanded=True):
        # IMPORTANT: Do not remove formulae
        st.markdown("Base mapping used in this app:")
        st.latex(
            r"\text{Equity Weight} = P(\mathrm{Bull\text{-}LowVol}) + 0.5 \cdot P(\mathrm{Bull\text{-}HighVol})"
        )
        st.markdown("Policy-adjusted mapping with your risk appetite parameter:")
        st.latex(
            rf"\text{{Equity Weight}} = P(\mathrm{{Bull\text{{-}}LowVol}}) + {bull_highvol_weight:.2f} \cdot P(\mathrm{{Bull\text{{-}}HighVol}})"
        )

    st.markdown("---")
    st.subheader("Generate exposure policy and interpret it")

    if st.session_state.y_prob_lstm is None or st.session_state.df_data is None:
        st.error("Please complete **5) Trust Check** and **1) Define Regimes** first.")
    else:
        # Scenario / vignette mode
        vignette = st.selectbox(
            "Optional: focus on a historical vignette (intuition builder)",
            [
                "Full test period (2020+)",
                "COVID shock window (2020-02 to 2020-06)",
                "Inflation / rate shock window (2022-01 to 2022-12)",
                "Recent window (last ~18 months of test set)",
            ],
            help="Use vignettes to build intuition: do probabilities rise when you would expect, and does exposure tighten?",
        )

        if st.button("Generate policy output"):
            with st.spinner("Deriving probabilities → exposure policy..."):
                test_mask = st.session_state.df_data.index >= "2020-01-01"
                lookback = int(st.session_state.lookback)

                # Probabilities dataframe
                prob_df = pd.DataFrame(
                    st.session_state.y_prob_lstm,
                    columns=[f"P({label})" for label in list(st.session_state.regime_label_map.keys())],
                    index=st.session_state.df_data.loc[test_mask].index[lookback:],
                )

                # Policy mapping (user-calibrated)
                prob_df["equity_weight"] = prob_df["P(Bull-LowVol)"] + (bull_highvol_weight * prob_df["P(Bull-HighVol)"])
                prob_df["bond_weight"] = 1 - prob_df["equity_weight"]

                # Reconstruct S&P 500 "price" from daily returns on test period (base = 100)
                sp500_test_returns = st.session_state.df_data.loc[test_mask, "sp500_ret"].iloc[lookback:]
                sp500_test_prices = (1 + sp500_test_returns).cumprod() * 100
                sp500_test_prices = sp500_test_prices.reindex(prob_df.index).ffill()

                st.session_state.prob_df = prob_df
                st.session_state.sp500_test_prices = sp500_test_prices

            st.success("Complete. Now interpret the policy output carefully (with assumptions in view).")

            # Apply vignette filter
            p = st.session_state.prob_df.copy()
            s = st.session_state.sp500_test_prices.copy()

            if vignette == "COVID shock window (2020-02 to 2020-06)":
                p = p.loc["2020-02-01":"2020-06-30"]
                s = s.loc[p.index.min():p.index.max()]
            elif vignette == "Inflation / rate shock window (2022-01 to 2022-12)":
                p = p.loc["2022-01-01":"2022-12-31"]
                s = s.loc[p.index.min():p.index.max()]
            elif vignette == "Recent window (last ~18 months of test set)":
                if len(p) > 21 * 18:
                    p = p.iloc[-21 * 18:]
                    s = s.loc[p.index.min():p.index.max()]

            st.pyplot(fig_regime_probabilities_and_signal(p, s), clear_figure=True)

            st.markdown("### How to read this chart (CFA-level)")
            st.markdown(
                "- **Probabilities panel:** rising P(Bear) signals increasing defensive regime belief.\n"
                "- **Equity weight panel:** this is your policy output; it should tighten as defensive probability rises.\n"
                "- **Shading:** dominant regime belief; watch for frequent flipping (turnover risk)."
            )

            st.info(
                "Decision translation:\n"
                "- If **P(Bear)** rises materially, consider tightening risk (lower equity beta, add hedges, raise cash, rotate defensively).\n"
                "- If **P(Bull-HighVol)** dominates, consider staying invested but reducing risk concentration (position sizing, factor tilts).\n"
                "- If the equity weight oscillates rapidly, consider adding smoothing/turnover controls before implementation."
            )

            # Average equity allocation by actual regime (sanity check)
            st.markdown("### Sanity check: average equity weight by actual labeled regime (test set slice shown)")
            test_mask = st.session_state.df_data.index >= "2020-01-01"
            lookback = int(st.session_state.lookback)
            test_actual_regimes = st.session_state.df_data.loc[test_mask, "regime_encoded"].iloc[lookback:]
            tmp = st.session_state.prob_df.copy()
            tmp["actual_regime_encoded"] = test_actual_regimes.values

            reverse_regime_map = {v: k for k, v in st.session_state.regime_label_map.items()}
            rows = []
            for encoded_regime, regime_name in reverse_regime_map.items():
                m = tmp["actual_regime_encoded"] == encoded_regime
                avg_eq = tmp.loc[m, "equity_weight"].mean()
                rows.append({"Actual Regime (Label)": regime_name, "Average Equity Weight": avg_eq})
            out = pd.DataFrame(rows).sort_values("Average Equity Weight", ascending=False)
            st.dataframe(out, use_container_width=True)

            st.warning(
                "Guardrail: If the ranking is not intuitive (e.g., Bear has higher average equity weight than Bull states), "
                "revisit either (a) the probability quality, or (b) the policy mapping parameters."
            )

        else:
            st.info("Click **Generate policy output** to produce probabilities → exposure mapping.")

    checkpoint_question(
        prompt="What is the most defensible interpretation of the 0.5 (or slider) applied to Bull-HighVol?",
        options=[
            "A risk appetite parameter: how much equity you choose to carry when conditions are risk-on but fragile.",
            "A proven optimal coefficient that maximizes Sharpe ratio.",
            "A guarantee that drawdowns will be cut in half.",
        ],
        correct_index=0,
        explanation_if_correct="It is explicitly a policy choice reflecting mandate and drawdown tolerance.",
        explanation_if_incorrect="This coefficient is not proven optimal by the app; it is a transparent risk policy parameter.",
    )

# License
st.caption('''
---
## QuantUniversity License

© QuantUniversity 2026  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
''')

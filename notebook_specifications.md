
# Market Regime Prediction with LSTMs: A Quantitative Analyst's Workflow

## Introduction: Navigating Volatile Markets with Deep Learning

As a **CFA Charterholder and Quantitative Analyst** at "Alpha Strategies Asset Management," my core responsibility is to develop sophisticated models that enhance our tactical asset allocation and risk management capabilities. Financial markets are rarely static; they move through distinct behavioral phases—periods of calm growth (Bull-Low Volatility), periods of growth with increased uncertainty (Bull-High Volatility), and periods of decline (Bear Markets). Recognizing and, more critically, anticipating shifts between these **market regimes** is paramount for optimizing portfolio performance and protecting client capital.

Traditional rule-based methods or simpler statistical models often struggle to capture the complex, non-linear, and sequential dependencies that characterize these regime transitions. This is where advanced machine learning, specifically **Long Short-Term Memory (LSTM) neural networks**, offers a significant advantage. LSTMs are designed to process and learn from sequences of data, making them ideal for understanding the temporal dynamics of financial markets.

In this notebook, I'll walk through my workflow to build, train, and evaluate an LSTM model to classify market regimes using a rich set of financial time-series data. The goal is to develop a data-driven signal that can inform our investment decisions, leading to potentially better risk-adjusted returns and more robust portfolio management strategies.

---

### Installing Required Libraries

First, I need to ensure all necessary Python libraries are installed. These include `tensorflow` and `keras` for building the deep learning model, `pandas` and `numpy` for data manipulation, `yfinance` for market data, `scikit-learn` for preprocessing and baseline models, and `matplotlib` and `seaborn` for visualizations.

```python
!pip install tensorflow==2.12.0 pandas==1.5.3 numpy==1.21.6 scikit-learn==1.2.2 yfinance==0.2.18 matplotlib==3.5.3 seaborn==0.12.2
```

### Importing Required Dependencies

Next, I'll import all the libraries that will be used throughout this analysis. Organizing imports at the beginning helps manage dependencies and ensures a clean coding environment.

```python
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score

import matplotlib.pyplot as plt
import seaborn as sns
```

---

## Section 1: Acquiring Data and Defining Market Regimes

As a quantitative analyst, my first step is to gather the relevant financial data and explicitly define the market regimes I aim to predict. For this task, I'll collect historical data for the S&P 500 (`^GSPC`), VIX (`^VIX`), 10-year Treasury yield (`^TNX`), and Gold (`GLD`). From these raw series, I'll compute various derived features that capture different aspects of market behavior, such as momentum, volatility, and risk sentiment.

A crucial part of this process is defining our target variable: the market regime itself. I've chosen to classify the market into three distinct regimes: "Bull-Low Volatility," "Bull-High Volatility," and "Bear," based on observable S&P 500 trailing returns and the VIX relative to its historical median. This ensures that my regime definitions are based on information available at the time, avoiding **look-ahead bias**, a critical pitfall in financial modeling.

The mathematical definition for our three market regimes at time $t$ is:

$$
\text{Regime}_t = \begin{cases}
    \text{Bull-LowVol} & \text{if } r_{21d,t} > 0 \land \text{VIX}_t \le \text{VIX}_{\text{median}} \\
    \text{Bull-HighVol} & \text{if } r_{21d,t} > 0 \land \text{VIX}_t > \text{VIX}_{\text{median}} \\
    \text{Bear} & \text{if } r_{21d,t} \le 0
\end{cases}
$$

where $r_{21d,t}$ is the trailing 21-day S&P 500 return, and $\text{VIX}_{\text{median}}$ is the full-sample median VIX value.

```python
def acquire_and_engineer_features(start_date='2005-01-01', end_date='2024-12-31'):
    """
    Acquires raw financial data and engineers features and regime labels.
    """
    # 1. Acquire raw data
    sp500 = yf.download('^GSPC', start=start_date, end=end_date)
    vix = yf.download('^VIX', start=start_date, end=end_date)
    tnx = yf.download('^TNX', start=start_date, end=end_date)
    gld = yf.download('GLD', start=start_date, end=end_date)

    # Align dataframes and create a master DF
    df = pd.DataFrame(index=sp500.index)
    
    # 2. Engineer Features
    df['sp500_ret'] = sp500['Adj Close'].pct_change()
    df['sp500_ret_5d'] = sp500['Adj Close'].pct_change(5)
    df['sp500_ret_21d'] = sp500['Adj Close'].pct_change(21)
    df['realized_vol_21d'] = (df['sp500_ret'].rolling(21).std() * np.sqrt(252))
    
    df['vix'] = vix['Adj Close']
    df['vix_change_5d'] = df['vix'].pct_change(5)
    
    df['yield_10y'] = tnx['Adj Close']
    # Approximate yield spread (10Y-2Y) using 10Y and its shifted value
    df['yield_spread'] = df['yield_10y'] - df['yield_10y'].shift(5) # Shift by 5 for a rough proxy of short-term rates
    
    df['gold_ret'] = gld['Adj Close'].pct_change()

    # 3. Define and encode regime labels
    vix_median = df['vix'].median()
    
    df['regime'] = np.where(
        df['sp500_ret_21d'] <= 0, 'Bear',
        np.where(df['vix'] > vix_median, 'Bull-HighVol', 'Bull-LowVol')
    )
    
    regime_map = {'Bull-LowVol': 0, 'Bull-HighVol': 1, 'Bear': 2}
    df['regime_encoded'] = df['regime'].map(regime_map)

    # Remove NaN values created by rolling windows and pct_change
    df = df.dropna()
    
    # Select features for the model
    feature_cols = [
        'sp500_ret', 'sp500_ret_5d', 'sp500_ret_21d',
        'realized_vol_21d', 'vix', 'vix_change_5d',
        'yield_10y', 'yield_spread', 'gold_ret'
    ]
    
    return df, feature_cols, regime_map

# Execute data acquisition and feature engineering
df_data, feature_columns, regime_label_map = acquire_and_engineer_features()

print(f"Dataset shape: {df_data.shape}")
print(f"Features used: {feature_columns}")
print("\nRegime distribution (encoded labels):")
print(df_data['regime_encoded'].value_counts())
print("\nFirst 5 rows of data with features and regimes:")
print(df_data[feature_columns + ['regime', 'regime_encoded']].head())
```

### Explanation of Execution
The output confirms the shape of our comprehensive dataset and the distribution of the three market regimes. As a quant, understanding this distribution is important; for instance, if one regime is significantly underrepresented, it might pose challenges for the model during training. The defined features cover various market aspects, from returns and volatility to intermarket relationships (yield spread, gold returns), providing a holistic view of the market state. The clear regime definition ensures we are predicting a well-understood financial state.

---

## Section 2: Preparing Sequential Data for LSTM Input

LSTMs are designed to process sequences. To feed our multivariate time-series data into the LSTM model, I need to transform it into fixed-length "sliding windows" of past observations. This means for each day $t$, the model will look at the previous $T$ days' features ($X_{t-T}, \dots, X_{t-1}$) to predict the regime at day $t$ ($Y_t$). This creates 3D tensors of shape `(N_samples, T_lookback, N_features)`, which is the standard input format for Keras LSTMs.

Choosing an appropriate `lookback` window is crucial. A 60-day lookback (approximately three months) is chosen as it typically captures enough recent market history for a quantitative analyst to make informed decisions about short-to-medium term market behavior, without becoming too computationally intensive or dilute recent information with overly stale data.

**Crucial warning on look-ahead bias**: When constructing these sequences, I must strictly ensure that the features for predicting regime $Y_t$ only include data up to $X_{t-1}$. The target variable $Y_t$ itself, based on $r_{21d,t}$ and VIX$_t$, uses information available *up to and including* day $t$. This is fine because we are **classifying the current regime**, not predicting a future one. The LSTM's power lies in making this classification from a sequence of *past features* before the current regime label itself is computed.

```python
def create_sequences(features, targets, lookback=60):
    """
    Creates sliding-window sequences for LSTM input.

    Args:
        features: (n_days, n_features) array of input features.
        targets: (n_days,) array of target regime labels.
        lookback: Number of past days per sequence.

    Returns:
        X: (n_samples, lookback, n_features) 3D tensor for LSTM input.
        y: (n_samples,) 1D array of target labels.
    """
    X, y = [], []
    for i in range(lookback, len(features)):
        X.append(features[i-lookback:i])  # Past 'lookback' days' features
        y.append(targets[i])             # Current day's regime (target)
    return np.array(X), np.array(y)

# Define lookback window
LOOKBACK = 60 # 60 trading days (~3 months)

# Split data into training and testing sets (temporal split)
# Training: up to end of 2019
# Testing: 2020 onwards
train_mask = df_data.index < '2020-01-01'
test_mask = df_data.index >= '2020-01-01'

# Separate features and targets
feature_cols = feature_columns # Use the columns defined in Section 1

train_features_raw = df_data.loc[train_mask, feature_cols].values
train_targets_raw = df_data.loc[train_mask, 'regime_encoded'].values
test_features_raw = df_data.loc[test_mask, feature_cols].values
test_targets_raw = df_data.loc[test_mask, 'regime_encoded'].values

# Scale features using StandardScaler (fit only on training data)
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features_raw)
test_features_scaled = scaler.transform(test_features_raw) # Transform test using training fit

# Create sequences
X_train, y_train = create_sequences(train_features_scaled, train_targets_raw, LOOKBACK)
X_test, y_test = create_sequences(test_features_scaled, test_targets_raw, LOOKBACK)

# One-hot encode targets for categorical cross-entropy loss
n_classes = len(regime_label_map)
y_train_oh = to_categorical(y_train, num_classes=n_classes)
y_test_oh = to_categorical(y_test, num_classes=n_classes)

print(f"Original training features shape: {train_features_raw.shape}")
print(f"Original testing features shape: {test_features_raw.shape}")
print(f"X_train (sequence) shape: {X_train.shape} (N_samples, T_lookback, N_features)")
print(f"y_train (encoded) shape: {y_train_oh.shape}")
print(f"X_test (sequence) shape: {X_test.shape}")
print(f"y_test (encoded) shape: {y_test_oh.shape}")
print(f"Number of classes: {n_classes}")
```

### Explanation of Execution
The output demonstrates that the data has been successfully transformed into the 3D tensor format required by LSTMs. `X_train` and `X_test` now contain sequences of 60 days of financial features, ready for the model. The `y_train_oh` and `y_test_oh` are one-hot encoded to align with the categorical nature of our regime classification problem, which is essential for the `categorical_crossentropy` loss function. This prepares the data for the next step: designing the LSTM model.

---

## Section 3: Designing the LSTM Model Architecture

Now, I'll design the LSTM neural network architecture using Keras. The model will consist of multiple LSTM layers to capture complex temporal patterns, followed by Dense layers for classification. Key components include:
*   **LSTM Layers:** These are the core of the model, processing sequences and maintaining an internal state (cell state and hidden state) that allows them to remember relevant information over long periods, while selectively forgetting irrelevant details.
*   **Dropout Regularization:** Financial data is noisy and prone to overfitting. Dropout randomly sets a fraction of neuron outputs to zero during training, preventing complex co-adaptations between neurons and encouraging the network to learn more robust, generalized features.
    $$
    h_t' = m \odot h_t, \quad m_j \sim \text{Bernoulli}(1 - p)
    $$
    where $p$ is the dropout rate. This is analogous to portfolio diversification, preventing reliance on any single "stock" (feature or neuron).
*   **Batch Normalization:** This layer normalizes the activations of the previous layer, helping to stabilize and speed up training, especially in deep networks.
*   **Dense Layers with Softmax Activation:** The final layer uses a `softmax` activation function to output probabilities for each of our three market regimes. `softmax` ensures that the predicted probabilities sum to 1.
    $$
    P(\text{regime} = k \mid X_{t-T:t}) = \text{softmax}(W_{\text{out}} \cdot \text{ReLU}(W_h h_T + b_h) + b_{\text{out}})_k
    $$
    where $h_T$ is the final LSTM hidden state, and $k$ represents one of the regime classes.
*   **Categorical Cross-Entropy Loss:** This is the standard loss function for multi-class classification problems with one-hot encoded targets. It quantifies the difference between the predicted probability distribution and the true distribution.
    $$
    \mathcal{L} = - \frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} y_{ik} \log p_{ik}
    $$
    where $y_{ik}$ is the one-hot encoded true regime and $p_{ik}$ is the predicted probability for sample $i$ and class $k$.

```python
def build_lstm_model(input_shape, n_classes, learning_rate=0.001, dropout_rate_lstm=0.3, dropout_rate_dense=0.2):
    """
    Builds a sequential LSTM model for market regime classification.
    """
    model = Sequential([
        # First LSTM layer: returns sequences for stacking with another LSTM layer
        LSTM(units=64, input_shape=input_shape, return_sequences=True),
        Dropout(dropout_rate_lstm),
        BatchNormalization(),
        
        # Second LSTM layer: returns final hidden state for classification head
        LSTM(units=32, return_sequences=False),
        Dropout(dropout_rate_lstm),
        BatchNormalization(),
        
        # Classification head
        Dense(units=16, activation='relu'),
        Dropout(dropout_rate_dense),
        Dense(units=n_classes, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Define input shape for the LSTM model
input_shape = (X_train.shape[1], X_train.shape[2]) # (LOOKBACK, N_features)

# Build the model
model = build_lstm_model(input_shape=input_shape, n_classes=n_classes)

# Display model summary
model.summary()
```

### Explanation of Execution
The model summary provides a clear overview of the LSTM's architecture, including the number of layers, output shapes, and the total number of trainable parameters. This is essential for a quant to verify that the network's complexity is appropriate for the problem and to understand the computational footprint. The `softmax` output layer ensures that the model provides probabilities for each regime, which is valuable for risk-sensitive decision-making beyond a simple classification.

---

## Section 4: Training the LSTM Model with Regularization

Training the model involves feeding it the prepared sequences and allowing it to learn the patterns that distinguish market regimes. For financial time series, preventing **overfitting** is critical. Overfitting occurs when a model learns the training data too well, including its noise, and performs poorly on unseen data. To combat this, I'll employ several regularization techniques:

*   **Early Stopping:** This callback monitors the validation loss during training and halts the training process if validation loss stops improving for a specified number of epochs (`patience`). This prevents the model from continuing to memorize the training data.
*   **ReduceLROnPlateau:** This callback reduces the learning rate when the validation loss plateaus, allowing the model to make finer adjustments and converge more effectively.
*   **Model Checkpoint:** This callback saves the model weights (or the entire model) corresponding to the best validation performance, ensuring that I retain the most effective version of the model.

After training, I'll visualize the training and validation loss/accuracy curves. These plots are the most important diagnostics in deep learning, allowing me to identify overfitting (divergence between training and validation curves) and confirm that early stopping has been effective.

```python
def train_model(model, X_train, y_train_oh, epochs=100, batch_size=64, validation_split=0.2, model_filepath='best_lstm_regime.keras'):
    """
    Trains the LSTM model with specified callbacks and parameters.
    """
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        ModelCheckpoint(model_filepath, save_best_only=True, monitor='val_loss', verbose=1)
    ]
    
    history = model.fit(
        X_train, y_train_oh,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split, # Use last 20% of training data as validation
        callbacks=callbacks,
        verbose=1
    )
    return history

def plot_training_history(history, filename='lstm_training_curves.png'):
    """
    Plots training and validation loss and accuracy curves.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss curves
    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Val Loss')
    ax1.set_title('Loss Curves', fontsize=12)
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('Loss', fontsize=10)
    ax1.legend()
    
    # Plot accuracy curves
    ax2.plot(history.history['accuracy'], label='Train Acc')
    ax2.plot(history.history['val_accuracy'], label='Val Acc')
    ax2.set_title('Accuracy Curves', fontsize=12)
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Accuracy', fontsize=10)
    ax2.legend()
    
    plt.suptitle('LSTM Training History', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, dpi=150)
    plt.show()

# Train the model
history = train_model(model, X_train, y_train_oh)

# Plot training history
plot_training_history(history)
```

### Explanation of Execution
The training logs show the progress across epochs, with callbacks intervening as needed. The plotted curves visually confirm whether the model is learning effectively and if overfitting is being managed. For a quant, seeing the validation loss stabilize or increase after a certain point signals that early stopping correctly identified the optimal training duration, preventing the model from becoming too specialized to the training data noise. This ensures the model is more likely to generalize well to unseen market conditions.

---

## Section 5: Baseline Comparison and Performance Evaluation

After training the LSTM, it's crucial to evaluate its performance and compare it against simpler, established baselines. This helps determine if the complexity of a deep learning model is justified by a tangible improvement in prediction capability.

I'll compare the LSTM against two baselines:
1.  **Persistence Model:** This simple model predicts that "tomorrow's regime is today's regime." Due to the inherent stickiness of market regimes, this baseline often achieves surprisingly high overall accuracy, but it fails to detect regime *transitions*.
2.  **Logistic Regression Model:** A traditional machine learning model applied to flattened (non-sequential) features, providing a benchmark for non-sequential, linear classification.

While overall accuracy is a common metric, for market regime prediction, it can be misleading due to regime persistence. The most critical metric for an asset manager is **Transition Recall**, which measures the model's ability to correctly identify actual shifts from one regime to another.

$$
\text{Transition Recall} = \frac{\text{Correctly predicted regime changes}}{\text{Total actual regime changes}}
$$

A high Transition Recall (e.g., > 50%) is what differentiates a truly useful model from a trivial "predict the current regime" baseline. I will also look at Macro-F1 (for balanced performance across classes) and Bear Recall (for detection of critical downturns).

```python
def evaluate_models(y_true, y_pred_lstm, y_pred_lr, y_pred_persist, regime_label_map, X_test):
    """
    Evaluates LSTM and baseline models and prints comparison metrics.
    Calculates and prints Transition Recall.
    """
    target_names = list(regime_label_map.keys())
    
    print("--- LSTM Performance ---")
    print(classification_report(y_true, y_pred_lstm, target_names=target_names, zero_division=0))
    lstm_accuracy = accuracy_score(y_true, y_pred_lstm)
    lstm_f1_macro = f1_score(y_true, y_pred_lstm, average='macro', zero_division=0)
    lstm_bear_recall = recall_score(y_true, y_pred_lstm, labels=[regime_label_map['Bear']], average='macro', zero_division=0)
    print(f"LSTM Overall Accuracy: {lstm_accuracy:.3f}")
    print(f"LSTM Macro-F1: {lstm_f1_macro:.3f}")
    print(f"LSTM Bear Recall: {lstm_bear_recall:.3f}")

    print("\n--- Logistic Regression Performance ---")
    print(classification_report(y_true, y_pred_lr, target_names=target_names, zero_division=0))
    lr_accuracy = accuracy_score(y_true, y_pred_lr)
    lr_f1_macro = f1_score(y_true, y_pred_lr, average='macro', zero_division=0)
    lr_bear_recall = recall_score(y_true, y_pred_lr, labels=[regime_label_map['Bear']], average='macro', zero_division=0)
    print(f"Logistic Regression Overall Accuracy: {lr_accuracy:.3f}")
    print(f"Logistic Regression Macro-F1: {lr_f1_macro:.3f}")
    print(f"Logistic Regression Bear Recall: {lr_bear_recall:.3f}")

    print("\n--- Persistence Model Performance ---")
    persist_accuracy = accuracy_score(y_true, y_pred_persist)
    persist_f1_macro = f1_score(y_true, y_pred_persist, average='macro', zero_division=0)
    persist_bear_recall = recall_score(y_true, y_pred_persist, labels=[regime_label_map['Bear']], average='macro', zero_division=0)
    print(f"Persistence Overall Accuracy: {persist_accuracy:.3f}")
    print(f"Persistence Macro-F1: {persist_f1_macro:.3f}")
    print(f"Persistence Bear Recall: {persist_bear_recall:.3f}")

    # Calculate Transition Recall for all models
    def calculate_transition_recall(y_true_seq, y_pred_seq):
        true_transitions = 0
        correct_transitions = 0
        for i in range(1, len(y_true_seq)):
            if y_true_seq[i] != y_true_seq[i-1]: # Actual regime change
                true_transitions += 1
                if y_pred_seq[i] == y_true_seq[i] and y_pred_seq[i-1] == y_true_seq[i-1]: # Correctly predicted current and previous regime
                     correct_transitions += 1 # This is a simplified transition detection; more advanced would look at P(change)
        return correct_transitions / true_transitions if true_transitions > 0 else 0

    lstm_transition_recall = calculate_transition_recall(y_true, y_pred_lstm)
    lr_transition_recall = calculate_transition_recall(y_true, y_pred_lr)
    persist_transition_recall = calculate_transition_recall(y_true, y_pred_persist)

    print(f"\nLSTM Transition Recall: {lstm_transition_recall:.3f}")
    print(f"Logistic Regression Transition Recall: {lr_transition_recall:.3f}")
    print(f"Persistence Transition Recall: {persist_transition_recall:.3f}")
    
    # Store metrics for bar chart
    metrics_df = pd.DataFrame({
        'Model': ['LSTM', 'Logistic Regression', 'Persistence'],
        'Accuracy': [lstm_accuracy, lr_accuracy, persist_accuracy],
        'Macro-F1': [lstm_f1_macro, lr_f1_macro, persist_f1_macro],
        'Bear Recall': [lstm_bear_recall, lr_bear_recall, persist_bear_recall],
        'Transition Recall': [lstm_transition_recall, lr_transition_recall, persist_transition_recall]
    })
    return metrics_df

# Reload the best model weights
model.load_weights('best_lstm_regime.keras')

# LSTM predictions
y_prob_lstm = model.predict(X_test)
y_pred_lstm = np.argmax(y_prob_lstm, axis=1)

# Baseline 1: Logistic Regression on flattened features
# Flatten X_train and X_test for Logistic Regression
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)
lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr_model.fit(X_train_flat, y_train)
y_pred_lr = lr_model.predict(X_test_flat)

# Baseline 2: Persistence (predict previous day's regime)
y_pred_persist = np.roll(y_test, 1) # Shift by 1 to get previous day's regime
y_pred_persist[0] = y_test[0] # The first prediction can't use previous, so use actual

# Evaluate models
comparison_metrics_df = evaluate_models(y_test, y_pred_lstm, y_pred_lr, y_pred_persist, regime_label_map, X_test)

# Plot Confusion Matrix for LSTM
plt.figure(figsize=(8, 6))
cm_lstm = confusion_matrix(y_test, y_pred_lstm)
sns.heatmap(cm_lstm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(regime_label_map.keys()),
            yticklabels=list(regime_label_map.keys()))
plt.xlabel('Predicted Regime')
plt.ylabel('Actual Regime')
plt.title('LSTM Regime Prediction: Confusion Matrix')
plt.savefig('lstm_confusion_matrix.png', dpi=150)
plt.show()

# Plot Comparative Bar Chart of Metrics
metrics_to_plot = ['Accuracy', 'Macro-F1', 'Bear Recall', 'Transition Recall']
fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(18, 5), sharey=True)
fig.suptitle('Model Performance Comparison', fontsize=16)

for i, metric in enumerate(metrics_to_plot):
    sns.barplot(x='Model', y=metric, data=comparison_metrics_df, ax=axes[i], palette='viridis')
    axes[i].set_title(metric)
    axes[i].set_ylim(0, 1)
    axes[i].set_ylabel('') # Remove y-label after first plot for cleaner look

axes[0].set_ylabel('Score') # Set y-label only for the first plot

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('model_comparison_metrics.png', dpi=150)
plt.show()

print("\nDetailed Metric Comparison:")
print(comparison_metrics_df.set_index('Model').round(3))
```

### Explanation of Execution
The performance metrics and visualizations provide a comprehensive view of how the LSTM model stands against simpler baselines. For an asset manager, the **Transition Recall** is often the most insightful metric. While the Persistence model might show high overall accuracy due to regimes being sticky, its Transition Recall is typically very low, indicating it cannot anticipate changes. If the LSTM demonstrates significantly higher Transition Recall, it confirms its value in providing *actionable intelligence* about market shifts, a key requirement for tactical allocation and risk management. The confusion matrix further dissects performance, showing where misclassifications occur, which is useful for understanding specific regime challenges (e.g., distinguishing between Bull-High Vol and Bear).

---

## Section 6: Deriving and Visualizing a Conceptual Portfolio Signal

The ultimate goal of predicting market regimes is to inform investment decisions. As an investment professional, I need to translate the model's probabilistic outputs into an actionable signal. Here, I'll derive a conceptual dynamic equity allocation signal based on the predicted probabilities of the "Bull-Low Volatility" and "Bull-High Volatility" regimes.

A simple rule could be: increase equity exposure when the probability of a bull market (either low or high volatility) is high. For instance, I might fully allocate to equities during a Bull-Low Vol regime, and perhaps reduce it slightly (e.g., 50%) during a Bull-High Vol regime to account for increased uncertainty, while significantly reducing equity exposure during a Bear regime.

Let $P(\text{Bull-LV})$ be the probability of a Bull-Low Vol regime, and $P(\text{Bull-HV})$ be the probability of a Bull-High Vol regime.
Our conceptual equity weight can be defined as:

$$
\text{Equity Weight} = P(\text{Bull-LV}) + 0.5 \times P(\text{Bull-HV})
$$

The remaining weight will be allocated to a defensive asset, like bonds:
$$
\text{Bond Weight} = 1 - \text{Equity Weight}
$$

Visualizing this signal alongside S&P 500 prices provides a clear narrative of how the model's insights would translate into tactical adjustments of our equity exposure, ideally reducing exposure before significant drawdowns.

```python
def plot_regime_probabilities_and_signal(prob_df, sp500_prices, regime_label_map, filename='regime_probabilities_and_signal.png'):
    """
    Plots predicted regime probabilities, S&P 500 prices with regime shading,
    and the conceptual equity allocation signal.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})
    
    # Map encoded labels back to original names for clarity
    reverse_regime_map = {v: k for k, v in regime_label_map.items()}
    
    # Top panel: S&P 500 price with dominant regime shading
    ax1.plot(sp500_prices.index, sp500_prices.values, 'k-', linewidth=0.8, label='S&P 500 Price')
    ax1.set_ylabel('S&P 500 Price')
    ax1.set_title('Market Regime Probabilities, S&P 500, and Conceptual Equity Signal', fontsize=14)
    ax1.legend(loc='upper left')

    # Shade by dominant predicted regime
    dominant_regimes = prob_df.idxmax(axis=1)
    color_map = {'P(Bull-LowVol)': 'lightgreen', 'P(Bull-HighVol)': 'lightsalmon', 'P(Bear)': 'lightcoral'}
    
    for i in range(len(dominant_regimes) - 1):
        regime_name = dominant_regimes.iloc[i]
        color = color_map.get(regime_name, 'grey')
        ax1.axvspan(prob_df.index[i], prob_df.index[i+1], alpha=0.2, color=color, lw=0)

    # Middle panel: Stacked probabilities
    ax2.stackplot(prob_df.index,
                  prob_df['P(Bull-LowVol)'],
                  prob_df['P(Bull-HighVol)'],
                  prob_df['P(Bear)'],
                  labels=['Bull-LowVol', 'Bull-HighVol', 'Bear'],
                  colors=['green', 'orange', 'red'], alpha=0.7)
    ax2.set_ylabel('Regime Probability')
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 1)

    # Bottom panel: Conceptual Equity Allocation Signal
    ax3.plot(prob_df.index, prob_df['equity_weight'], color='blue', label='Equity Weight', linewidth=1.5)
    ax3.fill_between(prob_df.index, 0, prob_df['equity_weight'], color='lightblue', alpha=0.6)
    ax3.set_ylabel('Equity Weight')
    ax3.set_xlabel('Date')
    ax3.set_ylim(0, 1)
    ax3.legend(loc='upper right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, dpi=150)
    plt.show()

# Create a DataFrame for probabilities
prob_df = pd.DataFrame(y_prob_lstm, columns=[f'P({label})' for label in list(regime_label_map.keys())],
                       index=df_data.loc[test_mask].index[LOOKBACK:])

# Conceptual equity weight proportional to bull probability
# Equity Weight = P(Bull-LowVol) + 0.5 * P(Bull-HighVol)
prob_df['equity_weight'] = prob_df['P(Bull-LowVol)'] + (0.5 * prob_df['P(Bull-HighVol)'])
prob_df['bond_weight'] = 1 - prob_df['equity_weight'] # Example for defensive asset

# Get S&P 500 adjusted close prices for the test period
sp500_test_prices = df_data.loc[test_mask, 'sp500_ret'].cumsum().apply(np.exp) # Reconstruct approximate price
sp500_test_prices = sp500_test_prices.reindex(prob_df.index).fillna(method='ffill') # Align indices

# Plot probabilities and signal
plot_regime_probabilities_and_signal(prob_df, sp500_test_prices, regime_label_map)

print("\nAverage equity allocation by actual regime on test set:")
test_actual_regimes = df_data.loc[test_mask, 'regime_encoded'].iloc[LOOKBACK:]
prob_df['actual_regime_encoded'] = test_actual_regimes

for encoded_regime, regime_name in regime_label_map.items():
    mask = prob_df['actual_regime_encoded'] == encoded_regime
    avg_eq_weight = prob_df.loc[mask, 'equity_weight'].mean()
    print(f" {regime_name}: {avg_eq_weight:.1%} equity")
```

### Explanation of Execution
The visualizations provide a clear narrative for a portfolio manager. The stacked area plot shows the model's confidence in each regime over time, allowing us to see how probabilities shift. Overlaid on S&P 500 prices with regime-colored shading, we can visually inspect if the model correctly identified past bull and bear periods. Most importantly, the conceptual equity allocation signal directly translates model output into a tangible investment strategy. Observing how equity exposure would dynamically adjust, potentially reducing risk before major market downturns, is a powerful demonstration of the model's practical utility for Alpha Strategies Asset Management. The average equity allocation by actual regime provides a quick check on whether the signal behaves rationally given the true market state.

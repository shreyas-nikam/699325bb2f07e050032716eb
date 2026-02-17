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

# --- Constants (can be overridden by function arguments) ---
LOOKBACK = 60
MODEL_FILEPATH = 'best_lstm_regime.keras' # Path to save/load the best LSTM model
TRAIN_TEST_SPLIT_DATE = '2020-01-01' # Date to split training and testing data


# --- 1. Data Acquisition and Feature Engineering ---
def acquire_and_engineer_features(start_date='2005-01-01', end_date='2024-12-31'):
    """
    Acquires raw financial data from Yahoo Finance and engineers features and market regime labels.

    Args:
        start_date (str): Start date for data acquisition (e.g., '2005-01-01').
        end_date (str): End date for data acquisition (e.g., '2024-12-31').

    Returns:
        tuple: A tuple containing:
            - df (pd.DataFrame): DataFrame with engineered features and regime labels.
            - feature_cols (list): List of column names used as features.
            - regime_map (dict): Mapping from regime names to encoded integer labels.
    """
    # 1. Acquire raw data
    print(f"Downloading data from {start_date} to {end_date}...")
    sp500 = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
    vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)
    tnx = yf.download('^TNX', start=start_date, end=end_date, progress=False)
    gld = yf.download('GLD', start=start_date, end=end_date, progress=False)
    print("Data download complete.")

    # Align dataframes and create a master DF
    df = pd.DataFrame(index=sp500.index)

    # 2. Engineer Features
    df['sp500_ret'] = sp500['Close'].pct_change()
    df['sp500_ret_5d'] = sp500['Close'].pct_change(5)
    df['sp500_ret_21d'] = sp500['Close'].pct_change(21)
    df['realized_vol_21d'] = (df['sp500_ret'].rolling(21).std() * np.sqrt(252))

    df['vix'] = vix['Close']
    df['vix_change_5d'] = df['vix'].pct_change(5)

    df['yield_10y'] = tnx['Close']
    # Approximate yield spread (10Y-2Y) using 10Y and its shifted value
    # Shift by 5 for a rough proxy of short-term rates (e.g., 1-week or 5-day difference)
    df['yield_spread'] = df['yield_10y'] - df['yield_10y'].shift(5)

    df['gold_ret'] = gld['Close'].pct_change()

    # 3. Define and encode regime labels
    # Calculate median VIX on the available data after initial processing
    vix_median = df['vix'].median()

    df['regime'] = np.where(
        df['sp500_ret_21d'] <= 0, 'Bear', # Bear regime if 21-day S&P 500 return is non-positive
        np.where(df['vix'] > vix_median, 'Bull-HighVol', 'Bull-LowVol') # Bull regimes based on VIX
    )

    # Define a consistent mapping for regime labels
    # Using specific order for intuitive interpretation (e.g., Bull-LowVol as 0)
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

    # Ensure all feature columns exist after dropping NaNs
    feature_cols_present = [col for col in feature_cols if col in df.columns]
    if len(feature_cols_present) != len(feature_cols):
        print(f"Warning: Some specified feature columns were not found or dropped: {set(feature_cols) - set(feature_cols_present)}")
        feature_cols = feature_cols_present

    print("Feature engineering complete.")
    return df, feature_cols, regime_map

# --- 2. Data Preprocessing for LSTM ---
def create_sequences(features, targets, lookback):
    """
    Creates sliding-window sequences for LSTM input.

    Args:
        features (np.array): (n_days, n_features) array of input features.
        targets (np.array): (n_days,) array of target regime labels.
        lookback (int): Number of past days per sequence.

    Returns:
        tuple: A tuple containing:
            - X (np.array): (n_samples, lookback, n_features) 3D tensor for LSTM input.
            - y (np.array): (n_samples,) 1D array of target labels.
    """
    X, y = [], []
    for i in range(lookback, len(features)):
        X.append(features[i-lookback:i])  # Past 'lookback' days' features
        y.append(targets[i])             # Current day's regime (target)
    return np.array(X), np.array(y)


def prepare_lstm_data(df_data, feature_columns, target_column, lookback, train_split_date):
    """
    Prepares data for LSTM model: splits, scales, creates sequences, and one-hot encodes targets.

    Args:
        df_data (pd.DataFrame): DataFrame containing features and target.
        feature_columns (list): List of column names to be used as features.
        target_column (str): Name of the column containing target labels (e.g., 'regime_encoded').
        lookback (int): Number of past days per sequence for LSTM.
        train_split_date (str): Date string to split training and testing data (e.g., '2020-01-01').

    Returns:
        tuple: A tuple containing:
            - X_train (np.array): Training features as sequences (N_samples, lookback, N_features).
            - y_train_oh (np.array): Training targets one-hot encoded.
            - X_test (np.array): Testing features as sequences (N_samples, lookback, N_features).
            - y_test_oh (np.array): Testing targets one-hot encoded.
            - y_train_non_oh (np.array): Training targets not one-hot encoded (for LR/evaluation).
            - y_test_non_oh (np.array): Testing targets not one-hot encoded (for LR/evaluation).
            - scaler (StandardScaler): Fitted StandardScaler object.
            - n_classes (int): Number of unique target classes.
            - train_mask (pd.Series): Boolean mask used for training data.
            - test_mask (pd.Series): Boolean mask used for testing data.
    """
    # Split data into training and testing sets (temporal split)
    train_mask = df_data.index < train_split_date
    test_mask = df_data.index >= train_split_date

    train_features_raw = df_data.loc[train_mask, feature_columns].values
    train_targets_raw = df_data.loc[train_mask, target_column].values
    test_features_raw = df_data.loc[test_mask, feature_columns].values
    test_targets_raw = df_data.loc[test_mask, target_column].values

    # Scale features using StandardScaler (fit only on training data)
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features_raw)
    test_features_scaled = scaler.transform(test_features_raw) # Transform test using training fit

    # Create sequences
    X_train, y_train_non_oh = create_sequences(train_features_scaled, train_targets_raw, lookback)
    X_test, y_test_non_oh = create_sequences(test_features_scaled, test_targets_raw, lookback)

    # One-hot encode targets for categorical cross-entropy loss
    n_classes = len(np.unique(train_targets_raw)) # Determine number of classes from training data
    y_train_oh = to_categorical(y_train_non_oh, num_classes=n_classes)
    y_test_oh = to_categorical(y_test_non_oh, num_classes=n_classes)

    return X_train, y_train_oh, X_test, y_test_oh, y_train_non_oh, y_test_non_oh, scaler, n_classes, train_mask, test_mask


# --- 3. Model Definition ---
def build_lstm_model(input_shape, n_classes, learning_rate=0.001, dropout_rate_lstm=0.3, dropout_rate_dense=0.2):
    """
    Builds a sequential LSTM model for market regime classification.

    Args:
        input_shape (tuple): Shape of the input sequences (lookback, n_features).
        n_classes (int): Number of output classes.
        learning_rate (float): Initial learning rate for the Adam optimizer.
        dropout_rate_lstm (float): Dropout rate for LSTM layers.
        dropout_rate_dense (float): Dropout rate for dense layers.

    Returns:
        tf.keras.models.Sequential: Compiled Keras LSTM model.
    """
    model = Sequential([
        LSTM(units=64, input_shape=input_shape, return_sequences=True, name='lstm_1'),
        Dropout(dropout_rate_lstm, name='dropout_1'),
        BatchNormalization(name='batch_norm_1'),

        LSTM(units=32, return_sequences=False, name='lstm_2'),
        Dropout(dropout_rate_lstm, name='dropout_2'),
        BatchNormalization(name='batch_norm_2'),

        Dense(units=16, activation='relu', name='dense_1'),
        Dropout(dropout_rate_dense, name='dropout_3'),
        Dense(units=n_classes, activation='softmax', name='output_dense')
    ])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# --- 4. Model Training ---
def train_model(model, X_train, y_train_oh, epochs=100, batch_size=64, validation_split=0.2, model_filepath=MODEL_FILEPATH):
    """
    Trains the LSTM model with specified callbacks and parameters.

    Args:
        model (tf.keras.models.Sequential): Compiled Keras model to train.
        X_train (np.array): Training features.
        y_train_oh (np.array): One-hot encoded training targets.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        validation_split (float): Fraction of training data to use for validation.
        model_filepath (str): Path to save the best model weights.

    Returns:
        tf.keras.callbacks.History: History object containing training metrics.
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
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1
    )
    return history


def plot_training_history(history, filename='lstm_training_curves.png'):
    """
    Plots training and validation loss and accuracy curves and saves the plot.

    Args:
        history (tf.keras.callbacks.History): History object returned by model.fit().
        filename (str): Name of the file to save the plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Val Loss')
    ax1.set_title('Loss Curves', fontsize=12)
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('Loss', fontsize=10)
    ax1.legend()

    ax2.plot(history.history['accuracy'], label='Train Acc')
    ax2.plot(history.history['val_accuracy'], label='Val Acc')
    ax2.set_title('Accuracy Curves', fontsize=12)
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Accuracy', fontsize=10)
    ax2.legend()

    plt.suptitle('LSTM Training History', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, dpi=150)
    plt.close(fig) # Close plot to prevent it from displaying automatically in some environments
    print(f"Training history plot saved to {filename}")


# --- 5. Model Evaluation ---
def predict_with_baselines(lstm_model, X_train_sequences, y_train_non_oh, X_test_sequences, y_test_non_oh, model_filepath=MODEL_FILEPATH):
    """
    Generates predictions from the trained LSTM model, a Logistic Regression baseline,
    and a Persistence model.

    Args:
        lstm_model (tf.keras.models.Sequential): The trained (or loaded) LSTM model.
        X_train_sequences (np.array): Training features as sequences (for LR flattening).
        y_train_non_oh (np.array): Non-one-hot encoded training targets (for LR training).
        X_test_sequences (np.array): Testing features as sequences (for LSTM predictions and LR flattening).
        y_test_non_oh (np.array): Non-one-hot encoded testing targets (for persistence and evaluation).
        model_filepath (str): Path to the saved best LSTM model weights.

    Returns:
        tuple: A tuple containing:
            - y_pred_lstm (np.array): Predicted class labels from LSTM.
            - y_pred_lr (np.array): Predicted class labels from Logistic Regression.
            - y_pred_persist (np.array): Predicted class labels from Persistence model.
            - y_prob_lstm (np.array): Predicted probabilities from LSTM.
    """
    # Load the best LSTM model weights if available
    try:
        lstm_model.load_weights(model_filepath)
        print(f"Loaded LSTM model weights from {model_filepath}")
    except Exception as e:
        print(f"Warning: Could not load LSTM model weights from {model_filepath}. Using current model state. Error: {e}")

    # LSTM predictions
    y_prob_lstm = lstm_model.predict(X_test_sequences)
    y_pred_lstm = np.argmax(y_prob_lstm, axis=1)

    # Baseline 1: Logistic Regression on flattened features
    # Flatten X_train and X_test sequences for Logistic Regression
    X_train_flat = X_train_sequences.reshape(X_train_sequences.shape[0], -1)
    X_test_flat = X_test_sequences.reshape(X_test_sequences.shape[0], -1)
    lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42, solver='liblinear')
    lr_model.fit(X_train_flat, y_train_non_oh)
    y_pred_lr = lr_model.predict(X_test_flat)

    # Baseline 2: Persistence (predict previous day's regime)
    y_pred_persist = np.roll(y_test_non_oh, 1) # Shift by 1 to get previous day's regime
    if len(y_test_non_oh) > 0:
        y_pred_persist[0] = y_test_non_oh[0] # The first prediction can't use previous, so use actual

    return y_pred_lstm, y_pred_lr, y_pred_persist, y_prob_lstm


def evaluate_models(y_true, y_pred_lstm, y_pred_lr, y_pred_persist, regime_label_map):
    """
    Evaluates LSTM and baseline models, prints comparison metrics, and calculates Transition Recall.

    Args:
        y_true (np.array): True target labels.
        y_pred_lstm (np.array): Predicted labels from the LSTM model.
        y_pred_lr (np.array): Predicted labels from the Logistic Regression model.
        y_pred_persist (np.array): Predicted labels from the Persistence model.
        regime_label_map (dict): Mapping from regime names to encoded integer labels.

    Returns:
        pd.DataFrame: DataFrame containing comparison metrics for all models.
    """
    target_names = list(regime_label_map.keys())

    print("\n--- LSTM Performance ---")
    print(classification_report(y_true, y_pred_lstm, target_names=target_names, zero_division=0))
    lstm_accuracy = accuracy_score(y_true, y_pred_lstm)
    lstm_f1_macro = f1_score(y_true, y_pred_lstm, average='macro', zero_division=0)
    lstm_recall_per_class = recall_score(y_true, y_pred_lstm, average=None, labels=list(regime_label_map.values()), zero_division=0)
    lstm_bear_recall_idx = regime_label_map['Bear']
    lstm_bear_recall = lstm_recall_per_class[lstm_bear_recall_idx]
    print(f"LSTM Overall Accuracy: {lstm_accuracy:.3f}")
    print(f"LSTM Macro-F1: {lstm_f1_macro:.3f}")
    print(f"LSTM Bear Recall: {lstm_bear_recall:.3f}")

    print("\n--- Logistic Regression Performance ---")
    print(classification_report(y_true, y_pred_lr, target_names=target_names, zero_division=0))
    lr_accuracy = accuracy_score(y_true, y_pred_lr)
    lr_f1_macro = f1_score(y_true, y_pred_lr, average='macro', zero_division=0)
    lr_recall_per_class = recall_score(y_true, y_pred_lr, average=None, labels=list(regime_label_map.values()), zero_division=0)
    lr_bear_recall = lr_recall_per_class[regime_label_map['Bear']]
    print(f"Logistic Regression Overall Accuracy: {lr_accuracy:.3f}")
    print(f"Logistic Regression Macro-F1: {lr_f1_macro:.3f}")
    print(f"Logistic Regression Bear Recall: {lr_bear_recall:.3f}")

    print("\n--- Persistence Model Performance ---")
    persist_accuracy = accuracy_score(y_true, y_pred_persist)
    persist_f1_macro = f1_score(y_true, y_pred_persist, average='macro', zero_division=0)
    persist_recall_per_class = recall_score(y_true, y_pred_persist, average=None, labels=list(regime_label_map.values()), zero_division=0)
    persist_bear_recall = persist_recall_per_class[regime_label_map['Bear']]
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
                if y_pred_seq[i] == y_true_seq[i] and y_pred_seq[i-1] == y_true_seq[i-1]:
                    correct_transitions += 1
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


def plot_confusion_matrix(y_true, y_pred, regime_label_map, filename='confusion_matrix.png', title='Confusion Matrix'):
    """
    Plots a confusion matrix and saves the plot.

    Args:
        y_true (np.array): True target labels.
        y_pred (np.array): Predicted target labels.
        regime_label_map (dict): Mapping from regime names to encoded integer labels.
        filename (str): Name of the file to save the plot.
        title (str): Title of the confusion matrix plot.
    """
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(regime_label_map.keys()),
                yticklabels=list(regime_label_map.keys()))
    plt.xlabel('Predicted Regime')
    plt.ylabel('Actual Regime')
    plt.title(title)
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Confusion matrix plot saved to {filename}")


def plot_comparison_metrics(metrics_df, filename='model_comparison_metrics.png'):
    """
    Plots a comparative bar chart of model performance metrics and saves the plot.

    Args:
        metrics_df (pd.DataFrame): DataFrame containing comparison metrics.
        filename (str): Name of the file to save the plot.
    """
    metrics_to_plot = ['Accuracy', 'Macro-F1', 'Bear Recall', 'Transition Recall']
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(18, 5), sharey=True)
    fig.suptitle('Model Performance Comparison', fontsize=16)

    for i, metric in enumerate(metrics_to_plot):
        sns.barplot(x='Model', y=metric, data=metrics_df, ax=axes[i], palette='viridis')
        axes[i].set_title(metric)
        axes[i].set_ylim(0, 1)
        axes[i].set_ylabel('')

    if axes: # Ensure axes list is not empty before accessing first element
        axes[0].set_ylabel('Score')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Model comparison metrics plot saved to {filename}")


# --- 6. Visualization of Predictions ---
def plot_regime_probabilities_and_signal(prob_df, sp500_prices, regime_label_map, filename='regime_probabilities_and_signal.png'):
    """
    Plots predicted regime probabilities, S&P 500 prices with regime shading,
    and the conceptual equity allocation signal, then saves the plot.

    Args:
        prob_df (pd.DataFrame): DataFrame containing regime probabilities and equity weights.
        sp500_prices (pd.Series): S&P 500 prices for the plotting period.
        regime_label_map (dict): Mapping from regime names to encoded integer labels.
        filename (str): Name of the file to save the plot.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})

    # Top panel: S&P 500 price with dominant regime shading
    ax1.plot(sp500_prices.index, sp500_prices.values, 'k-', linewidth=0.8, label='S&P 500 Price')
    ax1.set_ylabel('S&P 500 Price')
    ax1.set_title('Market Regime Probabilities, S&P 500, and Conceptual Equity Signal', fontsize=14)
    
    # Shade by dominant predicted regime
    dominant_regimes = prob_df[[f'P({label})' for label in regime_label_map.keys()]].idxmax(axis=1)
    color_map = {'P(Bull-LowVol)': 'lightgreen', 'P(Bull-HighVol)': 'lightsalmon', 'P(Bear)': 'lightcoral'}

    # Ensure shading aligns correctly with dates
    for i in range(len(dominant_regimes) - 1):
        regime_prob_col = dominant_regimes.iloc[i]
        color = color_map.get(regime_prob_col, 'grey')
        ax1.axvspan(prob_df.index[i], prob_df.index[i+1], alpha=0.2, color=color, lw=0)
    
    # Add legend for regime colors and S&P line
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='lightgreen', edgecolor='black', label='Bull-LowVol Dominant'),
                       Patch(facecolor='lightsalmon', edgecolor='black', label='Bull-HighVol Dominant'),
                       Patch(facecolor='lightcoral', edgecolor='black', label='Bear Dominant'),
                       plt.Line2D([0], [0], color='k', lw=0.8, label='S&P 500 Price')] # Add S&P line manually
    ax1.legend(handles=legend_elements, loc='upper left')

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
    plt.close(fig)
    print(f"Regime probabilities and signal plot saved to {filename}")


def analyze_and_plot_regime_predictions(y_prob_lstm, df_data, test_mask, lookback, regime_label_map, y_test_true_encoded):
    """
    Analyzes LSTM prediction probabilities, calculates a conceptual equity signal,
    and plots the results. Also prints average equity allocation by actual regime.

    Args:
        y_prob_lstm (np.array): Predicted probabilities from the LSTM model.
        df_data (pd.DataFrame): Original DataFrame with features and targets.
        test_mask (pd.Series): Boolean mask for the test data period.
        lookback (int): Lookback window used for sequence creation.
        regime_label_map (dict): Mapping from regime names to encoded integer labels.
        y_test_true_encoded (np.array): True encoded labels for the test set (non-one-hot).

    Returns:
        pd.DataFrame: DataFrame containing regime probabilities and calculated equity weights.
    """
    # Create a DataFrame for probabilities, aligning indices with the test data after lookback
    prob_df = pd.DataFrame(y_prob_lstm, columns=[f'P({label})' for label in list(regime_label_map.keys())],
                           index=df_data.loc[test_mask].index[lookback:])

    # Conceptual equity weight proportional to bull probability
    # Equity Weight = P(Bull-LowVol) + 0.5 * P(Bull-HighVol)
    prob_df['equity_weight'] = prob_df['P(Bull-LowVol)'] + (0.5 * prob_df['P(Bull-HighVol)'])
    prob_df['bond_weight'] = 1 - prob_df['equity_weight'] # Example for defensive asset

    # Get S&P 500 adjusted close prices for the test period
    # Reconstruct approximate price from daily returns for visualization
    sp500_test_prices = df_data.loc[test_mask, 'sp500_ret'].iloc[lookback:].cumsum().apply(np.exp)
    # Reindex to align with prob_df index in case of slight mismatches
    sp500_test_prices = sp500_test_prices.reindex(prob_df.index).fillna(method='ffill')

    # Plot probabilities and signal
    plot_regime_probabilities_and_signal(prob_df, sp500_test_prices, regime_label_map)

    print("\nAverage equity allocation by actual regime on test set:")
    # Align actual regimes with the prob_df index (which starts after lookback)
    prob_df['actual_regime_encoded'] = y_test_true_encoded

    reverse_regime_map = {v: k for k, v in regime_label_map.items()}
    for encoded_regime, regime_name in regime_label_map.items():
        mask = prob_df['actual_regime_encoded'] == encoded_regime
        if mask.any(): # Ensure there are entries for this regime
            avg_eq_weight = prob_df.loc[mask, 'equity_weight'].mean()
            print(f" {regime_name}: {avg_eq_weight:.1%} equity")
        else:
            print(f" {regime_name}: No occurrences in test set after lookback period.")

    return prob_df


# --- Main Pipeline Function ---
def run_regime_prediction_pipeline(
    start_date='2005-01-01',
    end_date='2024-12-31',
    train_split_date=TRAIN_TEST_SPLIT_DATE,
    lookback=LOOKBACK,
    epochs=100,
    batch_size=64,
    learning_rate=0.001,
    model_filepath=MODEL_FILEPATH,
    plot_history=True,
    plot_confusion=True,
    plot_comparison=True,
    plot_predictions=True
):
    """
    Runs the complete market regime prediction pipeline: data acquisition, feature engineering,
    data preparation, model building, training, evaluation, and visualization.

    Args:
        start_date (str): Start date for data acquisition.
        end_date (str): End date for data acquisition.
        train_split_date (str): Date to split training and testing data.
        lookback (int): Number of past days per sequence for LSTM.
        epochs (int): Number of training epochs for LSTM.
        batch_size (int): Batch size for LSTM training.
        learning_rate (float): Learning rate for LSTM optimizer.
        model_filepath (str): Path to save/load the best LSTM model weights.
        plot_history (bool): Whether to plot training history.
        plot_confusion (bool): Whether to plot LSTM confusion matrix.
        plot_comparison (bool): Whether to plot model comparison metrics.
        plot_predictions (bool): Whether to plot regime probabilities and signal.

    Returns:
        dict: A dictionary containing key artifacts from the pipeline,
              such as the trained model, scaler, regime map, test data, and results.
    """
    print("--- Starting Market Regime Prediction Pipeline ---")

    # 1. Acquire and Engineer Features
    df_data, feature_columns, regime_label_map = acquire_and_engineer_features(start_date, end_date)
    print(f"\nDataset shape: {df_data.shape}")
    print(f"Features used: {feature_columns}")
    print("\nRegime distribution (encoded labels):")
    print(df_data['regime_encoded'].value_counts())
    print("\nFirst 5 rows of data with features and regimes:")
    print(df_data[feature_columns + ['regime', 'regime_encoded']].head())

    # 2. Prepare Data for LSTM
    print("\n--- Preparing Data for LSTM Model ---")
    X_train, y_train_oh, X_test, y_test_oh, y_train_non_oh, y_test_non_oh, scaler, n_classes, train_mask, test_mask = \
        prepare_lstm_data(df_data, feature_columns, 'regime_encoded', lookback, train_split_date)

    print(f"X_train (sequence) shape: {X_train.shape} (N_samples, T_lookback, N_features)")
    print(f"y_train (encoded) shape: {y_train_oh.shape}")
    print(f"X_test (sequence) shape: {X_test.shape}")
    print(f"y_test (encoded) shape: {y_test_oh.shape}")
    print(f"Number of classes: {n_classes}")

    # 3. Build LSTM Model
    print("\n--- Building LSTM Model ---")
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape=input_shape, n_classes=n_classes, learning_rate=learning_rate)
    model.summary()

    # 4. Train LSTM Model
    print("\n--- Training LSTM Model ---")
    history = train_model(model, X_train, y_train_oh, epochs=epochs, batch_size=batch_size, model_filepath=model_filepath)
    if plot_history:
        plot_training_history(history)

    # 5. Evaluate Models and Baselines
    print("\n--- Evaluating Models and Baselines ---")
    y_pred_lstm, y_pred_lr, y_pred_persist, y_prob_lstm = \
        predict_with_baselines(model, X_train, y_train_non_oh, X_test, y_test_non_oh, model_filepath)

    comparison_metrics_df = evaluate_models(y_test_non_oh, y_pred_lstm, y_pred_lr, y_pred_persist, regime_label_map)
    print("\nDetailed Metric Comparison:")
    print(comparison_metrics_df.set_index('Model').round(3))

    if plot_confusion:
        plot_confusion_matrix(y_test_non_oh, y_pred_lstm, regime_label_map, filename='lstm_confusion_matrix.png', title='LSTM Regime Prediction: Confusion Matrix')
    if plot_comparison:
        plot_comparison_metrics(comparison_metrics_df)

    # 6. Analyze and Visualize Regime Predictions
    print("\n--- Analyzing and Visualizing Regime Predictions ---")
    prob_df = None
    if plot_predictions:
        prob_df = analyze_and_plot_regime_predictions(y_prob_lstm, df_data, test_mask, lookback, regime_label_map, y_test_non_oh)

    print("\n--- Pipeline Completed Successfully ---")
    return {
        "model": model,
        "scaler": scaler,
        "regime_label_map": regime_label_map,
        "feature_columns": feature_columns,
        "lookback": lookback,
        "X_test_sequences": X_test,
        "y_test_true_encoded": y_test_non_oh,
        "y_pred_lstm": y_pred_lstm,
        "y_prob_lstm": y_prob_lstm,
        "prob_df_visualization": prob_df, # Contains probabilities and equity weights for the test set
        "comparison_metrics": comparison_metrics_df
    }


if __name__ == "__main__":
    # Example usage when running the script directly
    # Reduced epochs for quicker example run during development/testing.
    # For actual training, use higher epochs (e.g., 100).
    results = run_regime_prediction_pipeline(
        start_date='2005-01-01',
        end_date='2024-12-31',
        train_split_date='2020-01-01',
        lookback=LOOKBACK,
        epochs=10, 
        batch_size=64,
        learning_rate=0.001,
        model_filepath='best_lstm_regime_example.keras',
        plot_history=True,
        plot_confusion=True,
        plot_comparison=True,
        plot_predictions=True
    )
    # You can now access pipeline artifacts from the 'results' dictionary
    print("\nPipeline artifacts available in 'results' dictionary (e.g., results['model'], results['scaler']).")

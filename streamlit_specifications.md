
# Streamlit Application Specification: Market Regime Prediction with LSTMs

## 1. Application Overview

This Streamlit application, "Market Sentinel: AI for Tactical Allocation," is designed for **CFA Charterholders and Investment Professionals** to demonstrate a practical workflow for developing and evaluating an LSTM-based market regime prediction model. The app moves beyond theoretical explanations by guiding the user through a real-world analytical pipeline, from data acquisition and feature engineering to model training, rigorous evaluation against baselines, and the derivation of an actionable portfolio signal.

The high-level story flow is as follows:

1.  **Introduction**: Set the stage, explaining the critical role of market regime detection in tactical asset allocation and risk management for investment professionals.
2.  **Data Acquisition & Regime Definition**: Users will see how raw financial time-series data is acquired and transformed into a rich feature set. The core concept of market regimes (Bull-Low Vol, Bull-High Vol, Bear) is defined mathematically and applied to the data.
3.  **Data Preparation for LSTM**: The process of structuring the financial time-series into sequential "sliding windows" suitable for LSTM input is demonstrated, along with crucial data preprocessing steps like scaling and temporal splitting.
4.  **LSTM Model Architecture**: The app details the construction of the LSTM neural network, emphasizing key components like LSTM layers, dropout for regularization, and softmax output for probabilistic regime classification.
5.  **Model Training**: The LSTM model is trained with advanced regularization techniques (Early Stopping, ReduceLROnPlateau) to prevent overfitting, a common pitfall in financial modeling. Training progress is visualized.
6.  **Evaluation & Baselines**: The LSTM's performance is rigorously evaluated against simpler, traditional baselines (Persistence, Logistic Regression) using financially relevant metrics, most notably "Transition Recall" to assess the model's ability to detect regime shifts. A confusion matrix provides insights into misclassifications.
7.  **Portfolio Signal & Visualization**: Finally, the model's probabilistic regime predictions are translated into a conceptual dynamic equity allocation signal. This signal is visualized alongside market prices, demonstrating how the model's insights can directly inform tactical portfolio adjustments.

The application emphasizes practical application, avoiding look-ahead bias, and interpreting results in a financially meaningful context for asset managers and quantitative analysts.

## 2. Code Requirements

The Streamlit application (`app.py`) will import all necessary functions directly from `source.py`.

```python
from source import *
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from contextlib import redirect_stdout
```

---

### `st.session_state` Design

`st.session_state` will be used to store and pass data between different sections (simulated pages) of the application, ensuring that computationally expensive operations are not re-run unnecessarily.

**Initialization (at the top of `app.py`)**:

```python
if 'page' not in st.session_state:
    st.session_state.page = 'Introduction'

# Data Acquisition & Regime Definition outputs
if 'df_data' not in st.session_state:
    st.session_state.df_data = None
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = None
if 'regime_label_map' not in st.session_state:
    st.session_state.regime_label_map = None

# Data Preparation for LSTM outputs
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'y_train_oh' not in st.session_state:
    st.session_state.y_train_oh = None
if 'y_test_oh' not in st.session_state:
    st.session_state.y_test_oh = None
if 'n_classes' not in st.session_state:
    st.session_state.n_classes = None
if 'lookback' not in st.session_state:
    st.session_state.lookback = 60 # Default value

# LSTM Model Design outputs
if 'model' not in st.session_state:
    st.session_state.model = None

# Model Training outputs
if 'history' not in st.session_state:
    st.session_state.history = None

# Evaluation & Baselines outputs
if 'y_prob_lstm' not in st.session_state:
    st.session_state.y_prob_lstm = None
if 'y_pred_lstm' not in st.session_state:
    st.session_state.y_pred_lstm = None
if 'y_pred_lr' not in st.session_state:
    st.session_state.y_pred_lr = None
if 'y_pred_persist' not in st.session_state:
    st.session_state.y_pred_persist = None
if 'comparison_metrics_df' not in st.session_state:
    st.session_state.comparison_metrics_df = None

# Portfolio Signal & Visualization outputs
if 'prob_df' not in st.session_state:
    st.session_state.prob_df = None
if 'sp500_test_prices' not in st.session_state:
    st.session_state.sp500_test_prices = None
```

---

### Application Structure and Flow

The application will use a sidebar for navigation, simulating a multi-page experience.

**Sidebar Navigation**:

```python
st.sidebar.title("Navigation")
page_options = [
    "Introduction",
    "1. Data Acquisition & Regimes",
    "2. Data Preparation for LSTM",
    "3. LSTM Model Design",
    "4. Model Training",
    "5. Evaluation & Baselines",
    "6. Portfolio Signal"
]
st.session_state.page = st.sidebar.selectbox("Go to", page_options, index=page_options.index(st.session_state.page))
```

---

**Main Content Rendering (Conditional based on `st.session_state.page`)**:

---

#### Page: `Introduction`

```python
if st.session_state.page == "Introduction":
    st.title("Market Sentinel: AI for Tactical Allocation")
    st.subheader("Navigating Volatile Markets with Deep Learning")

    st.markdown(f"As a **CFA Charterholder and Quantitative Analyst** at \"Alpha Strategies Asset Management,\" my core responsibility is to develop sophisticated models that enhance our tactical asset allocation and risk management capabilities. Financial markets are rarely static; they move through distinct behavioral phases—periods of calm growth (Bull-Low Volatility), periods of growth with increased uncertainty (Bull-High Volatility), and periods of decline (Bear Markets).")
    st.markdown(f"Recognizing and, more critically, anticipating shifts between these **market regimes** is paramount for optimizing portfolio performance and protecting client capital. Traditional rule-based methods or simpler statistical models often struggle to capture the complex, non-linear, and sequential dependencies that characterize these regime transitions. This is where advanced machine learning, specifically **Long Short-Term Memory (LSTM) neural networks**, offers a significant advantage. LSTMs are designed to process and learn from sequences of data, making them ideal for understanding the temporal dynamics of financial markets.")
    st.markdown(f"In this application, I'll walk through my workflow to build, train, and evaluate an LSTM model to classify market regimes using a rich set of financial time-series data. The goal is to develop a data-driven signal that can inform our investment decisions, leading to potentially better risk-adjusted returns and more robust portfolio management strategies.")
    st.markdown("---")
    st.markdown(f"### Core Learning Objectives:")
    st.markdown(f"- **Prepare sequential financial data** for LSTM input: construct sliding-window samples, normalize features, and create the 3D tensor shape.")
    st.markdown(f"- **Define and train an LSTM model** in TensorFlow/Keras, applying regularization (dropout, early stopping) to prevent overfitting.")
    st.markdown(f"- **Evaluate the LSTM against baselines**: compare regime classification accuracy to a logistic regression baseline and a random-walk baseline, using walk-forward validation.")
    st.markdown(f"- **Interpret LSTM outputs financially**: translate regime probabilities into actionable signals (e.g., reduce equity exposure when bear-regime probability exceeds 60%).")

    st.info("💡 **Key Insight for Investment Professionals**: LSTMs are chosen deliberately as the first deep learning architecture because they operate on sequences, which is how financial professionals naturally think about markets. This approach captures non-linear temporal patterns that simpler models often miss, offering a probabilistic signal useful in a portfolio management framework.")
```

---

#### Page: `1. Data Acquisition & Regimes`

```python
elif st.session_state.page == "1. Data Acquisition & Regimes":
    st.title("1. Data Acquisition & Defining Market Regimes")
    st.markdown(f"As a quantitative analyst, my first step is to gather the relevant financial data and explicitly define the market regimes I aim to predict. For this task, I'll collect historical data for the S&P 500 (`^GSPC`), VIX (`^VIX`), 10-year Treasury yield (`^TNX`), and Gold (`GLD`). From these raw series, I'll compute various derived features that capture different aspects of market behavior, such as momentum, volatility, and risk sentiment.")
    st.markdown(f"A crucial part of this process is defining our target variable: the market regime itself. I've chosen to classify the market into three distinct regimes: \"Bull-Low Volatility,\" \"Bull-High Volatility,\" and \"Bear,\" based on observable S&P 500 trailing returns and the VIX relative to its historical median. This ensures that my regime definitions are based on information available at the time, avoiding **look-ahead bias**, a critical pitfall in financial modeling.")

    st.markdown(r"The mathematical definition for our three market regimes at time $t$ is:")
    st.markdown(r"$$ \text{{Regime}}_t = \begin{{cases}} \text{{Bull-LowVol}} & \text{{if }} r_{{21d,t}} > 0 \land \text{{VIX}}_t \le \text{{VIX}}_{{\text{{median}}}} \\ \text{{Bull-HighVol}} & \text{{if }} r_{{21d,t}} > 0 \land \text{{VIX}}_t > \text{{VIX}}_{{\text{{median}}}} \\ \text{{Bear}} & \text{{if }} r_{{21d,t}} \le 0 \end{{cases}} $$")
    st.markdown(r"where $r_{{21d,t}}$ is the trailing 21-day S&P 500 return, and $\text{{VIX}}_{{\text{{median}}}}$ is the full-sample median VIX value.")

    st.markdown(f"---")
    st.subheader("Execute Data Acquisition & Feature Engineering")
    if st.button("Run Data Acquisition"):
        with st.spinner("Acquiring data and engineering features... This may take a moment."):
            df_data, feature_columns, regime_label_map = acquire_and_engineer_features()
            st.session_state.df_data = df_data
            st.session_state.feature_columns = feature_columns
            st.session_state.regime_label_map = regime_label_map
            
            st.success("Data acquisition and feature engineering complete!")
            st.markdown(f"**Dataset shape:** `{st.session_state.df_data.shape}`")
            st.markdown(f"**Features used:** `{st.session_state.feature_columns}`")
            st.markdown(f"**Regime distribution (encoded labels):**")
            st.dataframe(st.session_state.df_data['regime_encoded'].value_counts())
            st.markdown(f"**First 5 rows of data with features and regimes:**")
            st.dataframe(st.session_state.df_data[st.session_state.feature_columns + ['regime', 'regime_encoded']].head())
            st.markdown(f"The output confirms the shape of our comprehensive dataset and the distribution of the three market regimes. As a quant, understanding this distribution is important; for instance, if one regime is significantly underrepresented, it might pose challenges for the model during training. The defined features cover various market aspects, from returns and volatility to intermarket relationships (yield spread, gold returns), providing a holistic view of the market state. The clear regime definition ensures we are predicting a well-understood financial state.")
    else:
        st.info("Click 'Run Data Acquisition' to process the financial data and define market regimes.")

    if st.session_state.df_data is not None:
        st.subheader("Current Data Snapshot")
        st.dataframe(st.session_state.df_data.head())
```

---

#### Page: `2. Data Preparation for LSTM`

```python
elif st.session_state.page == "2. Data Preparation for LSTM":
    st.title("2. Preparing Sequential Data for LSTM Input")
    st.markdown(f"LSTMs are designed to process sequences. To feed our multivariate time-series data into the LSTM model, I need to transform it into fixed-length \"sliding windows\" of past observations. This means for each day $t$, the model will look at the previous $T$ days' features ($X_{{t-T}}, \\dots, X_{{t-1}}$) to predict the regime at day $t$ ($Y_t$). This creates 3D tensors of shape `(N_samples, T_lookback, N_features)`, which is the standard input format for Keras LSTMs.")
    st.markdown(f"Choosing an appropriate `lookback` window is crucial. A 60-day lookback (approximately three months) is chosen as it typically captures enough recent market history for a quantitative analyst to make informed decisions about short-to-medium term market behavior, without becoming too computationally intensive or dilute recent information with overly stale data.")
    st.warning("**Crucial warning on look-ahead bias**:")
    st.markdown(f"When constructing these sequences, I must strictly ensure that the features for predicting regime $Y_t$ only include data up to $X_{{t-1}}$. The target variable $Y_t$ itself, based on $r_{{21d,t}}$ and VIX$_t$, uses information available *up to and including* day $t$. This is fine because we are **classifying the current regime**, not predicting a future one. The LSTM's power lies in making this classification from a sequence of *past features* before the current regime label itself is computed.")

    st.markdown("---")
    st.subheader("Process Data for LSTM")

    if st.session_state.df_data is None:
        st.error("Please complete '1. Data Acquisition & Regimes' first.")
    else:
        if st.button("Prepare Data"):
            LOOKBACK = st.session_state.lookback
            
            # Split data into training and testing sets (temporal split)
            train_mask = st.session_state.df_data.index < '2020-01-01'
            test_mask = st.session_state.df_data.index >= '2020-01-01'

            # Separate features and targets
            train_features_raw = st.session_state.df_data.loc[train_mask, st.session_state.feature_columns].values
            train_targets_raw = st.session_state.df_data.loc[train_mask, 'regime_encoded'].values
            test_features_raw = st.session_state.df_data.loc[test_mask, st.session_state.feature_columns].values
            test_targets_raw = st.session_state.df_data.loc[test_mask, 'regime_encoded'].values

            # Scale features using StandardScaler (fit only on training data)
            scaler = StandardScaler()
            train_features_scaled = scaler.fit_transform(train_features_raw)
            test_features_scaled = scaler.transform(test_features_raw) # Transform test using training fit
            st.session_state.scaler = scaler

            # Create sequences
            X_train, y_train = create_sequences(train_features_scaled, train_targets_raw, LOOKBACK)
            X_test, y_test = create_sequences(test_features_scaled, test_targets_raw, LOOKBACK)
            st.session_state.X_train = X_train
            st.session_state.y_train = y_train
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test

            # One-hot encode targets for categorical cross-entropy loss
            n_classes = len(st.session_state.regime_label_map)
            y_train_oh = to_categorical(y_train, num_classes=n_classes)
            y_test_oh = to_categorical(y_test, num_classes=n_classes)
            st.session_state.n_classes = n_classes
            st.session_state.y_train_oh = y_train_oh
            st.session_state.y_test_oh = y_test_oh

            st.success("Data prepared for LSTM!")
            st.markdown(f"**Original training features shape:** `{train_features_raw.shape}`")
            st.markdown(f"**Original testing features shape:** `{test_features_raw.shape}`")
            st.markdown(f"**X_train (sequence) shape:** `{X_train.shape}` `(N_samples, T_lookback, N_features)`")
            st.markdown(f"**y_train (encoded) shape:** `{y_train_oh.shape}`")
            st.markdown(f"**X_test (sequence) shape:** `{X_test.shape}`")
            st.markdown(f"**y_test (encoded) shape:** `{y_test_oh.shape}`")
            st.markdown(f"**Number of classes:** `{n_classes}`")
            st.markdown(f"The output demonstrates that the data has been successfully transformed into the 3D tensor format required by LSTMs. `X_train` and `X_test` now contain sequences of {LOOKBACK} days of financial features, ready for the model. The `y_train_oh` and `y_test_oh` are one-hot encoded to align with the categorical nature of our regime classification problem, which is essential for the `categorical_crossentropy` loss function. This prepares the data for the next step: designing the LSTM model.")
        else:
            st.info("Click 'Prepare Data' to transform the time-series into LSTM-ready sequences.")
```

---

#### Page: `3. LSTM Model Design`

```python
elif st.session_state.page == "3. LSTM Model Design":
    st.title("3. Designing the LSTM Model Architecture")
    st.markdown(f"Now, I'll design the LSTM neural network architecture using Keras. The model will consist of multiple LSTM layers to capture complex temporal patterns, followed by Dense layers for classification. Key components include:")
    st.markdown(f"*   **LSTM Layers:** These are the core of the model, processing sequences and maintaining an internal state (cell state and hidden state) that allows them to remember relevant information over long periods, while selectively forgetting irrelevant details.")
    st.markdown(f"*   **Dropout Regularization:** Financial data is noisy and prone to overfitting. Dropout randomly sets a fraction of neuron outputs to zero during training, preventing complex co-adaptations between neurons and encouraging the network to learn more robust, generalized features.")
    st.markdown(r"$$ h_t' = m \odot h_t, \quad m_j \sim \text{{Bernoulli}}(1 - p) $$")
    st.markdown(r"where $p$ is the dropout rate, $h_t$ is the activation, and $m$ is a binary mask. This is analogous to portfolio diversification, preventing reliance on any single \"stock\" (feature or neuron).")
    st.markdown(f"*   **Batch Normalization:** This layer normalizes the activations of the previous layer, helping to stabilize and speed up training, especially in deep networks.")
    st.markdown(f"*   **Dense Layers with Softmax Activation:** The final layer uses a `softmax` activation function to output probabilities for each of our three market regimes. `softmax` ensures that the predicted probabilities sum to 1.")
    st.markdown(r"$$ P(\text{{regime}} = k \mid X_{{t-T:t}}) = \text{{softmax}}(W_{{\text{{out}}}} \cdot \text{{ReLU}}(W_h h_T + b_h) + b_{{\text{{out}}}})_k $$")
    st.markdown(r"where $h_T$ is the final LSTM hidden state, $k$ represents one of the regime classes, $W_{out}$, $W_h$, $b_h$, $b_{out}$ are weights and biases, and ReLU is the Rectified Linear Unit activation function.")
    st.markdown(f"*   **Categorical Cross-Entropy Loss:** This is the standard loss function for multi-class classification problems with one-hot encoded targets. It quantifies the difference between the predicted probability distribution and the true distribution.")
    st.markdown(r"$$ \mathcal{{L}} = - \frac{{1}}{{N}} \sum_{{i=1}}^{{N}} \sum_{{k=1}}^{{K}} y_{{ik}} \log p_{{ik}} $$")
    st.markdown(r"where $\mathcal{{L}}$ is the loss, $N$ is the number of samples, $K$ is the number of classes, $y_{{ik}}$ is the one-hot encoded true regime for sample $i$ and class $k$, and $p_{{ik}}$ is the predicted probability for sample $i$ and class $k$.")

    st.markdown("---")
    st.subheader("Build LSTM Model")

    if st.session_state.X_train is None or st.session_state.n_classes is None:
        st.error("Please complete '2. Data Preparation for LSTM' first.")
    else:
        if st.button("Build Model"):
            input_shape = (st.session_state.X_train.shape[1], st.session_state.X_train.shape[2])
            model = build_lstm_model(input_shape=input_shape, n_classes=st.session_state.n_classes)
            st.session_state.model = model
            
            st.success("LSTM Model Architecture Built!")
            st.markdown(f"**Model Summary:**")
            
            # Capture model summary output
            f = io.StringIO()
            with redirect_stdout(f):
                st.session_state.model.summary()
            s = f.getvalue()
            st.code(s)
            
            st.markdown(f"The model summary provides a clear overview of the LSTM's architecture, including the number of layers, output shapes, and the total number of trainable parameters. This is essential for a quant to verify that the network's complexity is appropriate for the problem and to understand the computational footprint. The `softmax` output layer ensures that the model provides probabilities for each regime, which is valuable for risk-sensitive decision-making beyond a simple classification.")
        else:
            st.info("Click 'Build Model' to initialize the LSTM network.")

    if st.session_state.model is not None:
        st.subheader("LSTM Cell Architecture (Theoretical Insight)")
        st.markdown(f"Understanding the internal workings of an LSTM cell helps grasp its power for sequence data. Each gate (forget, input, output) uses sigmoid activation ($\sigma$) to control information flow (values between 0 and 1), while $\tanh$ is used to generate candidate memory and output values (between -1 and 1).")
        st.markdown(r"**Forget Gate** (what to discard from cell state):")
        st.markdown(r"$$f_t = \sigma(W_f[h_{t-1}, x_t] + b_f)$$")
        st.markdown(r"where $f_t$ is the forget gate output, $\sigma$ is the sigmoid function, $W_f$ is the weight matrix, $h_{t-1}$ is the previous hidden state, $x_t$ is the current input, and $b_f$ is the bias vector. *Financial interpretation: This gate learns which past signals (e.g., an old volatility spike) are no longer relevant to the current market state.*")

        st.markdown(r"**Input Gate** (what new information to store):")
        st.markdown(r"$$i_t = \sigma(W_i[h_{t-1}, x_t] + b_i)$$")
        st.markdown(r"where $i_t$ is the input gate output, $\sigma$ is the sigmoid function, $W_i$ is the weight matrix, $h_{t-1}$ is the previous hidden state, $x_t$ is the current input, and $b_i$ is the bias vector. *Financial interpretation: This gate decides which new observations (e.g., today's sharp VIX increase) are important to incorporate.*")

        st.markdown(r"**Candidate Cell State** (new information):")
        st.markdown(r"$$\tilde{C}_t = \tanh(W_c[h_{t-1}, x_t] + b_c)$$")
        st.markdown(r"where $\tilde{C}_t$ is the candidate cell state, $\tanh$ is the hyperbolic tangent function, $W_c$ is the weight matrix, $h_{t-1}$ is the previous hidden state, $x_t$ is the current input, and $b_c$ is the bias vector.")

        st.markdown(r"**Cell State Update** (selective memory):")
        st.markdown(r"$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$")
        st.markdown(r"where $C_t$ is the current cell state, $\odot$ denotes element-wise multiplication, $f_t$ is the forget gate output, $C_{t-1}$ is the previous cell state, $i_t$ is the input gate output, and $\tilde{C}_t$ is the candidate cell state. *Financial interpretation: The cell state maintains a running memory of the market's trajectory, selectively remembering important long-term trends and forgetting irrelevant short-term noise. This is why LSTMs can capture regime persistence.*")

        st.markdown(r"**Output Gate** (what to output):")
        st.markdown(r"$$o_t = \sigma(W_o[h_{t-1}, x_t] + b_o)$$")
        st.markdown(r"where $o_t$ is the output gate output, $\sigma$ is the sigmoid function, $W_o$ is the weight matrix, $h_{t-1}$ is the previous hidden state, $x_t$ is the current input, and $b_o$ is the bias vector.")

        st.markdown(r"**Hidden State** (the output):")
        st.markdown(r"$$h_t = o_t \odot \tanh(C_t)$$")
        st.markdown(r"where $h_t$ is the current hidden state, $o_t$ is the output gate output, $\odot$ denotes element-wise multiplication, and $\tanh(C_t)$ is the hyperbolic tangent of the current cell state. *Financial interpretation: This is the actual prediction or feature representation output by the LSTM at time $t$.*")
```

---

#### Page: `4. Model Training`

```python
elif st.session_state.page == "4. Model Training":
    st.title("4. Training the LSTM Model with Regularization")
    st.markdown(f"Training the model involves feeding it the prepared sequences and allowing it to learn the patterns that distinguish market regimes. For financial time series, preventing **overfitting** is critical. Overfitting occurs when a model learns the training data too well, including its noise, and performs poorly on unseen data. To combat this, I'll employ several regularization techniques:")
    st.markdown(f"*   **Early Stopping:** This callback monitors the validation loss during training and halts the training process if validation loss stops improving for a specified number of epochs (`patience`). This prevents the model from continuing to memorize the training data.")
    st.markdown(f"*   **ReduceLROnPlateau:** This callback reduces the learning rate when the validation loss plateaus, allowing the model to make finer adjustments and converge more effectively.")
    st.markdown(f"*   **Model Checkpoint:** This callback saves the model weights (or the entire model) corresponding to the best validation performance, ensuring that I retain the most effective version of the model.")
    st.markdown(f"After training, I'll visualize the training and validation loss/accuracy curves. These plots are the most important diagnostics in deep learning, allowing me to identify overfitting (divergence between training and validation curves) and confirm that early stopping has been effective.")

    st.markdown("---")
    st.subheader("Train LSTM Model")

    if st.session_state.model is None or st.session_state.X_train is None:
        st.error("Please complete '3. LSTM Model Design' and '2. Data Preparation for LSTM' first.")
    else:
        if st.button("Start Training"):
            with st.spinner("Training LSTM model... This may take several minutes."):
                # Ensure the model is fresh for training if it was already built
                input_shape = (st.session_state.X_train.shape[1], st.session_state.X_train.shape[2])
                model = build_lstm_model(input_shape=input_shape, n_classes=st.session_state.n_classes)
                
                # Using a temporary file path for model checkpointing in Streamlit context
                model_filepath = 'best_lstm_regime.keras' 
                history = train_model(model, st.session_state.X_train, st.session_state.y_train_oh, model_filepath=model_filepath)
                
                # Load the best weights back into the model after training
                model.load_weights(model_filepath)
                st.session_state.model = model
                st.session_state.history = history

                st.success("Model training complete!")
                
                st.markdown(f"**Training History Plots:**")
                # Plot and save to a temporary file
                plt.figure() # Clear previous plots
                plot_training_history(history, filename='lstm_training_curves.png')
                st.image('lstm_training_curves.png', caption='LSTM Training History: Loss and Accuracy Curves')
                plt.close() # Close figure to free memory

                st.markdown(f"The training logs show the progress across epochs, with callbacks intervening as needed. The plotted curves visually confirm whether the model is learning effectively and if overfitting is being managed. For a quant, seeing the validation loss stabilize or increase after a certain point signals that early stopping correctly identified the optimal training duration, preventing the model from becoming too specialized to the training data noise. This ensures the model is more likely to generalize well to unseen market conditions.")
        else:
            st.info("Click 'Start Training' to train the LSTM model.")
```

---

#### Page: `5. Evaluation & Baselines`

```python
elif st.session_state.page == "5. Evaluation & Baselines":
    st.title("5. Baseline Comparison and Performance Evaluation")
    st.markdown(f"After training the LSTM, it's crucial to evaluate its performance and compare it against simpler, established baselines. This helps determine if the complexity of a deep learning model is justified by a tangible improvement in prediction capability. I'll compare the LSTM against two baselines:")
    st.markdown(f"1.  **Persistence Model:** This simple model predicts that \"tomorrow's regime is today's regime.\" Due to the inherent stickiness of market regimes, this baseline often achieves surprisingly high overall accuracy, but it fails to detect regime *transitions*.")
    st.markdown(f"2.  **Logistic Regression Model:** A traditional machine learning model applied to flattened (non-sequential) features, providing a benchmark for non-sequential, linear classification.")
    st.markdown(f"While overall accuracy is a common metric, for market regime prediction, it can be misleading due to regime persistence. The most critical metric for an asset manager is **Transition Recall**, which measures the model's ability to correctly identify actual shifts from one regime to another.")
    st.markdown(r"$$ \text{{Transition Recall}} = \frac{{\text{{Correctly predicted regime changes}}}}{{\text{{Total actual regime changes}}}} $$")
    st.markdown(r"where \"Correctly predicted regime changes\" refers to instances where a true regime change occurs, and the model correctly predicts both the new regime and the previous regime, and \"Total actual regime changes\" is the total count of days where the actual regime differs from the previous day's actual regime.")
    st.markdown(f"A high Transition Recall (e.g., > 50%) is what differentiates a truly useful model from a trivial \"predict the current regime\" baseline. I will also look at Macro-F1 (for balanced performance across classes) and Bear Recall (for detection of critical downturns).")

    st.markdown("---")
    st.subheader("Evaluate Models")

    if st.session_state.model is None or st.session_state.X_test is None:
        st.error("Please complete '4. Model Training' and '2. Data Preparation for LSTM' first.")
    else:
        if st.button("Run Evaluation"):
            with st.spinner("Evaluating LSTM and baseline models..."):
                # LSTM predictions
                y_prob_lstm = st.session_state.model.predict(st.session_state.X_test)
                y_pred_lstm = np.argmax(y_prob_lstm, axis=1)
                st.session_state.y_prob_lstm = y_prob_lstm
                st.session_state.y_pred_lstm = y_pred_lstm

                # Baseline 1: Logistic Regression on flattened features
                X_train_flat = st.session_state.X_train.reshape(st.session_state.X_train.shape[0], -1)
                X_test_flat = st.session_state.X_test.reshape(st.session_state.X_test.shape[0], -1)
                lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
                lr_model.fit(X_train_flat, st.session_state.y_train)
                y_pred_lr = lr_model.predict(X_test_flat)
                st.session_state.y_pred_lr = y_pred_lr

                # Baseline 2: Persistence (predict previous day's regime)
                y_pred_persist = np.roll(st.session_state.y_test, 1)
                y_pred_persist[0] = st.session_state.y_test[0] # Handle first element
                st.session_state.y_pred_persist = y_pred_persist

                # Evaluate all models
                comparison_metrics_df = evaluate_models(
                    st.session_state.y_test, 
                    st.session_state.y_pred_lstm, 
                    st.session_state.y_pred_lr, 
                    st.session_state.y_pred_persist, 
                    st.session_state.regime_label_map,
                    st.session_state.X_test # This X_test is not used by evaluate_models, but needed for the signature
                )
                st.session_state.comparison_metrics_df = comparison_metrics_df

                st.success("Model evaluation complete!")
                
                st.markdown(f"**LSTM Confusion Matrix:**")
                # Plot Confusion Matrix for LSTM
                plt.figure(figsize=(8, 6))
                cm_lstm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred_lstm)
                sns.heatmap(cm_lstm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=list(st.session_state.regime_label_map.keys()),
                            yticklabels=list(st.session_state.regime_label_map.keys()))
                plt.xlabel('Predicted Regime')
                plt.ylabel('Actual Regime')
                plt.title('LSTM Regime Prediction: Confusion Matrix')
                plt.savefig('lstm_confusion_matrix.png', dpi=150)
                st.image('lstm_confusion_matrix.png', caption='LSTM Regime Prediction: Confusion Matrix')
                plt.close()

                st.markdown(f"**Model Performance Comparison:**")
                # Plot Comparative Bar Chart of Metrics
                metrics_to_plot = ['Accuracy', 'Macro-F1', 'Bear Recall', 'Transition Recall']
                fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(18, 5), sharey=True)
                fig.suptitle('Model Performance Comparison', fontsize=16)

                for i, metric in enumerate(metrics_to_plot):
                    sns.barplot(x='Model', y=metric, data=st.session_state.comparison_metrics_df, ax=axes[i], palette='viridis')
                    axes[i].set_title(metric)
                    axes[i].set_ylim(0, 1)
                    axes[i].set_ylabel('')
                axes[0].set_ylabel('Score')

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.savefig('model_comparison_metrics.png', dpi=150)
                st.image('model_comparison_metrics.png', caption='Model Performance Comparison Across Key Metrics')
                plt.close()

                st.markdown(f"**Detailed Metric Comparison Table:**")
                st.dataframe(st.session_state.comparison_metrics_df.set_index('Model').round(3))
                
                st.markdown(f"The performance metrics and visualizations provide a comprehensive view of how the LSTM model stands against simpler baselines. For an asset manager, the **Transition Recall** is often the most insightful metric. While the Persistence model might show high overall accuracy due to regimes being sticky, its Transition Recall is typically very low, indicating it cannot anticipate changes. If the LSTM demonstrates significantly higher Transition Recall, it confirms its value in providing *actionable intelligence* about market shifts, a key requirement for tactical allocation and risk management. The confusion matrix further dissects performance, showing where misclassifications occur, which is useful for understanding specific regime challenges (e.g., distinguishing between Bull-High Vol and Bear).")
        else:
            st.info("Click 'Run Evaluation' to compare LSTM performance against baselines.")
```

---

#### Page: `6. Portfolio Signal`

```python
elif st.session_state.page == "6. Portfolio Signal":
    st.title("6. Deriving and Visualizing a Conceptual Portfolio Signal")
    st.markdown(f"The ultimate goal of predicting market regimes is to inform investment decisions. As an investment professional, I need to translate the model's probabilistic outputs into an actionable signal. Here, I'll derive a conceptual dynamic equity allocation signal based on the predicted probabilities of the \"Bull-Low Volatility\" and \"Bull-High Volatility\" regimes.")
    st.markdown(f"A simple rule could be: increase equity exposure when the probability of a bull market (either low or high volatility) is high. For instance, I might fully allocate to equities during a Bull-Low Vol regime, and perhaps reduce it slightly (e.g., 50%) during a Bull-High Vol regime to account for increased uncertainty, while significantly reducing equity exposure during a Bear regime.")
    st.markdown(r"Let $P(\text{{Bull-LV}})$ be the probability of a Bull-Low Vol regime, and $P(\text{{Bull-HV}})$ be the probability of a Bull-High Vol regime. Our conceptual equity weight can be defined as:")
    st.markdown(r"$$ \text{{Equity Weight}} = P(\text{{Bull-LV}}) + 0.5 \times P(\text{{Bull-HV}}) $$")
    st.markdown(r"where $P(\text{{Bull-LV}})$ is the probability of a Bull-Low Vol regime and $P(\text{{Bull-HV}})$ is the probability of a Bull-High Vol regime.")
    st.markdown(r"The remaining weight will be allocated to a defensive asset, like bonds:")
    st.markdown(r"$$ \text{{Bond Weight}} = 1 - \text{{Equity Weight}} $$")
    st.markdown(r"where \"Bond Weight\" is the allocation to a defensive asset.")
    st.markdown(f"Visualizing this signal alongside S&P 500 prices provides a clear narrative of how the model's insights would translate into tactical adjustments of our equity exposure, ideally reducing exposure before significant drawdowns.")

    st.markdown("---")
    st.subheader("Generate Portfolio Signal")

    if st.session_state.y_prob_lstm is None or st.session_state.df_data is None:
        st.error("Please complete '5. Evaluation & Baselines' and '1. Data Acquisition & Regimes' first.")
    else:
        if st.button("Derive & Visualize Signal"):
            with st.spinner("Generating portfolio signal and visualizations..."):
                # Create a DataFrame for probabilities
                test_mask = st.session_state.df_data.index >= '2020-01-01'
                prob_df = pd.DataFrame(st.session_state.y_prob_lstm, 
                                       columns=[f'P({label})' for label in list(st.session_state.regime_label_map.keys())],
                                       index=st.session_state.df_data.loc[test_mask].index[st.session_state.lookback:])
                st.session_state.prob_df = prob_df

                # Conceptual equity weight proportional to bull probability
                prob_df['equity_weight'] = prob_df['P(Bull-LowVol)'] + (0.5 * prob_df['P(Bull-HighVol)'])
                prob_df['bond_weight'] = 1 - prob_df['equity_weight']
                
                # Get S&P 500 adjusted close prices for the test period (reconstructed from returns)
                sp500_test_returns = st.session_state.df_data.loc[test_mask, 'sp500_ret'].iloc[st.session_state.lookback:]
                sp500_test_prices = (1 + sp500_test_returns).cumprod() * 100 # Start from an arbitrary base
                sp500_test_prices = sp500_test_prices.reindex(prob_df.index).fillna(method='ffill')
                st.session_state.sp500_test_prices = sp500_test_prices

                # Plot probabilities and signal
                plt.figure() # Clear previous plots
                plot_regime_probabilities_and_signal(st.session_state.prob_df, st.session_state.sp500_test_prices, st.session_state.regime_label_map, filename='regime_probabilities_and_signal.png')
                st.image('regime_probabilities_and_signal.png', caption='Market Regime Probabilities, S&P 500, and Conceptual Equity Signal')
                plt.close()
                
                st.success("Portfolio signal generated and visualized!")

                st.markdown(f"**Average equity allocation by actual regime on test set:**")
                test_actual_regimes = st.session_state.df_data.loc[test_mask, 'regime_encoded'].iloc[st.session_state.lookback:]
                prob_df['actual_regime_encoded'] = test_actual_regimes

                # Map encoded labels back to original names for clarity
                reverse_regime_map = {v: k for k, v in st.session_state.regime_label_map.items()}

                for encoded_regime, regime_name in reverse_regime_map.items():
                    mask = prob_df['actual_regime_encoded'] == encoded_regime
                    avg_eq_weight = prob_df.loc[mask, 'equity_weight'].mean()
                    st.markdown(f" - **{regime_name}**: `{avg_eq_weight:.1%}` equity")

                st.markdown(f"The visualizations provide a clear narrative for a portfolio manager. The stacked area plot shows the model's confidence in each regime over time, allowing us to see how probabilities shift. Overlaid on S&P 500 prices with regime-colored shading, we can visually inspect if the model correctly identified past bull and bear periods. Most importantly, the conceptual equity allocation signal directly translates model output into a tangible investment strategy. Observing how equity exposure would dynamically adjust, potentially reducing risk before major market downturns, is a powerful demonstration of the model's practical utility for Alpha Strategies Asset Management. The average equity allocation by actual regime provides a quick check on whether the signal behaves rationally given the true market state.")
        else:
            st.info("Click 'Derive & Visualize Signal' to see the conceptual dynamic equity allocation.")
```

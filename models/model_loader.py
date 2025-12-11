import os
import warnings
from typing import List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# try/except tensorflow import so non-ML devs can still run parts of app
try:
    import tensorflow as tf
except Exception:
    tf = None

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Files
MODEL_FILE_PATH = os.path.join(os.path.dirname(__file__), 'lstm_model.h5')
SCALER_FILE_PATH = os.path.join(os.path.dirname(__file__), 'scaler.pkl')

# Defaults (will be inferred if a model is present)
SEQUENCE_LENGTH = 24
FEATURE_COLUMNS = [
    # a candidate list of features we will try to pull from incoming ERA5 frames
    'pm25', 'pm10', 'u10', 'v10', 't2m', 'd2m', 'rh',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos',
    # optional lag/rolling placeholders
    'pm25_lag1', 'pm25_lag2', 'pm25_roll6h'
]
INPUT_FEATURE_COUNT = len(FEATURE_COLUMNS)

# Global model + scaler
TRAINED_MODEL = None
GLOBAL_SCALER = None

def _create_placeholder_model(seq_len, n_features):
    """Create a tiny LSTM model placeholder in case a saved model is missing."""
    if tf is None:
        return None
    m = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(seq_len, n_features)),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1)
    ])
    m.compile(optimizer='adam', loss='mse')
    return m

def load_trained_model():
    """Loads the serialized Keras model and sets up the mock scaler."""
    global TRAINED_MODEL, GLOBAL_SCALER
    try:
        loaded = tf.keras.models.load_model(MODEL_FILE_PATH)
        # Inspect input shape: model expects (batch, seq_len, n_features)
        try:
            input_shape = loaded.input_shape  # could be tuple or list
            # normalize to (None, seq, features)
            if isinstance(input_shape, list):
                input_shape = input_shape[0]
            seq_len = input_shape[1] if len(input_shape) > 2 else None
            feat_cnt = input_shape[2] if len(input_shape) > 2 else None

            # If model dimensions don't match expected, fallback to mock
            if seq_len != SEQUENCE_LENGTH or feat_cnt != INPUT_FEATURE_COUNT:
                print(f"WARNING: Loaded model shape mismatch. Model expects sequence length {seq_len}, features {feat_cnt}.")
                print("Using mock model instead (to avoid runtime errors).")
                TRAINED_MODEL = create_mock_model()
            else:
                TRAINED_MODEL = loaded
                print(f"INFO: Keras/LSTM model loaded successfully.")
        except Exception as e:
            print(f"WARNING: Couldn't inspect model.input_shape: {e}. Using loaded model but will fallback if predict fails.")
            TRAINED_MODEL = loaded

        # Setup a simple scaler so code continues
        GLOBAL_SCALER = MinMaxScaler()
        GLOBAL_SCALER.fit(np.array([1, 500]).reshape(-1, 1))

    except FileNotFoundError:
        print(f"WARNING: LSTM model not found at {MODEL_FILE_PATH}. Using mock model.")
        TRAINED_MODEL = create_mock_model()
        GLOBAL_SCALER = MinMaxScaler()
        GLOBAL_SCALER.fit(np.array([1, 500]).reshape(-1, 1))
    except Exception as e:
        print(f"CRITICAL ERROR loading Keras model: {e}. Falling back to mock model.")
        TRAINED_MODEL = create_mock_model()
        GLOBAL_SCALER = MinMaxScaler()
        GLOBAL_SCALER.fit(np.array([1, 500]).reshape(-1, 1))
    return TRAINED_MODEL


def _prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    From an ERA5-like dataframe, build a (N_samples, N_features) feature matrix
    matching FEATURE_COLUMNS up to INPUT_FEATURE_COUNT. Missing features are filled with zeros.
    Returns (feature_matrix, used_feature_columns).
    """
    global FEATURE_COLUMNS, INPUT_FEATURE_COUNT
    df_local = df.copy()
    n_samples = len(df_local)

    # Create time features if not present
    now = pd.Timestamp.now()
    if 'hour_sin' not in df_local.columns:
        df_local['hour_sin'] = np.sin(2 * np.pi * now.hour / 24)
        df_local['hour_cos'] = np.cos(2 * np.pi * now.hour / 24)
    if 'month_sin' not in df_local.columns:
        df_local['month_sin'] = np.sin(2 * np.pi * now.month / 12)
        df_local['month_cos'] = np.cos(2 * np.pi * now.month / 12)
    if 'day_sin' not in df_local.columns:
        df_local['day_sin'] = np.sin(2 * np.pi * now.day / 31)
        df_local['day_cos'] = np.cos(2 * np.pi * now.day / 31)

    # Choose feature columns to use: prefer existing ones in df, but ensure count matches INPUT_FEATURE_COUNT
    available = [c for c in FEATURE_COLUMNS if c in df_local.columns]
    used = available.copy()

    # If we don't have enough features, append other numeric columns from df
    if len(used) < INPUT_FEATURE_COUNT:
        extras = [c for c in df_local.select_dtypes(include=[np.number]).columns if c not in used]
        for c in extras:
            used.append(c)
            if len(used) >= INPUT_FEATURE_COUNT:
                break

    # If still not enough, pad with dummy feature names and zeros
    while len(used) < INPUT_FEATURE_COUNT:
        pad_name = f"_pad_{len(used)}"
        df_local[pad_name] = 0.0
        used.append(pad_name)

    # If too many features, trim to expected count
    used = used[:INPUT_FEATURE_COUNT]

    # Build matrix (N_samples, N_features)
    X = df_local[used].astype(float).to_numpy()
    return X, used

def run_model_prediction(raw_data_frame: pd.DataFrame) -> np.ndarray:
    """
    Accepts forecasted ERA5 rows (one row per station) and returns predicted AQI values (array length = n_rows).
    - Builds features to match the model's expected feature count.
    - Repeats each row to create a sequence of length SEQUENCE_LENGTH if necessary.
    - Scales features with GLOBAL_SCALER if available, otherwise fits a MinMaxScaler on the sample batch.
    """
    global TRAINED_MODEL, GLOBAL_SCALER, SEQUENCE_LENGTH, INPUT_FEATURE_COUNT

    # If TensorFlow not available, return mock
    if tf is None or TRAINED_MODEL is None:
        print("TF/model not available, returning mock predictions.")
        return np.random.randint(50, 300, size=len(raw_data_frame))

    try:
        df = raw_data_frame.copy().reset_index(drop=True)
        n = len(df)
        if n == 0:
            return np.array([])

        # 1. Build (n, n_features) aligned to INPUT_FEATURE_COUNT
        X_basic, used_cols = _prepare_features(df)  # shape (n, INPUT_FEATURE_COUNT)

        # 2. Fit scaler if needed (fit on this batch)
        if GLOBAL_SCALER is None:
            GLOBAL_SCALER = MinMaxScaler()
            try:
                GLOBAL_SCALER.fit(X_basic)
                print("Fitted on-the-fly MinMaxScaler from batch.")
            except Exception as e:
                print(f"Scaler fit failed: {e}. Proceeding without scaling.")
                GLOBAL_SCALER = None

        # 3. Scale X
        if GLOBAL_SCALER is not None:
            X_scaled = GLOBAL_SCALER.transform(X_basic)
        else:
            X_scaled = X_basic

        # 4. Create sequences: (n_samples, seq_len, n_features)
        # If the model was trained on sequences but we only have snapshot, repeat across time axis
        seq_len = SEQUENCE_LENGTH
        if seq_len <= 1:
            input_seq = X_scaled.reshape((n, 1, INPUT_FEATURE_COUNT))
        else:
            # repeat each row seq_len times
            input_seq = np.repeat(X_scaled[:, np.newaxis, :], seq_len, axis=1)

        # 5. Predict
        preds = TRAINED_MODEL.predict(input_seq, verbose=0)
        # preds shape should be (n, 1) or (n,)
        preds = np.asarray(preds).reshape(-1)

        # 6. If scaler was applied to target during training, you need to inverse transform here.
        # We don't know that, so assume preds are in PM2.5 units (or normalized). We attempt a sensible mapping:
        # If scaler exists and was fit on features only, we cannot reliably inverse. We'll return positive values and scale to AQI approximation:
        pred_pm25 = np.maximum(0.0, preds)
        # crude conversion to AQI (approx): AQI ≈ PM2.5 * 2
        aqi_preds = pred_pm25 * 2.0
        # ensure minimum 1
        return np.maximum(1.0, aqi_preds)

    except Exception as e:
        print(f"CRITICAL ERROR during model prediction: {e}. Returning mock values.")
        return np.random.randint(50, 300, size=len(raw_data_frame))

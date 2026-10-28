# ============================================================
# CAUC-AeroRULNet - Deployment Inference Backend
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from config import (
    DL_CHECKPOINT_PATH,
    FD002_MODEL_CANDIDATES,
    FD003_MODEL_CANDIDATES,
    SEQUENCE_LENGTH,
    NUM_SENSORS,
    MODEL_PARAMETERS,
    OPERATIONAL_COLUMNS,
    SELECTED_SENSORS,
    PRIORITY_ORDER,
    RUL_CAP,
)


@tf.keras.utils.register_keras_serializable(package="CAUC")
class TemporalSum(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def get_config(self):
        return super().get_config()


def _first_existing_path(candidates, label):
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"{label} model not found. Checked:\n" + "\n".join(candidates)
    )


@st.cache_resource(show_spinner=False)
def load_preprocessing_artifacts():
    if not os.path.exists(DL_CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Deep-learning checkpoint not found:\n{DL_CHECKPOINT_PATH}"
        )

    checkpoint = joblib.load(DL_CHECKPOINT_PATH)

    required_keys = [
        "model_sensor_cols",
        "fd002_op_scaler_model",
        "fd002_kmeans_model",
        "fd002_model_condition_means",
        "fd002_model_safe_stds",
        "fd003_model_means",
        "fd003_model_safe_stds",
    ]

    missing = [k for k in required_keys if k not in checkpoint]
    if missing:
        raise KeyError("Missing preprocessing objects: " + ", ".join(missing))

    model_sensor_cols = list(checkpoint["model_sensor_cols"])
    if len(model_sensor_cols) != NUM_SENSORS:
        raise ValueError(
            f"Checkpoint contains {len(model_sensor_cols)} model sensor columns; "
            f"expected {NUM_SENSORS}."
        )

    return {
        "model_sensor_cols": model_sensor_cols,
        "fd002_op_scaler": checkpoint["fd002_op_scaler_model"],
        "fd002_kmeans": checkpoint["fd002_kmeans_model"],
        "fd002_means": checkpoint["fd002_model_condition_means"],
        "fd002_stds": checkpoint["fd002_model_safe_stds"],
        "fd003_means": checkpoint["fd003_model_means"],
        "fd003_stds": checkpoint["fd003_model_safe_stds"],
    }


@st.cache_resource(show_spinner=False)
def load_models():
    fd002_path = _first_existing_path(FD002_MODEL_CANDIDATES, "FD002 improved final")
    fd003_path = _first_existing_path(FD003_MODEL_CANDIDATES, "FD003 final")

    custom_objects = {
        "TemporalSum": TemporalSum,
        "CAUC>TemporalSum": TemporalSum,
    }

    fd002_model = tf.keras.models.load_model(
        fd002_path,
        custom_objects=custom_objects,
        compile=False,
        safe_mode=False,
    )
    fd003_model = tf.keras.models.load_model(
        fd003_path,
        custom_objects=custom_objects,
        compile=False,
        safe_mode=False,
    )

    for label, model in (("FD002", fd002_model), ("FD003", fd003_model)):
        if model.count_params() != MODEL_PARAMETERS:
            raise ValueError(
                f"{label} model parameter mismatch: {model.count_params():,} "
                f"!= expected {MODEL_PARAMETERS:,}."
            )

    return {
        "FD002": fd002_model,
        "FD003": fd003_model,
        "_paths": {"FD002": fd002_path, "FD003": fd003_path},
    }


def get_loaded_model_paths():
    return load_models()["_paths"]


def classify_risk(rul):
    if rul <= 30:
        return "Critical"
    if rul <= 60:
        return "High Risk"
    if rul <= 90:
        return "Moderate"
    return "Healthy"


def get_priority(risk):
    return {
        "Critical": "P1",
        "High Risk": "P2",
        "Moderate": "P3",
        "Healthy": "P4",
    }[risk]


def get_recommendation(risk):
    return {
        "Critical": "Immediate maintenance required",
        "High Risk": "Schedule maintenance soon",
        "Moderate": "Increase monitoring frequency",
        "Healthy": "Continue routine monitoring",
    }[risk]


def validate_engine_dataframe(engine_df, dataset_type):
    if not isinstance(engine_df, pd.DataFrame):
        raise TypeError("Engine input must be a Pandas DataFrame.")
    if engine_df.empty:
        raise ValueError("Engine data is empty.")

    dataset_type = dataset_type.upper().strip()
    if dataset_type not in ("FD002", "FD003"):
        raise ValueError("Dataset profile must be FD002 or FD003.")

    required = ["cycle", *SELECTED_SENSORS]
    if dataset_type == "FD002":
        required += OPERATIONAL_COLUMNS

    missing = [c for c in required if c not in engine_df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    numeric = engine_df[required].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("Required columns contain missing or non-numeric values.")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Engine input contains NaN/Inf values.")
    if (numeric["cycle"] <= 0).any():
        raise ValueError("Cycle values must be greater than zero.")
    if numeric["cycle"].duplicated().any():
        raise ValueError("Duplicate cycle values detected for the same engine.")


def preprocess_fd002(engine_df, artifacts):
    df = engine_df.copy().sort_values("cycle").reset_index(drop=True)

    operational_scaled = artifacts["fd002_op_scaler"].transform(
        df[OPERATIONAL_COLUMNS]
    )
    df["operating_condition_model"] = artifacts["fd002_kmeans"].predict(
        operational_scaled
    )

    model_sensor_cols = artifacts["model_sensor_cols"]
    means = artifacts["fd002_means"]
    stds = artifacts["fd002_stds"]
    conditions = df["operating_condition_model"].to_numpy()

    for raw_sensor, norm_sensor in zip(SELECTED_SENSORS, model_sensor_cols):
        raw_values = df[raw_sensor].to_numpy(dtype=np.float32)
        normalized = np.empty(len(df), dtype=np.float32)

        for condition in np.unique(conditions):
            if condition not in means.index:
                raise ValueError(f"Unknown FD002 operating condition: {condition}")

            mask = conditions == condition
            mean_value = float(means.loc[condition, raw_sensor])
            std_value = float(stds.loc[condition, raw_sensor])

            if not np.isfinite(std_value) or std_value <= 0:
                raise ValueError(
                    f"Invalid normalization std for {raw_sensor}, condition {condition}."
                )

            normalized[mask] = (raw_values[mask] - mean_value) / std_value

        df[norm_sensor] = normalized

    return df


def preprocess_fd003(engine_df, artifacts):
    df = engine_df.copy().sort_values("cycle").reset_index(drop=True)
    model_sensor_cols = artifacts["model_sensor_cols"]
    means = artifacts["fd003_means"]
    stds = artifacts["fd003_stds"]

    for raw_sensor, norm_sensor in zip(SELECTED_SENSORS, model_sensor_cols):
        mean_value = float(means[raw_sensor])
        std_value = float(stds[raw_sensor])
        if not np.isfinite(std_value) or std_value <= 0:
            raise ValueError(f"Invalid FD003 std for {raw_sensor}.")
        df[norm_sensor] = (df[raw_sensor].astype(float) - mean_value) / std_value

    return df


def create_sequence(processed_df, model_sensor_cols):
    data = processed_df[model_sensor_cols].to_numpy(dtype=np.float32)
    if len(data) == 0:
        raise ValueError("No sensor data available after preprocessing.")

    observed_cycles = len(data)
    if observed_cycles >= SEQUENCE_LENGTH:
        sequence = data[-SEQUENCE_LENGTH:]
        padding_count = 0
    else:
        padding_count = SEQUENCE_LENGTH - observed_cycles
        padding = np.repeat(data[0:1], repeats=padding_count, axis=0)
        sequence = np.concatenate([padding, data], axis=0)

    sequence = np.expand_dims(sequence, axis=0)
    expected = (1, SEQUENCE_LENGTH, NUM_SENSORS)
    if sequence.shape != expected:
        raise ValueError(f"Invalid model input shape {sequence.shape}; expected {expected}.")
    if not np.isfinite(sequence).all():
        raise ValueError("Model sequence contains NaN/Inf.")

    return sequence, padding_count


def _sequence_diagnostics(sequence):
    flat = sequence.reshape(-1)
    return {
        "Input_Mean": float(np.mean(flat)),
        "Input_Std": float(np.std(flat)),
        "Input_Min": float(np.min(flat)),
        "Input_Max": float(np.max(flat)),
    }


def predict_engine(engine_df, dataset_type):
    dataset_type = dataset_type.upper().strip()
    validate_engine_dataframe(engine_df, dataset_type)

    artifacts = load_preprocessing_artifacts()
    models = load_models()

    if dataset_type == "FD002":
        processed = preprocess_fd002(engine_df, artifacts)
    else:
        processed = preprocess_fd003(engine_df, artifacts)

    sequence, padding_count = create_sequence(
        processed, artifacts["model_sensor_cols"]
    )

    prediction = models[dataset_type].predict(sequence, verbose=0)
    predicted_rul_raw = float(np.asarray(prediction).reshape(-1)[0])
    predicted_rul = max(predicted_rul_raw, 0.0)

    # Do not force predictions to the training cap. Preserve the model output;
    # the cap was used for target construction, not as a deployment clipping rule.
    risk = classify_risk(predicted_rul)
    priority = get_priority(risk)

    observed_cycles = len(engine_df)
    if observed_cycles >= SEQUENCE_LENGTH:
        history_quality = "Recommended"
    elif observed_cycles >= 40:
        history_quality = "Acceptable"
    else:
        history_quality = "Limited"

    diagnostics = _sequence_diagnostics(sequence)

    result = {
        "Dataset": dataset_type,
        "Observed_Cycles": observed_cycles,
        "Padding_Used": padding_count,
        "History_Quality": history_quality,
        "Predicted_RUL": round(predicted_rul, 2),
        "Predicted_RUL_Raw": round(predicted_rul_raw, 4),
        "Risk_Level": risk,
        "Maintenance_Priority": priority,
        "Recommendation": get_recommendation(risk),
        "RUL_Cap_Reference": RUL_CAP,
    }
    result.update({k: round(v, 4) for k, v in diagnostics.items()})
    return result


def predict_fleet(dataframe, dataset_type):
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Uploaded data must be a Pandas DataFrame.")
    if dataframe.empty:
        raise ValueError("Uploaded dataset is empty.")

    if "engine_id" in dataframe.columns:
        groups = dataframe.groupby("engine_id", sort=True)
    else:
        groups = [(1, dataframe)]

    results = []
    for engine_id, engine_df in groups:
        try:
            result = predict_engine(engine_df.copy(), dataset_type)
            result["Engine_ID"] = engine_id
            result["Status"] = "Success"
            result["Error"] = ""
        except Exception as error:
            result = {
                "Engine_ID": engine_id,
                "Dataset": dataset_type,
                "Observed_Cycles": len(engine_df),
                "Status": "Failed",
                "Error": str(error),
            }
        results.append(result)

    results_df = pd.DataFrame(results)

    if "Maintenance_Priority" in results_df.columns:
        results_df["Priority_Order"] = results_df["Maintenance_Priority"].map(
            PRIORITY_ORDER
        )
        results_df = results_df.sort_values(
            ["Priority_Order", "Predicted_RUL"],
            ascending=[True, True],
            na_position="last",
        ).reset_index(drop=True)

    return results_df

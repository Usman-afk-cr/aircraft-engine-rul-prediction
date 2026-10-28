# ============================================================
# CAUC-AeroRULNet - Application Configuration
# ============================================================

import os

PROJECT_DIR = r"D:\IST\Aircraft_Engine_RUL"
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints")
RESULT_DIR = os.path.join(PROJECT_DIR, "results")
FINAL_RESULT_DIR = os.path.join(RESULT_DIR, "final")
DECISION_RESULT_DIR = os.path.join(RESULT_DIR, "maintenance_decision")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")

# ------------------------------------------------------------
# Saved artifacts
# ------------------------------------------------------------
DL_CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "phase_02_deep_learning_checkpoint.joblib"
)

# IMPORTANT:
# The improved FD002 model is the official final FD002 model.
# The first existing path is used by inference.py.
FD002_MODEL_CANDIDATES = [
    os.path.join(MODEL_DIR, "cauc_aerorulnet_fd002_improved_final.keras"),
    os.path.join(MODEL_DIR, "final", "CAUC_AeroRULNet_FD002_IMPROVED.keras"),
]

FD003_MODEL_CANDIDATES = [
    os.path.join(MODEL_DIR, "cauc_aerorulnet_fd003_final.keras"),
    os.path.join(MODEL_DIR, "final", "CAUC_AeroRULNet_FD003.keras"),
]

# ------------------------------------------------------------
# Final model configuration
# ------------------------------------------------------------
SEQUENCE_LENGTH = 50
NUM_SENSORS = 16
MODEL_PARAMETERS = 97378
RUL_CAP = 125
FINAL_LEARNING_RATE = 0.0005
FINAL_DROPOUT = 0.20

OPERATIONAL_COLUMNS = [
    "operational_setting_1",
    "operational_setting_2",
    "operational_setting_3",
]

SELECTED_SENSORS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_6",
    "sensor_7", "sensor_8", "sensor_9", "sensor_10",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14",
    "sensor_15", "sensor_17", "sensor_20", "sensor_21",
]

RISK_ORDER = ["Critical", "High Risk", "Moderate", "Healthy"]
PRIORITY_ORDER = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}

# ------------------------------------------------------------
# FINAL VERIFIED VALIDATION RESULTS
# ------------------------------------------------------------
# FD002 CAUC values below correspond to the improved final model.
VALIDATION_RESULTS = {
    "FD002": {
        "Random Forest": {"MAE": 13.5646, "RMSE": 18.5966, "R2": 0.8023},
        "XGBoost": {"MAE": 12.8267, "RMSE": 17.9837, "R2": 0.8151},
        "LightGBM": {"MAE": 12.8498, "RMSE": 18.0478, "R2": 0.8130},
        "BiLSTM": {"MAE": 9.9731, "RMSE": 14.1695, "R2": 0.8802},
        "CAUC-AeroRULNet": {"MAE": 10.1087, "RMSE": 13.3602, "R2": 0.8935},
    },
    "FD003": {
        "Random Forest": {"MAE": 10.0615, "RMSE": 14.5246, "R2": 0.8753},
        "XGBoost": {"MAE": 9.5612, "RMSE": 14.3204, "R2": 0.8788},
        "LightGBM": {"MAE": 9.6838, "RMSE": 14.5837, "R2": 0.8736},
        "BiLSTM": {"MAE": 8.2702, "RMSE": 11.3660, "R2": 0.9263},
        "CAUC-AeroRULNet": {"MAE": 8.5568, "RMSE": 11.3140, "R2": 0.9270},
    },
}

# Existing decision-support validation results retained from project outputs.
DECISION_RESULTS = {
    "FD002": {
        "Accuracy": 0.780818,
        "Precision": 0.754072,
        "Recall": 0.746200,
        "F1": 0.748004,
        "Critical_Recall": 0.926799,
        "Critical_F1": 0.898106,
    },
    "FD003": {
        "Accuracy": 0.845225,
        "Precision": 0.807250,
        "Recall": 0.796790,
        "F1": 0.801162,
        "Critical_Recall": 0.922581,
        "Critical_F1": 0.898665,
    },
}

# UI colors
MODEL_COLOR_MAP = {
    "Random Forest": "#64748B",
    "XGBoost": "#0EA5E9",
    "LightGBM": "#8B5CF6",
    "BiLSTM": "#334155",
    "CAUC-AeroRULNet": "#F97316",
}

RISK_COLOR_MAP = {
    "Critical": "#DC2626",
    "High Risk": "#F97316",
    "Moderate": "#EAB308",
    "Healthy": "#16A34A",
}

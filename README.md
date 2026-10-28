
# Aircraft Engine Remaining Useful Life (RUL) Prediction

A machine learning and deep learning project for predicting the Remaining Useful Life (RUL) of aircraft engines using NASA C-MAPSS FD002 and FD003 datasets.

The project introduces **CAUC-AeroRULNet**, a condition-aware hybrid deep learning framework, and provides maintenance decision support through risk categories and a Streamlit dashboard.

## Project Overview

Remaining Useful Life (RUL) estimates how many operational cycles an engine may have left before reaching its end of useful life. Reliable RUL estimates can help maintenance teams plan inspections and reduce unexpected failures.

This project combines traditional machine learning baselines, deep learning models, model evaluation, and a dashboard for exploring predictions.

## Key Features

- Aircraft engine RUL prediction.
- Support for NASA C-MAPSS FD002 and FD003.
- Machine learning baselines: Random Forest, XGBoost, and LightGBM.
- BiLSTM and CAUC-AeroRULNet deep learning models.
- Sensor selection and preprocessing.
- Model evaluation using MAE, RMSE, and R².
- Maintenance risk classification based on predicted RUL.
- Streamlit dashboard for individual predictions, fleet monitoring, and research results.

## Proposed Model: CAUC-AeroRULNet

The proposed framework combines:

- Multi-branch 1D convolutional layers to learn temporal patterns.
- Bidirectional GRU layers to capture sequence information.
- Temporal and feature attention mechanisms.
- A regression head to predict RUL.

The workflow uses 50-cycle input sequences and 16 selected engine sensors.

For FD002, the preprocessing pipeline includes operating-condition clustering and condition-aware sensor normalization. FD003 uses global sensor normalization.

## Project Structure

```text
aircraft-engine-rul-prediction/
├── app/                  # Streamlit application and inference code
├── checkpoints/          # Saved training and preprocessing artifacts
├── data/
│   └── raw/              # NASA C-MAPSS dataset files
├── deployment/           # Deployment configuration and sample outputs
├── models/               # Trained deep learning models
├── notebooks/            # Experiments and analysis
├── results/              # Evaluation tables, predictions, and figures
├── .gitignore
└── README.md
```

## Dataset

The project uses the NASA C-MAPSS turbofan engine simulation datasets:

- **FD002:** Multiple operating conditions and fault modes.
- **FD003:** A single operating condition with multiple fault modes.

The data includes engine operating settings and sensor measurements collected over successive cycles.

Dataset files are located in `data/raw/`.

## Evaluation

The project evaluates models using:

- **MAE:** Mean Absolute Error.
- **RMSE:** Root Mean Squared Error.
- **R²:** Coefficient of Determination.
- **NASA scoring function:** An asymmetric penalty-based evaluation metric.

Reported validation results include:

| Dataset | Model | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| FD002 | CAUC-AeroRULNet | 10.1087 | 13.3602 | 0.8935 |
| FD003 | CAUC-AeroRULNet | 8.5568 | 11.3140 | 0.9270 |

These are validation results, not official NASA test-set results. See the files in `results/` for additional model comparisons and evaluation outputs.

## Maintenance Decision Support

Predicted RUL values are mapped to maintenance priority categories using the following rule-based policy:

| Predicted RUL | Risk Category | Priority |
|---|---|---|
| 0–30 cycles | Critical | P1 |
| 31–60 cycles | High Risk | P2 |
| 61–90 cycles | Moderate | P3 |
| Above 90 cycles | Healthy | P4 |

These categories are derived from predicted RUL thresholds; they are not the output of a separately trained four-class classifier.

## Installation

Python dependencies will be listed in `requirements.txt`.

1. Clone the repository:

   ```bash
   git clone https://github.com/Usman-afk-cr/aircraft-engine-rul-prediction.git
   cd aircraft-engine-rul-prediction
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   Windows:

   ```bash
   .venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Run the Streamlit Application

From the repository root, run:

```bash
streamlit run app/app.py
```

The application provides RUL prediction, fleet monitoring, model performance, research analysis, and system checks.

**Note:** The app depends on the trained models and preprocessing artifacts in this repository. The environment must include compatible versions of the required libraries.

## Limitations

- Reported results are based on validation evaluations.
- Official NASA test-set evaluation is not reported here.
- RUL-based maintenance categories use predefined thresholds.
- Results may vary with preprocessing, model configuration, and evaluation methodology.

## Future Work

- Evaluate on official NASA test sets.
- Conduct repeated-seed experiments and ablation studies.
- Improve prediction uncertainty estimates.
- Expand deployment monitoring and model reliability checks.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements

The project uses the NASA C-MAPSS turbofan engine simulation datasets for prognostics research.

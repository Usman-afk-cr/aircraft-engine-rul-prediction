# Aircraft Engine Remaining Useful Life (RUL) Prediction

**Research Implementation of CAUC-AeroRULNet — A Condition-Aware Hybrid Deep Learning Framework for Aircraft Engine Prognostics**

This repository contains the implementation and experimental evaluation of a deep learning framework for predicting aircraft engine Remaining Useful Life (RUL) using NASA C-MAPSS FD002 and FD003 datasets.

The project investigates a hybrid architecture combining multiscale convolutional feature extraction, bidirectional recurrent modelling, and attention mechanisms for sensor time-series analysis.

## Key Features

- CAUC-AeroRULNet model implementation
- RUL prediction using multivariate sensor sequences
- Comparison with Random Forest, XGBoost, LightGBM, and BiLSTM
- Operating-condition-aware preprocessing for FD002
- Sensor feature analysis
- Maintenance risk categorization
- Streamlit dashboard for RUL prediction and fleet monitoring

## Methodology

The workflow includes sensor selection, data preprocessing, sequence generation, deep learning model training, and validation against baseline models. The implementation also maps predicted RUL values to predefined maintenance priority categories.

## Experimental Results

The recorded validation results show that CAUC-AeroRULNet achieved the best RMSE and R² among the evaluated models on both FD002 and FD003. BiLSTM achieved lower MAE on both datasets.

Detailed evaluation results and visualizations are available in the [`results/`](results/) directory.

*Note: These are project validation results, not official NASA test-set benchmark results.*

## Technology Stack

Python · TensorFlow/Keras · Scikit-learn · XGBoost · LightGBM · Pandas · NumPy · Streamlit

## Installation and Usage

Clone the repository:

```bash
git clone https://github.com/Usman-afk-cr/aircraft-engine-rul-prediction.git
cd aircraft-engine-rul-prediction
```

Install the project's dependencies, then run:

```bash
streamlit run app/app.py
```

## Research Paper

**Status:** Written — not yet submitted.

The associated research manuscript has been prepared but has not been submitted for publication. No publication or peer-review acceptance is claimed.

## Limitations and Future Work

Further work includes official NASA test-set evaluation, uncertainty estimation, expanded robustness testing, and improved reproducibility.

## Author

**Muhammad Usman**

## Acknowledgements

NASA C-MAPSS turbofan engine degradation datasets.

## License

Released under the MIT License. See [`LICENSE`](LICENSE) for details.

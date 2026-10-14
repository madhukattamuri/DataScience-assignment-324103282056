# Aerisense — Air Quality Prediction & Analysis

A beginner-friendly Data Science academic project that runs locally in VS Code.

## Features
- Dashboard with AQI-like score, pollutant readings, trend chart and health guidance
- AQI estimator powered by a Random Forest regression model
- Dataset inspection, descriptive statistics, correlation heatmap and feature importance
- A step-by-step Data Science workflow section
- Standalone model training/evaluation script

## Important dataset note
`data/air_quality_demo.csv` is **simulated teaching data** generated for this project. It is not live sensor data, not official CPCB/OpenAQ data, and its `aqi` target is a synthetic AQI-like value rather than an official regulatory AQI calculation. Do not use its predictions for health or public-safety decisions.

## Run in VS Code (Windows)
1. Install Python 3.10 or newer and VS Code's Python extension.
2. Extract the ZIP and open the `Aerisense_Air_Quality_Project` folder in VS Code.
3. Open Terminal → New Terminal.
4. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use Command Prompt:
   ```bat
   .venv\Scripts\activate.bat
   ```

5. Install dependencies:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
6. Run the app:
   ```bash
   streamlit run app.py
   ```
7. A browser tab should open automatically. If not, use the local URL printed in the terminal (usually `http://localhost:8501`).

## Retrain the model
```bash
python train_model.py
```
The script validates required columns and missing values, creates a train/test split, trains a Random Forest Regressor, prints MAE/RMSE/R² and saves the model to `models/aqi_random_forest.joblib`.

## Data Science lifecycle covered
1. Problem definition
2. Data collection / loading
3. Data understanding
4. Data cleaning checks
5. Exploratory data analysis
6. Feature selection
7. Train/test split
8. Model training
9. Evaluation (MAE, RMSE, R²)
10. Prediction and communication

## Project structure
```text
Aerisense_Air_Quality_Project/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── air_quality_demo.csv
└── models/
    └── .gitkeep
```

## Before academic submission
- Confirm the course rubric and required libraries with your friend.
- If the instructor expects real data, replace the simulated CSV with a documented, appropriately licensed dataset and rerun evaluation.
- Explain the target definition, feature choices, train/test split, metrics, limitations, and why this is not yet a future-time forecast.

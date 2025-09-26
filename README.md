# Calories Burnt Prediction — Model Training & Evaluation

This project trains and evaluates multiple regression models to predict calories burnt using a preprocessed dataset.

It includes:
- Train/test split
- Models: Linear Regression (with scaling), Random Forest, XGBoost (optional)
- Metrics: MAE, RMSE, R²
- Clear comparison summary (sorted by RMSE)
- Feature importance plots for tree-based models
- Saving the best model to a `.pkl` file for deployment

## Project Structure
- `calories_burn_prediction_modeling.py` — Main training and evaluation script.
- `calories_burn_prediction_dataset.csv` — Your preprocessed dataset (15 cleaned features + target).
- `best_model.pkl` — Output: serialized best model (created after running the script).
- `plots/` — Output: feature importance charts saved as PNGs.

## Dependencies
- Python 3.9+ recommended
- pandas, numpy, scikit-learn, xgboost (optional), matplotlib, seaborn, joblib

Install with pip:
```bash
pip install -r requirements.txt
```

If you don’t need XGBoost, you can omit it; the script will auto-skip that model.

## How to Run
By default the script points to files inside the project folder. Adjust paths as needed.

Windows PowerShell (example):
```powershell
python "d:/AIML project/calories_burn_prediction_modeling.py" ^
  --data-path "d:/AIML project/calories_burn_prediction_dataset.csv" ^
  --model-output-path "d:/AIML project/best_model.pkl" ^
  --plots-output-dir "d:/AIML project/plots" ^
  --test-size 0.2
```

Linux/Mac (example):
```bash
python3 "d:/AIML project/calories_burn_prediction_modeling.py" \
  --data-path "d:/AIML project/calories_burn_prediction_dataset.csv" \
  --model-output-path "d:/AIML project/best_model.pkl" \
  --plots-output-dir "d:/AIML project/plots" \
  --test-size 0.2
```

### Optional: Force a Specific Target Column
If your dataset’s target column has a specific name, you can override auto-detection:
```bash
--target-col "Calories_Burned"
```
Auto-detection tries common names: `Calories_Burned`, `Calories_Burnt`, `CaloriesBurned`, `CaloriesBurnt`, `Calories`.
If none match, it uses the last numeric column.

## Workflow
1. Load dataset from `--data-path`.
2. Detect target column (or use `--target-col` if provided).
3. Split into train/test using `--test-size` and `--random-state`.
4. Define models:
   - Linear Regression with `StandardScaler` in a `Pipeline`.
   - `RandomForestRegressor`.
   - `XGBRegressor` (used if XGBoost is installed).
5. Train each model on the training set.
6. Predict on the test set and compute metrics: MAE, RMSE, R².
7. Print a comparison table sorted by RMSE.
8. Plot and save feature importances for Random Forest and XGBoost under `plots/`.
9. Save the best model (lowest RMSE) to `--model-output-path` as `.pkl` (using `joblib`).

## Outputs
- Console summary with metrics per model (MAE, RMSE, R²).
- `plots/feature_importance_random_forest.png` (if RF trained).
- `plots/feature_importance_xgboost.png` (if XGBoost trained).
- `best_model.pkl` containing the best-performing model.

## Tips & Troubleshooting
- Ensure your CSV is fully numeric for features; non-numeric columns should be encoded during preprocessing.
- If XGBoost import fails, install it:
  ```bash
  pip install xgboost
  ```
  or ignore; the script will proceed without it.
- To reproduce results, fix `--random-state` (default is 42).
- If the target detection is wrong, pass `--target-col` explicitly.

## Next Steps (Optional Enhancements)
- Add cross-validation and hyperparameter tuning (`GridSearchCV` or `RandomizedSearchCV`).
- Add experiment tracking (e.g., MLflow).
- Export a finalized `Pipeline` that includes preprocessing transformers.
- Dockerize the environment for deployment.

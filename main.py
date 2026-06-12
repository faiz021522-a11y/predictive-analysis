import os
import json
import traceback
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# Machine Learning & Stats imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

app = FastAPI(title="Predictive Analytics & Forecasting API")

# Setup directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Helper function to get dataset path
def get_csv_path(filename: str) -> str:
    # Prevent path traversal
    safe_filename = os.path.basename(filename)
    path = os.path.join(DATA_DIR, safe_filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Dataset {filename} not found.")
    return path

# ----------------- Models for Requests -----------------

class ImputationConfig(BaseModel):
    column: str
    strategy: str  # "mean", "median", "mode", "drop"

class RegressionTrainRequest(BaseModel):
    dataset_name: str
    target_column: str
    feature_columns: List[str]
    imputation_strategies: Dict[str, str] = {}  # column -> strategy
    scale_features: bool = False
    scaler_type: str = "standard"  # "standard", "minmax"
    model_type: str = "linear"  # "linear", "ridge", "random_forest"
    test_size: float = 0.2
    hyperparameters: Dict[str, Any] = {}

class TimeSeriesTrainRequest(BaseModel):
    dataset_name: str
    date_column: str
    target_column: str
    model_type: str = "arima"  # "arima", "exponential_smoothing", "moving_average"
    test_periods: int = 12
    forecast_horizon: int = 12
    hyperparameters: Dict[str, Any] = {}

# ----------------- Endpoints -----------------

# Serve index.html on root
@app.get("/")
def read_root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        return {"message": "Welcome to Predictive Analytics API. Frontend files not found yet in static/."}
    return FileResponse(index_path)

# Upload dataset
@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    
    file_path = os.path.join(DATA_DIR, file.filename)
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
        return {"message": "Dataset uploaded successfully", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload dataset: {str(e)}")

# List available datasets
@app.get("/api/datasets")
def list_datasets():
    try:
        files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
        return {"datasets": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Get dataset schema and preview
@app.get("/api/datasets/{filename}")
def get_dataset_details(filename: str):
    path = get_csv_path(filename)
    try:
        df = pd.read_csv(path)
        
        # Summary information
        num_rows, num_cols = df.shape
        columns_info = []
        
        for col in df.columns:
            null_count = int(df[col].isnull().sum())
            dtype = str(df[col].dtype)
            unique_count = df[col].nunique()
            
            # Get stats
            stats = {}
            if np.issubdtype(df[col].dtype, np.number):
                stats = {
                    "mean": float(df[col].mean()) if not df[col].isnull().all() else None,
                    "min": float(df[col].min()) if not df[col].isnull().all() else None,
                    "max": float(df[col].max()) if not df[col].isnull().all() else None,
                    "std": float(df[col].std()) if not df[col].isnull().all() else None,
                }
            
            columns_info.append({
                "name": col,
                "type": dtype,
                "null_count": null_count,
                "unique_count": unique_count,
                "stats": stats
            })
            
        # Sample rows (first 10)
        sample_data = df.head(10).replace({np.nan: None}).to_dict(orient="records")
        
        return {
            "filename": filename,
            "shape": [num_rows, num_cols],
            "columns": columns_info,
            "preview": sample_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading dataset: {str(e)}")

# Data preprocess helper
def preprocess_df(df: pd.DataFrame, target: str, features: List[str], imputations: Dict[str, str]):
    # Keep only target and features
    all_cols = list(set([target] + features))
    df = df[all_cols].copy()
    
    # Handle missing values column by column
    for col in all_cols:
        if col not in df.columns:
            continue
        strategy = imputations.get(col, "drop")
        
        if df[col].isnull().sum() > 0:
            if strategy == "drop":
                df = df.dropna(subset=[col])
            elif strategy == "mean" and np.issubdtype(df[col].dtype, np.number):
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == "median" and np.issubdtype(df[col].dtype, np.number):
                df[col] = df[col].fillna(df[col].median())
            elif strategy == "mode" or strategy in ["mean", "median"]:
                # If non-numeric mean/median, fallback to mode
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val[0])
                else:
                    df = df.dropna(subset=[col])
                    
    return df

# ----------------- Model Training - Regression -----------------
@app.post("/api/train/regression")
def train_regression(req: RegressionTrainRequest):
    path = get_csv_path(req.dataset_name)
    try:
        df = pd.read_csv(path)
        
        # Check columns
        if req.target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{req.target_column}' not in dataset.")
        for col in req.feature_columns:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Feature column '{col}' not in dataset.")
                
        # 1. Preprocess (Imputation)
        df = preprocess_df(df, req.target_column, req.feature_columns, req.imputation_strategies)
        if len(df) < 5:
            raise HTTPException(status_code=400, detail="Not enough data remaining after cleaning (minimum 5 rows).")
            
        X = df[req.feature_columns].copy()
        y = df[req.target_column].copy()
        
        # 2. Categorical Encoding (One-Hot Encoding)
        # Find categorical variables
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if len(categorical_features) > 0:
            X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
            
        # Convert bools (if any, from get_dummies) to ints
        for col in X.columns:
            if X[col].dtype == bool:
                X[col] = X[col].astype(int)
                
        # 3. Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=req.test_size, random_state=42
        )
        
        # Keep track of column names after OHE
        final_feature_names = X.columns.tolist()
        
        # 4. Feature Scaling
        scaled_features = []
        if req.scale_features:
            scaler = StandardScaler() if req.scaler_type == "standard" else MinMaxScaler()
            # Fit scaler only on numerical features
            num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
                X_test[num_cols] = scaler.transform(X_test[num_cols])
                
        # 5. Model Initialization & Training
        model_name = req.model_type.lower()
        if model_name == "linear":
            model = LinearRegression()
        elif model_name == "ridge":
            alpha = float(req.hyperparameters.get("alpha", 1.0))
            model = Ridge(alpha=alpha)
        elif model_name == "random_forest":
            n_estimators = int(req.hyperparameters.get("n_estimators", 100))
            max_depth = req.hyperparameters.get("max_depth", None)
            if max_depth is not None:
                max_depth = int(max_depth)
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model type: {req.model_type}")
            
        model.fit(X_train, y_train)
        
        # 6. Evaluation
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        metrics = {
            "train": {
                "r2": float(r2_score(y_train, y_train_pred)),
                "mae": float(mean_absolute_error(y_train, y_train_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_train, y_train_pred)))
            },
            "test": {
                "r2": float(r2_score(y_test, y_test_pred)),
                "mae": float(mean_absolute_error(y_test, y_test_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
            }
        }
        
        # 7. Predictions & Residuals
        # Actual vs Predicted for test set
        test_predictions = []
        residuals = []
        for act, pred, idx in zip(y_test, y_test_pred, y_test.index):
            res = float(act - pred)
            residuals.append(res)
            test_predictions.append({
                "index": int(idx),
                "actual": float(act),
                "predicted": float(pred),
                "residual": res
            })
            
        # 8. Feature Importances / Coefficients
        importances = []
        if model_name in ["linear", "ridge"]:
            for name, coef in zip(final_feature_names, model.coef_):
                importances.append({"feature": name, "importance": float(coef)})
        elif model_name == "random_forest":
            for name, imp in zip(final_feature_names, model.feature_importances_):
                importances.append({"feature": name, "importance": float(imp)})
                
        # Sort importances by absolute value
        importances = sorted(importances, key=lambda x: abs(x["importance"]), reverse=True)
        
        return {
            "success": True,
            "metrics": metrics,
            "predictions": test_predictions,
            "residuals": residuals,
            "feature_importances": importances
        }
        
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "detail": f"Training failed: {str(e)}"}
        )

# ----------------- Model Training - Time Series -----------------
@app.post("/api/train/timeseries")
def train_timeseries(req: TimeSeriesTrainRequest):
    path = get_csv_path(req.dataset_name)
    try:
        df = pd.read_csv(path)
        
        # Validate columns
        if req.date_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Date column '{req.date_column}' not found.")
        if req.target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{req.target_column}' not found.")
            
        # Convert date column and sort
        df[req.date_column] = pd.to_datetime(df[req.date_column])
        df = df.sort_values(by=req.date_column).reset_index(drop=True)
        
        # Missing values in date or target?
        df = df.dropna(subset=[req.date_column, req.target_column])
        
        if len(df) < 10:
            raise HTTPException(status_code=400, detail="Time-series dataset must have at least 10 valid data points.")
            
        # Establish dates as index
        df.set_index(req.date_column, inplace=True)
        
        # Try to infer index frequency
        inferred_freq = pd.infer_freq(df.index)
        if inferred_freq:
            df = df.asfreq(inferred_freq, method='ffill')
        else:
            # Fallback to resetting frequency to numeric/arbitrary if not datetime,
            # or try to set monthly/daily frequency if regular spacing is detected
            # For simplicity, we just force resample to Daily or Monthly if inferred fails
            # Let's inspect differences.
            diffs = df.index.to_series().diff().dropna()
            mean_days = diffs.mean().days
            if 27 <= mean_days <= 32:
                df = df.asfreq('ME', method='ffill')  # Monthly
            elif mean_days == 1:
                df = df.asfreq('D', method='ffill')   # Daily
            else:
                # Set a generic period frequency if datetime index isn't clean
                df.index.freq = pd.infer_freq(df.index) or 'D'
                
        series = df[req.target_column].astype(float)
        
        # Split train/test
        test_periods = min(req.test_periods, len(series) - 4)
        if test_periods <= 0:
            test_periods = 1
            
        train_series = series.iloc[:-test_periods]
        test_series = series.iloc[-test_periods:]
        
        model_name = req.model_type.lower()
        
        # Initialize outputs
        fitted_values = []
        test_predictions = []
        future_forecast = []
        metrics = {}
        
        if model_name == "moving_average":
            window = int(req.hyperparameters.get("window", 3))
            # Train fits (rolling mean)
            train_fit = train_series.rolling(window=window).mean().bfill()
            
            # Forecast (for moving average, the forecast is just the last rolling mean value)
            last_val = train_fit.iloc[-1]
            
            # Test predictions
            test_preds_series = pd.Series([last_val] * test_periods, index=test_series.index)
            
            # Future Forecast
            future_dates = pd.date_range(
                start=series.index[-1],
                periods=req.forecast_horizon + 1,
                freq=series.index.freq
            )[1:]
            future_forecast_series = pd.Series([last_val] * req.forecast_horizon, index=future_dates)
            
        elif model_name == "exponential_smoothing":
            trend = req.hyperparameters.get("trend", None)  # "add", "mul", None
            seasonal = req.hyperparameters.get("seasonal", None)  # "add", "mul", None
            sp = req.hyperparameters.get("seasonal_periods", None)
            if sp is not None:
                sp = int(sp)
                
            model = ExponentialSmoothing(
                train_series,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=sp,
                initialization_method="estimated"
            )
            res = model.fit()
            
            train_fit = res.fittedvalues
            test_preds_series = res.forecast(steps=test_periods)
            
            # Future forecast (fit model on FULL series for future prediction)
            full_model = ExponentialSmoothing(
                series,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=sp,
                initialization_method="estimated"
            )
            full_res = full_model.fit()
            future_forecast_series = full_res.forecast(steps=req.forecast_horizon)
            
        elif model_name == "arima":
            p = int(req.hyperparameters.get("p", 1))
            d = int(req.hyperparameters.get("d", 1))
            q = int(req.hyperparameters.get("q", 1))
            
            model = ARIMA(train_series, order=(p, d, q))
            res = model.fit()
            
            train_fit = res.fittedvalues
            test_preds_series = res.forecast(steps=test_periods)
            
            # Future forecast on full series
            full_model = ARIMA(series, order=(p, d, q))
            full_res = full_model.fit()
            future_forecast_series = full_res.forecast(steps=req.forecast_horizon)
            
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model type: {req.model_type}")
            
        # Calculate test metrics
        mae = mean_absolute_error(test_series, test_preds_series)
        rmse = np.sqrt(mean_squared_error(test_series, test_preds_series))
        
        # Avoid division by zero in MAPE
        non_zeros = test_series != 0
        if non_zeros.any():
            mape = np.mean(np.abs((test_series[non_zeros] - test_preds_series[non_zeros]) / test_series[non_zeros])) * 100
        else:
            mape = 0.0
            
        metrics = {
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape)
        }
        
        # Structure outputs
        historical_list = [{"date": d.strftime('%Y-%m-%d'), "value": float(v)} for d, v in series.items()]
        fitted_list = [{"date": d.strftime('%Y-%m-%d'), "value": float(v)} for d, v in train_fit.items()]
        test_preds_list = [{"date": d.strftime('%Y-%m-%d'), "value": float(v)} for d, v in test_preds_series.items()]
        future_forecast_list = [{"date": d.strftime('%Y-%m-%d'), "value": float(v)} for d, v in future_forecast_series.items()]
        
        return {
            "success": True,
            "metrics": metrics,
            "historical": historical_list,
            "fitted": fitted_list,
            "predictions": test_preds_list,
            "forecast": future_forecast_list
        }
        
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "detail": f"Time-series training failed: {str(e)}"}
        )

# Mount static files (must be at the bottom so it doesn't block APIs)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

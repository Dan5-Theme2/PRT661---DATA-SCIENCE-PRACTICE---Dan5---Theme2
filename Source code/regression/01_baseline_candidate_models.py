import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor


# Find the model-ready dataset
possible_folders = [
    Path.cwd(),
    Path.cwd() / "Data Science",
    Path.home() / "Desktop" / "Data Science",
]

project_folder = None

for folder in possible_folders:
    checkpoint = folder / "Fire2Air_model_ready_checkpoint.csv"

    if checkpoint.exists():
        project_folder = folder
        break

if project_folder is None:
    raise FileNotFoundError(
        "Fire2Air_model_ready_checkpoint.csv could not be found. "
        "Run the data preparation pipeline first."
    )

print("Project folder:", project_folder)


# Load the shared model-ready dataset
integrated_daily = pd.read_csv(
    project_folder / "Fire2Air_model_ready_checkpoint.csv",
    parse_dates=["date"]
)

print("Dataset shape:", integrated_daily.shape)


# Define the regression target and shared predictors
regression_target = "target_pm25_next_day"

model_features = [
    "station",
    "pm25_mean",
    "pm25_max",
    "pm10_mean",
    "pm10_max",
    "humidity_mean",
    "temperature_mean",
    "wind_speed_mean",
    "pressure_mean",
    "wind_dir_sin_mean",
    "wind_dir_cos_mean",
    "fire_count_0_25km",
    "fire_count_25_50km",
    "fire_count_50_100km",
    "fire_count_100_200km",
    "frp_sum_0_200km",
    "month_sin",
    "month_cos",
    "pm25_lag1",
    "pm25_3day_mean"
]


# Use 2018-2022 for training and 2023 for validation
train_reg = integrated_daily[
    (integrated_daily["date"].dt.year <= 2022)
    & integrated_daily[regression_target].notna()
].copy()

validation_reg = integrated_daily[
    (integrated_daily["date"].dt.year == 2023)
    & integrated_daily[regression_target].notna()
].copy()

print("\nRegression development data:")
print("Training observations:", len(train_reg))
print("Validation observations:", len(validation_reg))


# Separate predictors and target
X_train = train_reg[model_features]
y_train = train_reg[regression_target]

X_validation = validation_reg[model_features]
y_validation = validation_reg[regression_target]


# Separate categorical and numerical predictors
categorical_features = ["station"]

numeric_features = [
    feature
    for feature in model_features
    if feature not in categorical_features
]


# Preprocess station
categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)


# Ridge uses median imputation and numerical scaling
ridge_numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

ridge_preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", categorical_pipeline, categorical_features),
        ("numeric", ridge_numeric_pipeline, numeric_features)
    ]
)


# Tree models only require missing-value imputation
tree_numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

tree_preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", categorical_pipeline, categorical_features),
        ("numeric", tree_numeric_pipeline, numeric_features)
    ]
)


# Calculate regression evaluation metrics
def regression_metrics(y_true, predictions):
    return {
        "MAE": mean_absolute_error(y_true, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_true, predictions)),
        "R2": r2_score(y_true, predictions)
    }


# Persistence baseline assumes tomorrow's PM2.5 equals today's PM2.5
persistence_mask = validation_reg["pm25_mean"].notna()

persistence_actual = validation_reg.loc[
    persistence_mask,
    regression_target
]

persistence_predictions = validation_reg.loc[
    persistence_mask,
    "pm25_mean"
]

persistence_results = regression_metrics(
    persistence_actual,
    persistence_predictions
)

print("\nPersistence baseline - 2023 validation:")
print(f"Observations: {persistence_mask.sum()}")
print(f"MAE: {persistence_results['MAE']:.3f}")
print(f"RMSE: {persistence_results['RMSE']:.3f}")
print(f"R²: {persistence_results['R2']:.3f}")


# Define the initial candidate regression models
ridge_initial = Pipeline(
    steps=[
        ("preprocessor", ridge_preprocessor),
        ("model", Ridge(alpha=1.0))
    ]
)

rf_initial = Pipeline(
    steps=[
        ("preprocessor", tree_preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

xgb_initial = Pipeline(
    steps=[
        ("preprocessor", tree_preprocessor),
        (
            "model",
            XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

initial_models = {
    "Ridge Regression": ridge_initial,
    "Random Forest": rf_initial,
    "XGBoost": xgb_initial
}


# Train and evaluate the initial candidate models
initial_results = []

for model_name, model in initial_models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_validation)

    metrics = regression_metrics(
        y_validation,
        predictions
    )

    initial_results.append(
        {
            "Model": model_name,
            **metrics
        }
    )

initial_results = pd.DataFrame(initial_results)

print("\nInitial 2023 validation results:")
print(
    initial_results[
        ["Model", "MAE", "RMSE", "R2"]
    ]
    .round(4)
    .to_string(index=False)
)


# Compare all candidates and persistence on identical validation rows
common_validation_mask = validation_reg["pm25_mean"].notna()

X_validation_common = X_validation.loc[
    common_validation_mask
]

y_validation_common = y_validation.loc[
    common_validation_mask
]

common_results = [
    {
        "Model": "Persistence",
        **regression_metrics(
            y_validation_common,
            validation_reg.loc[
                common_validation_mask,
                "pm25_mean"
            ]
        )
    }
]

for model_name, model in initial_models.items():

    predictions = model.predict(
        X_validation_common
    )

    common_results.append(
        {
            "Model": model_name,
            **regression_metrics(
                y_validation_common,
                predictions
            )
        }
    )

common_results = pd.DataFrame(common_results)

print("\nCommon-row 2023 validation comparison:")
print(
    common_results
    .round(4)
    .to_string(index=False)
)

print("\nBaseline and candidate model stage complete.")
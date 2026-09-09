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


# Define regression target and predictors
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


# Create chronological training and validation periods
train_reg = integrated_daily[
    (integrated_daily["date"].dt.year <= 2022)
    & integrated_daily[regression_target].notna()
].copy()

validation_reg = integrated_daily[
    (integrated_daily["date"].dt.year == 2023)
    & integrated_daily[regression_target].notna()
].copy()


# Separate predictors and targets
X_train = train_reg[model_features]
y_train = train_reg[regression_target]

X_validation = validation_reg[model_features]
y_validation = validation_reg[regression_target]


# Use common validation rows for fair model comparison
common_validation_mask = validation_reg["pm25_mean"].notna()

X_validation_common = X_validation.loc[
    common_validation_mask
]

y_validation_common = y_validation.loc[
    common_validation_mask
]


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


# Ridge requires scaling after median imputation
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


# Calculate regression metrics
def regression_metrics(y_true, predictions):
    return {
        "MAE": mean_absolute_error(y_true, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_true, predictions)),
        "R2": r2_score(y_true, predictions)
    }


# Tune Ridge regularisation strength
ridge_alphas = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0
]

ridge_tuning_results = []

for alpha in ridge_alphas:

    model = Pipeline(
        steps=[
            ("preprocessor", ridge_preprocessor),
            ("model", Ridge(alpha=alpha))
        ]
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_validation_common
    )

    metrics = regression_metrics(
        y_validation_common,
        predictions
    )

    ridge_tuning_results.append(
        {
            "Alpha": alpha,
            **metrics
        }
    )

ridge_tuning_results = pd.DataFrame(
    ridge_tuning_results
).sort_values(
    "RMSE"
)

print("\nRidge tuning:")
print(
    ridge_tuning_results
    .round(4)
    .to_string(index=False)
)


# Tune Random Forest
rf_settings = [
    {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 1
    },
    {
        "n_estimators": 300,
        "max_depth": 10,
        "min_samples_leaf": 4
    },
    {
        "n_estimators": 300,
        "max_depth": 15,
        "min_samples_leaf": 4
    },
    {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 4
    }
]

rf_tuning_results = []

for settings in rf_settings:

    model = Pipeline(
        steps=[
            ("preprocessor", tree_preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=settings["n_estimators"],
                    max_depth=settings["max_depth"],
                    min_samples_leaf=settings["min_samples_leaf"],
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_validation_common
    )

    metrics = regression_metrics(
        y_validation_common,
        predictions
    )

    rf_tuning_results.append(
        {
            **settings,
            **metrics
        }
    )

rf_tuning_results = pd.DataFrame(
    rf_tuning_results
).sort_values(
    "RMSE"
)

print("\nRandom Forest tuning:")
print(
    rf_tuning_results
    .round(4)
    .to_string(index=False)
)


# Tune XGBoost
xgb_settings = [
    {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 2
    },
    {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 2
    },
    {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 2
    },
    {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 3
    }
]

xgb_tuning_results = []

for settings in xgb_settings:

    model = Pipeline(
        steps=[
            ("preprocessor", tree_preprocessor),
            (
                "model",
                XGBRegressor(
                    n_estimators=settings["n_estimators"],
                    learning_rate=settings["learning_rate"],
                    max_depth=settings["max_depth"],
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_validation_common
    )

    metrics = regression_metrics(
        y_validation_common,
        predictions
    )

    xgb_tuning_results.append(
        {
            **settings,
            **metrics
        }
    )

xgb_tuning_results = pd.DataFrame(
    xgb_tuning_results
).sort_values(
    "RMSE"
)

print("\nXGBoost tuning:")
print(
    xgb_tuning_results
    .round(4)
    .to_string(index=False)
)


# Compare the best configuration from each model family
best_ridge = ridge_tuning_results.iloc[0]
best_rf = rf_tuning_results.iloc[0]
best_xgb = xgb_tuning_results.iloc[0]

selected_validation_results = pd.DataFrame(
    [
        {
            "Model": "Ridge Regression",
            "MAE": best_ridge["MAE"],
            "RMSE": best_ridge["RMSE"],
            "R2": best_ridge["R2"]
        },
        {
            "Model": "Random Forest",
            "MAE": best_rf["MAE"],
            "RMSE": best_rf["RMSE"],
            "R2": best_rf["R2"]
        },
        {
            "Model": "XGBoost",
            "MAE": best_xgb["MAE"],
            "RMSE": best_xgb["RMSE"],
            "R2": best_xgb["R2"]
        }
    ]
)

print("\nSelected 2023 validation comparison:")
print(
    selected_validation_results
    .round(4)
    .to_string(index=False)
)


# Freeze Ridge as the selected regression model
selected_ridge_alpha = float(
    best_ridge["Alpha"]
)

print("\nSelected final model: Ridge Regression")
print("Selected alpha:", selected_ridge_alpha)

print("\nRegression model tuning and selection stage complete.")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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


# Define the regression target
regression_target = "target_pm25_next_day"


# Use the shared predictor set
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


# Create chronological development and test periods
train_reg = integrated_daily[
    (integrated_daily["date"].dt.year <= 2022)
    & integrated_daily[regression_target].notna()
].copy()

validation_reg = integrated_daily[
    (integrated_daily["date"].dt.year == 2023)
    & integrated_daily[regression_target].notna()
].copy()

test_reg = integrated_daily[
    (integrated_daily["date"].dt.year == 2024)
    & integrated_daily[regression_target].notna()
].copy()


# Prepare the unseen 2024 test data
X_test = test_reg[model_features]
y_test = test_reg[regression_target]


# Separate categorical and numerical predictors
categorical_features = ["station"]

numeric_features = [
    feature
    for feature in model_features
    if feature not in categorical_features
]


# Preprocess the station variable
categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)


# Ridge uses median imputation and numerical scaling
ridge_numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


ridge_preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        ),
        (
            "numeric",
            ridge_numeric_pipeline,
            numeric_features
        )
    ]
)


# Calculate regression evaluation metrics
def regression_metrics(y_true, predictions):
    return {
        "MAE": mean_absolute_error(
            y_true,
            predictions
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                y_true,
                predictions
            )
        ),
        "R2": r2_score(
            y_true,
            predictions
        )
    }


# Freeze the model selected using 2023 validation performance
final_ridge_model = Pipeline(
    steps=[
        (
            "preprocessor",
            ridge_preprocessor
        ),
        (
            "model",
            Ridge(alpha=0.01)
        )
    ]
)

print("\nSelected final model: Ridge Regression")
print("Selected alpha: 0.01")


# Refit the selected model using 2018-2023 development data
development_reg = pd.concat(
    [
        train_reg,
        validation_reg
    ],
    ignore_index=True
)

X_development = development_reg[
    model_features
]

y_development = development_reg[
    regression_target
]


final_ridge_model.fit(
    X_development,
    y_development
)

print(
    "Final refit observations:",
    len(development_reg)
)


# Evaluate once on the unseen 2024 test set
test_predictions = final_ridge_model.predict(
    X_test
)

test_metrics = regression_metrics(
    y_test,
    test_predictions
)

print("\nFinal 2024 regression performance:")
print(
    f"MAE: {test_metrics['MAE']:.3f}"
)
print(
    f"RMSE: {test_metrics['RMSE']:.3f}"
)
print(
    f"R²: {test_metrics['R2']:.3f}"
)


# Compare Ridge and persistence on identical 2024 rows
test_common_mask = test_reg[
    "pm25_mean"
].notna()

y_test_common = y_test.loc[
    test_common_mask
]

persistence_test_predictions = test_reg.loc[
    test_common_mask,
    "pm25_mean"
]

ridge_test_common_predictions = final_ridge_model.predict(
    X_test.loc[
        test_common_mask
    ]
)


persistence_test_metrics = regression_metrics(
    y_test_common,
    persistence_test_predictions
)

ridge_test_common_metrics = regression_metrics(
    y_test_common,
    ridge_test_common_predictions
)


test_common_comparison = pd.DataFrame(
    [
        {
            "Model": "Persistence",
            **persistence_test_metrics
        },
        {
            "Model": "Ridge Regression",
            **ridge_test_common_metrics
        }
    ]
)

print("\nCommon-row 2024 comparison:")
print(
    "Observations:",
    test_common_mask.sum()
)

print(
    test_common_comparison
    .round(4)
    .to_string(index=False)
)


# Calculate residuals
residuals = (
    y_test.to_numpy()
    - test_predictions
)

print("\nResidual diagnostics:")
print(
    "Mean residual:",
    round(residuals.mean(), 3)
)
print(
    "Median residual:",
    round(np.median(residuals), 3)
)
print(
    "Minimum residual:",
    round(residuals.min(), 3)
)
print(
    "Maximum residual:",
    round(residuals.max(), 3)
)


# Compare performance on normal and elevated PM2.5 days
normal_mask = (
    y_test.to_numpy() <= 25
)

elevated_mask = (
    y_test.to_numpy() > 25
)

normal_metrics = regression_metrics(
    y_test.to_numpy()[normal_mask],
    test_predictions[normal_mask]
)

elevated_metrics = regression_metrics(
    y_test.to_numpy()[elevated_mask],
    test_predictions[elevated_mask]
)


print("\nObserved next-day PM2.5 <= 25:")
print(
    "Observations:",
    normal_mask.sum()
)
print(
    f"MAE: {normal_metrics['MAE']:.3f}"
)
print(
    f"RMSE: {normal_metrics['RMSE']:.3f}"
)
print(
    "Mean residual:",
    round(
        residuals[normal_mask].mean(),
        3
    )
)


print("\nObserved next-day PM2.5 > 25:")
print(
    "Observations:",
    elevated_mask.sum()
)
print(
    f"MAE: {elevated_metrics['MAE']:.3f}"
)
print(
    f"RMSE: {elevated_metrics['RMSE']:.3f}"
)
print(
    "Mean residual:",
    round(
        residuals[elevated_mask].mean(),
        3
    )
)


# Plot observed against predicted PM2.5
plt.figure(figsize=(7, 6))

plt.scatter(
    y_test,
    test_predictions,
    alpha=0.35,
    s=18
)

plot_min = min(
    y_test.min(),
    test_predictions.min()
)

plot_max = max(
    y_test.max(),
    test_predictions.max()
)

plt.plot(
    [plot_min, plot_max],
    [plot_min, plot_max],
    linestyle="--"
)

plt.xlabel(
    "Observed next-day PM2.5 (µg/m³)"
)

plt.ylabel(
    "Predicted next-day PM2.5 (µg/m³)"
)

plt.title(
    "Observed vs Predicted PM2.5 - 2024"
)

plt.tight_layout()
plt.show()


# Plot residual distribution
plt.figure(figsize=(7, 5))

plt.hist(
    residuals,
    bins=40
)

plt.xlabel(
    "Residual: observed - predicted (µg/m³)"
)

plt.ylabel("Frequency")

plt.title(
    "Ridge Regression Residual Distribution - 2024"
)

plt.tight_layout()
plt.show()


print(
    "\nFinal regression evaluation and diagnostics stage complete."
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import shap

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge


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


# Prepare development and test data
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

X_test = test_reg[
    model_features
]

y_test = test_reg[
    regression_target
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


# Recreate the selected final Ridge model
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


# Refit on 2018-2023 development data
final_ridge_model.fit(
    X_development,
    y_development
)


# Generate 2024 predictions
test_predictions = final_ridge_model.predict(
    X_test
)

residuals = (
    y_test.to_numpy()
    - test_predictions
)


# Prepare transformed data for SHAP
final_preprocessor = (
    final_ridge_model
    .named_steps["preprocessor"]
)

final_ridge = (
    final_ridge_model
    .named_steps["model"]
)

X_test_transformed = (
    final_preprocessor.transform(
        X_test
    )
)


# Retrieve transformed feature names
transformed_feature_names = (
    final_preprocessor
    .get_feature_names_out()
)


# Make feature names easier to read
clean_feature_names = [
    name
    .replace("categorical__", "")
    .replace("numeric__", "")
    for name in transformed_feature_names
]


# Explain the final Ridge model
explainer = shap.LinearExplainer(
    final_ridge,
    X_test_transformed
)

shap_values = explainer(
    X_test_transformed
)


# Create and save the SHAP beeswarm plot
plt.figure()

shap.summary_plot(
    shap_values.values,
    X_test_transformed,
    feature_names=clean_feature_names,
    show=False,
    max_display=15
)

plt.title(
    "SHAP Feature Contributions - Final Ridge Model"
)

plt.tight_layout()

shap_output = (
    project_folder
    / "ridge_shap_beeswarm_2024.png"
)

plt.savefig(
    shap_output,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Save 2024 regression predictions
regression_predictions = pd.DataFrame(
    {
        "date": test_reg["date"].values,
        "station": test_reg["station"].values,
        "observed_pm25_next_day": y_test.values,
        "predicted_pm25_next_day": test_predictions,
        "residual": residuals
    }
)

regression_predictions.to_csv(
    project_folder
    / "Fire2Air_regression_predictions_2024.csv",
    index=False
)


# Save the final fitted Ridge pipeline
joblib.dump(
    final_ridge_model,
    project_folder
    / "Fire2Air_final_ridge_model.joblib"
)


print(
    "\nRegression predictions saved:",
    regression_predictions.shape
)

print(
    "Final Ridge model saved."
)

print(
    "SHAP plot saved:",
    shap_output.name
)

print(
    "\nRegression explainability and output stage complete."
)
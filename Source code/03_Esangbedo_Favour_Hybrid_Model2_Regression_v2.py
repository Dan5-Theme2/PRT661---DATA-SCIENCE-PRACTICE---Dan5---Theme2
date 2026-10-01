# %% [markdown]
# # PART 3 — ESANGBEDO FAVOUR: MODEL 2 REGRESSION
#
# Extracted directly from Fire2Air_Darwin_Complete_Hybrid_All_Team_FIXED_v2.py
# Team responsibility version — code preserved from the corrected combined file.

# ==================================================================================================
# PART 3 — ESANGBEDO FAVOUR: MODEL 2 REGRESSION
# ==================================================================================================

# %% [markdown]
# # Fire2Air Darwin — 03 Model 2 Regression
# **Member:** Esangbedo Favour  
# **Responsibility:** Build the next-day PM₂.₅ concentration model and evaluate it using MAE and RMSE.
# 
# This notebook keeps the stronger newer regression workflow: persistence baseline, Ridge, Random Forest and XGBoost, validation tuning, common-row baseline comparison, residual diagnostics and final 2024 evaluation. The only structural safeguard added is splitting by `target_date` from Thi's hybrid checkpoint.
# 
# Project-wide explainability is kept in Navodya's notebook to match the team responsibility table.

# %%
from pathlib import Path
import json
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except Exception as exc:
    HAS_XGBOOST = False
    print("XGBoost unavailable:", exc)

RANDOM_STATE = 42
possible_folders = [Path.cwd(), Path.cwd() / "Data Science", Path.home() / "Desktop" / "Data Science"]
project_folder = next(
    (folder.resolve() for folder in possible_folders if (folder / "Fire2Air_model_ready_checkpoint.csv").exists()),
    None,
)
if project_folder is None:
    raise FileNotFoundError("Fire2Air_model_ready_checkpoint.csv not found. Run Thi's notebook first.")

output_root = project_folder / "outputs_prt661"
table_dir = output_root / "tables"
figure_dir = output_root / "figures"
model_dir = output_root / "models"
for folder in [table_dir, figure_dir, model_dir]:
    folder.mkdir(parents=True, exist_ok=True)

integrated_daily = pd.read_csv(
    project_folder / "Fire2Air_model_ready_checkpoint.csv",
    parse_dates=["date", "target_date"],
)
with open(project_folder / "Fire2Air_feature_manifest.json", "r", encoding="utf-8") as f:
    manifest = json.load(f)
model_features = manifest["features"]
regression_target = manifest["targets"]["regression"]
print("Project folder:", project_folder)
print("Dataset shape:", integrated_daily.shape)

# %% [markdown]
# ## A. Chronological regression split
# The split is based on the predicted day (`target_date`): 2018–2022 training, 2023 validation, 2024 final test.

# %%
regression_data = integrated_daily.dropna(subset=[regression_target, "target_date"]).copy()
regression_data["target_year"] = regression_data["target_date"].dt.year

train_reg = regression_data[regression_data["target_year"] <= 2022].copy()
validation_reg = regression_data[regression_data["target_year"] == 2023].copy()
test_reg = regression_data[regression_data["target_year"] == 2024].copy()

if min(len(train_reg), len(validation_reg), len(test_reg)) == 0:
    raise ValueError(
        f"Insufficient chronological data: train={len(train_reg)}, "
        f"validation={len(validation_reg)}, test={len(test_reg)}"
    )

assert train_reg["target_date"].max() < validation_reg["target_date"].min()
assert validation_reg["target_date"].max() < test_reg["target_date"].min()

X_train, y_train = train_reg[model_features], train_reg[regression_target]
X_validation, y_validation = validation_reg[model_features], validation_reg[regression_target]
X_test, y_test = test_reg[model_features], test_reg[regression_target]

print("Train rows:", len(train_reg))
print("Validation rows:", len(validation_reg))
print("Test rows:", len(test_reg))

# %% [markdown]
# ## B. Preprocessing, persistence baseline and model metrics
# MAE and RMSE are primary. R² is secondary. Persistence predicts tomorrow's PM₂.₅ as today's PM₂.₅.

# %%
categorical_features = ["station"]
numeric_features = [f for f in model_features if f not in categorical_features]

def onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", onehot()),
])

ridge_preprocessor = ColumnTransformer([
    ("categorical", categorical_pipeline, categorical_features),
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric_features),
])

tree_preprocessor = ColumnTransformer([
    ("categorical", categorical_pipeline, categorical_features),
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ]), numeric_features),
])

def regression_metrics(y_true, predictions):
    return {
        "MAE": mean_absolute_error(y_true, predictions),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "R2": r2_score(y_true, predictions),
    }

# Persistence validation baseline on rows where today's PM2.5 is available.
persistence_validation = X_validation["pm25_mean"].to_numpy()
valid_persistence = ~pd.isna(persistence_validation)

persistence_metrics = regression_metrics(
    y_validation.loc[valid_persistence],
    persistence_validation[valid_persistence],
)

print(
    "\nPersistence validation baseline:",
    {k: round(v, 4) for k, v in persistence_metrics.items()},
)


# %% [markdown]
# ## C. Validation tuning
# Ridge, Random Forest and XGBoost are tuned using only 2023. The selected configuration is the one with the lowest validation RMSE.

# %%
configurations = []
for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
    configurations.append((
        "Ridge", f"alpha={alpha}",
        Pipeline([
            ("preprocessor", ridge_preprocessor),
            ("model", Ridge(alpha=alpha)),
        ])
    ))

for max_depth, min_samples_leaf in [(None, 1), (None, 2), (10, 2), (15, 4)]:
    configurations.append((
        "Random Forest", f"max_depth={max_depth};min_samples_leaf={min_samples_leaf}",
        Pipeline([
            ("preprocessor", tree_preprocessor),
            ("model", RandomForestRegressor(
                n_estimators=300,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ])
    ))

if HAS_XGBOOST:
    for n_estimators, learning_rate, max_depth in [
        (200, 0.03, 2), (300, 0.03, 3), (300, 0.05, 3), (400, 0.05, 4)
    ]:
        configurations.append((
            "XGBoost", f"n_estimators={n_estimators};learning_rate={learning_rate};max_depth={max_depth}",
            Pipeline([
                ("preprocessor", tree_preprocessor),
                ("model", XGBRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    objective="reg:squarederror",
                    eval_metric="rmse",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )),
            ])
        ))

validation_rows = [{"Model": "Persistence", "Config": "y(t+1)=PM2.5(t)", **persistence_metrics}]
model_templates = {}

for model_name, config_id, pipeline in configurations:
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_validation)
    metrics = regression_metrics(y_validation, predictions)
    validation_rows.append({"Model": model_name, "Config": config_id, **metrics})
    model_templates[(model_name, config_id)] = pipeline

validation_results = pd.DataFrame(validation_rows).sort_values("RMSE").reset_index(drop=True)
print("\n2023 validation results:")
print(validation_results.round(4).to_string(index=False))
validation_results.to_csv(table_dir / "model2_regression_validation_results.csv", index=False)

best_nonbaseline = validation_results[validation_results["Model"] != "Persistence"].iloc[0]
best_model_name = best_nonbaseline["Model"]
best_config_id = best_nonbaseline["Config"]
selected_template = model_templates[(best_model_name, best_config_id)]
print("\nSelected fitted model by validation RMSE:", best_model_name)
print("Configuration:", best_config_id)

# %% [markdown]
# ## D. Final 2024 evaluation and diagnostics
# The selected model is refit on 2018–2023, then evaluated once on 2024. A fair persistence comparison uses the same test rows.

# %%
development_reg = pd.concat([train_reg, validation_reg], ignore_index=True)
X_development = development_reg[model_features]
y_development = development_reg[regression_target]

final_regression_model = clone(selected_template)
final_regression_model.fit(X_development, y_development)
test_predictions = final_regression_model.predict(X_test)
final_metrics = regression_metrics(y_test, test_predictions)

# Persistence on the same 2024 rows.
persistence_test = X_test["pm25_mean"].to_numpy()
common_mask = ~pd.isna(persistence_test)
persistence_test_metrics = regression_metrics(
    y_test.loc[common_mask], persistence_test[common_mask]
)
selected_common_metrics = regression_metrics(
    y_test.loc[common_mask], test_predictions[common_mask]
)

final_results = pd.DataFrame([
    {"Model": best_model_name, "Scope": "All 2024 model rows", **final_metrics},
    {"Model": best_model_name, "Scope": "Common rows with persistence", **selected_common_metrics},
    {"Model": "Persistence", "Scope": "Common rows with selected model", **persistence_test_metrics},
])
print("\nFinal 2024 regression comparison:")
print(final_results.round(4).to_string(index=False))
final_results.to_csv(table_dir / "model2_final_2024_metrics.csv", index=False)

residuals = y_test.to_numpy() - test_predictions

# Normal vs elevated next-day PM2.5 performance.
strata_rows = []
for label, mask in [
    ("PM2.5 < 25", y_test < 25),
    ("PM2.5 >= 25", y_test >= 25),
]:
    if int(mask.sum()) > 0:
        strata_rows.append({"Stratum": label, "Rows": int(mask.sum()), **regression_metrics(y_test[mask], test_predictions[mask])})
strata_results = pd.DataFrame(strata_rows)
print("\n2024 performance by PM2.5 level:")
print(strata_results.round(4).to_string(index=False))
strata_results.to_csv(table_dir / "model2_2024_stratified_metrics.csv", index=False)

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test, test_predictions, alpha=0.35, s=15)
limits = [min(y_test.min(), np.min(test_predictions)), max(y_test.max(), np.max(test_predictions))]
ax.plot(limits, limits, linestyle="--")
ax.set_xlim(limits); ax.set_ylim(limits)
ax.set_xlabel("Observed next-day PM₂.₅")
ax.set_ylabel("Predicted next-day PM₂.₅")
ax.set_title(f"2024 Observed vs Predicted — {best_model_name}")
fig.tight_layout()
fig.savefig(figure_dir / "model2_2024_observed_vs_predicted.png", dpi=180, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(residuals, bins=35)
ax.axvline(0, linestyle="--")
ax.set_xlabel("Residual (observed − predicted)")
ax.set_ylabel("Frequency")
ax.set_title("2024 Regression Residual Distribution")
fig.tight_layout()
fig.savefig(figure_dir / "model2_2024_residual_distribution.png", dpi=180, bbox_inches="tight")
plt.show()

# %%
regression_predictions = test_reg[["date", "target_date", "station", regression_target]].copy()
regression_predictions["prediction"] = test_predictions
regression_predictions["residual"] = residuals
regression_predictions.to_csv(table_dir / "model2_2024_predictions.csv", index=False)

joblib.dump(final_regression_model, model_dir / "model2_regression.joblib")
with open(model_dir / "model2_regression_config.json", "w", encoding="utf-8") as f:
    json.dump({
        "model": best_model_name,
        "config": best_config_id,
        "primary_metrics": ["MAE", "RMSE"],
        "features": model_features,
    }, f, indent=2)

print("\nSaved regression model, metrics, predictions and diagnostic figures.")

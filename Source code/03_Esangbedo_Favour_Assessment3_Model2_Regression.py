# %% [markdown]
# # Fire2Air Darwin — Assessment 3
# ## 03 Model 2 Regression: Next-Day Mean PM2.5 Concentration
# **Member:** Esangbedo Favour
#
# Assessment 3 upgrades:
# - consumes the same Assessment 3 FSI-aware feature table as the other models;
# - retains persistence, Ridge, Random Forest and XGBoost;
# - selects the model/configuration using 2023 validation RMSE;
# - adds A–F ablation analysis, elevated-PM2.5 error analysis,
#   station-level metrics, leave-one-station-out validation and bootstrap CIs;
# - explicitly measures elevated-event underprediction before/after advanced features.

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
BOOTSTRAP_REPEATS = 300
PM25_THRESHOLD = 25.0

possible_folders = [Path.cwd(), Path.cwd() / "Data Science", Path.home() / "Desktop" / "Data Science"]
project_folder = next(
    (folder.resolve() for folder in possible_folders if (folder / "Fire2Air_model_ready_checkpoint_A3.csv").exists()),
    None,
)
if project_folder is None:
    raise FileNotFoundError(
        "Fire2Air_model_ready_checkpoint_A3.csv not found. Run the Assessment 3 Thi preprocessing script first."
    )

output_root = project_folder / "outputs_prt661_a3"
table_dir = output_root / "tables"
figure_dir = output_root / "figures"
model_dir = output_root / "models"
for folder in [table_dir, figure_dir, model_dir]:
    folder.mkdir(parents=True, exist_ok=True)

integrated_daily = pd.read_csv(
    project_folder / "Fire2Air_model_ready_checkpoint_A3.csv",
    parse_dates=["date", "target_date"],
)
with open(project_folder / "Fire2Air_feature_manifest_A3.json", "r", encoding="utf-8") as f:
    manifest = json.load(f)
model_features = manifest["features"]
feature_groups = manifest["feature_groups"]
regression_target = manifest["targets"]["regression"]
PM25_THRESHOLD = float(manifest["threshold_pm25_ug_m3"])

print("Project folder:", project_folder)
print("Dataset shape:", integrated_daily.shape)


# %% [markdown]
# ## A. Chronological regression split
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


# %% [markdown]
# ## B. Model builders and metrics

def onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(features, scale_numeric=False):
    categorical = [f for f in features if f == "station"]
    numeric = [f for f in features if f not in categorical]
    transformers = []
    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", onehot()),
            ]),
            categorical,
        ))
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    if numeric:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric))
    return ColumnTransformer(transformers)


def build_regressor(model_name, params, features):
    if model_name == "Ridge":
        estimator = Ridge(alpha=params["alpha"])
        prep = build_preprocessor(features, scale_numeric=True)
    elif model_name == "Random Forest":
        estimator = RandomForestRegressor(
            n_estimators=300,
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        prep = build_preprocessor(features, scale_numeric=False)
    elif model_name == "XGBoost":
        if not HAS_XGBOOST:
            raise RuntimeError("XGBoost was selected but is unavailable.")
        estimator = XGBRegressor(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        prep = build_preprocessor(features, scale_numeric=False)
    else:
        raise ValueError(f"Unknown regressor: {model_name}")
    return Pipeline([("preprocessor", prep), ("model", estimator)])


def regression_metrics(y_true, predictions):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(predictions, dtype=float)
    return {
        "MAE": mean_absolute_error(y, p),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "R2": r2_score(y, p) if len(y) >= 2 else np.nan,
        "Bias_PredMinusObs": float(np.mean(p - y)),
        "Underprediction_Rate": float(np.mean(p < y)),
    }


def elevated_metrics(y_true, predictions, threshold=PM25_THRESHOLD):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(predictions, dtype=float)
    mask = y >= threshold
    if mask.sum() == 0:
        return {
            "Elevated_Rows": 0,
            "Elevated_MAE": np.nan,
            "Elevated_RMSE": np.nan,
            "Elevated_Bias_PredMinusObs": np.nan,
            "Elevated_Underprediction_Rate": np.nan,
        }
    return {
        "Elevated_Rows": int(mask.sum()),
        "Elevated_MAE": mean_absolute_error(y[mask], p[mask]),
        "Elevated_RMSE": float(np.sqrt(mean_squared_error(y[mask], p[mask]))),
        "Elevated_Bias_PredMinusObs": float(np.mean(p[mask] - y[mask])),
        "Elevated_Underprediction_Rate": float(np.mean(p[mask] < y[mask])),
    }


# %% [markdown]
# ## C. Persistence baseline and candidate-model tuning
persistence_validation = X_validation["pm25_mean"].to_numpy()
valid_persistence = ~pd.isna(persistence_validation)
persistence_metrics = regression_metrics(
    y_validation.loc[valid_persistence],
    persistence_validation[valid_persistence],
)
persistence_elevated = elevated_metrics(
    y_validation.loc[valid_persistence],
    persistence_validation[valid_persistence],
)

candidate_specs = []
for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
    candidate_specs.append(("Ridge", {"alpha": alpha}, f"alpha={alpha}"))
for max_depth, min_samples_leaf in [(None, 1), (None, 2), (10, 2), (15, 4)]:
    candidate_specs.append((
        "Random Forest",
        {"max_depth": max_depth, "min_samples_leaf": min_samples_leaf},
        f"max_depth={max_depth};min_samples_leaf={min_samples_leaf}",
    ))
if HAS_XGBOOST:
    for n_estimators, learning_rate, max_depth in [
        (200, 0.03, 2), (300, 0.03, 3), (300, 0.05, 3), (400, 0.05, 4)
    ]:
        candidate_specs.append((
            "XGBoost",
            {"n_estimators": n_estimators, "learning_rate": learning_rate, "max_depth": max_depth},
            f"n_estimators={n_estimators};learning_rate={learning_rate};max_depth={max_depth}",
        ))

validation_rows = [{
    "Model": "Persistence",
    "Config": "y(t+1)=PM2.5(t)",
    **persistence_metrics,
    **persistence_elevated,
}]
model_templates = {}
params_lookup = {}
for model_name, params, config_id in candidate_specs:
    model = build_regressor(model_name, params, model_features)
    model.fit(X_train, y_train)
    pred = model.predict(X_validation)
    validation_rows.append({
        "Model": model_name,
        "Config": config_id,
        **regression_metrics(y_validation, pred),
        **elevated_metrics(y_validation, pred),
    })
    model_templates[(model_name, config_id)] = model
    params_lookup[(model_name, config_id)] = params

validation_results = pd.DataFrame(validation_rows).sort_values("RMSE").reset_index(drop=True)
validation_results.to_csv(table_dir / "a3_model2_regression_validation_results.csv", index=False)
print("\n2023 regression validation results:")
print(validation_results.round(4).to_string(index=False))

best_nonbaseline = validation_results[validation_results["Model"] != "Persistence"].iloc[0]
best_model_name = best_nonbaseline["Model"]
best_config_id = best_nonbaseline["Config"]
selected_params = params_lookup[(best_model_name, best_config_id)]
selected_template = model_templates[(best_model_name, best_config_id)]
print("\nSelected regression model by validation RMSE:", best_model_name)
print("Configuration:", best_config_id)


# %% [markdown]
# ## D. A–F ablation study
# The selected model family/configuration is fixed while the advanced feature groups are added.
ablation_rows = []
for stage, stage_features in feature_groups.items():
    stage_features = [f for f in stage_features if f in regression_data.columns]
    stage_model = build_regressor(best_model_name, selected_params, stage_features)
    stage_model.fit(train_reg[stage_features], y_train)
    pred = stage_model.predict(validation_reg[stage_features])
    ablation_rows.append({
        "Stage": stage,
        "Feature_Count": len(stage_features),
        **regression_metrics(y_validation, pred),
        **elevated_metrics(y_validation, pred),
    })
ablation_results = pd.DataFrame(ablation_rows)
ablation_results.to_csv(table_dir / "a3_model2_ablation_results.csv", index=False)
print("\nRegression ablation results:")
print(ablation_results.round(4).to_string(index=False))


# %% [markdown]
# ## E. Final 2024 evaluation
development_reg = pd.concat([train_reg, validation_reg], ignore_index=True)
X_development = development_reg[model_features]
y_development = development_reg[regression_target]

final_regression_model = clone(selected_template)
final_regression_model.fit(X_development, y_development)
test_predictions = final_regression_model.predict(X_test)

# Fair persistence comparison on the same test rows.
persistence_test = X_test["pm25_mean"].to_numpy()
common_mask = ~pd.isna(persistence_test)

final_results = pd.DataFrame([
    {
        "Model": best_model_name,
        "Scope": "All 2024 model rows",
        **regression_metrics(y_test, test_predictions),
        **elevated_metrics(y_test, test_predictions),
    },
    {
        "Model": best_model_name,
        "Scope": "Common rows with persistence",
        **regression_metrics(y_test.loc[common_mask], test_predictions[common_mask]),
        **elevated_metrics(y_test.loc[common_mask], test_predictions[common_mask]),
    },
    {
        "Model": "Persistence",
        "Scope": "Common rows with selected model",
        **regression_metrics(y_test.loc[common_mask], persistence_test[common_mask]),
        **elevated_metrics(y_test.loc[common_mask], persistence_test[common_mask]),
    },
])
final_results.to_csv(table_dir / "a3_model2_final_2024_metrics.csv", index=False)
print("\nFinal 2024 regression comparison:")
print(final_results.round(4).to_string(index=False))

# Normal vs elevated next-day performance.
strata_rows = []
for label, mask in [
    ("PM2.5 < 25", y_test < PM25_THRESHOLD),
    ("PM2.5 >= 25", y_test >= PM25_THRESHOLD),
]:
    if int(mask.sum()) > 0:
        strata_rows.append({
            "Stratum": label,
            "Rows": int(mask.sum()),
            **regression_metrics(y_test[mask], test_predictions[mask]),
        })
strata_results = pd.DataFrame(strata_rows)
strata_results.to_csv(table_dir / "a3_model2_2024_stratified_metrics.csv", index=False)

# Observed vs predicted.
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test, test_predictions, alpha=0.35, s=15)
limits = [min(y_test.min(), np.min(test_predictions)), max(y_test.max(), np.max(test_predictions))]
ax.plot(limits, limits, linestyle="--")
ax.axvline(PM25_THRESHOLD, linestyle=":", linewidth=1)
ax.axhline(PM25_THRESHOLD, linestyle=":", linewidth=1)
ax.set_xlim(limits)
ax.set_ylim(limits)
ax.set_xlabel("Observed next-day PM2.5")
ax.set_ylabel("Predicted next-day PM2.5")
ax.set_title(f"2024 Observed vs Predicted — {best_model_name}")
fig.tight_layout()
fig.savefig(figure_dir / "a3_model2_2024_observed_vs_predicted.png", dpi=180, bbox_inches="tight")
plt.close(fig)

residuals = y_test.to_numpy() - test_predictions
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(residuals, bins=35)
ax.axvline(0, linestyle="--")
ax.set_xlabel("Residual (observed - predicted)")
ax.set_ylabel("Frequency")
ax.set_title("2024 Regression Residual Distribution")
fig.tight_layout()
fig.savefig(figure_dir / "a3_model2_2024_residual_distribution.png", dpi=180, bbox_inches="tight")
plt.close(fig)


# %% [markdown]
# ## F. Station-level performance
station_rows = []
test_reset = test_reg.reset_index(drop=True)
y_reset = y_test.reset_index(drop=True)
for station, g in test_reset.groupby("station"):
    idx = g.index.to_numpy()
    station_rows.append({
        "Station": station,
        "Rows": len(idx),
        **regression_metrics(y_reset.iloc[idx], test_predictions[idx]),
        **elevated_metrics(y_reset.iloc[idx], test_predictions[idx]),
    })
station_metrics = pd.DataFrame(station_rows)
station_metrics.to_csv(table_dir / "a3_model2_2024_metrics_by_station.csv", index=False)
print("\n2024 regression by station:")
print(station_metrics.round(4).to_string(index=False))


# %% [markdown]
# ## G. Leave-one-station-out transfer evaluation
loso_rows = []
for held_out in sorted(regression_data["station"].dropna().unique()):
    train_loso = development_reg[development_reg["station"] != held_out].copy()
    test_loso = test_reg[test_reg["station"] == held_out].copy()
    if train_loso.empty or test_loso.empty:
        continue
    model = build_regressor(best_model_name, selected_params, model_features)
    model.fit(train_loso[model_features], train_loso[regression_target])
    pred = model.predict(test_loso[model_features])
    loso_rows.append({
        "Held_Out_Station": held_out,
        "Train_Stations": ", ".join(sorted(train_loso["station"].unique())),
        "Test_Rows_2024": len(test_loso),
        **regression_metrics(test_loso[regression_target], pred),
        **elevated_metrics(test_loso[regression_target], pred),
    })
loso_results = pd.DataFrame(loso_rows)
loso_results.to_csv(table_dir / "a3_model2_leave_one_station_out.csv", index=False)


# %% [markdown]
# ## H. Bootstrap uncertainty
rng = np.random.default_rng(RANDOM_STATE)
y_arr = np.asarray(y_test)
p_arr = np.asarray(test_predictions)
bootstrap_rows = []
for _ in range(BOOTSTRAP_REPEATS):
    idx = rng.integers(0, len(y_arr), len(y_arr))
    y_b = y_arr[idx]
    p_b = p_arr[idx]
    m = regression_metrics(y_b, p_b)
    e = elevated_metrics(y_b, p_b)
    bootstrap_rows.append({
        "MAE": m["MAE"],
        "RMSE": m["RMSE"],
        "Elevated_MAE": e["Elevated_MAE"],
        "Elevated_RMSE": e["Elevated_RMSE"],
    })
bootstrap_df = pd.DataFrame(bootstrap_rows)
ci_rows = []
for metric in ["MAE", "RMSE", "Elevated_MAE", "Elevated_RMSE"]:
    vals = bootstrap_df[metric].dropna()
    if len(vals):
        ci_rows.append({
            "Metric": metric,
            "CI_2.5%": vals.quantile(0.025),
            "CI_97.5%": vals.quantile(0.975),
            "Bootstrap_Repeats_Used": len(vals),
        })
pd.DataFrame(ci_rows).to_csv(table_dir / "a3_model2_bootstrap_95ci.csv", index=False)


# %% [markdown]
# ## I. Save predictions and model
regression_predictions = test_reg[["date", "target_date", "station", regression_target]].copy()
regression_predictions["prediction"] = test_predictions
regression_predictions["residual_observed_minus_predicted"] = residuals
regression_predictions["underprediction"] = (test_predictions < y_test.to_numpy()).astype(int)
regression_predictions["elevated_actual"] = (y_test.to_numpy() >= PM25_THRESHOLD).astype(int)
regression_predictions.to_csv(table_dir / "a3_model2_2024_predictions.csv", index=False)

joblib.dump(final_regression_model, model_dir / "a3_model2_regression.joblib")
with open(model_dir / "a3_model2_regression_config.json", "w", encoding="utf-8") as f:
    json.dump({
        "model": best_model_name,
        "config": best_config_id,
        "params": selected_params,
        "primary_selection_metric": "RMSE",
        "reported_metrics": ["MAE", "RMSE", "R2", "elevated-event error", "bias", "underprediction rate"],
        "features": model_features,
        "assessment3_additions": [
            "FSI and wind-aligned features",
            "A-F ablation",
            "elevated-event analysis",
            "station-level metrics",
            "leave-one-station-out validation",
            "bootstrap uncertainty",
        ],
    }, f, indent=2)

print("\nSaved Assessment 3 Model 2 outputs.")

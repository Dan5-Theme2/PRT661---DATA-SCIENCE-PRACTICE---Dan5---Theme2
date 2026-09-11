# Fire2Air Darwin — Assessment 2 Technical Workflow

## Run order
1. `01_Thi_Pham_Hybrid_Data_Processing_Feature_Engineering.ipynb`
2. `02_Lihini_Jinanjalie_Hybrid_Model1_Classification.ipynb`
3. `03_Esangbedo_Favour_Hybrid_Model2_Regression.ipynb`
4. `04_Navodya_Piumanthi_Hybrid_Model3_Count_Explainability.ipynb`
5. `05_Dieu_Yen_Diep_Hybrid_Visualisation_Documentation.ipynb`

## Team responsibilities
- **Thi Pham:** data processing and feature engineering.
- **Lihini Jinanjalie:** Model 1 classification; PR-AUC-first model selection.
- **Esangbedo Favour:** Model 2 next-day PM2.5 regression; MAE/RMSE evaluation.
- **Navodya Piumanthi:** Model 3 elevated-hours count prediction and explainability.
- **Dieu Yen Diep:** visualisation, diagrams, evidence manifest and documentation.

## Shared modelling design
- Prediction horizon: day t → day t+1.
- Classification target: next-day 24-hour mean PM2.5 > 25 µg/m³.
- Regression target: next-day 24-hour mean PM2.5.
- Count target: next-day number of hours with PM2.5 ≥ 25 µg/m³.
- FIRMS scope: station-specific S-NPP VIIRS vegetation-fire features out to 500 km.
- Training: target dates in 2018–2022.
- Validation: target dates in 2023.
- Final untouched test: target dates in 2024.
- Target continuity: next-day targets are created only when the next station record is exactly one calendar day later.

## Key shared artefacts
- `Fire2Air_model_ready_checkpoint.csv`
- `Fire2Air_feature_manifest.json`
- `outputs_prt661/tables/`
- `outputs_prt661/figures/`
- `outputs_prt661/models/`

## Governance note
Do not fabricate GitHub commits, pull requests, Jira tickets, sprint boards or meeting records. Add screenshots/links from the team's real repository and project-management tools.

## Decision-support statement
This project is a research/decision-support prototype. It is not an official NT EPA warning and is not medical advice. Feature importance and SHAP explain model behaviour; they do not prove causal relationships.

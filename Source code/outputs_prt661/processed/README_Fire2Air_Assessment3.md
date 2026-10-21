# Fire2Air Darwin — Assessment 3 Technical Workflow

## Assessment 3 question
Can Fire2Air become more physically meaningful, robust, transferable and actionable?

## Run order
1. `01_Thi_Pham_Assessment3_Data_Processing_Feature_Engineering.py`
2. `02_Lihini_Jinanjalie_Assessment3_Model1_Classification.py`
3. `03_Esangbedo_Favour_Assessment3_Model2_Regression.py`
4. `04_Navodya_Piumanthi_Assessment3_Model3_Count_Explainability.py`
5. `05_Dieu_Yen_Diep_Assessment3_Visualisation_Documentation.py`

## Core novelty
Fire Smoke Influence Index (FSI): a transparent research feature combining fire intensity,
distance decay, wind alignment, wind speed, fire age and confidence weighting.
FSI is not a validated atmospheric dispersion model.

## Shared modelling design
- Predictor day: t
- Target day: exactly t+1
- Classification: next-day 24-hour mean PM2.5 > 25 ug/m3
- Regression: next-day 24-hour mean PM2.5
- Count: next-day number of hours >= 25 ug/m3
- Training target dates: 2018–2022
- Validation target dates: 2023
- Untouched final test target dates: 2024
- FIRMS range: station-specific features to 500 km

## Assessment 3 evidence
- A–F ablation: air/weather -> basic fire -> FRP -> age -> wind alignment -> full FSI
- High-PM2.5/rare-event evaluation
- Station-level evaluation
- Leave-one-station-out validation
- Classifier calibration
- SHAP/permutation importance
- Tomorrow Smoke Outlook

## Future automation
The preprocessing script contains an optional NASA FIRMS Area API helper. Historical
2018–2024 evaluation remains based on the supplied archive files for reproducibility.
Cloud storage/dashboard automation should be described as future work unless actually implemented.

## Decision-support statement
Fire2Air Darwin is a research/decision-support prototype. It is not an official NT EPA
warning and is not medical advice. Feature importance and FSI show predictive relationships,
not proof of causation.

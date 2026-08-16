# Fire2Air Darwin

## Explainable Next-Day Smoke and PM₂.₅ Forecasting for Greater Darwin

**Unit:** PRT661 – Data Science Practice  
**Theme:** Theme 2 – Predictive Analytics and Forecasting  
**Semester:** Semester 2, 2026  
**Group:** DAN5 – Theme 2

---

## Project Overview

Fire2Air Darwin is a predictive analytics project that aims to forecast next-day PM₂.₅ conditions in Greater Darwin.

The project combines air-quality and weather data from the NT EPA Air Quality Network with satellite fire data from NASA FIRMS VIIRS.

The study focuses on Palmerston, Winnellie and Stokes Hill using data from 2018–2024.

---

## Project Objectives

The project has three main prediction tasks:

1. **Classification** – Predict whether the next-day 24-hour average PM₂.₅ will exceed 25 µg/m³.
2. **Regression** – Predict the next-day average PM₂.₅ concentration.
3. **Count Prediction** – Predict the number of hours during the next day with elevated PM₂.₅.

SHAP and feature importance methods will also be used to explain the model results.

---

## Data Sources

### NT EPA Air Quality Network

Air-quality and weather data including:

- PM₂.₅
- PM₁₀
- Temperature
- Humidity
- Wind speed and direction
- Pressure
- Rainfall

Monitoring stations:

- Palmerston
- Winnellie
- Stokes Hill

### NASA FIRMS VIIRS

Satellite-detected fire information including:

- Fire location
- Fire Radiative Power (FRP)
- Confidence
- Detection date and time
- Fire distance from Greater Darwin

---

## Candidate Models

| Task | Candidate Models |
|---|---|
| Model 1 – Classification | Logistic Regression, Random Forest, XGBoost |
| Model 2 – Regression | Ridge Regression, Random Forest Regressor, XGBoost Regressor |
| Model 3 – Count Prediction | Poisson Regression, Tweedie Regression, Random Forest, XGBoost |

Simple baseline models will also be used for comparison.

---

## System Architecture

The project follows an end-to-end data science pipeline from data collection and storage to modelling, evaluation and visualisation.

Planned AWS services include:

- **Amazon S3** – raw, processed and modelling data storage
- **AWS Glue / Python** – data processing and feature engineering
- **Amazon SageMaker / Python ML** – model development and evaluation

![Fire2Air Darwin System Architecture](architecture/system-architecture.png)

---

## Model Evaluation

Different evaluation measures will be used for each prediction task.

**Classification**
- PR-AUC
- Recall
- Precision
- F1-score

**Regression**
- MAE
- RMSE
- R²

**Count Prediction**
- MAE
- RMSE
- Count prediction performance

A chronological data split will be used:

- **Training:** 2018–2022
- **Validation:** 2023
- **Testing:** 2024

---

## Team Allocation

| Member | Role |
|---|---|
| Thi Pham | Data processing and feature engineering |
| Lihini Jinanjalie | Model 1 – Classification |
| Esangbedo Favour | Model 2 – Regression |
| Navodya Piumanthi | Model 3 – Count Prediction and Explainability |
| Dieu Yen Diep | Visualisation and documentation |

![Task Allocation](planning/task-allocation.png)

---

## Project Management

The group uses:

- **Jira** – task allocation, sprint planning and progress tracking
- **GitHub** – version control, project files and documentation
- **Microsoft Teams** – team communication and meeting recordings

### Sprint 1 – Project Planning

Sprint 1 includes:

- Topic and dataset research
- Dataset finalisation
- Project overview and objectives
- Problem statement
- Workflow plan
- Initial risk analysis
- Architecture planning

![Sprint 1](planning/sprint-1.png)

**Jira Board:** [Click here](YOUR-JIRA-LINK)

---

## Repository Structure

```text
Fire2Air-Darwin/
│
├── README.md
├── assessment-reports/
├── architecture/
├── workflow/
├── planning/
├── task-allocation/
├── meeting-records/
├── data/
├── notebooks/
└── src/
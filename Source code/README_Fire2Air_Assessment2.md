# Fire2Air Darwin — Assessment 2 Technical Workflow

# Fire2Air Darwin

## Explainable Next-Day Smoke and PM₂.₅ Forecasting for Greater Darwin

**Unit:** PRT661 – Data Science Practice  
**Theme:** Theme 2 – Predictive Analytics and Forecasting  
**Semester:** Semester 2, 2026  
**Group:** DAN5 – Theme 2

---

## Project Overview

Fire2Air Darwin is a predictive analytics project developed to forecast next-day PM₂.₅ conditions in Greater Darwin.

The project combines historical air-quality and weather observations from the NT EPA Air Quality Network with NASA FIRMS Suomi-NPP VIIRS fire-detection data. The study focuses on Palmerston, Winnellie and Stokes Hill using data from 2018–2024.

The project contains three prediction tasks:

1. **Classification** – predict whether the next-day 24-hour mean PM₂.₅ will exceed 25 µg/m³.
2. **Regression** – predict the next-day mean PM₂.₅ concentration.
3. **Count Prediction** – predict the number of next-day hours with PM₂.₅ at or above 25 µg/m³.

For Model 3, 25 µg/m³ is used as a project reference level for counting elevated hours. It is not treated as an official hourly PM₂.₅ standard.

---

## Data Sources

### NT EPA Air Quality Network

Hourly air-quality and weather observations are used from:

- Palmerston
- Winnellie
- Stokes Hill

Main variables include:

- PM₂.₅
- PM₁₀
- Temperature
- Relative humidity
- Wind speed
- Wind direction
- Atmospheric pressure
- Rainfall

### NASA FIRMS Suomi-NPP VIIRS

Satellite fire-detection data include:

- Fire latitude and longitude
- Detection date and time
- Fire Radiative Power (FRP)
- Confidence information

Fire-related features are calculated using distance bands extending up to 500 km from each monitoring station.

---

## Team Responsibilities

| Team Member | Responsibility |
|---|---|
| Huynh Anh Thi Pham | Data processing and feature engineering |
| Lihini Jinanjalie Deniyelge | Model 1 – Classification |
| Esangbedo Favour Ikponwosa | Model 2 – Regression |
| Navodya Piumanthi Siriwardhana | Model 3 – Count Prediction and Explainability |
| Dieu Yen Diep | EDA, visualisation and documentation |

---

## Environment Setup

### 1. Requirements

The project is designed to run in Python using VS Code and Jupyter Notebook.

Recommended software:

- Python 3
- Visual Studio Code
- VS Code Python extension
- VS Code Jupyter extension

### 2. Install Required Python Packages

After cloning or downloading the repository, open the VS Code terminal in the project folder.

Upgrade `pip` first:

```bash
python -m pip install --upgrade pip
```

---

## Project File Setup

The main project code is stored under the `Source code` area of the repository.

The main executable files are:

```text
Source code/
│
├── 01_Huynh_Hybrid_Data_Processing_Feature_Engineering(1).ipynb
├── Lihini Model 1 Classification.ipynb
├── 03_Esangbedo_Favour_Hybrid_Model2_Regression_v2.py
├── Model3_Count_Explainability(2).ipynb
└── Jenny - Visualization and Documentation.ipynb
```

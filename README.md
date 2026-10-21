# Fire2Air Darwin

## Explainable Next-Day Smoke and PM₂.₅ Forecasting for Greater Darwin

**Unit:** PRT661 – Data Science Practice  
**Theme:** Theme 2 – Predictive Analytics and Forecasting  
**Group:** DAN5 – Theme 2  
**Semester:** Semester 2, 2026

## Project Overview

Fire2Air Darwin combines NT EPA air-quality and weather observations with NASA FIRMS VIIRS satellite fire detections to investigate next-day PM₂.₅ conditions at Palmerston, Winnellie and Stokes Hill.

The project uses historical data from 2018–2024. Three modelling tasks produce complementary predictions:

| Task | Output |
|---|---|
| Classification | Probability that next-day daily mean PM₂.₅ exceeds 25 µg/m³ |
| Regression | Expected next-day daily mean PM₂.₅ concentration in µg/m³ |
| Count / Duration Prediction | Expected number of next-day hours with PM₂.₅ at or above 25 µg/m³ |

The daily classification target uses **greater than 25 µg/m³**, while the hourly duration target uses **at or above 25 µg/m³**. The hourly threshold is a project-defined indicator.

Predictions are integrated into a Tomorrow Smoke Outlook and presented through public and administrator dashboards. The demonstrated forecasts are historical research outputs, not live warnings.

## Quick Guide for Lecturers and Reviewers

| Review task | Where to start | AWS access required? |
|---|---|---|
| Inspect local notebooks | `Source code/` | No |
| Inspect AWS-integrated notebooks | `AWS/AWS_integrations/` | No, to read the code |
| Review committed tables and figures | `Source code/outputs_prt661/` | No |
| Review AWS screenshots | `AWS/AWS_evidences/` | No |
| Inspect shared S3 objects | Fire2Air S3 bucket link below | Yes |
| Run AWS-dependent notebook cells | AWS-integrated workflow below | Yes |
| Run dashboards using supplied local inputs | Dashboard instructions below | No |
| Review contribution evidence | GitHub history, pull requests and final report | Repository access |

Access to the GitHub repository does not grant access to AWS.

### AWS Console Access

**AWS account ID:** `646468992009`  
**Project region:** `ap-southeast-2`  
**S3 bucket:** `fire2air-darwin-dan5-646468992009-ap-southeast-2-an`

- [Open the AWS account sign-in page](https://646468992009.signin.aws.amazon.com/console)
- [Open the Fire2Air S3 bucket](https://ap-southeast-2.console.aws.amazon.com/s3/buckets/fire2air-darwin-dan5-646468992009-ap-southeast-2-an?region=ap-southeast-2&tab=objects)

These links require an authorised AWS account or session and appropriate permissions.

If using an IAM user, use the username and password provided privately by the project team. If the team provides a different access method, follow those instructions instead. Contact the project team through the course communication channel if access has not been arranged.

After signing in, open S3 and select the project bucket. Saved outputs can be inspected without rerunning the models.

If access is denied, ask the team to check the assigned permissions. Reading outputs and publishing new outputs require different permissions.

Passwords, access keys, secret keys and session tokens are not included in this repository.

### Checking Saved AWS Outputs

Within the bucket:

1. Open `processed/model_ready/` to inspect the modelling checkpoint and feature manifest.
2. Open `outputs/model1/`, `outputs/model2/` and `outputs/model3/` to inspect model results.
3. Open `outputs/visualisation/` to inspect integration tables and figures.
4. Open `dashboard/` to inspect dashboard-ready outputs.
5. Open `documentation/` to inspect generated documentation.

Check object names, sizes and Last modified timestamps. Download CSV, JSON or image files where needed for inspection.

Names and timestamps alone do not establish that all objects belong to the same approved run. Use available manifests and file inventories when checking version consistency.

Running notebooks may upload or replace project objects. Reviewing saved outputs does not require execution.

## Data and Modelling

### Data Sources

- **NT EPA Air Quality Network:** PM₂.₅, PM₁₀ and available weather observations, including temperature, humidity, wind speed, wind direction, pressure and rainfall.
- **NASA FIRMS VIIRS:** satellite fire locations, acquisition times, Fire Radiative Power (FRP), confidence and fire type where available.

Observation availability and coverage differ between stations and periods.

### Feature Engineering

The notebooks include data auditing, daily aggregation, next-day target alignment and feature engineering using:

- Air-quality and weather measurements
- Lagged and rolling summaries
- Seasonal indicators
- Fire activity, distance and bearing
- Fire intensity and age
- Wind alignment and upwind fire measures
- Fire Smoke Influence Index (FSI)
- Input coverage and completeness indicators

FSI is an engineered predictor. It is not a validated atmospheric dispersion model, measured smoke concentration or probability.

The local preprocessing implementation also includes versioned outputs, validation checks, run-status records and file inventories.

### Evaluation

The chronological design uses:

| Stage | Target years |
|---|---|
| Initial training | 2018–2022 |
| Validation | 2023 |
| Historical testing | 2024 |

Model selection, refitting and evaluation details are documented in the individual notebooks.

| Task | Main evaluation measures |
|---|---|
| Classification | PR-AUC, precision, recall and F1-score |
| Regression | MAE, RMSE and R² |
| Count / Duration | MAE, RMSE, R² and mean Poisson deviance |

Additional evaluation includes baseline comparisons, station-level results, feature-group ablation, elevated-event diagnostics and uncertainty analysis.

Overall metrics should be interpreted alongside missed elevated events and errors during high-concentration or prolonged-pollution conditions.

SHAP and permutation importance describe predictive relationships, not causation.

## Repository Structure

The main folders and subfolders are organised as follows:

```text
PRT661---DATA-SCIENCE-PRACTICE---Dan5---Theme2/
├── ArchitectureDiagrams/              # System architecture diagrams
├── Assesment Reports/                 # Assessment reports
├── AWS/
│   ├── AWS_evidences/                 # AWS implementation evidence
│   └── AWS_integrations/              # AWS-integrated notebooks
├── Change log/                        # Project change records
├── Dashboard/                         # Streamlit apps and local input files
├── Documentation/
│   └── Research Papers/               # Supporting research papers
├── Environment setup instructions/    # Environment setup guidance
├── Execution instructions/            # Workflow execution guidance
├── Models/                            # Model-related files
├── Outputs/                           # Project output files
├── Planning/                          # Project planning material
├── Project Planning Records/
│   ├── Meeting Minutes/               # Written meeting minutes
│   ├── Meeting Records/               # Meeting records
│   └── Spring Plannning Screenshots/   # Planning screenshots
├── Reproducibility Artefacts/          # Reproducibility supporting files
├── Requirements files/                # Dependency requirements
├── Source code/
│   ├── Datasets/                      # Local source datasets
│   └── outputs_prt661/
│       ├── figures/                   # Generated figures
│       ├── metadata/                  # Manifests and inventories
│       ├── models/                    # Saved model artefacts
│       ├── processed/                 # Processed checkpoints
│       ├── runs/                      # Versioned preprocessing runs
│       └── tables/                    # Evaluation and integration tables
├── Task Allocation/                   # Team task allocation records
├── Technical Artefacts/               # Technical supporting material
├── Visualisations/                    # Visualisation material
├── Workflow Diagrams/                 # Workflow diagrams
└── README.md                          # Project overview and running instructions
```

This structure shows the main folders and subfolders only. Folder names retain the spelling used in the repository.

## Separate Code Sets: AWS and Local

The repository contains two separate notebook sets:

| Workflow | Location | Data and outputs |
|---|---|---|
| Primary — AWS-integrated | `AWS/AWS_integrations/` | Downloads shared inputs from S3 and uploads generated outputs to S3 |
| Fallback — Local | `Source code/` | Reads locally available files and saves outputs on the computer |

The AWS-integrated notebooks execute Python in the selected computer or notebook environment while using S3 for storage. They do not automatically execute on AWS compute infrastructure.

The local workflow requires the necessary datasets, checkpoints, manifests and model outputs on the computer. Switching workflows is manual.

The two code sets are separate implementations and are not automatically synchronised. Check the relevant notebook version and configuration rather than assuming equivalent behaviour.

The dashboard applications read local input files in both cases.

## Setup

Clone the repository and open it in VS Code:

```powershell
git clone https://github.com/Dan5-Theme2/PRT661---DATA-SCIENCE-PRACTICE---Dan5---Theme2.git
cd PRT661---DATA-SCIENCE-PRACTICE---Dan5---Theme2
code .
```

Use VS Code with Python and Jupyter support. Select the intended Python environment as the notebook kernel.

Install common notebook dependencies into that environment:

```powershell
python -m pip install numpy pandas matplotlib scikit-learn joblib
```

For AWS-integrated notebooks, also install:

```powershell
python -m pip install "boto3[crt]" s3fs
```

Install optional modelling and explainability packages when required:

```powershell
python -m pip install xgboost shap
```

These commands cover common imports. They are not a pinned environment specification. Check individual notebook imports and record package versions when reproducing results.

## AWS Workflow

**Notebook folder:** `AWS/AWS_integrations/`  
**Region:** `ap-southeast-2`

### Programmatic Access Requirements

Running AWS-dependent code requires valid credentials available to the notebook's Python environment.

Console sign-in alone does not configure Python authentication. Reading S3 inputs requires read permissions; publishing outputs requires write permissions. Glue and Athena operations require additional service permissions.

Use the team's approved authentication method and request access from the project team where necessary.

### Authentication and Verification

For the team's AWS CLI login workflow, run:

```powershell
aws login
```

Use AWS CLI supporting this command. Enter `ap-southeast-2` when prompted for the project region.

Verify authentication:

```powershell
aws sts get-caller-identity
```

Verify shared S3 access:

```powershell
aws s3 ls s3://fire2air-darwin-dan5-646468992009-ap-southeast-2-an/
```

Ensure the notebook uses the intended authorised AWS session. If a notebook explicitly configures a named profile, that profile must also be configured on the current computer.

Reauthenticate if the session expires. Do not paste credentials into notebook cells or commit them to GitHub.

### Notebook Order

Run these notebooks under `AWS/AWS_integrations/`:

| Order | Notebook |
|---|---|
| 1 | `Huynh_Data Processing Feature Engineering_Assessment 3.ipynb` |
| 2 | `Lihini Model 1 Classification AWS copy.ipynb` |
| 3 | `Favour_Model 2 Regression_Assessment 3.ipynb` |
| 4 | `Navodya - Model 3_Count_Explainability.ipynb` |
| 5 | `Jenny - Visualisation and Documentation AWS.ipynb` |

For each notebook:

1. Check configured inputs, output destinations and local working paths.
2. Select the intended kernel.
3. Restart the kernel and run all cells.
4. Review errors, warnings and missing-input messages.
5. Confirm expected outputs before continuing.

The models depend on preprocessing outputs. Integration requires predictions from all three models using compatible input versions.

Several AWS notebooks use `/tmp/fire2air_a3` as a local workspace. Check that the configured workspace is appropriate and writable on the execution machine.

### S3 Locations

| S3 prefix | Contents |
|---|---|
| `raw/ntepa/` | Source air-quality and weather observations |
| `raw/firms/` | Source fire detections |
| `processed/ntepa_daily/` | Daily station observations |
| `processed/firms_daily/` | Daily fire features |
| `processed/model_ready/` | Shared modelling checkpoint and manifest |
| `models/model1/` | Classification model artefacts |
| `models/model2/` | Regression model artefacts |
| `models/model3/` | Count-model artefacts |
| `outputs/data_processing/` | Preprocessing outputs and audits |
| `outputs/model1/` | Classification results |
| `outputs/model2/` | Regression results |
| `outputs/model3/` | Count-model results |
| `outputs/visualisation/` | Integration tables and figures |
| `dashboard/` | Dashboard-ready outputs |
| `documentation/` | Generated documentation |
| `athena-results/` | Athena query results |

Committing code to GitHub does not refresh S3 outputs. Publishing requires successful execution of the relevant notebooks.

## Local Fallback

**Notebook folder:** `Source code/`  
**Source datasets:** `Source code/Datasets/`  
**Shared outputs:** `Source code/outputs_prt661/`

The local workflow does not require AWS credentials once the necessary files are available on the computer. Downloading inputs from private S3 storage still requires authorised access.

### Notebook Order

Run these notebooks under `Source code/`:

| Order | Notebook |
|---|---|
| 1 | `Huynh_Data Processing Feature Engineering_Assessment 3.ipynb` |
| 2 | `Lihini Model 1 Classification.ipynb` |
| 3 | `Favour Model 2 Regression.ipynb` |
| 4 | `Navodya - Model 3_Count_Explainability.ipynb` |
| 5 | `Jenny - Visualisation and Documentation.ipynb` |

Restart the kernel and run all cells in each notebook.

If compatible preprocessing and model outputs already exist, integration can run without retraining every model.

### Path Configuration

Check paths before execution. The notebooks do not yet share one portable path configuration.

| Notebook | Path considerations |
|---|---|
| Preprocessing | Searches for a source directory containing `Datasets` |
| Classification | Contains a machine-specific `SOURCE_CODE_DIR` setting |
| Regression | Uses configured checkpoint and manifest inputs; writes outputs under the current user's Downloads folder |
| Count Prediction | Uses source-folder selection and checkpoint-verification logic |
| Visualisation | Uses `Path.cwd()` and reads model tables from the shared output directory |

The supplied local regression notebook writes outputs under:

```text
<user home>/Downloads/Fire2Air_Model2/outputs_prt661_a3/
```

The local integration notebook reads model tables from:

```text
Source code/outputs_prt661/tables/
```

Copy the intended regression tables into the shared location, or configure integration to read their actual location.

Include predictions, final metrics, station-level results, leave-one-station-out results and ablation tables as required.

### Visualisation Working Directory

From the repository root, open the source folder separately:

```powershell
code "Source code"
```

Open the local visualisation notebook in that window and verify the paths printed by its setup cell.

Expected locations, relative to `Source code/`, are:

```text
Processed checkpoints:
outputs_prt661/processed/

Feature manifest:
outputs_prt661/metadata/Fire2Air_feature_manifest_A3.json

Model result tables:
outputs_prt661/tables/
```

When all model tables are stored in the shared directory, the setup uses:

```python
regression_table_dir = table_dir
```

### Required Integration Inputs

Key inputs include:

```text
Fire2Air_ntepa_daily_checkpoint_A3.csv
Fire2Air_firms_daily_checkpoint_A3.csv
Fire2Air_model_ready_checkpoint_A3.csv
Fire2Air_feature_manifest_A3.json
```

The integrated outlook requires:

```text
a3_model1_2024_predictions.csv
a3_model2_2024_predictions.csv
a3_model3_2024_predictions.csv
```

The integration notebook does not train the models. Missing prediction files prevent generation of the complete outlook.

### Output Locations

Paths below are relative to `Source code/`:

| Folder | Contents |
|---|---|
| `outputs_prt661/processed/` | Processed checkpoints and local dashboard-ready output |
| `outputs_prt661/metadata/` | Feature manifests and inventories |
| `outputs_prt661/runs/` | Versioned preprocessing outputs |
| `outputs_prt661/tables/` | Evaluation and integration tables |
| `outputs_prt661/figures/` | Exported figures |
| `outputs_prt661/models/` | Saved models where configured |

Individual notebooks may write to separate locations. Check their configured and printed paths.

The offline local workflow does not reproduce S3 publishing, IAM checks, Glue operations or Athena queries.

## Tomorrow Smoke Outlook Outputs

Integration joins the three prediction tables using `date`, `target_date` and `station`.

| File | Purpose |
|---|---|
| `Fire2Air_Tomorrow_Smoke_Outlook_2024_A3.csv` | Integrated three-model outlook |
| `Fire2Air_dashboard_ready_outlook_A3.csv` | Dashboard-ready output |
| `a3_smoke_outlook_thresholds.json` | FSI thresholds and input-completeness rules |
| `a3_08_tomorrow_smoke_outlook_example.png` | Example outlook figure |
| `a3_final_2024_model_summary.csv` | Consolidated model metrics |
| `a3_all_models_station_level_metrics.csv` | Station-level comparisons |
| `a3_all_models_leave_one_station_out.csv` | Cross-station results |
| `a3_all_models_ablation_results.csv` | Feature-group comparisons |
| `a3_evidence_manifest.csv` | Supporting evidence index |

The local integration generates `README_Fire2Air_Assessment3.md`, which is separate from the repository-root `README.md`.

The `input_reliability` field reflects input completeness under the implemented rule. It is not calibrated forecast confidence.

## Run the Dashboards

The dashboard scripts read local files. They do not automatically download refreshed results from S3.

Install dependencies from the repository root:

```powershell
python -m pip install -r "Dashboard/requirements.txt"
```

Start the public dashboard:

```powershell
python -m streamlit run "Dashboard/Fire2Air_User_Dashboard_V3_UI_FIXED.py"
```

Or start the administrator dashboard:

```powershell
python -m streamlit run "Dashboard/Fire2Air_Admin_Analytics_Dashboard.py"
```

### Dashboard Inputs

The public dashboard reads its main inputs directly from `Dashboard/`:

```text
Fire2Air_dashboard_ready_outlook_A3.csv
Fire2Air_ntepa_daily_checkpoint_A3.csv
Fire2Air_model_ready_checkpoint_A3.csv
Fire2Air_feature_manifest_A3.json
```

The administrator dashboard searches several local directories, including `Dashboard/` and source-code output folders. An older duplicate may be selected before a newer file.

After regenerating results:

1. Verify the intended matching input versions.
2. Update dashboard input copies.
3. Restart the application or clear its cache.
4. Check displayed dates and values against the selected source files.

Starting a dashboard does not regenerate data or train models.

The optional `Dashboard/Dashboard_ready_csv_file_create.py` utility contains machine-specific Windows paths. Update those paths before using it on another computer.

## Reproducibility and Verification

Use compatible checkpoints, feature manifests and model outputs throughout the selected workflow.

The updated local preprocessing exports run identifiers, status records and SHA-256 inventories. Compare files against the approved run where those records are available.

On Windows:

```powershell
Get-FileHash -Algorithm SHA256 "path\to\file"
```

Compare the calculated value with a trusted inventory. A hash without a reference does not establish that the file is approved.

Matching filenames, dimensions or timestamps alone do not establish compatibility.

AWS and local preprocessing notebooks contain different implementations. Do not assume that a safeguard or update in one exists in the other.

Repository inspection does not establish successful execution. Identical results across both workflows and a complete verified source-to-dashboard rerun have not been confirmed by this README review.

## Team Contributions

| Member | Main responsibility |
|---|---|
| Huynh Anh Thi Pham | Data processing, feature engineering and data-quality audits |
| Lihini Jinanjalie Deniyelge | Classification, evaluation, AWS integration and public dashboard development |
| Esangbedo Favour Ikponwosa | Regression, evaluation, AWS integration and initial administrator dashboard contribution |
| Navodya Piumanthi Siriwardhana | Count prediction, explainability, AWS checks and administrator dashboard development |
| Dieu Yen Diep / Jenny | Initial dashboard prototype, visualisation, integration, documentation and dashboard-ready outputs |

Detailed individual contributions are documented in the final report and supported by repository activity.

## Project Management and Evidence

The team uses GitHub for version control and pull requests, Jira for task tracking, and Microsoft Teams for communication.

Supporting material is organised under:

- `Assesment Reports/`
- `AWS/AWS_evidences/`
- `ArchitectureDiagrams/`
- `Documentation/`
- `Project Planning Records/`
- `Reproducibility Artefacts/`
- `Task Allocation/`

GitHub commits, pull requests, meeting records and Jira evidence support contribution tracking and project decisions.

## Limitations

- Data availability and coverage differ between stations and periods.
- Satellite detections do not directly measure ground-level smoke exposure.
- FSI uses simplified transport assumptions.
- Overall metrics can conceal missed elevated events and severe-day underprediction.
- Historical performance does not establish reliable transfer to future periods or new locations.
- Notebook paths and local output copies require manual management.
- A complete operational refresh, monitoring and retraining service has not been demonstrated.

Further work includes calibration, improved elevated-event performance, broader temporal evaluation, portable configuration and verified source-to-dashboard version consistency.

## Responsible Use

Fire2Air Darwin is an educational research and decision-support prototype. It is not an official smoke-warning service or a source of medical advice.

The demonstrated forecasts use historical data. Low predicted risk does not rule out elevated pollution.

FSI describes engineered fire influence, and input completeness describes data availability. Neither establishes personal exposure or forecast certainty.

Consult official air-quality information for current conditions and health-related decisions.
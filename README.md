# Fire2Air Darwin

## Explainable Next-Day Smoke and PM₂.₅ Forecasting for Greater Darwin

**Unit:** PRT661 – Data Science Practice  
**Theme:** Theme 2 – Predictive Analytics and Forecasting  
**Semester:** Semester 2, 2026  
**Group:** DAN5 – Theme 2

## Project Overview

Fire2Air Darwin investigates next-day PM₂.₅ conditions at Palmerston, Winnellie and Stokes Hill in Greater Darwin.

The project combines air-quality and weather observations from the NT EPA Air Quality Network with satellite fire detections from NASA FIRMS VIIRS. Historical data from 2018–2024 support preprocessing, feature engineering, modelling and evaluation.

Three complementary predictions are combined into a Tomorrow Smoke Outlook:

- Probability of elevated next-day daily mean PM₂.₅
- Expected next-day mean PM₂.₅ concentration
- Expected number of elevated hours during the next day

The repository contains local and AWS-integrated notebooks, model evaluation outputs, visualisations, public and administrator dashboards, and project evidence.

Fire2Air Darwin is an educational research and decision-support prototype. It is not an operational air-quality warning service.

## Prediction Tasks

| Task | Target |
|---|---|
| Model 1 – Classification | Whether the next-day 24-hour mean PM₂.₅ exceeds 25 µg/m³ |
| Model 2 – Regression | Next-day mean PM₂.₅ concentration in µg/m³ |
| Model 3 – Count Prediction | Number of next-day hours with PM₂.₅ at or above 25 µg/m³ |

The daily classification target uses **greater than 25 µg/m³**, while the hourly count uses **at or above 25 µg/m³**.

The hourly count is a project-defined indicator, not a validated hourly health standard. The three targets describe different aspects of air quality and need not produce equivalent conclusions.

## Data Sources

### NT EPA Air Quality Network

Monitoring stations:

- Palmerston
- Winnellie
- Stokes Hill

Variables include PM₂.₅, PM₁₀, temperature, humidity, wind speed, wind direction, pressure and rainfall.

Observation coverage and variable availability differ between stations and periods.

### NASA FIRMS VIIRS

Fire-detection information includes:

- Latitude and longitude
- Fire Radiative Power (FRP)
- Detection confidence
- Acquisition date and time
- Fire type, where available

Station-specific predictors are derived using fire distance, bearing, age and wind alignment.

Satellite fire detections do not independently establish that smoke reached a monitoring station.

## Data Processing and Feature Engineering

The preprocessing workflow audits source data, creates daily station-level summaries and aligns observation-day predictors with next-day targets.

The updated local preprocessing implementation includes:

- Archive-year, schema and timestamp checks
- Duplicate detection and numeric-conversion audits
- Pressure-unit and rainfall processing
- Observation-coverage checks
- Calendar continuity before lag and rolling calculations
- Missing-wind, FRP and confidence safeguards
- Target-alignment validation
- Versioned run outputs, status records and file hashes

The Assessment 4 report documents a successful updated preprocessing run containing **7,530 station-days**, **93 predictors** and **120 total columns**, with all **20 implemented validation checks** passing.

These figures describe the reported preprocessing run. They do not establish that every downstream model or dashboard output uses that same approved version.

### Feature Groups

Predictors include:

- Air-quality and weather measurements
- Lagged and rolling summaries
- Seasonal indicators
- Fire activity and proximity
- FRP summaries
- Fire-age measures
- Wind alignment and upwind fire measures
- Fire Smoke Influence Index (FSI)
- Input availability and completeness indicators

### Fire Smoke Influence Index

FSI combines fire intensity, distance decay, directional alignment, wind speed, fire age and confidence weighting.

It is an engineered research feature, not a validated atmospheric dispersion model, measured concentration or probability.

The updated local preprocessing preserves unknown wind-dependent contributions rather than treating missing wind as zero smoke influence.

### Shared Checkpoints

Key files include:

```text
Fire2Air_ntepa_daily_checkpoint_A3.csv
Fire2Air_firms_daily_checkpoint_A3.csv
Fire2Air_model_ready_checkpoint_A3.csv
Fire2Air_feature_manifest_A3.json
```

## Modelling and Evaluation

| Task | Model families investigated |
|---|---|
| Classification | Logistic Regression, Random Forest and optional XGBoost |
| Regression | Ridge Regression, Random Forest and optional XGBoost |
| Count Prediction | Count-regression and tree-based approaches, including Poisson and Tweedie |

The Assessment 4 report identifies Logistic Regression, Ridge Regression and Poisson Regression as the reported final models.

Consult each notebook and its exported tables for the implemented candidates, selected configurations and input versions. Optional package availability can affect which candidates run.

### Chronological Design

| Stage | Target years |
|---|---|
| Initial training | 2018–2022 |
| Validation and selection | 2023 |
| Historical test evaluation | 2024 |

Imputation, encoding and applicable scaling are fitted within the training pipelines. Selected models may subsequently be refitted on 2018–2023, as documented in their notebooks.

The 2024 results have informed project review. Further tuning based on those results would reuse the benchmark; independent confirmation requires a genuinely unseen period.

### Evaluation Measures

| Task | Main measures |
|---|---|
| Classification | PR-AUC, precision, recall and F1-score |
| Regression | MAE, RMSE and R² |
| Count Prediction | MAE, RMSE, R² and mean Poisson deviance |

Additional diagnostics include station-level evaluation, leave-one-station-out testing, feature-group ablation, elevated-event analysis, baseline comparisons and bootstrap uncertainty.

Bootstrap procedures differ between notebook versions. Interpret confidence intervals using the corresponding implementation.

SHAP and permutation importance describe predictive relationships, not causation.

### Main Findings

The reported evaluation highlights several limitations:

- Classification missed many elevated station-days.
- Regression frequently underestimated concentrations on elevated days.
- Count prediction became less accurate during prolonged elevated episodes.
- Fire features did not consistently improve all three tasks.
- Performance and data coverage varied between stations.

Aggregate metrics should therefore be read alongside station-level and elevated-event results.

## Tomorrow Smoke Outlook

The integration notebook joins the three prediction tables using:

```text
date
target_date
station
```

The resulting output contains:

- Elevated-PM₂.₅ probability
- Predicted daily mean PM₂.₅
- Expected elevated hours
- Fire-smoke influence category
- Input completeness score and category
- Decision-support summary
- Research-use disclaimer

The field named `input_reliability` reflects input completeness under the implemented rule. It is not a calibrated forecast-confidence measure.

The demonstrated outlook uses historical 2024 predictions. It is not evidence of live operational forecasting.

### Key Outputs

| File | Purpose |
|---|---|
| `Fire2Air_Tomorrow_Smoke_Outlook_2024_A3.csv` | Integrated three-model outlook |
| `Fire2Air_dashboard_ready_outlook_A3.csv` | Dashboard-ready data |
| `a3_smoke_outlook_thresholds.json` | FSI thresholds and completeness rules |
| `a3_08_tomorrow_smoke_outlook_example.png` | Example outlook figure |
| `a3_final_2024_model_summary.csv` | Consolidated final metrics |
| `a3_all_models_station_level_metrics.csv` | Station-level results |
| `a3_all_models_leave_one_station_out.csv` | Cross-station results |
| `a3_all_models_ablation_results.csv` | Feature-group comparisons |
| `a3_evidence_manifest.csv` | Supporting evidence index |

## System Architecture

The workflow connects:

1. NT EPA and FIRMS inputs
2. Auditing and preprocessing
3. Station-specific feature engineering
4. Shared model-ready checkpoints
5. Three modelling tasks
6. Evaluation and explainability
7. Forecast integration and visualisation
8. Dashboard-ready outputs and documentation

### AWS Components

| Component | Role |
|---|---|
| Amazon S3 | Shared data, checkpoints, model artefacts and outputs |
| AWS IAM | Authorised access to project resources |
| AWS Glue | Catalogue and crawler support where documented |
| Amazon Athena | SQL checks and queries where documented |

AWS-integrated notebooks can execute in VS Code on a local computer while accessing S3. This does not mean computation runs in SageMaker or another AWS compute service.

The dashboard scripts in this repository read local files. They do not automatically retrieve refreshed data from S3.

Scheduled refresh, live ingestion and a complete production monitoring pipeline have not been demonstrated.


## Repository Structure

The main folders and subfolders are organised in GitHub's alphabetical order:

```text
PRT661---DATA-SCIENCE-PRACTICE---Dan5---Theme2/
├── ArchitectureDiagrams/              # System architecture diagrams
├── Assesment Reports/                 # Assessment reports
├── AWS/
│   ├── AWS_evidences/                 # AWS implementation evidence
│   └── AWS_integrations/              # AWS-integrated notebooks
├── Change log/                        # Project change records
├── Dashboard/                         # Streamlit apps and dashboard inputs
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

This structure shows the main folders and subfolders only.

## Separate Code Sets: AWS and Local

The repository contains two separate sets of notebooks:

| Code set | Location | Purpose |
|---|---|---|
| AWS-integrated | `AWS/AWS_integrations/` | Reads shared inputs from Amazon S3 and publishes outputs to S3 |
| Local | `Source code/` | Reads inputs from local folders and saves outputs on the computer |

### Primary Workflow — AWS

Use the notebooks in `AWS/AWS_integrations/` for the primary AWS-integrated workflow.

These notebooks can run locally in VS Code while accessing AWS storage. They do not automatically execute on AWS compute infrastructure.

### Local Fallback

Use the notebooks in `Source code/` when the required datasets, checkpoints, feature manifests and model outputs are already available on the computer.

Check each notebook's configured paths before execution. The local notebooks do not automatically download inputs from S3 or publish results to AWS.

### Version Consistency and Dashboard Inputs

The AWS and local code sets are separate implementations and are not automatically synchronised. Changes to one set must be reviewed separately before being applied to the other.

Use matching data and model versions. Identical results across both workflows have not been verified.

The Streamlit applications in `Dashboard/` read local files. To display AWS-generated results, download the matching outputs and place them in the locations expected by the dashboard.

## AWS Credentials and Access Requirements

Running and verifying the AWS-integrated workflow requires valid AWS credentials and appropriate permissions.

Access to this GitHub repository does not grant access to the project's AWS resources. Reviewers need an authorised AWS account or role to read S3 inputs, execute AWS-dependent code and verify uploaded outputs. Publishing outputs requires write permissions. Glue and Athena checks require additional service permissions.

Authenticate using the team's approved login method before running AWS-dependent notebooks. Request access from the team's AWS administrator if required.

Never include passwords, access keys, secret keys or session tokens in the repository.

The local workflow does not require AWS credentials once all required input files are available on the computer. Downloading those files from private S3 storage still requires authorised access.

## Running the Project

### Primary Workflow — AWS Integration

The AWS-integrated workflow uses shared S3 inputs and publishes results to S3. It requires an authorised account and a configured Python environment.

#### Environment Setup

Use VS Code with Python and Jupyter support. Select the intended Python environment as the notebook kernel.

Install the common scientific and AWS packages into that environment:

```powershell
python -m pip install numpy pandas matplotlib scikit-learn joblib "boto3[crt]" s3fs
```

For optional XGBoost candidates and SHAP analysis:

```powershell
python -m pip install xgboost shap
```

These commands provide common dependencies; the repository does not define a single pinned environment for every notebook. Check imports and record package versions when reproducing results.

#### AWS Authentication

Use AWS CLI compatible with the team's login method:

```powershell
aws login
```

Use the project region when prompted:

```text
ap-southeast-2
```

Verify the identity and shared S3 access:

```powershell
aws sts get-caller-identity
```

```powershell
aws s3 ls s3://fire2air-darwin-dan5-646468992009-ap-southeast-2-an/
```

Reauthenticate if the session expires.

Never commit passwords, access keys, secret keys or session tokens.

#### AWS Notebook Order

Open these notebooks under `AWS/AWS_integrations/`:

| Order | Notebook |
|---|---|
| 1 | `Huynh_Data Processing Feature Engineering_Assessment 3.ipynb` |
| 2 | `Lihini Model 1 Classification AWS copy.ipynb` |
| 3 | `Favour_Model 2 Regression_Assessment 3.ipynb` |
| 4 | `Navodya - Model 3_Count_Explainability.ipynb` |
| 5 | `Jenny - Visualisation and Documentation AWS.ipynb` |

For each notebook:

1. Check its configured inputs and output destinations.
2. Select the intended Python kernel.
3. Restart the kernel and select **Run All**.
4. Review errors, warnings and conditional outputs.
5. Confirm the expected S3 files before continuing.

The three model workflows depend on preprocessing outputs. Integration requires predictions from all three models.

The AWS notebooks use local working directories while transferring files to and from S3. Several use `/tmp/fire2air_a3`; check that this location is appropriate and writable on the execution machine.

#### Shared S3 Layout

```text
raw/
    ntepa/
    firms/
processed/
    ntepa_daily/
    firms_daily/
    model_ready/
models/
    model1/
    model2/
    model3/
outputs/
    data_processing/
    model1/
    model2/
    model3/
    visualisation/
dashboard/
documentation/
athena-results/
```

Check uploaded objects and Last modified timestamps. Timestamps alone do not prove that files belong to the same approved run.

Committing code to GitHub does not refresh S3 outputs. Publishing requires successful execution of the relevant notebooks.

### Local Fallback — Files on the Computer

Use the local workflow when AWS access is unavailable and the required files are already available on the computer.

#### Dependencies

Install the common local notebook packages:

```powershell
python -m pip install numpy pandas matplotlib scikit-learn joblib
```

Install optional modelling and explainability packages when needed:

```powershell
python -m pip install xgboost shap
```

Use the same environment for installation and notebook execution.

#### Local Notebook Order

Open the notebooks under `Source code/`:

| Order | Notebook |
|---|---|
| 1 | `Huynh_Data Processing Feature Engineering_Assessment 3.ipynb` |
| 2 | `Lihini Model 1 Classification.ipynb` |
| 3 | `Favour Model 2 Regression.ipynb` |
| 4 | `Navodya - Model 3_Count_Explainability.ipynb` |
| 5 | `Jenny - Visualisation and Documentation.ipynb` |

Check the source-data configuration before running preprocessing. The preprocessing notebook searches for a source directory containing `Datasets`.

The local notebooks do not yet use a fully portable, unified path configuration.

#### Important Path Differences

| Notebook | Configuration to check |
|---|---|
| Preprocessing | Source directory, `Datasets` location and approved-run outputs |
| Classification | Machine-specific `SOURCE_CODE_DIR` setting |
| Regression | Checkpoint and manifest selection; outputs are written under the current user's Downloads folder |
| Count Prediction | Source-folder selection and checkpoint/hash verification |
| Visualisation | Current working directory and shared result-table locations |

The local regression output directory is:

```text
<user home>/Downloads/Fire2Air_Model2/outputs_prt661_a3/
```

The visualisation notebook instead reads model tables from:

```text
Source code/outputs_prt661/tables/
```

Before integration, copy the intended regression result tables into the shared tables directory, or explicitly configure the integration notebook to read the regression output directory.

Include predictions, final metrics, station-level results, leave-one-station-out results and ablation tables where required. Do not assume that running regression automatically refreshes the shared copies.

#### Visualisation Working Directory

The local visualisation notebook uses `Path.cwd()`.

From the repository root, open the source folder separately:

```powershell
code "Source code"
```

Open the notebook in that window and verify its printed input and output paths.

Its configured locations are:

```text
Processed inputs:
Source code/outputs_prt661/processed/

Feature manifest:
Source code/outputs_prt661/metadata/Fire2Air_feature_manifest_A3.json

Model tables:
Source code/outputs_prt661/tables/
```

When all model tables are stored in the shared directory, the setup uses:

```python
regression_table_dir = table_dir
```

#### Required Prediction Files

The integrated outlook requires:

```text
a3_model1_2024_predictions.csv
a3_model2_2024_predictions.csv
a3_model3_2024_predictions.csv
```

The integration notebook does not train the three models. Missing prediction files prevent creation of the complete outlook.

If matching preprocessing and model outputs already exist, integration can run without retraining every model.

#### Version Verification

Use matching checkpoints, manifests and model outputs.

The updated local preprocessing exports run identifiers, status records and SHA-256 inventories. Compare files against the approved run where those records are available.

On Windows:

```powershell
Get-FileHash -Algorithm SHA256 "path\to\file"
```

Compare the calculated value with a trusted inventory. A hash without a reference does not establish that the file is approved.

Matching filenames or dataset dimensions alone do not establish version consistency.

#### Run and Check Outputs

Restart the kernel and select **Run All**.

Review missing-input messages and errors. Confirm that the expected files were created under:

```text
Source code/outputs_prt661/figures/
Source code/outputs_prt661/tables/
Source code/outputs_prt661/processed/
```

The local integration writes `Fire2Air_dashboard_ready_outlook_A3.csv` and `README_Fire2Air_Assessment3.md` into its configured processed-data folder.

The generated workflow README is separate from the repository-root `README.md`.

A tracked file with unchanged contents will not appear in VS Code Source Control. Check the actual output folder as well as the Changes list.

#### Local Limitations

The offline local workflow does not reproduce:

- S3 publishing
- IAM permission checks
- Glue catalogue or crawler operations
- Athena queries

Required inputs must already be available locally. Downloading additional S3 files requires connectivity and authorised access.

### Dashboard Execution

Both dashboards read local CSV and JSON files. Neither automatically downloads refreshed outputs from S3.

From the repository root, install dependencies:

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

#### Dashboard Inputs

The public dashboard reads its main inputs directly from `Dashboard/`:

```text
Fire2Air_dashboard_ready_outlook_A3.csv
Fire2Air_ntepa_daily_checkpoint_A3.csv
Fire2Air_model_ready_checkpoint_A3.csv
Fire2Air_feature_manifest_A3.json
```

The administrator dashboard searches several local locations, including `Dashboard/` and `Source code/outputs_prt661/`.

Older copies can be selected before newer results because the administrator application uses an ordered file search.

After regenerating results:

1. Verify the intended input versions.
2. Update the dashboard input copies.
3. Restart the application or clear its cached data.
4. Check displayed dates and values against the source files.

Starting a dashboard does not regenerate data, train models or upload results to S3.

The optional utility `Dashboard/Dashboard_ready_csv_file_create.py` contains machine-specific Windows paths. Update its input and output paths before using it on another computer.

### Reproducibility Status

The updated local preprocessing includes versioned exports, validation checks and SHA-256 inventories. The duration workflow includes checkpoint-verification logic.

The AWS and local preprocessing notebooks contain different implementations. The latest local safeguards must not be assumed to exist in the AWS version without checking and testing it separately.

A complete source-to-dashboard rerun confirming that all three models and both dashboards use the same approved preprocessing version remains outstanding in the Assessment 4 report.

Identical results across local and AWS-integrated workflows have not been established. Such a claim requires matching inputs, code, package versions and model configurations.

## Visualisation and Dashboard Features

The project includes public and administrator dashboard views for exploring station-specific predictions and historical results.

The visualisation workflow generates:

- End-to-end architecture figure
- PM₂.₅ trends by station
- Station-level concentration distributions
- Exceedance counts by year and station
- FSI, aligned FRP and upwind FRP relationships
- Input completeness summaries
- Consolidated evaluation tables
- Example Tomorrow Smoke Outlook

The initial dashboard prototype was extended through subsequent team contributions.

## Team Contributions

| Member | Main responsibility |
|---|---|
| Huynh Anh Thi Pham | Data processing, feature engineering and data-quality audits |
| Lihini Jinanjalie Deniyelge | Classification, evaluation, AWS integration and public dashboard development |
| Esangbedo Favour Ikponwosa | Regression, evaluation, AWS integration and initial administrator dashboard contribution |
| Navodya Piumanthi Siriwardhana | Count prediction, explainability, AWS checks and administrator dashboard development |
| Dieu Yen Diep / Jenny | Initial dashboard prototype, visualisation, integration, documentation and dashboard-ready outputs |

Team members also contributed to troubleshooting, review, presentation preparation and coordination. Detailed attribution is recorded in project documentation and repository history.

## Project Management and Evidence

The team uses:

- **GitHub** for version control, pull requests, review and documentation
- **Jira** for task allocation and progress tracking
- **Microsoft Teams** for communication and meetings

Supporting evidence includes meeting minutes, AWS screenshots, notebooks, exported results and GitHub activity.

Only genuine project activity should be presented as contribution or implementation evidence.

## Limitations and Future Work

Current limitations include:

- Missing observations and uneven station coverage
- Missed elevated-pollution events
- Underprediction of high concentrations and prolonged elevated episodes
- Limited evidence of transfer to new stations or future periods
- Inconsistent incremental benefit from fire features
- Simplified smoke-transport assumptions in FSI
- Different path configurations across notebooks
- Duplicate local data copies and incomplete downstream version verification
- Manual execution without a demonstrated operational refresh pipeline

Future priorities include:

- Probability calibration and risk-sensitive thresholds
- Improved severe-event performance
- Rolling historical backtesting and unseen-period evaluation
- Consistent dependence-aware uncertainty analysis
- FSI sensitivity and transport validation
- Portable path and dependency configuration
- Verified source-to-dashboard traceability
- Automated refresh, monitoring and controlled retraining

These are improvement priorities, not completed operational capabilities.

## Responsible Use

Fire2Air Darwin is an educational research and decision-support prototype.

Historical performance does not guarantee future accuracy. Low predicted risk does not rule out elevated pollution.

FSI describes engineered fire influence, while input completeness describes data availability. Neither is a verified measure of personal exposure or forecast certainty.

The project is not an official NT EPA warning system and does not provide medical advice. Consult official air-quality information when making health or outdoor-activity decisions.

# Fire2Air Darwin

## Explainable Next-Day Smoke and PM₂.₅ Forecasting

**Unit:** PRT661 – Data Science Practice  
**Group:** DAN5 – Theme 2  
**Semester:** Semester 2, 2026

Fire2Air Darwin combines NT EPA air-quality and weather observations with NASA FIRMS satellite fire detections to investigate next-day PM₂.₅ conditions at Palmerston, Winnellie and Stokes Hill.

The project uses historical data from 2018–2024 and includes three complementary prediction tasks:

| Model | Prediction task |
|---|---|
| Model 1 – Classification | Probability that next-day daily mean PM₂.₅ exceeds 25 µg/m³ |
| Model 2 – Regression | Next-day daily mean PM₂.₅ concentration in µg/m³ |
| Model 3 – Duration / Count | Number of hours during the next day with PM₂.₅ at or above 25 µg/m³ |

The results support visualisation, model evaluation and public and administrator dashboards.

Displayed forecasts are historical research outputs rather than live warnings. The hourly threshold used in the duration task is a project-defined indicator.

## Quick Guide for Lecturers and Evaluators

The project can be reviewed through the following resources:

| Resource | Purpose |
|---|---|
| [User dashboard](https://fire2airdarwin-cdu-prt661-datasciencepractice-user-dashboard.streamlit.app/) | Explore the public-facing presentation of historical forecasting outputs |
| [Administrator dashboard](https://prt661---data-science-practice---dan5---theme2-vdb3z6nuf8appxk.streamlit.app/) | Explore analytical results and supporting project information |
| [AWS account sign-in](https://646468992009.signin.aws.amazon.com/console) | Sign in using the authorised account details supplied privately |
| [Fire2Air S3 bucket](https://ap-southeast-2.console.aws.amazon.com/s3/buckets/fire2air-darwin-dan5-646468992009-ap-southeast-2-an?region=ap-southeast-2&tab=objects) | Inspect stored project data and outputs after signing in |
| `AWS/AWS_evidences/` | Review recorded AWS implementation evidence |
| `Assesment Reports/` | Read the assessment reports |

AWS access requires appropriate permissions. The sign-in and S3 links do not grant access by themselves.

Lecturer access details must be supplied privately by the project team. Do not place passwords, access keys, secret keys or session tokens in this README or other repository files.

## Two Execution Options

The repository contains separate AWS-integrated and local code sets.

The AWS workflow uses Amazon S3 for shared storage. The local workflow is available when AWS access or services are unavailable.

| Workflow | Notebook location | Storage |
|---|---|---|
| AWS-integrated | `AWS/AWS_integrations/` | Reads inputs from S3 and writes configured outputs to S3 |
| Local | `Source code/` | Reads locally available datasets and writes outputs to local folders |

These are separate implementations. Paths, configuration and available outputs should be checked before execution.

Switching between workflows is manual. Running a notebook does not automatically switch the dashboard to its new outputs.

## Repository Structure

The main folders are shown below in repository order. Comments briefly describe their purpose.

```text
PRT661---DATA-SCIENCE-PRACTICE---Dan5---Theme2/
├── ArchitectureDiagrams/                 # System architecture diagrams
├── Assesment Reports/                    # Assessment reports
├── AWS/
│   ├── AWS_evidences/                    # Recorded AWS implementation evidence
│   └── AWS_integrations/                 # AWS-integrated notebooks
├── Change log/                          # Project change records
├── Dashboard/                           # Dashboard scripts, dependencies and inputs
├── Documentation/
│   └── Research Papers/                 # Supporting research material
├── Environment setup instructions/      # Environment setup guidance
├── Execution instructions/              # Execution guidance
├── Models/                              # Model-related project files
├── Outputs/                             # Project output files
├── Planning/                            # Planning material
├── Project Planning Records/
│   ├── Meeting Minutes/                 # Meeting minutes
│   ├── Meeting Records/                 # Meeting records
│   └── Spring Plannning Screenshots/     # Planning screenshots
├── Reproducibility Artefacts/            # Supporting reproducibility material
├── Requirements files/                  # Dependency-related files
├── Source code/
│   ├── Datasets/                        # Local source datasets
│   └── outputs_prt661/
│       ├── figures/                     # Exported plots
│       ├── metadata/                    # Feature manifests and inventories
│       ├── models/                      # Saved model artefacts
│       ├── processed/                   # Processed data and checkpoints
│       ├── runs/                        # Versioned preprocessing runs
│       └── tables/                      # Predictions and evaluation tables
├── Task Allocation/                     # Team allocation records
├── Technical Artefacts/                 # Supporting technical material
├── Visualisations/                      # Visualisation-related files
├── Workflow Diagrams/                   # Workflow diagrams
└── README.md                            # Project overview and execution guidance
```

This is a simplified folder view. Notebook filenames and execution order are listed below.

Folder names retain the spelling used in the repository.

## Setup

Clone the repository and enter its folder:

```powershell
git clone https://github.com/Dan5-Theme2/PRT661---DATA-SCIENCE-PRACTICE---Dan5---Theme2.git
cd PRT661---DATA-SCIENCE-PRACTICE---Dan5---Theme2
```

Open the project in VS Code:

```powershell
code .
```

Install Python and enable Python and Jupyter support in VS Code. Select the intended Python environment as the notebook kernel.

Common notebook dependencies include:

```powershell
python -m pip install numpy pandas matplotlib scikit-learn joblib "boto3[crt]" s3fs
```

Install additional modelling and explainability packages when required by the selected notebook:

```powershell
python -m pip install xgboost shap
```

These commands provide common dependencies. Check each notebook's imports and any relevant requirements files for additional packages.

Dashboard dependencies are installed separately using `Dashboard/requirements.txt`.

## AWS Workflow

**Notebook folder:** `AWS/AWS_integrations/`  
**AWS region:** `ap-southeast-2`  
**S3 bucket:** `fire2air-darwin-dan5-646468992009-ap-southeast-2-an`

The AWS-integrated notebooks execute Python in the selected computer or notebook environment and use Amazon S3 for shared storage.

### Authentication and Access

Before running the notebooks:

1. Obtain authorised programmatic access from the project team.
2. Configure the AWS authentication method required by the notebook.
3. Check whether the notebook explicitly selects an AWS profile.
4. Confirm that the session is valid and has the required S3 permissions.

Where a notebook selects the `fire2air` profile, that profile must be available in the execution environment.

AWS Console sign-in and programmatic authentication are separate. Signing in through a browser does not, by itself, configure credentials for Python or the AWS CLI.

If the AWS CLI is installed, the following commands can help check access using the configured default session:

```powershell
aws sts get-caller-identity
aws s3 ls s3://fire2air-darwin-dan5-646468992009-ap-southeast-2-an/
```

To check the named `fire2air` profile instead:

```powershell
aws sts get-caller-identity --profile fire2air
aws s3 ls s3://fire2air-darwin-dan5-646468992009-ap-southeast-2-an/ --profile fire2air
```

An identity response confirms the active AWS identity. A successful S3 listing confirms listing access to the requested location; it does not establish every permission needed to execute the workflow.

### Notebook Execution Order

Run the notebooks in the following order:

| Order | Notebook |
|---|---|
| 1 | `Huynh_Data Processing Feature Engineering_Assessment 3.ipynb` |
| 2 | `Lihini Model 1 Classification AWS copy.ipynb` |
| 3 | `Favour_Model 2 Regression_Assessment 3.ipynb` |
| 4 | `Navodya - Model 3_Count_Explainability.ipynb` |
| 5 | `Jenny - Visualisation and Documentation AWS.ipynb` |

For each notebook:

1. Select the intended kernel.
2. Check its configuration, input locations and output locations.
3. Restart the kernel and run all cells.
4. Resolve errors and confirm the required outputs before continuing.

Run preprocessing before the model notebooks. Run visualisation and integration after the required predictions from all three models are available.

The model and integration notebooks should use compatible preprocessing outputs and matching evaluation periods.

### S3 Locations

The AWS workflow uses the following S3 prefixes:

| Prefix | Purpose |
|---|---|
| `raw/ntepa/` | Source air-quality and weather data |
| `raw/firms/` | Source satellite fire detections |
| `processed/ntepa_daily/` | Daily station observations |
| `processed/firms_daily/` | Daily fire data |
| `processed/model_ready/` | Shared modelling data and feature information |
| `models/model1/` | Classification model artefacts |
| `models/model2/` | Regression model artefacts |
| `models/model3/` | Duration/count model artefacts |
| `outputs/data_processing/` | Preprocessing outputs and supporting evidence |
| `outputs/model1/` | Classification results |
| `outputs/model2/` | Regression results |
| `outputs/model3/` | Duration/count results |
| `outputs/visualisation/` | Integrated results and visualisations |
| `dashboard/` | Dashboard-related outputs |
| `documentation/` | Generated documentation and supporting evidence |

Check each notebook's configured S3 keys and execution messages for the exact files it reads and writes.

### Lecturer Access to Saved Outputs

- [Open the AWS account sign-in page](https://646468992009.signin.aws.amazon.com/console)
- [Open the Fire2Air S3 bucket](https://ap-southeast-2.console.aws.amazon.com/s3/buckets/fire2air-darwin-dan5-646468992009-ap-southeast-2-an?region=ap-southeast-2&tab=objects)

Authorised lecturers can inspect saved outputs without rerunning the full modelling workflow.

Access details must be obtained privately from the project team. If access is denied, confirm the account, session and assigned permissions with the team.

## Local Workflow

**Notebook folder:** `Source code/`  
**Local dataset folder:** `Source code/Datasets/`  
**Shared output folder:** `Source code/outputs_prt661/`

The local workflow requires the source datasets and matching project files to be available on the computer.

### Notebook Execution Order

| Order | Notebook |
|---|---|
| 1 | `Huynh_Data Processing Feature Engineering_Assessment 3.ipynb` |
| 2 | `Lihini Model 1 Classification.ipynb` |
| 3 | `Favour Model 2 Regression.ipynb` |
| 4 | `Navodya - Model 3_Count_Explainability.ipynb` |
| 5 | `Jenny - Visualisation and Documentation.ipynb` |

Before running each notebook, inspect its path configuration. Some notebooks contain paths that must be adjusted for the current computer.

Restart the kernel and run all cells. Complete preprocessing first, then the three models, followed by visualisation and integration.

### Local Output Locations

The following paths are relative to `Source code/`:

| Folder | Contents |
|---|---|
| `outputs_prt661/processed/` | Processed data, checkpoints and integrated dashboard-ready output |
| `outputs_prt661/metadata/` | Feature manifests and file inventories |
| `outputs_prt661/models/` | Saved model artefacts where configured |
| `outputs_prt661/tables/` | Evaluation tables and prediction files |
| `outputs_prt661/figures/` | Exported figures |
| `outputs_prt661/runs/` | Versioned preprocessing runs |

Individual notebooks may write to different output locations.

In particular, check the local regression notebook's configuration for:

```text
<user home>/Downloads/Fire2Air_Model2/outputs_prt661_a3/
```

Ensure the integration notebook reads the intended regression predictions. If files need to be copied or paths changed, use outputs from the matching model run.

### Integration Inputs

The visualisation and integration workflow requires prediction tables from all three models:

```text
a3_model1_2024_predictions.csv
a3_model2_2024_predictions.csv
a3_model3_2024_predictions.csv
```

Check that:

- All three files are available at the configured input locations.
- Required date and station fields are present.
- Predictions refer to compatible evaluation periods.
- The files belong to the intended preprocessing and model runs.

Missing or incompatible prediction files can prevent the combined outlook from being generated.

**Checkpoint compatibility:** Some filenames retain `a3_` or `_A3` for compatibility. These suffixes alone do not identify the preprocessing version or prove that the files are current. Before integration, confirm that all three prediction files correspond to the intended approved checkpoint, model configuration and compatible station/date evaluation records. In the Assessment 4 report, the duration workflow had verified the regenerated checkpoint; equivalent verification for classification, regression and dashboard outputs was still outstanding.

### Integrated Outputs

Outputs produced by the visualisation and integration notebook include:

| File | Purpose |
|---|---|
| `a3_final_2024_model_summary.csv` | Consolidated model evaluation summary |
| `a3_all_models_station_level_metrics.csv` | Station-level evaluation results |
| `a3_all_models_leave_one_station_out.csv` | Available station holdout results |
| `a3_all_models_ablation_results.csv` | Available feature ablation results |
| `a3_evidence_manifest.csv` | Inventory of supporting output files |
| `Fire2Air_Tomorrow_Smoke_Outlook_2024_A3.csv` | Combined historical outlook table |
| `Fire2Air_dashboard_ready_outlook_A3.csv` | Dashboard-ready integrated output |
| `a3_smoke_outlook_thresholds.json` | Outlook threshold information |
| `a3_08_tomorrow_smoke_outlook_example.png` | Example outlook visualisation |

Availability depends on the required inputs and successful execution of the corresponding notebook sections. Check the notebook's printed output paths.

## Run the Dashboards

### Hosted Dashboard Links

- [Fire2Air User Dashboard](https://fire2airdarwin-cdu-prt661-datasciencepractice-user-dashboard.streamlit.app/)
- [Fire2Air Administrator Dashboard](https://prt661---data-science-practice---dan5---theme2-vdb3z6nuf8appxk.streamlit.app/)

The hosted dashboards present historical project outputs. Their deployment does not establish live forecasting or automatic synchronisation with newly generated notebook results.

### Run Locally

The dashboard scripts read local files.

After running either the AWS or local workflow, make the intended dashboard input files available in the locations expected by the dashboard scripts. For AWS outputs, this may require downloading the relevant files from S3.

Install dashboard dependencies from the repository root:

```powershell
python -m pip install -r "Dashboard/requirements.txt"
```

Start the user dashboard:

```powershell
python -m streamlit run "Dashboard/Fire2Air_User_Dashboard_V3_UI_FIXED.py"
```

Start the administrator dashboard:

```powershell
python -m streamlit run "Dashboard/Fire2Air_Admin_Analytics_Dashboard.py"
```

Run each command in a separate terminal if both dashboards are needed at the same time.

### Dashboard Inputs

Main dashboard inputs include:

```text
Fire2Air_dashboard_ready_outlook_A3.csv
Fire2Air_ntepa_daily_checkpoint_A3.csv
Fire2Air_model_ready_checkpoint_A3.csv
Fire2Air_feature_manifest_A3.json
```

Place the required files in `Dashboard/` or check the input-search paths configured in the selected dashboard script.

Keep the input files consistent with the model run being displayed. Check for older copies in other searched folders, as these may cause outdated results to be loaded.

After replacing inputs, restart the application or clear its cache and reload it.

`Dashboard_ready_csv_file_create.py` is an additional preparation script. Inspect and update its local paths before using it.

## Evaluation and Interpretation

The chronological evaluation design uses:

| Period | Purpose |
|---|---|
| 2018–2022 | Initial training |
| 2023 | Validation and model selection |
| 2024 | Historical testing |

Model-specific selection, final refitting and evaluation procedures are documented in the notebooks and assessment reports.

Evaluation includes task-specific metrics and supporting analyses where available:

| Task | Example evaluation measures |
|---|---|
| Classification | PR-AUC, recall, precision and F1-score |
| Regression | MAE, RMSE and R² |
| Duration / Count | MAE, RMSE and duration-related error analysis |

Supporting analyses include persistence baselines, station-level results, feature ablation and elevated-pollution diagnostics.

### Model 3: Seven-Day Moving-Block Bootstrap

Evaluation uncertainty for Model 3 (elevated-hour duration) was estimated using **300 moving-block bootstrap samples**, with blocks of **seven consecutive calendar days**. All eligible station observations were kept together within each sampled date. The resulting **95% percentile intervals** were **0.9625–1.7068 hours for MAE** and **1.6465–2.6460 hours for RMSE**. These intervals reflect uncertainty in the fitted model's overall evaluation metrics; they are **not prediction intervals for individual daily forecasts**. Sensitivity to other block lengths has not yet been tested.

### Key Findings from the 2024 Historical Evaluation

| Task | Reported result | Important limitation |
|---|---|---|
| Model 1 – Classification | Detected **12 of 37** elevated station-days, missing **25** (**32.43% recall**, threshold **0.87**) | A low predicted risk does not reliably exclude an elevated-PM₂.₅ day. |
| Model 2 – Regression | Overall **RMSE 4.32 µg/m³**; elevated-day **RMSE 13.59 µg/m³** | **91.89%** of elevated-day concentrations were underpredicted. The model performed slightly worse than persistence on elevated days despite improving matched-cohort RMSE overall. |
| Model 3 – Duration / Count | Overall **MAE 1.30 hours**; **MAE 8.35 hours** among the five observations with 13–24 elevated hours | **All five** prolonged observations were underpredicted. On observations matched to duration persistence, persistence had lower MAE but Poisson had lower RMSE. |

These numbers refer to the saved historical evaluation reported in Assessment 4. They should not be treated as evidence that the dashboards are live or that all three models have been independently rerun against the latest preprocessing checkpoint.

Overall metrics should be interpreted alongside missed elevated events and errors during prolonged or severe pollution episodes.

The Fire Smoke Influence Index is an engineered predictor rather than a validated physical smoke-dispersion model. Input Data Completeness describes input availability rather than forecast certainty.

## Limitations

The project has several limitations. The key 2024 results are summarised above; more detailed diagnostics are provided in the Assessment 4 report.

- **Elevated-event detection:** Strong overall results can coexist with missed elevated-PM₂.₅ events. Classification performance should therefore be assessed using event-level recall as well as overall metrics.
- **Errors during severe pollution:** Regression and duration predictions can underestimate unusually high concentrations or prolonged elevated episodes.
- **Limited geographical and temporal coverage:** The study covers three monitoring stations and uses 2024 as the historical test year. Performance elsewhere or in future years is not established.
- **Data quality and availability:** Missing observations, monitoring coverage and preprocessing decisions can affect the modelling data and interpretation of results.
- **Satellite fire information:** Satellite detections do not directly measure smoke exposure at monitoring stations. Detection timing, observation conditions and transport processes limit their interpretation.
- **Engineered fire predictors:** Fire-related features and the Fire Smoke Influence Index simplify complex processes. Their inclusion does not guarantee improved predictive performance.
- **Prediction uncertainty:** Input completeness and model outputs should not be treated as calibrated confidence estimates.
- **Reproducibility and integration:** Separate AWS and local implementations, computer-specific paths and manual dashboard input updates require configuration checks. The duration run has checkpoint verification, but a clean, independently reproduced source-to-all-models-to-both-dashboards run has not yet been demonstrated.
- **Operational readiness:** The project evaluates historical data. Live ingestion, scheduled execution, monitoring and controlled retraining would require further development.

Detailed findings and discussion are provided in the assessment reports and model notebooks.


## Team Contributions

Fire2Air Darwin was developed collaboratively by five team members. Each member had primary responsibilities while also contributing to integration, evaluation, documentation and the final project delivery.

| Team Member | Main Responsibilities | Key Contributions |
|---|---|---|
| **Huynh Anh Thi Pham** | Data Engineering and Feature Engineering | Developed and maintained the shared preprocessing workflow, expanding the feature set from 22 to 93 predictors. Implemented fire-related features, FSI, data coverage checks and missing-value safeguards. Corrected missing-wind handling, audited 657 affected station-days and supported publication of updated checkpoints to AWS S3. Prepared the project update summary and the governance and operational readiness sections of the final report. |
| **Lihini Jinanjalie Deniyelge** | Model 1 – Classification, System Integration and User Dashboard | Developed and evaluated the next-day elevated-PM₂.₅ classification model using Logistic Regression, Random Forest and XGBoost. Conducted threshold selection, feature ablation, station-level and robustness analyses. Contributed to the system architecture and AWS integration. Further developed the public-facing Streamlit dashboard from the initial prototype, improving its interface, fire visualisation, historical forecast presentation and input-data labels. |
| **Esangbedo Favour Ikponwosa** | Model 2 – Regression and Initial Admin Dashboard | Developed and evaluated the next-day PM₂.₅ concentration regression model, comparing Ridge Regression, Random Forest, XGBoost and persistence. Conducted station-level evaluation, feature ablation and elevated-pollution diagnostics. Updated the regression notebook to work with revised project data, regenerated evaluation results and implemented seven-day date-block bootstrap uncertainty analysis. Also contributed to the early administrator dashboard development. |
| **Navodya Piumanthi Siriwardhana Senarath Mudiyanselage** | Model 3 – Duration Forecasting, AWS Integration, Admin Dashboard and Jira Coordination | Developed the elevated-PM₂.₅ duration model and evaluated candidate approaches, including persistence and zero baselines, feature ablation, prolonged-event diagnostics and seven-day moving-block bootstrap analysis. Contributed to preprocessing validation and reproducibility checks. Supported AWS S3, Glue and Athena integration, developed the final administrator dashboard and coordinated project tasks through Jira. Prepared the data engineering, feature calculations and results interpretation sections of the final report. |
| **Dieu Yen (Jenny) Diep** | Initial Dashboard, Visualisation Integration and Project Documentation | Developed the initial Streamlit user dashboard prototype for exploring model predictions, weather, fire activity and historical evaluation results. Updated the visualisation and documentation notebooks, corrected integration paths and generated combined forecasting outputs. Revised the project README with AWS and local execution instructions, dashboard links, dependencies and reproducibility limitations. Prepared the Limitations and Conclusion sections of the Assessment 4 report and reviewed references, citations, figures, tables and overall document formatting. |

### Contribution Evidence

Individual contributions can be verified through GitHub commits, pull requests, Jira tasks and the Individual Work and Contribution Summary pages in the Assessment 4 final report.

Some components were developed collaboratively or improved by different members at different stages. In particular, Jenny developed an initial user dashboard prototype, which Lihini subsequently extended into the final public dashboard. The initial administrator dashboard involved contributions from multiple members, while Navodya developed its final analytical interface.

The responsibilities above describe documented project work and should not be interpreted as evidence that every proposed system capability has been fully deployed or independently reproduced.


## Project Evidence and Documentation

Supporting material is organised under:

| Location | Material |
|---|---|
| `Assesment Reports/` | Assessment reports and project discussion |
| `AWS/AWS_evidences/` | Recorded AWS implementation evidence |
| `ArchitectureDiagrams/` | Architecture diagrams |
| `Documentation/` | Supporting documentation and research material |
| `Project Planning Records/` | Meeting and planning records |
| `Reproducibility Artefacts/` | Supporting reproducibility material |
| `Task Allocation/` | Team responsibility records |

GitHub commits, pull requests and Jira records provide additional evidence of development and individual contributions. Relevant evidence links are included in the final report.

## Responsible Use

Fire2Air Darwin is an educational research prototype.

It is not an official smoke-warning service or a source of medical advice. Historical performance does not guarantee future accuracy, and a low predicted risk does not rule out elevated pollution.

Consult official air-quality information for current conditions.

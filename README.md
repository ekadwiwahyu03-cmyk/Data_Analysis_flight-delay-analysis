# Indian Flight Delay Analysis

An exploratory data analysis (EDA) and machine-learning preparation project using an Indian flight-delay dataset covering 2023–2025. The project investigates data quality, temporal patterns, relationships between operational/weather-related variables and departure delay, and potential data leakage before predictive modeling.

> **Research note:** The dataset is distributed through Kaggle and is described as model input for predictive analytics. The original data-generation process, primary operational source, and definitions of several engineered features are not fully documented in the materials available for this project. Therefore, findings are interpreted as patterns **within this dataset**, not as verified estimates of real-world Indian aviation performance.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Environment and Installation](#environment-and-installation)
- [How to Run](#how-to-run)
- [Analysis Workflow](#analysis-workflow)
- [Key Findings](#key-findings)
- [Data Leakage and Modeling Considerations](#data-leakage-and-modeling-considerations)
- [Limitations](#limitations)
- [Reproducibility and Responsible Interpretation](#reproducibility-and-responsible-interpretation)
- [References](#references)

## Project Overview

Flight delays are influenced by multiple factors, potentially including schedule timing, airport congestion, weather, and aircraft turnaround operations. This project uses a structured flight-level dataset to practice and demonstrate a reproducible analytical workflow:

1. Inspect the raw dataset and its schema.
2. Assess completeness, validity, consistency, and duplication.
3. Investigate identifiers, routes, airlines, and temporal coverage.
4. Perform descriptive and exploratory data analysis.
5. Examine temporal patterns by year, month, weekday, hour, and airline.
6. Analyze numerical and categorical relationships with departure delay.
7. Investigate non-linear patterns and potential feature leakage.
8. Establish a defensible basis for later predictive modeling.

The project is implemented in Python and organized into numbered scripts so that each stage can be run and reviewed independently.

## Objectives

- Understand the structure and quality of the supplied flight dataset.
- Identify missing values, duplicate records, invalid values, and inconsistencies.
- Describe the distribution of departure delays and delay labels.
- Explore delay patterns across time, airlines, routes, and weather categories.
- Measure associations between candidate predictors and departure delay.
- Investigate whether candidate features may contain information unavailable at prediction time.
- Document dataset limitations and avoid unsupported real-world or causal claims.
- Prepare a reproducible foundation for regression and classification experiments.

## Dataset

### Source

- Platform: Kaggle
- Dataset slug: [`leofire123/indian-flight-delay-datasets`](https://www.kaggle.com/datasets/leofire123/indian-flight-delay-datasets)
- Project input file: `india_flight_delay_model_input.csv`
- Period represented in the file: 1 January 2023 – 30 December 2025

The project archive contains the CSV locally at:

```text
data/flight_delay_dataset/india_flight_delay_model_input.csv
```

### Dataset dimensions

| Property | Value |
|---|---:|
| Rows | 150,000 |
| Columns | 23 |
| Date range | 2023-01-01 to 2025-12-30 |
| Unique calendar dates | 1,095 |
| Missing values in the initial quality check | 0 |
| Exact duplicate rows in the initial quality check | 0 |

### Column reference

| Column | Description |
|---|---|
| `Flight_Date` | Flight date |
| `Airline` | Airline name |
| `Flight_Number` | Flight identifier/number |
| `Origin_Airport` | Origin airport code |
| `Destination_Airport` | Destination airport code |
| `Scheduled_Departure_Hour` | Scheduled departure hour |
| `Scheduled_Departure_Minute` | Scheduled departure minute |
| `Day_of_Week` | Day-of-week value |
| `Month` | Month value |
| `Is_Weekend` | Weekend indicator |
| `Peak_Hour` | Peak-hour indicator |
| `Weather` | Weather category |
| `Temperature_C` | Temperature in degrees Celsius |
| `Humidity_pct` | Relative humidity percentage |
| `Wind_Speed_kmh` | Wind speed in km/h |
| `Visibility_km` | Visibility in kilometres |
| `Rainfall_mm` | Rainfall in millimetres |
| `Cloud_Cover_pct` | Cloud cover percentage |
| `Origin_Congestion_Index` | Origin-airport congestion index; exact construction is not documented in the supplied project materials |
| `Previous_Flight_Delay_Minutes` | Previous-flight delay in minutes; linkage and calculation method are not documented in the supplied project materials |
| `Turnaround_Risk_Index` | Turnaround risk index; exact construction is not documented in the supplied project materials |
| `Departure_Delay` | Departure delay value, treated as the continuous outcome in regression-oriented analysis |
| `Delay_Target` | Binary delay label |

**Important:** The descriptions above reflect column names and the dataset description available to the project. They should not be interpreted as independently verified operational definitions. In particular, the exact formulas and timestamps behind engineered indices are unknown.

## Repository Structure

```text
flight-delay-analysis/
├── .vscode/
│   └── settings.json
├── data/
│   └── flight_delay_dataset/
│       └── india_flight_delay_model_input.csv
├── reports/
│   ├── figures/
│   ├── temporal/
│   │   ├── figures/
│   │   ├── 09_hourly_by_airline.csv
│   │   ├── 09_hourly_by_year.csv
│   │   ├── 09_monthly_analysis.csv
│   │   ├── 09_overall_hourly_summary.csv
│   │   ├── 09_weekday_analysis.csv
│   │   └── 09_yearly_analysis.csv
│   ├── relationships/
│   │   ├── figures/
│   │   ├── 10a_nonlinear_quantile_analysis.csv
│   │   ├── 10b_features_by_delay_status.csv
│   │   ├── 10b_feature_correlation.csv
│   │   ├── 10_numeric_quality_check.csv
│   │   ├── 10_numeric_relationships.csv
│   │   └── 10_weather_delay_summary.csv
│   ├── data_quality_summary.json
│   ├── duplicate_group_analysis.csv
│   ├── historical_consistency.csv
│   └── other EDA CSV outputs
├── src/
│   ├── 01_download_and_inspect.py
│   ├── 02_data_quality.py
│   ├── 03_investigate_duplicates_target.py
│   ├── 04_deep_duplicate_check.py
│   ├── 05_variable_investigation.py
│   ├── 06_airline_temporal_check.py
│   ├── 07_historical_consistency_check.py
│   ├── 08_eda_basic.py
│   ├── 09_temporal_pattern_investigation.py
│   ├── 10_relationship_analysis.py
│   ├── 10a_nonlinear_investigation.py
│   └── 10b_feature_leakage_audit.py
└── requirements.txt
```

The `reports/` directory stores tabular outputs and generated figures. The scripts write analysis artifacts there so that results can be reviewed without rerunning every calculation.

## Environment and Installation

### Requirements

- Python 3.10 or later recommended
- Visual Studio Code or another Python IDE
- Git (optional)
- Google Colab can also be used, with paths adjusted to the Colab environment

### Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```bat
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

From the project root:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install matplotlib
```

`matplotlib` is installed separately because the analysis scripts generate figures and import it, while it is not listed in the supplied `requirements.txt`.

## How to Run

Run commands from the project root (`flight-delay-analysis/`).

```bash
python src/01_download_and_inspect.py
python src/02_data_quality.py
python src/03_investigate_duplicates_target.py
python src/04_deep_duplicate_check.py
python src/05_variable_investigation.py
python src/06_airline_temporal_check.py
python src/07_historical_consistency_check.py
python src/08_eda_basic.py
python src/09_temporal_pattern_investigation.py
python src/10_relationship_analysis.py
python src/10a_nonlinear_investigation.py
python src/10b_feature_leakage_audit.py
```

Run the scripts in sequence because later analyses build on the understanding developed in earlier checks. If a script is launched from a different working directory or moved to another folder, verify that its project-root and dataset-path resolution still points to the intended CSV.

## Analysis Workflow

| Stage | Script | Purpose |
|---|---|---|
| 1 | `01_download_and_inspect.py` | Dataset discovery, loading, dimensions, schema, and initial inspection |
| 2 | `02_data_quality.py` | Missing values, data types, date checks, value ranges, and target distribution |
| 3 | `03_investigate_duplicates_target.py` | Repeated flight/date identifiers and their relationship with the target |
| 4 | `04_deep_duplicate_check.py` | Duplicate investigation using a more detailed composite key |
| 5 | `05_variable_investigation.py` | Variable distributions, airline counts, routes, and feature characteristics |
| 6 | `06_airline_temporal_check.py` | Airline coverage across years and months |
| 7 | `07_historical_consistency_check.py` | Comparison of airline records with known historical operating timelines |
| 8 | `08_eda_basic.py` | Descriptive statistics, airline comparison, delay distribution, and hourly analysis |
| 9 | `09_temporal_pattern_investigation.py` | Yearly, monthly, weekday, hourly, and airline-hour analysis |
| 10 | `10_relationship_analysis.py` | Numeric correlations, weather summaries, and relationship plots |
| 11 | `10a_nonlinear_investigation.py` | Quantile-based investigation of potentially non-linear relationships |
| 12 | `10b_feature_leakage_audit.py` | Preliminary feature/target relationship and leakage-risk investigation |

## Key Findings

The following findings are based on the analysis outputs available in this project.

### Data quality and structure

- The working dataset contains 150,000 rows and 23 columns.
- The initial quality assessment found no missing values and no exact duplicate rows.
- `Flight_Date` was successfully parsed for all records.
- The date range is 1 January 2023 through 30 December 2025, covering 1,095 unique dates.
- Repeated `(Flight_Date, Flight_Number)` combinations were found. However, these combinations can represent different airlines or routes, so they should not automatically be deleted as duplicates.
- No duplicate groups were found using the more detailed composite key of date, flight number, airline, origin, and destination in the reported investigation.

### Target distribution

The dataset contains:

| `Delay_Target` | Records | Share |
|---|---:|---:|
| 0 | 20,075 | 13.38% |
| 1 | 129,925 | 86.62% |

The supplied analysis found that `Delay_Target` is directly aligned with whether `Departure_Delay` is zero or positive. Positive delay records begin at 19 minutes in the observed data. This unusual distribution should be documented and investigated rather than assumed to represent a general aviation rule.

### Departure delay summary

| Statistic | Value (minutes) |
|---|---:|
| Count | 150,000 |
| Mean | 42.664 |
| Median | 47 |
| Standard deviation | 19.017 |
| Minimum | 0 |
| Maximum | 119 |

For records with a positive delay, the reported mean is approximately 49.257 minutes and the median is 48 minutes.

### Temporal patterns

- Annual mean delay was broadly stable in the three years:
  - 2023: approximately 42.647 minutes
  - 2024: approximately 42.743 minutes
  - 2025: approximately 42.601 minutes
- The hourly analysis showed higher average delays in several departure-hour bands, particularly around 07:00–10:00 and 17:00–21:00.
- The overall hourly average was highest at 21:00 (approximately 50.78 minutes) and lowest at 14:00 (approximately 35.01 minutes).
- Monthly averages were also relatively close to one another in the supplied analysis.

These are descriptive patterns in the supplied dataset. They do not establish that departure hour or month causes delays.

### Relationships with departure delay

Reported Pearson correlations with `Departure_Delay`:

| Feature | Pearson correlation |
|---|---:|
| `Origin_Congestion_Index` | 0.6214 |
| `Turnaround_Risk_Index` | 0.5352 |
| `Previous_Flight_Delay_Minutes` | 0.2185 |
| `Rainfall_mm` | 0.1057 |
| `Cloud_Cover_pct` | 0.0271 |
| `Scheduled_Departure_Hour` | 0.0239 |
| `Humidity_pct` | 0.0176 |
| `Scheduled_Departure_Minute` | 0.0031 |
| `Wind_Speed_kmh` | 0.0029 |
| `Temperature_C` | 0.0017 |
| `Visibility_km` | -0.0276 |

The congestion and turnaround indices show the strongest positive associations among the numeric variables tested. Quantile analysis also showed increasing average delay across higher groups of these indices. These findings are associations, not causal effects.

### Weather categories

The supplied category summary showed higher mean delay for `Rain` and `Heavy Rain` than for `Clear/Partly Cloudy`. The `Cloudy` category contained only four records, which is too small a group for a reliable comparison. Category-level differences should be interpreted descriptively and with attention to group sizes.

### Historical consistency

The historical consistency investigation identified records that appear inconsistent with known airline timelines:

- Vistara records appear after its final operating date in November 2024.
- Fly91 records appear before its commercial operations began in March 2024.

These inconsistencies are important dataset limitations. They should be documented and investigated in sensitivity analysis; they should not be silently removed without a justified rule.

## Data Leakage and Modeling Considerations

The project distinguishes statistical association from data leakage.

### Direct target leakage

`Departure_Delay` must not be used as a predictor when `Delay_Target` is the target, because the binary target is directly determined by whether the departure delay is zero or positive.

Likewise, `Delay_Target` must not be used as a predictor when `Departure_Delay` is the regression target, because it is derived from the outcome being predicted.

### Candidate features requiring provenance checks

| Feature | Observed association with `Departure_Delay` | Required verification |
|---|---|---|
| `Origin_Congestion_Index` | Relatively strong positive | Determine whether it is measured before departure and whether its calculation uses realized delay information |
| `Turnaround_Risk_Index` | Moderate positive | Obtain its formula and confirm that it uses only information available before prediction |
| `Previous_Flight_Delay_Minutes` | Weak positive | Confirm that it refers to the relevant previous leg/aircraft and is available before the current departure |

The available dataset description does not specify the construction formulas or timestamps for these three features. Consequently, the statistical audit cannot certify them as leakage-free. A high correlation alone is not proof of leakage, and a low correlation does not prove that a feature is safe.

### Recommended modeling protocol

For a defensible predictive experiment:

1. Define the prediction moment (for example, a fixed time before scheduled departure).
2. Define the target precisely: regression on `Departure_Delay`, or classification on `Delay_Target`.
3. Exclude direct target-derived variables from predictors.
4. Include a candidate feature only after confirming it would be available at the defined prediction moment.
5. Use chronological train/validation/test splits where appropriate; avoid randomly mixing future and past records when evaluating a time-dependent prediction task.
6. Fit preprocessing steps only on training data.
7. Compare against a simple baseline before interpreting more complex models.
8. Report suitable metrics:
   - Regression: MAE, RMSE, and R².
   - Classification: precision, recall, F1-score, ROC-AUC, and a confusion matrix; include class balance and threshold rationale.
9. Record random seeds, package versions, split dates, and all model parameters.
10. Treat model performance as dataset-specific unless the dataset provenance and external validity are established.

**Model results are intentionally not stated in this README.** The supplied project archive contains EDA and preliminary leakage-audit scripts and reports, but no model-training scripts or verified model evaluation outputs. Add model names, split strategy, metrics, and selected features only after those experiments have actually been run and checked.

## Limitations

1. **Unverified provenance:** The original operational source, collection procedure, and whether the data are real, simulated, or synthetically generated are not established by the supplied documentation.
2. **Engineered feature definitions:** The exact formulas and timestamps for congestion and turnaround indices are unavailable.
3. **Previous-flight linkage:** The dataset does not provide an explicit aircraft identifier or documented linkage proving that `Previous_Flight_Delay_Minutes` belongs to the same aircraft's preceding leg.
4. **Historical airline inconsistencies:** Some records conflict with known airline operating timelines.
5. **Target construction:** The observed relationship between `Departure_Delay` and `Delay_Target`, including the absence of positive delays from 1 to 18 minutes, needs clarification from the dataset creator.
6. **No causal inference:** Correlation and grouped averages do not establish causality.
7. **External validity:** Results should not be generalized to all Indian flights, airlines, airports, or real-world operations without independent validation against authoritative operational data.

## Reproducibility and Responsible Interpretation

- Keep the original CSV unchanged and store derived outputs separately.
- Record any cleaning or filtering rule and the number of affected records.
- Do not remove records solely because they appear unusual; investigate and document the reason first.
- Do not infer missing feature definitions.
- Preserve the scripts and generated CSV/figure outputs used to support conclusions.
- Re-run the workflow when the input dataset changes.
- Clearly distinguish verified results, exploratory observations, and assumptions.

## References

1. Kaggle dataset: [`leofire123/indian-flight-delay-datasets`](https://www.kaggle.com/datasets/leofire123/indian-flight-delay-datasets).
2. Python documentation: https://docs.python.org/3/
3. pandas documentation: https://pandas.pydata.org/docs/
4. NumPy documentation: https://numpy.org/doc/
5. Matplotlib documentation: https://matplotlib.org/stable/

---

## Project Status

This repository documents the dataset inspection, data-quality assessment, exploratory analysis, temporal analysis, relationship analysis, and preliminary leakage investigation available in the project archive.

The project should be considered ready for a carefully controlled modeling phase only after the prediction-time definition and provenance of candidate engineered features have been addressed. Any final machine-learning claims must be added from verified experiment outputs rather than estimated or invented.

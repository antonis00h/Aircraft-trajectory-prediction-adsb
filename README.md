# Aircraft-trajectory-prediction-adsb
Thesis on the Prediction of Aircraft Location/Trajectory

MSc thesis repository for **Thesis on the Prediction of Aircraft Location/Trajectory**.

This repository contains the source code, Google Colab notebook, software requirements, final thesis document, and reproduction instructions for the experimental pipeline developed in the thesis.

## Repository Structure

```text
Aircraft-trajectory-prediction-adsb/
├── README.md
├── requirements.txt
├── code/
│   ├── opensky_thesis.py
│   └── opensky_thesis.ipynb
└── thesis/
    └── Thesis_on_the_Prediction_of_Aircraft_Location_Trajectory.pdf
```

- `code/opensky_thesis.py`: cleaned Python implementation of the complete experimental pipeline.
- `code/opensky_thesis.ipynb`: the same workflow in Jupyter/Google Colab notebook format.
- `requirements.txt`: Python package requirements.
- `thesis/`: final thesis document.

## Study Overview

The thesis investigates deterministic aircraft trajectory prediction in three-dimensional space using ADS-B state-vector data and Long Short-Term Memory (LSTM) neural networks.

The prediction task is formulated as **sequence-to-one regression**. A historical sequence of aircraft observations is used to predict the future displacement:

- latitude displacement,
- longitude displacement,
- altitude displacement.

The evaluated forecasting horizons are:

- **5 minutes** (`300 s`)
- **10 minutes** (`600 s`)
- **15 minutes** (`900 s`)

The final full-day robustness evaluation uses the selected `ENRICH_LSTM96` architecture for the **10-minute** and **15-minute** horizons.

## Data Source

The data used in the thesis were obtained from the **OpenSky Network** historical ADS-B database.

The raw ADS-B files are **not included in this repository** because of their size and data-access constraints.

To reproduce the experiments, corresponding OpenSky state-vector data must be obtained separately and exported as hourly CSV files.

### Spatial and Altitude Filtering

The thesis uses observations within the following study area:

- Latitude: **35° to 55° N**
- Longitude: **5° to 30° E**
- Barometric altitude: **above 6096 m (FL200)**

### Expected CSV Schema

Each hourly CSV file is expected to contain the following columns, in this order and without a header:

```text
icao24
callsign
time
lat
lon
baroaltitude
velocity
heading
vertrate
```

The code reads the files using:

- UTF-16 encoding,
- comma separator,
- no header row.

### Expected File Naming

The code assumes filenames of the form:

```text
opensky_FL200_YYYYMMDD_HH.csv
```

Examples:

```text
opensky_FL200_20190501_00.csv
opensky_FL200_20190501_01.csv
opensky_FL200_20190502_00.csv
...
opensky_FL200_20190502_23.csv
```

## Experimental Data Organization

The experimental workflow uses two stages.

### Development Stage

Initial baseline experiments, feature-enrichment experiments, and architecture comparison use the first hourly development file:

```text
opensky_FL200_20190501_00.csv
```

### Final Model Training

The final horizon-specific `ENRICH_LSTM96` models are retrained using the first two hourly development files:

```text
opensky_FL200_20190501_00.csv
opensky_FL200_20190501_01.csv
```

### Full-Day Evaluation

The final robustness and ETA-style temporal analyses use 24 consecutive hourly files labelled as May 2, 2019:

```text
opensky_FL200_20190502_00.csv
...
opensky_FL200_20190502_23.csv
```

The timestamps contained in these files should be interpreted separately from the filename labels, as described in the thesis.

## Preprocessing

The main preprocessing procedure includes:

1. Removal of observations with missing required variables.
2. Sorting observations by aircraft identifier (`icao24`) and timestamp.
3. Computation of time differences between consecutive observations.
4. Segmentation of trajectories whenever the temporal gap exceeds **60 seconds**.
5. Removal of trajectory segments shorter than **300 observations**.
6. Construction of first-order positional differences:
   - `dlat`
   - `dlon`
   - `dalt`
7. For the enriched representation:
   - ground velocity,
   - vertical rate,
   - sine of heading,
   - cosine of heading.
8. Clipping of:
   - velocity to `[0, 400] m/s`,
   - vertical rate to `[-50, 50] m/s`.

The input history length is:

```text
INPUT_LEN = 120
```

This represents **120 consecutive observations**, not a fixed number of elapsed seconds.

## Prediction Target Construction

Future targets are selected using **elapsed time**, not a fixed row offset.

For a current state at time `t` and forecasting horizon `H`, the code searches for the first observation at or after:

```text
t + H
```

A target is accepted only if the actual time gap differs from the requested horizon by no more than:

```text
10 seconds
```

The target is the future displacement relative to the current position:

```text
[Δlatitude, Δlongitude, Δaltitude]
```

## Dataset Sampling

To control computational cost and reduce over-representation of long trajectory segments:

```text
MAX_SAMPLES = 200000
per_traj_cap = 200
SEED = 42
```

The development datasets are randomly shuffled and partitioned into:

- **80% training**
- **10% validation**
- **10% test**

The split is performed at the window level.

## Input Representations

### Baseline Representation

The baseline model uses three features:

```text
dlat
dlon
dalt
```

### Enriched Representation

The enriched models use seven features:

```text
dlat
dlon
dalt
velocity
sin_hdg
cos_hdg
vertrate
```

## Model Architectures

Three enriched LSTM configurations are compared.

### ENRICH_LSTM64

```text
Input (120 × 7)
→ LSTM(64)
→ Dense(64, ReLU)
→ Dense(32, ReLU)
→ Dense(3, Linear)
```

### ENRICH_LSTM96

```text
Input (120 × 7)
→ LSTM(96)
→ Dense(64, ReLU)
→ Dense(32, ReLU)
→ Dense(3, Linear)
```

### ENRICH_LSTM64x32_DO20

```text
Input (120 × 7)
→ LSTM(64, return_sequences=True)
→ Dropout(0.20)
→ LSTM(32)
→ Dropout(0.20)
→ Dense(64, ReLU)
→ Dense(32, ReLU)
→ Dense(3, Linear)
```

`ENRICH_LSTM96` is selected as the common architecture for the final 10-minute and 15-minute evaluation.

## Training Configuration

The main training settings are:

```text
Optimizer: Adam
Learning rate: 1e-3
Loss: Mean Squared Error
Maximum epochs: 20
Batch size: 256
Early stopping patience: 3
Random seed: 42
```

Input features and target displacements are standardized using `StandardScaler`.

The scalers are fitted **only on the training subset** and then applied unchanged to validation and test data.

Predicted target values are inverse-transformed before physical-domain evaluation.

## Evaluation Metrics

### Horizontal Error

Horizontal prediction error is calculated using the Haversine great-circle distance and reported in kilometers.

The following summaries are used:

- Mean error
- Median error
- 90th percentile error (`p90`)

### Vertical Error

Vertical performance is measured using absolute barometric-altitude error in meters.

The following summaries are used:

- Mean altitude error
- Median altitude error

## Reproduction Instructions

The recommended environment is **Google Colab with Python 3**.

### 1. Clone or Download the Repository

Either clone the repository or download it as a ZIP file.

### 2. Install Required Packages

In a Python environment:

```bash
pip install -r requirements.txt
```

Google Colab already includes several of these packages, but the command can be used to ensure the required dependencies are available.

### 3. Prepare Google Drive

The code is configured to use Google Drive.

Create the following directories:

```text
/content/drive/MyDrive/exports_hourly
/content/drive/MyDrive/opensky_models
/content/drive/MyDrive/opensky_results
```

Place the required hourly OpenSky CSV files in:

```text
/content/drive/MyDrive/exports_hourly
```

### 4. Verify File Names

At minimum, the development stage requires:

```text
opensky_FL200_20190501_00.csv
opensky_FL200_20190501_01.csv
```

The full-day evaluation additionally requires:

```text
opensky_FL200_20190502_00.csv
...
opensky_FL200_20190502_23.csv
```

### 5. Run the Notebook or Python Script

Two equivalent entry points are provided:

```text
code/opensky_thesis.ipynb
```

or

```text
code/opensky_thesis.py
```

For the notebook version, open it in Google Colab and execute the cells sequentially from top to bottom.

For the Python version, execute the script in a compatible environment with Google Drive mounted and the expected directory structure available.

### 6. Baseline Experiments

The pipeline first:

- loads the first development hourly file,
- preprocesses and segments the trajectories,
- constructs 3-feature baseline datasets,
- trains a 64-unit LSTM separately for the 5-, 10-, and 15-minute horizons,
- evaluates horizontal and vertical prediction errors.

### 7. Feature-Enrichment Experiments

The code then adds:

- ground velocity,
- vertical rate,
- sine and cosine heading representations,

and repeats the 5-, 10-, and 15-minute experiments using the enriched 7-feature input representation.

### 8. Architecture Comparison

The following models are compared for the 10-minute and 15-minute horizons:

```text
ENRICH_LSTM64
ENRICH_LSTM96
ENRICH_LSTM64x32_DO20
```

The architecture-comparison results are exported to:

```text
/content/drive/MyDrive/opensky_results/architecture_comparison_results.csv
```

### 9. Final ENRICH_LSTM96 Training

The final 10-minute and 15-minute `ENRICH_LSTM96` models are retrained using the first two development hourly files.

The trained models and fitted scalers are stored in:

```text
/content/drive/MyDrive/opensky_models
```

Expected files include:

```text
ENRICH_LSTM96_10min.keras
ENRICH_LSTM96_10min_x_scaler.pkl
ENRICH_LSTM96_10min_y_scaler.pkl

ENRICH_LSTM96_15min.keras
ENRICH_LSTM96_15min_x_scaler.pkl
ENRICH_LSTM96_15min_y_scaler.pkl
```

### 10. Full-Day Robustness Evaluation

The final models are applied independently to the 24 hourly evaluation files.

The resulting hourly spatial metrics are saved to:

```text
/content/drive/MyDrive/opensky_results/full_day_20190502_ENRICH_LSTM96.csv
```

The reported full-day summary is obtained by aggregating the independently calculated hourly metrics.

In particular, the reported full-day `p90` is the **mean of the hourly p90 values**, not a single percentile calculated after pooling all full-day prediction errors.

## Derived ETA-Style Temporal Metric

The thesis also evaluates a derived fixed-horizon temporal proxy.

For each prediction sample:

- `d_true` is the realized horizontal displacement from the current position to the true future position.
- `d_pred` is the predicted horizontal displacement from the current position to the predicted future position.
- `H` is the forecasting horizon in seconds.

The derived temporal estimate is computed as:

```text
eta_hat = H × (d_true / d_pred)
```

and the temporal error is:

```text
eta_error = eta_hat - H
```

For numerical stability:

```text
d_pred >= 1e-3 km
```

and temporal errors are clipped to:

```text
±3600 seconds
```

This metric is **not a conventional destination-specific Estimated Time of Arrival prediction**. It is used only as a horizon-based temporal interpretation of the spatial trajectory-prediction results.

The hourly ETA-style results are saved to:

```text
/content/drive/MyDrive/opensky_results/full_day_20190502_ENRICH_LSTM96_with_ETA.csv
```

## Horizon-Normalized Analysis

The final stage computes:

```text
horizontal error / prediction-horizon duration
```

and

```text
ETA-style absolute error / prediction-horizon duration
```

for the 10-minute and 15-minute full-day results.

These values are used only to describe the scaling behavior observed across the two evaluated horizons.

## Expected Main Output Files

After a complete run, the principal generated files are:

```text
opensky_results/
├── architecture_comparison_results.csv
├── full_day_20190502_ENRICH_LSTM96.csv
└── full_day_20190502_ENRICH_LSTM96_with_ETA.csv

opensky_models/
├── ENRICH_LSTM96_10min.keras
├── ENRICH_LSTM96_10min_x_scaler.pkl
├── ENRICH_LSTM96_10min_y_scaler.pkl
├── ENRICH_LSTM96_15min.keras
├── ENRICH_LSTM96_15min_x_scaler.pkl
└── ENRICH_LSTM96_15min_y_scaler.pkl
```

Plots used to inspect baseline performance, feature enrichment, architecture comparison, hourly robustness, ETA-style errors, and horizon-normalized error behavior are generated during execution.

## Reproducibility Notes and Limitations

Several points should be considered when reproducing the experiments:

- The raw OpenSky dataset is not distributed with this repository.
- Access to the same historical OpenSky data is required for exact replication.
- The development split is a random **window-level** 80/10/10 split.
- Overlapping windows may therefore occur across development subsets.
- The historical input contains **120 observations**, not a fixed 120-second interval.
- Future targets are selected using timestamp-based elapsed-time matching.
- Separate models and scalers are used for each forecasting horizon.
- The final full-day evaluation is temporally distinct from the development files but remains within the same general geographical and temporal study setting.
- TensorFlow training can show small numerical differences across hardware, software versions, and runtime environments even when the same random seed is used.
- Exact package versions from the original Google Colab runtime were not permanently recorded; `requirements.txt` therefore lists the required packages without asserting unverified historical version numbers.

## Software Requirements

The project uses:

- Python 3
- NumPy
- pandas
- Matplotlib
- scikit-learn
- TensorFlow / Keras
- joblib
- Google Colab / Google Drive integration

See:

```text
requirements.txt
```

for the package list.

## Thesis

The final MSc thesis document is included in the `thesis/` directory.

**Title:**  
*Thesis on the Prediction of Aircraft Location/Trajectory*

## Author

**Antonios Chamilos**  
MSc in Applied Statistics  
University of Piraeus

## Data Availability

The raw ADS-B observations used in this work were obtained from the OpenSky Network historical database.

The raw data are not redistributed in this repository because of their large size and access constraints. The repository provides the complete preprocessing, model-training, evaluation, and analysis pipeline required to reproduce the study when the corresponding source data are available.

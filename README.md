# F1 Standings Predictor

A work-in-progress Python project that pulls Formula 1 race data from [FastF1](https://docs.fastf1.dev/), builds lap-by-lap datasets for each Grand Prix, analyses which factors affect lap time, and is starting to build a neural network that predicts lap times.

> **Current status:** The data collection and exploratory plotting are in place. The model (`model.py`) is an early skeleton: it doesn't run yet, and it has no loss function, optimiser or training loop. Nothing in the code predicts championship **standings** yet. The model's target is currently **lap time**. A to-do note in `model.py` mentions adding "the standings matrix", but the code doesn't yet show how lap-time predictions would become standings.

---

## How it works

The pipeline has three stages.

### 1. Data collection (FastF1)

FastF1 downloads official timing data for each race session. For every lap, the scripts keep things like lap time, tyre compound, tyre age and track status. They then attach weather readings with `pandas.merge_asof`, which matches each lap to the most recent weather sample taken before that lap's timestamp. Responses are cached locally in `fastf1_cache/`, so data that's already been downloaded isn't fetched again.

There are two dataset builders:

| Script | Output folder | Purpose |
|---|---|---|
| `buildDataset.py` | `all/` | The original, feature-rich builder used for the analysis plots |
| `buildDataset2.py` | `all2/` | A slimmer builder with the columns the model uses |

### 2. Feature engineering

Key concepts used across the builders:

- **Fuel load:** a stand-in for fuel weight, calculated as `(total laps − current lap) / total laps`. It's 1.0 at the start and 0.0 at the chequered flag. Cars get faster as fuel burns off.
- **Tyre life and compound:** the number of laps on the current set and the tyre type. The older 2018 compounds (ULTRASOFT, HYPERSOFT, SUPERSOFT) are merged into `SOFT`.
- **Track status flags** (`buildDataset.py` only): FastF1's `TrackStatus` codes are turned into true/false columns for safety car (`4`) and virtual safety car (`6`). A **restart lap** is any lap that comes straight after a SC or VSC lap, when the cars are still bunched up.
- **In-laps and out-laps** (`buildDataset.py` only): laps where the driver enters or leaves the pits. These are much slower than normal racing laps.
- **Lag features** (`buildDataset.py` only): the driver's lap times from 1, 2 and 3 laps earlier (`LapTime_lag1..3`) and the change from the previous lap (`LapTimeDelta`).
- **Pace vs median** (`buildDataset.py` only): the driver's lap time minus the median lap time of the whole field on that lap. `HistPaceVsMedian` is the driver's average of this value at the same circuit in earlier years, using clean laps only.
- **Regulation era** (`buildDataset.py` only): `GroudEffectEra` is true from 2022 onwards, when the ground-effect aerodynamic rules came in.
- **Track key** (`buildDataset2.py` only): races are grouped by **venue and layout**, not by event name. For example, the Styrian and Austrian GPs share the Spielberg key. Circuits with a major layout change are split into `_old` and `_new` versions:

  | Venue | New layout from |
  |---|---|
  | Melbourne | 2022 |
  | Yas Marina | 2021 |
  | Barcelona | 2023 |
  | Singapore | 2023 |

  The 2020 Sakhir GP (outer loop) has its own key, `Sakhir Outer`.

### 3. Analysis and modelling

- **Correlation analysis** (`plot.py`): calculates Pearson's *r* between each numeric or true/false feature and `LapTime`, and draws it as a bar chart.
- **Tyre degradation** (`plot.py`): plots lap time against tyre life for each compound with 95% confidence intervals. It uses only accurate laps, leaving out SC, VSC, in-laps, out-laps and restart laps. A comment in the code says the result is "all over the place".
- **Outlier cleaning** (`cleanData.py`): two helper functions, one using the z-score (number of standard deviations from the mean) and one using the IQR rule (keep values within 1.5 × the interquartile range).
- **Model** (`model.py`):
  - Loads one race (`all2/all2_Australian Grand Prix.csv`).
  - One-hot encodes the text columns (Driver, Compound, Car, Track, Race).
  - Z-score normalises `LapNumber`, `TyreLife` and `TrackTemp`.
  - Defines a two-hidden-layer MLP with learnable **embeddings** (small learned vectors) for each driver, car and track.

  See [Current status and known issues](#current-status-and-known-issues) for what doesn't work yet.

---

## Data and inputs

- **Source:** the FastF1 library (official F1 live timing data). An internet connection is needed the first time each session is downloaded.
- **Seasons:** 2018–2025. `build_all_data.py` loops over every non-testing event in those years.
- **Session type:** race (`'R'`) by default.
- **Rate limit:** FastF1's API allows about 500 calls per hour. `build_all_data.py` sleeps for an hour when it hits the limit, then retries. `buildDataset.py` writes skipped sessions, along with the reason, to `SKIPPED.txt`.

**Columns in the model dataset (`all2/all2_<Event>.csv`):**

| Column | Meaning |
|---|---|
| `Driver` | Three-letter driver code (e.g. `HAM`) |
| `LapNumber` | Lap number in the race |
| `Compound` | Tyre compound |
| `TyreLife` | Laps on the current tyre set |
| `TrackTemp` | Track temperature (°C) at the time of the lap |
| `FuelLoad` | 1.0 at the start down to 0.0 at the finish |
| `Car` | Team name |
| `Track` | Venue and layout key (see above) |
| `Year` | Season |
| `Race` | Event name |
| `LapTime` | **Target:** lap time in seconds |

The `all/` datasets have more columns: position, stint, pit times, air temperature, humidity, rainfall, starting grid position, and all the engineered features listed above.

> **Note:** All `.csv` files and `fastf1_cache/` are git-ignored, so a fresh clone has **no data**. You have to build it yourself (see below).

---

## How to run it

Run everything from inside `F1-Project/`, because the scripts use relative paths.

### Set up

```bash
cd F1-Project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No minimum Python version is stated. The local virtual environment uses Python 3.14.

### Build the model datasets (`all2/`)

```bash
python build_all_data.py
```

This builds one CSV per Grand Prix name and skips any that already exist in `all2/`. Because of the rate limit, a full run can take several hours.

To build a single event:

```bash
python -c "import buildDataset2 as bd; bd.build_dataset('Australian Grand Prix')"
```

### Build the analysis datasets (`all/`)

```bash
python -c "import buildDataset as bd; bd.build_dataset('Australian Grand Prix')"
```

> ⚠️ This currently fails with a `KeyError`. See [known issues](#current-status-and-known-issues).

### Make the plots

These read from `all/`:

```bash
python -c "import plot_all as pa; pa.save_plots_cwl(save_dir='plots', data_dir='all')"     # correlation charts
python -c "import plot_all as pa; pa.save_plots_td(save_dir='tyre_deg', data_dir='all')"   # tyre degradation charts
```

The `save_dir` values above match the folders the existing PNGs are in. However, the code doesn't record that these exact commands produced them.

### Model

```bash
python model.py
```

> ⚠️ This currently fails. See below.

---

## Project structure

```
F1StandingsPredictor/
├── README.md
└── F1-Project/
    ├── buildDataset.py        # Original builder → all/ (feature-rich, used for plots)
    ├── buildDataset2.py       # Model builder → all2/ (venue/layout track keys)
    ├── build_all_data.py      # Builds all2/ for every GP 2018–2025, handles the rate limit
    ├── cleanData.py           # Z-score and IQR outlier helpers
    ├── plot.py                # Correlation, tyre-degradation and scatter plots
    ├── plot_all.py            # Runs plot.py over every CSV in a folder and saves PNGs
    ├── model.py               # PyTorch MLP with embeddings (work in progress)
    ├── main.py                # Imports only; no logic yet
    ├── testbuildset.py        # Imports buildDataset only; no tests
    ├── test/buildtest.py      # Imports buildDataset only; no tests
    ├── untitled.py            # Empty
    ├── Untitled.ipynb         # Scratch notebook: exploring all/ data and the event schedule
    ├── Untitled1.ipynb        # Scratch notebook: exploring a 2024 Melbourne session (pit durations, weather)
    ├── requirements.txt       # Pinned dependencies (torch, fastf1, pandas, scikit-learn, seaborn, …)
    ├── howtoFastf1.txt        # Notes on FastF1 session codes and a usage example
    ├── useful.txt             # Saved printouts of FastF1 results and locations, for reference
    ├── testtimes.txt          # Saved printouts of lap times per race, for reference
    ├── SKIPPED.txt            # Log of sessions skipped by buildDataset.py
    ├── plots/                 # Correlation-with-lap-time charts, one per GP
    ├── tyre_deg/              # Tyre degradation charts, one per GP
    ├── all/                   # (git-ignored) datasets from buildDataset.py
    ├── all2/                  # (git-ignored) datasets from buildDataset2.py
    └── fastf1_cache/          # (git-ignored) FastF1 download cache
```

There are also some git-ignored CSVs in `F1-Project/` (`ver_austria*.csv`, `HAM_silverstone.csv`, `stint.csv`). The `ver_austria_zscores.csv` and `ver_austria_IQR.csv` files match the hard-coded output names in `cleanData.py`. **It's unclear from the code how the others were made.**

---

## Current status and known issues

These come from reading and running the code as it is now:

1. **`model.py` doesn't run.** `F1MLP` calls `Embeddings(driver_dim, car_dim, track_dim)`, but `Embeddings.__init__` needs six arguments, so the script stops with a `TypeError`. `Embeddings.forward` also calls `nn.Parameter` objects as if they were functions, which won't work.
2. **The model's input size looks wrong.** `input_dim` is set to `len(X)`, which is the number of **rows**, not the number of features. The hidden sizes `n/2` and `n/4` are floats, but `nn.Linear` needs whole numbers.
3. **Normalisation doesn't reach the model inputs.** `LapNumber`, `TyreLife` and `TrackTemp` are normalised in `d` *after* `X` has already been built from `d`, so `X` still has the raw values.
4. **It's unclear whether the model uses one-hot encoding, embeddings or both.** `model.py` one-hot encodes Driver, Car and Track and *also* defines embeddings for them.
5. **There's no training, evaluation or testing yet.** The to-do list in `model.py` plans Huber loss, the Adam optimiser, training, evaluation and test loops, lag-lap (autoregressive) inputs, pit stops and a standings matrix. `LR = 0.001` is defined but not used.
6. **`buildDataset.build_dataset` fails.** It calls `dropna` on `LapTimeDelta` and `LapTime_lag1..3` before `add_features` creates those columns, which raises a `KeyError`. The existing `all/` data and plots must have been made with an earlier version of this code.
7. **`plot_all.make_super_plot` and `plot_all_in_dir` probably don't work.**
   - `make_super_plot` saves `All_Races_Ever.csv` in the current folder, then passes that name to `plot.py`. `plot.py` adds another `.csv` and looks inside `all/`, so it would look for `all/All_Races_Ever.csv.csv`.
   - `plot_all_in_dir` passes full paths where `plot.py` expects bare file names.
8. **The `driver` argument in `buildDataset2.build_dataset` only changes the output path.** It doesn't filter the data to one driver.
9. **The plot folders and the local data don't line up.** `plots/` and `tyre_deg/` hold charts for 30 GPs, but the local `all/` folder currently has only 13 CSVs.

---

## Results

> _To be filled in once the model trains._

| Metric | Value |
|---|---|
| Training set | [races / seasons used] |
| Test set | [held-out races / seasons] |
| Lap-time MAE (s) | [placeholder] |
| Lap-time RMSE (s) | [placeholder] |
| Huber loss | [placeholder] |
| Standings accuracy | [placeholder — metric to be decided] |

**Feature correlation findings:** [summary of the `plots/` charts]

**Tyre degradation findings:** [summary of the `tyre_deg/` charts]

**Example prediction vs actual:** [chart or table]

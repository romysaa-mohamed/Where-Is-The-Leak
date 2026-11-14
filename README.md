# Where Is The Leak?

Machine-learning leak **detection** and **localization** for water distribution networks, built on the BattLeDIM 2018 benchmark (L-Town network) and delivered as an interactive Streamlit dashboard.

The app loads SCADA pressure and flow data, runs two trained models, and shows:

- when the network looks like it is leaking (detection),
- which pipes are most likely responsible (localization),
- where those pipes sit on a 3D map of the real L-Town network.

---

## Features

| View | What it shows |
| --- | --- |
| **Map** | 3D network map. Each stem is a monitored pipe; height and colour show leak confidence at the chosen time. Status, alert count, and most suspected pipe float on top. A time slider moves through the analysed period. |
| **Timeline** | Leak confidence over time against the detection threshold, plus the documented leak periods. When ground truth is available it also reports MCC and F1 scores. |
| **Pipes** | Ranking of pipes by mean confidence and share of time above the threshold, and a heatmap of confidence per pipe over time. |
| **Sensors** | Raw readings for any selected sensors over the analysed period. |
| **Model** | Model details, the most important detection features, and a CSV download of all predictions. |

**Settings** (top right) holds everything you can change: data upload, detection and localization thresholds, alert persistence, the network file, and the analysis period.

---

## Project structure

```
project/
├── app.py                          # the dashboard
├── requirements.txt
├── README.md
├── L-TOWN.inp                      # EPANET network file (draws the real network)
├── .streamlit/
│   └── config.toml                 # theme (created automatically on first run)
├── artifacts/
│   └── water_leakage_models.joblib # trained models, produced by the notebook
└── Dataset/
    ├── 2018_SCADA_Pressures.csv
    ├── 2018_SCADA_Flows.csv
    └── 2018_Leakages.csv           # optional, ground truth for evaluation
```

The app looks for the models file, the `Dataset` folder, and `L-TOWN.inp` next to `app.py` or in its parent folder. If something is missing, you can upload it from **Settings**.

---

## Getting started

**Requirements:** Python 3.9 or newer.

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

> `scikit-learn` is pinned to `1.6.1` because the models were trained with that exact version. Loading them with a different version can fail or give wrong results.

### Theme

`app.py` creates `.streamlit/config.toml` in the folder you run the command from. Streamlit only reads the theme at startup, so after the first run **stop the app (`Ctrl+C`) and start it again once** to apply the colours. On Windows the `.streamlit` folder is hidden; enable *View → Show → Hidden items* in File Explorer to see it.

---

## Data format

| File | Content |
| --- | --- |
| `2018_SCADA_Pressures.csv` | `Timestamp` plus the pressure sensors `n1 … n769` (the sensors the models use). |
| `2018_SCADA_Flows.csv` | `Timestamp` plus the flow sensors, including `p227`, `p235` and `PUMP_1`. |
| `2018_Leakages.csv` | Optional. Per-pipe leak flow used as ground truth for the evaluation metrics. |

- CSV files are read with `;` as separator and `,` as decimal mark. A comma-separated file with a dot decimal is also accepted.
- Both the pressure and the flow files are required. The app reports exactly which sensor columns are missing.
- The models expect a **5-minute** sampling interval. Other intervals trigger a warning because results may be unreliable.
- At least **200 rows** (about 17 hours) are needed to build the rolling features.
- Small gaps in sensor values are interpolated automatically.

### Network layout

- If `L-TOWN.inp` is present (or uploaded), the map uses its real coordinates.
- Otherwise you can upload a coordinates CSV with the columns `id,x,y`.
- With neither, the app falls back to a **schematic** layout, which is clearly labelled as not geographic.

---

## How it works

1. **Feature engineering.** The app rebuilds exactly the features the models were trained on (differences, rolling means and standard deviations, deviation from a rolling baseline). All of them are causal, so no future data is used.
2. **Detection.** A classifier outputs a leak confidence for every time step. A step counts as an alert when the confidence stays above the detection threshold for the chosen number of consecutive steps (persistence).
3. **Localization.** One estimator per monitored pipe outputs a confidence that this pipe is leaking, which drives the map and the pipe ranking.
4. **Chronic leak.** Pipe `p257` has a long-lasting leak documented from 8 January 2018 that was never repaired. It is shown separately on the map and excluded from the ranking and from the ground truth used for scoring, so it does not hide the other pipes.

Scores are model confidence values (tree-ensemble votes), not calibrated probabilities. The models were trained on one year and one set of events, so scores show recognition of known leak patterns, not guaranteed generalisation to other networks.

---

## Settings reference

| Setting | Default | Meaning |
| --- | --- | --- |
| Detection threshold | 0.50 | Confidence above which a time step is counted as a leak. |
| Localization threshold | 0.50 | Confidence above which a pipe is highlighted as leaking. |
| Alert persistence | 6 steps (30 min) | Consecutive steps above the threshold needed to raise an alert. |
| Analysis period | last 14 days | Part of the data that is analysed and shown. |

---

## Troubleshooting

| Problem | Fix |
| --- | --- |
| "Models file not found" | Run the notebook to produce `artifacts/water_leakage_models.joblib` and place it next to `app.py`. |
| "Upload the SCADA files…" message | Open **Settings** and upload the pressure and flow CSVs, or put the `Dataset` folder next to the app. |
| `streamlit` command not found | Use `python -m streamlit run app.py`. |
| Colours did not change | Stop the app with `Ctrl+C` and start it again. The theme is read at startup only. |
| Model loading error | Check that `scikit-learn==1.6.1` is installed. |

---

## Credits

Built as a graduation project on the BattLeDIM 2018 (Battle of the Leakage Detection and Isolation Methods) benchmark and the L-Town network.

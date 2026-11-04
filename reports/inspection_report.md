# MAI391 — Dataset inspection (Khoa)

- Source: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
- Source SHA-256: `eed1f222f07f0ff48153f437bffd97d2a8cc60d3158bc458ed14b0a271f9cd3f`
- Dataset shape: **1599 × 12**
- Input features: **11**; target: **`quality`**
- Missing cells: **0**
- Duplicate rows: **240**, retained for baseline
- Training samples: **1279**
- Testing samples: **320**
- Split: **80/20**, `random_state=42`
- Quality observed range: **3–8**
- Test rows matching a full training record: **64/320**
- Unique full-record values present in both split sets: **62** (limitation)
- StandardScaler optional, fitted on train only

Files: `data/train.csv`, `data/test.csv`, `reports/feature_order.json`,
`reports/descriptive_statistics.csv`, `reports/quality_distribution.csv`,
`reports/preprocessing_manifest.json`, `models/standard_scaler.pkl`.

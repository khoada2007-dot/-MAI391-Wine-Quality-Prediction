# MAI391 — Wine Quality Prediction

**Project:** Red Wine Quality Prediction using Multiple Linear Regression  
**Course:** MAI391 — Mathematics for Machine Learning  
**Repository:** [MAI391-Wine-Quality-Prediction](https://github.com/khoada2007-dot/-MAI391-Wine-Quality-Prediction)

## Dataset

The [UCI Red Wine Quality dataset](https://archive.ics.uci.edu/dataset/186/wine+quality) has **1,599 samples**, **11 physicochemical input features**, and one target column, **`quality`**. Preprocessing found **0 missing values** and **240 duplicate rows** (retained for the agreed baseline).

## Repository layout

```text
.
├── data/
│   ├── winequality-red.csv           # Official UCI dataset (semicolon-separated)
│   ├── train.csv                     # Generated: 1,279 rows including quality
│   └── test.csv                      # Generated: 320 rows including quality
├── notebooks/
│   └── Khoa_Dataset_Preprocessing.ipynb
├── reports/
│   ├── column_quality_report.csv
│   ├── descriptive_statistics.csv
│   ├── quality_distribution.csv
│   ├── feature_order.json
│   ├── train_indices.csv
│   ├── test_indices.csv
│   ├── preprocessing_manifest.json
│   └── inspection_report.md
├── models/
│   └── standard_scaler.pkl           # Optional; fitted on training data only
└── README.md
```

Generated files appear after running the notebook. Repository links below are **relative** to this README, so they work if the repository name or owner changes.

## Run preprocessing (Jupyter Notebook or VS Code)

1. Clone or download this repository as ZIP and extract it **with its folder structure intact**.
2. Make sure `data/winequality-red.csv` exists. If not, the notebook attempts to download it from [UCI CSV](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv) when online.
3. Install dependencies in the selected Python environment:

   ```bash
   python -m pip install pandas numpy scikit-learn joblib notebook ipykernel
   ```

4. Open [`notebooks/Khoa_Dataset_Preprocessing.ipynb`](notebooks/Khoa_Dataset_Preprocessing.ipynb) in Jupyter Notebook or VS Code, select the correct Python kernel, then click **Run All**.
5. Results are saved automatically under `data/`, `reports/`, and `models/`. No Colab imports or downloads are required.

Run Jupyter from the repository root (`jupyter notebook`) or open the `.ipynb` in VS Code. The notebook detects the root from the `data/` and `notebooks/` directories, including when the working directory is `notebooks/`.

## Fixed preprocessing configuration

- X: 11 features in the original order, y: `quality`.
- Split: `train_test_split(X, y, test_size=0.2, random_state=42)`.
- Train: **1,279 samples**; test: **320 samples**.
- Duplicates **retained** for the baseline. This can produce identical full-record values in both train and test; discuss this limitation in the presentation.
- Scaling is **optional**, not required for baseline. `StandardScaler` is fitted on **training data only**, then applied to test data without refitting.
- Do not independently re-split or reorder features in M4/M5 notebooks. `train.csv` / `test.csv` are the common datasets for the experiments.

## Files for team members

| Member | Required files |
|---|---|
| Thao — EDA | [`reports/quality_distribution.csv`](reports/quality_distribution.csv), [`reports/descriptive_statistics.csv`](reports/descriptive_statistics.csv) |
| Dao — Math | [`reports/feature_order.json`](reports/feature_order.json) |
| Khue — Baseline | [`notebooks/Khoa_Dataset_Preprocessing.ipynb`](notebooks/Khoa_Dataset_Preprocessing.ipynb), [`data/train.csv`](data/train.csv), [`data/test.csv`](data/test.csv) |
| Thu — Experiments | [`data/train.csv`](data/train.csv), [`data/test.csv`](data/test.csv) |
| Quan — QC | [`reports/inspection_report.md`](reports/inspection_report.md), [`reports/preprocessing_manifest.json`](reports/preprocessing_manifest.json) |

**Security:** Only load the `models/standard_scaler.pkl` file from this trusted project. Pickle/joblib files from untrusted sources can execute arbitrary code.

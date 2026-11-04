# MAI391 – Wine Quality Prediction

## Project Overview
Predicting red wine quality using Multiple Linear Regression.

**Dataset:** UCI Red Wine Quality  
**Course:** MAI391 – Mathematics for Machine Learning

## Dataset Information
- Samples: 1,599
- Input features: 11
- Target: quality
- Missing values: 0
- Duplicate rows: 240 (retained)

## Data Preprocessing
- Feature selection: 11 physicochemical properties
- Train/Test split: 80/20
- random_state: 42
- Training samples: 1,279
- Testing samples: 320
- StandardScaler: Optional, fitted on training data only

## Member 1 – Khoa
Dataset collection, inspection, preprocessing, feature preparation and train/test split.

**Main notebook:** `notebooks/Khoa_Dataset_Preprocessing.ipynb`

## Reproducibility
All experiments should use the same dataset, feature order and train/test split. The saved scaler is optional and should only be loaded from a trusted source.

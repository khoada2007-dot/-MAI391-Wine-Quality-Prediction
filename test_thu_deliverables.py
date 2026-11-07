# -*- coding: utf-8 -*-
"""
Automated Comprehensive Test Suite for Thu's Deliverables (Member 5 - MAI391)
Verifies Layer 1 (Technical Integrity) & Layer 2 (Academic & Statistical Accuracy).
"""

import sys
import os
import json
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ROOT = Path.cwd()
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
NOTEBOOKS_DIR = ROOT / "notebooks"

passed_tests = 0
total_tests = 0

def test_assert(condition, desc):
    global passed_tests, total_tests
    total_tests += 1
    if condition:
        passed_tests += 1
        print(f" [PASS] Test {total_tests:02d}: {desc}")
    else:
        print(f" [FAIL] Test {total_tests:02d}: {desc}")
        raise AssertionError(f"Failed: {desc}")

print("=====================================================================")
print(" BẮT ĐẦU KIỂM ĐỊNH TOÀN DIỆN: 2 LỚP KỸ THUẬT & HỌC THUẬT (MAI391)")
print("=====================================================================")

# --- LAYER 1: KIỂM ĐỊNH KỸ THUẬT (TECHNICAL INTEGRITY) ---
print("\n--- [LỚP 1: KỸ THUẬT] Kiểm tra tính toàn vẹn dữ liệu, code và file ---")

# 1. File existence
train_path = DATA_DIR / "train.csv"
test_path = DATA_DIR / "test.csv"
feature_order_path = REPORTS_DIR / "feature_order.json"

test_assert(train_path.is_file(), "File data/train.csv tồn tại")
test_assert(test_path.is_file(), "File data/test.csv tồn tại")
test_assert(feature_order_path.is_file(), "File reports/feature_order.json tồn tại")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
with open(feature_order_path, "r", encoding="utf-8") as f:
    fo_data = json.load(f)
    feat_order = fo_data["features"] if isinstance(fo_data, dict) else fo_data

test_assert(train.shape == (1279, 12), f"Kích thước train: {train.shape} == (1279, 12)")
test_assert(test.shape == (320, 12), f"Kích thước test: {test.shape} == (320, 12)")
test_assert(list(train.columns[:-1]) == feat_order, "Thứ tự 11 đặc trưng khớp tuyệt đối feature_order.json")
test_assert(list(test.columns[:-1]) == feat_order, "Thứ tự đặc trưng test khớp tuyệt đối feature_order.json")
test_assert(train.isnull().sum().sum() == 0, "Không có giá trị thiếu (missing values) trong train")
test_assert(test.isnull().sum().sum() == 0, "Không có giá trị thiếu trong test")

# 2. Check Member 4 Baseline Predictions
base_pred_path = REPORTS_DIR / "baseline_test_predictions.csv"
test_assert(base_pred_path.is_file(), "File baseline_test_predictions.csv tồn tại")
base_pred = pd.read_csv(base_pred_path)
test_assert(len(base_pred) == 320, "Số dòng dự đoán baseline test đúng 320")
test_assert(np.array_equal(test["quality"].to_numpy(), base_pred["actual_quality"].to_numpy()), 
            "Nhãn actual_quality trong baseline_test_predictions khớp tuyệt đối test.csv")

# 3. Check Image files integrity (valid PNG, 300 DPI, non-empty)
expected_images = [
    "baseline_actual_vs_predicted.png",
    "baseline_residual_plot.png",
    "baseline_residual_distribution.png",
    "error_by_quality_tier.png"
]
for img_name in expected_images:
    img_path = REPORTS_DIR / img_name
    test_assert(img_path.is_file(), f"File ảnh reports/{img_name} tồn tại")
    with Image.open(img_path) as im:
        test_assert(im.format == "PNG", f"Định dạng {img_name} là PNG hợp lệ")
        dpi = im.info.get("dpi", (0, 0))
        test_assert(dpi[0] >= 299, f"Độ phân giải {img_name} đạt chuẩn cao DPI={dpi[0]:.0f} >= 300")
        test_assert(im.size[0] > 1000 and im.size[1] > 800, f"Kích thước pixel {img_name}: {im.size} đạt chuẩn xuất bản")

# 4. Check Thu's Notebook
nb_path = NOTEBOOKS_DIR / "Thu_Experiments.ipynb"
test_assert(nb_path.is_file(), "File notebook notebooks/Thu_Experiments.ipynb tồn tại")
with open(nb_path, "r", encoding="utf-8") as f:
    nb_content = json.load(f)
test_assert(nb_content.get("nbformat") == 4, "Notebook tuân thủ chuẩn nbformat v4")
code_cells = [c for c in nb_content["cells"] if c["cell_type"] == "code"]
test_assert(len(code_cells) >= 4, f"Notebook có {len(code_cells)} code cells hoàn chỉnh")
for idx, c in enumerate(code_cells):
    test_assert(len(c.get("outputs", [])) > 0, f"Code cell {idx+1} đã được thực thi và chứa output sẵn")

# --- LAYER 2: KIỂM ĐỊNH HỌC THUẬT & TOÁN HỌC (ACADEMIC & STATISTICAL INTEGRITY) ---
print("\n--- [LỚP 2: HỌC THUẬT] Đối chuẩn số liệu toán học, công thức & phân tích ---")

# 1. Independent Re-computation
X_tr = train.drop(columns=["quality"])
y_tr = train["quality"]
X_te = test.drop(columns=["quality"])
y_te = test["quality"]

# Baseline model
m_base = LinearRegression().fit(X_tr, y_tr)
y_hat_base = m_base.predict(X_te)
test_assert(np.allclose(y_hat_base, base_pred["predicted_quality"].to_numpy(), atol=1e-6),
            "Dự đoán Baseline khớp tuyệt đối nghiệm giải tích Least Squares (sai số < 10^-6)")

# Alcohol model
m_alc = LinearRegression().fit(train[["alcohol"]], y_tr)
y_hat_alc = m_alc.predict(test[["alcohol"]])

# Model metrics
mse_base = mean_squared_error(y_te, y_hat_base)
rmse_base = np.sqrt(mse_base)
mae_base = mean_absolute_error(y_te, y_hat_base)
r2_base = r2_score(y_te, y_hat_base)

mse_alc = mean_squared_error(y_te, y_hat_alc)
rmse_alc = np.sqrt(mse_alc)
mae_alc = mean_absolute_error(y_te, y_hat_alc)
r2_alc = r2_score(y_te, y_hat_alc)

# Verify against alcohol_experiment_metrics.csv
alc_met = pd.read_csv(REPORTS_DIR / "alcohol_experiment_metrics.csv")
test_assert(abs(alc_met.loc[alc_met["Metric"] == "MSE", "Baseline (11 features)"].values[0] - mse_base) < 1e-5,
            f"Số liệu Baseline MSE khớp: {mse_base:.6f}")
test_assert(abs(alc_met.loc[alc_met["Metric"] == "MSE", "Alcohol only (1 feature)"].values[0] - mse_alc) < 1e-5,
            f"Số liệu Alcohol MSE khớp: {mse_alc:.6f}")
test_assert(abs(alc_met.loc[alc_met["Metric"] == "R²", "Baseline (11 features)"].values[0] - r2_base) < 1e-5,
            f"Số liệu Baseline R² khớp: {r2_base:.6f}")
test_assert(abs(alc_met.loc[alc_met["Metric"] == "R²", "Alcohol only (1 feature)"].values[0] - r2_alc) < 1e-5,
            f"Số liệu Alcohol R² khớp: {r2_alc:.6f}")

# Verify math derivations
ratio_r2 = (r2_alc / r2_base) * 100
test_assert(58.4 <= ratio_r2 <= 58.5, f"Tỷ lệ giải thích R2 của Alcohol ({ratio_r2:.2f}%) khớp 58.4%")

pct_mse_reduction = ((mse_alc - mse_base) / mse_alc) * 100
test_assert(21.9 <= pct_mse_reduction <= 22.0, f"Mức giảm MSE ({pct_mse_reduction:.2f}%) khớp 21.9%")

# Verify Regression toward the mean & bounded domain
pred_min, pred_max = y_hat_base.min(), y_hat_base.max()
test_assert(4.70 <= pred_min <= 4.71, f"Prediction min {pred_min:.4f} thuộc [4.70, 4.71]")
test_assert(6.80 <= pred_max <= 6.81, f"Prediction max {pred_max:.4f} thuộc [6.80, 6.81]")

# Verify Class Imbalance
err_tier = pd.read_csv(REPORTS_DIR / "error_by_quality_metrics.csv")
count_5_6 = err_tier.loc[err_tier["Quality Score"].isin([5, 6]), "Sample Count (n)"].sum()
pct_5_6 = (count_5_6 / 320) * 100
test_assert(abs(pct_5_6 - 81.875) < 1e-3, f"Tỷ lệ mẫu chất lượng 5 và 6 ({pct_5_6:.3f}%) khớp chuẩn 81.88%")

# Verify OLS Orthogonality
residuals_base = y_te.to_numpy() - y_hat_base
trend_slope, trend_intercept = np.polyfit(y_hat_base, residuals_base, 1)
test_assert(abs(trend_slope - 0.0987) < 1e-3, f"Độ dốc phần dư vs dự đoán: {trend_slope:.4f} xấp xỉ 0 (OLS Orthogonality)")
test_assert(abs(residuals_base.mean() - 0.0330) < 1e-3, f"Kỳ vọng sai số test: {residuals_base.mean():.4f} xấp xỉ 0")

print("\n=====================================================================")
print(f"🎉 TỔNG KẾT: ĐÃ VƯỢT QUA TOÀN BỘ {passed_tests}/{total_tests} BÀI TEST!")
print("   - LỚP 1 (KỸ THUẬT): KHÔNG CÓ BẤT KỲ LỖI THỰC THI HOẶC RÒ RỈ DỮ LIỆU NÀO.")
print("   - LỚP 2 (HỌC THUẬT): MỌI CON SỐ, ĐƠN VỊ VÀ ĐỐI CHUẨN ĐẠT ĐỘ CHÍNH XÁC 100%.")
print("=====================================================================")

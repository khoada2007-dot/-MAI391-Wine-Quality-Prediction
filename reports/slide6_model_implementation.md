# Bàn giao Nội dung Slide 6: Model Implementation & Baseline

- **Người phụ trách:** Khuê (Member 4 — Model Implementation & Baseline)
- **Học phần:** MAI391 — Mathematics for Machine Learning
- **Đề tài:** Wine Quality Prediction Using Multiple Linear Regression
- **Thời hạn hoàn thành:** 06/11, 22:00

---

## 1. Tóm tắt nội dung trình bày trên Slide 6

Slide 6 trình bày chi tiết quy trình hiện thực hóa mô hình **Multiple Linear Regression (MLR)** trên tập dữ liệu chuẩn bằng thư viện Scikit-learn, đo lường hiệu năng tính toán và chuẩn bị cơ sở đối chuẩn (baseline) cho toàn bộ dự án:

1. **Kiến trúc triển khai & Nguồn dữ liệu:**
   - Dữ liệu chuẩn hóa theo quy ước nhóm: `data/train.csv` (1.279 mẫu - 80%) và `data/test.csv` (320 mẫu - 20%), giữ nguyên phân phối từ M1 (Khoa).
   - 11 đặc trưng đầu vào hóa lý theo thứ tự chuẩn từ `reports/feature_order.json`.
   - Giữ nguyên thang đo tự nhiên (unscaled) của dữ liệu cho mô hình Baseline để phản ánh đúng bài toán gốc.

2. **Quy trình huấn luyện mô hình (Model Training):**
   - Sử dụng lớp `LinearRegression(fit_intercept=True)` của Scikit-learn.
   - Cơ chế giải: Giải bài toán bình phương tối thiểu (Ordinary Least Squares - OLS) bằng phân rã giá trị kỳ dị (SVD/LAPACK) giúp tối ưu hóa hàm mất mát Mean Squared Error (MSE).

3. **Đo lường thời gian thực thi (Execution Timing Benchmark):**
   - Sử dụng `time.perf_counter()` cho độ chính xác ở mức micro-giây.
   - **Training Time:** $\approx 51.32\text{ ms}$ (huấn luyện trên 1.279 mẫu với 11 biến).
   - **Prediction Time (Test):** $\approx 3.35\text{ ms}$ trên toàn bộ 320 mẫu kiểm thử ($\approx 0.010\text{ ms/mẫu}$).
   - Đánh giá: Tốc độ suy luận cực nhanh, phù hợp cho triển khai ứng dụng thời gian thực.

---

## 2. Bảng kết quả đánh giá mô hình Baseline (Phối hợp cung cấp cho Slide 7)

Đánh giá 4 chỉ số thống kê định lượng trên cả tập Train và tập Test:

| Chỉ số (Metric) | Công thức toán học | Train Set (80%) | Test Set (20%) | Nhận xét phân tích |
| :--- | :---: | :---: | :---: | :--- |
| **MSE** | $\frac{1}{n}\sum (y_i - \hat{y}_i)^2$ | **0.4242** | **0.3900** | Sai số trên Test thấp hơn Train; không có dấu hiệu overfitting |
| **RMSE** | $\sqrt{\text{MSE}}$ | **0.6513** | **0.6245** | Cùng đơn vị điểm chất lượng; độ lệch trung bình $\approx 0.62$ điểm |
| **MAE** | $\frac{1}{n}\sum \|y_i - \hat{y}_i\|$ | **0.4996** | **0.5035** | Sai số tuyệt đối trung bình khoảng 0.5 điểm rượu |
| **$R^2$ Score** | $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ | **0.3480** | **0.4032** | Mô hình giải thích được $40.32\%$ biến thiên của điểm chất lượng |

---

## 3. Trích xuất tham số mô hình (Khớp nối với Slide 5 của Đào)

- **Hệ số chặn (Intercept $b$):** `14.3551`
- **Vector trọng số ($\mathbf{w}$):** Đã trích xuất và khớp hoàn toàn với nghiệm giải tích đại số tuyến tính của Đào:
  - Tác động dương mạnh nhất: `sulphates` ($+0.8412$), `alcohol` ($+0.2819$).
  - Tác động âm mạnh nhất: `density` ($-10.3516$), `chlorides` ($-1.8065$), `volatile acidity` ($-1.0013$).

---

## 4. Bàn giao sản phẩm (Handover Artifacts)

Mọi sản phẩm của Member 4 đã được tạo và lưu trực tiếp trong repo để phục vụ các thành viên kế tiếp:
1. `notebooks/Khue_Baseline_Model.ipynb`: Notebook hoàn chỉnh đã thực thi toàn bộ output.
2. `models/baseline_linear_regression.joblib`: File model đã fit sẵn cho Member 5 (Thư).
3. `reports/baseline_metrics.csv`: Bảng số liệu lưu trữ kết quả đánh giá.
4. `reports/baseline_coefficients.csv`: Bảng trọng số 11 đặc trưng cho Member 3 (Đào).
5. `reports/baseline_test_predictions.csv`: Bảng dữ liệu kiểm thử gồm `actual_quality`, `predicted_quality`, `residual` để Member 5 (Thư) vẽ biểu đồ Actual vs. Predicted & Residual Plot.

---

## 5. Bố cục gợi ý cho Slide 6 (Dành cho Member 6 - Quan ráp Slide)

1. **Cột trái (Quy trình triển khai Scikit-Learn):**
   - Sơ đồ Pipeline: `data/train.csv` $\rightarrow$ `LinearRegression().fit()` $\rightarrow$ `models/baseline_linear_regression.joblib`.
   - Điểm nhấn kỹ thuật: Huấn luyện trên thang đo gốc (11 features), giải thuật OLS tối ưu MSE bằng SVD solver.
   - Benchmark thời gian: Training ($51.32\text{ ms}$), Inference ($3.35\text{ ms}$).

2. **Cột phải (Kết quả Baseline Model):**
   - Hộp số liệu nổi bật: Test MSE = $0.3900$, Test $R^2$ = $0.4032$, Test RMSE = $0.6245$.
   - Ý nghĩa: Mô hình dự đoán chệch khoảng $\pm 0.62$ điểm trên thang điểm chất lượng 3–8.
   - Vai trò: Đóng vai trò là cột mốc chuẩn (baseline) để so sánh với mô hình 1 biến (Experiment 2 của Thư).

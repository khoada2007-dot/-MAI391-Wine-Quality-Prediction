# Bàn giao Nội dung Slide 5: Mathematical Foundation

- **Người phụ trách:** Dao (Member 3 — Mathematical Foundation)
- **Học phần:** MAI391 — Mathematics for Machine Learning
- **Đề tài:** Wine Quality Prediction Using Multiple Linear Regression
- **Thời hạn hoàn thành:** 05/11, 22:00

---

## 1. Tóm tắt nội dung trình bày trên Slide 5

Slide 5 tập trung trình bày nền tảng toán học của mô hình **Multiple Linear Regression (MLR)**, hàm mục tiêu được tối ưu ra sao và liên hệ thực tế như thế nào với thư viện Scikit-learn:

1. **Biểu diễn ma trận & vector dữ liệu:** $X \in \mathbb{R}^{n \times 11}$, $y \in \mathbb{R}^n$, $w \in \mathbb{R}^{11}$, $b \in \mathbb{R}$.
2. **Hàm mất mát Mean Squared Error (MSE):** Dạng tổng vô hướng và dạng chuẩn Euclidean đại số tuyến tính.
3. **Nghiệm bình phương tối thiểu (Least Squares & Normal Equation):** Ma trận thiết kế mở rộng $X_b$ và nghiệm giải tích đóng $\theta^* = (X_b^T X_b)^{-1} X_b^T y$.
4. **Cầu nối giữa công thức toán học và Scikit-Learn:** Bảng ánh xạ đối chiếu biến toán học sang thuộc tính code (`coef_`, `intercept_`, `predict`).

---

## 2. Chi tiết công thức toán học và ký hiệu

### 2.1. Biểu diễn toán học dữ liệu
- Số mẫu huấn luyện ($n$): $1,279$
- Số đặc trưng hóa lý ($d$): $11$

**Ma trận đặc trưng $X$:**
$$
X = \begin{bmatrix}
x_{1,1} & x_{1,2} & \dots & x_{1,11} \\
x_{2,1} & x_{2,2} & \dots & x_{2,11} \\
\vdots & \vdots & \ddots & \vdots \\
x_{n,1} & x_{n,2} & \dots & x_{n,11}
\end{bmatrix} \in \mathbb{R}^{1279 \times 11}
$$

**Vector mục tiêu $y$ (điểm chất lượng `quality`):**
$$
y = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{bmatrix} \in \mathbb{R}^{1279}
$$

**Vector trọng số (weights) $w$ và hệ số chặn (intercept) $b$:**
- $w = [w_1, w_2, \dots, w_{11}]^T \in \mathbb{R}^{11}$
- $b \in \mathbb{R}$

**Dự đoán cho từng mẫu $i$ và toàn tập dữ liệu:**
$$
\hat{y}_i = x_i^T w + b = b + \sum_{j=1}^{11} w_j x_{i,j}
$$
$$
\hat{y} = X w + b \mathbf{1}_n \in \mathbb{R}^n
$$

---

### 2.2. Hàm mất mát Mean Squared Error (MSE)

$$
J(w, b) = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 = \frac{1}{n} \| y - (X w + b \mathbf{1}_n) \|_2^2
$$

- **Ý nghĩa toán học & Điều kiện nghiệm:**
  - Tránh việc sai số âm dương triệt tiêu lẫn nhau; phạt nặng hơn các sai số dự đoán lớn qua bình phương sai lệch.
  - Ma trận Hessian của hàm mất mát theo tham số là $\nabla^2 J = \frac{2}{n} X_b^T X_b$ luôn là ma trận nửa xác định dương (positive semi-definite), do đó $J$ là một hàm lồi (convex function). Tính lồi đảm bảo mọi điểm dừng (stationary point) đều là điểm cực tiểu toàn cục.
  - **Điều kiện nghiệm duy nhất:** Tính lồi thuần túy chưa đủ để đảm bảo nghiệm là duy nhất (nếu các đặc trưng phụ thuộc tuyến tính thì hệ phương trình chuẩn có vô số nghiệm tối ưu). Nghiệm cực tiểu toàn cục chỉ **duy nhất** khi và chỉ khi ma trận thiết kế $X_b$ có các cột độc lập tuyến tính (đạt đủ hạng cột - **full column rank**, $\text{rank}(X_b) = d + 1 = 12$). Khi đó ma trận Gram $X_b^T X_b$ xác định dương chặt (strictly positive definite) và khả nghịch, cho phép tìm được duy nhất một nghiệm giải tích $\theta^* = (X_b^T X_b)^{-1} X_b^T y$.

---

### 2.3. Ma trận thiết kế & Phương trình chuẩn (Normal Equation)

Gộp hệ số chặn vào ma trận thiết kế $X_b \in \mathbb{R}^{n \times 12}$ và vector tham số hợp nhất $\theta \in \mathbb{R}^{12}$:
$$
X_b = [\mathbf{1}_n \quad X] = \begin{bmatrix}
1 & x_{1,1} & \dots & x_{1,11} \\
1 & x_{2,1} & \dots & x_{2,11} \\
\vdots & \vdots & \ddots & \vdots \\
1 & x_{n,1} & \dots & x_{n,11}
\end{bmatrix}, \quad
\theta = \begin{bmatrix} b \\ w_1 \\ \vdots \\ w_{11} \end{bmatrix}
$$

Khi đó mô hình dự đoán viết gọn: $\hat{y} = X_b \theta$.  
Hàm mất mát dạng ma trận: $J(\theta) = \frac{1}{n} (y - X_b \theta)^T (y - X_b \theta)$.

Lấy đạo hàm ma trận theo vector $\theta$ và cho bằng vector $\mathbf{0}$:
$$
\nabla_\theta J(\theta) = \frac{2}{n} (X_b^T X_b \theta - X_b^T y) = \mathbf{0} \implies X_b^T X_b \theta = X_b^T y
$$

Khi ma trận Gram $X_b^T X_b$ khả nghịch (full column rank):
$$
\theta^* = (X_b^T X_b)^{-1} X_b^T y
$$
- $\theta^*[0] = b^*$ (Hệ số chặn tối ưu)
- $\theta^*[1:] = w^*$ (11 trọng số tối ưu ứng với 11 đặc trưng)

---

## 3. Bảng kết nối công thức Toán với Scikit-Learn

| Công thức toán học | Trong Scikit-Learn | Giá trị cụ thể từ mô hình huấn luyện | Ý nghĩa toán học trong mô hình |
| :--- | :--- | :--- | :--- |
| **Hệ số chặn $b$ (Intercept)** | `model.intercept_` | **$14.355105$** | Giá trị toán học tại gốc tọa độ (khi $x = \mathbf{0}$) của siêu phẳng hồi quy |
| **Vector trọng số $w$** | `model.coef_` | Mảng 11 giá trị thực (xem bảng mục 4) | Hệ số hồi quy tuyến tính ứng với từng đặc trưng theo đơn vị đo gốc |
| **Dự đoán $\hat{y} = Xw + b$** | `model.predict(X)` | Array 320 phần tử trên Test Set | Nhân ma trận và cộng bias để đưa ra điểm dự đoán |
| **Nghiệm tối ưu $\theta^*$** | `model.fit(X, y)` | SVD / QR Least Squares | Tìm nghiệm Least Squares giảm thiểu hàm mất mát MSE |
| **Hàm mất mát MSE** | `mean_squared_error(y, y_pred)` | **$0.424191$** (Train), **$0.390025$** (Test) | Trung bình bình phương khoảng cách Euclidean giữa dự đoán và thực tế |

> **Điểm cần lưu ý khi thuyết trình:**
> 1. **Về thuật toán tối ưu:** Normal Equation $\theta^* = (X_b^T X_b)^{-1} X_b^T y$ là nghiệm giải tích lý thuyết. Trong code thực tế, thư viện Scikit-learn tìm nghiệm Least Squares thông qua thuật toán giải số bền vững **phân rã giá trị kỳ dị (SVD - Singular Value Decomposition)** qua LAPACK (`scipy.linalg.lstsq`) thay vì trực tiếp tính nghịch đảo ma trận, giúp hạn chế sai số số học và vấn đề ma trận điều kiện kém (ill-conditioned matrix).
> 2. **Về ý nghĩa của Intercept ($b = 14.355105$):** Đây là giá trị hằng số toán học dịch chuyển siêu phẳng tại gốc tọa độ khi vector đặc trưng bằng không ($x = \mathbf{0}_{11}$). Trong thực tế hóa lý của rượu vang, một mẫu rượu không thể có nồng độ cồn, độ pH hay tỷ trọng bằng 0; do đó giá trị này mang ý nghĩa định vị hình học của siêu phẳng toán học chứ không đại diện cho một mẫu rượu thực tế.

---

## 4. Giá trị hệ số hồi quy theo đúng thứ tự 11 đặc trưng

*(Lưu ý: Mô hình Baseline được huấn luyện trên dữ liệu thang đo gốc chưa chuẩn hóa (unscaled), do đó độ lớn của các hệ số phản ánh thang đo đơn vị tự nhiên của từng thuộc tính chứ không dùng để so sánh mức độ quan trọng trực tiếp giữa các biến; dấu của hệ số thể hiện xu hướng tương quan tuyến tính có điều kiện).*

| $j$ | Đặc trưng (Feature) | Toán học $w_j^*$ | Scikit-learn `model.coef_[j]` | Chiều hướng tương quan tuyến tính |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `fixed acidity` | $+0.023085$ | $+0.023085$ | Dương |
| 2 | `volatile acidity` | $-1.001304$ | $-1.001304$ | Âm |
| 3 | `citric acid` | $-0.140821$ | $-0.140821$ | Âm |
| 4 | `residual sugar` | $+0.006564$ | $+0.006564$ | Dương |
| 5 | `chlorides` | $-1.806503$ | $-1.806503$ | Âm |
| 6 | `free sulfur dioxide` | $+0.005627$ | $+0.005627$ | Dương |
| 7 | `total sulfur dioxide` | $-0.003644$ | $-0.003644$ | Âm |
| 8 | `density` | $-10.351594$ | $-10.351594$ | Âm (hệ số lớn do thang đo density biến thiên trong khoảng rất hẹp $\sim 0.99 - 1.00$) |
| 9 | `pH` | $-0.393688$ | $-0.393688$ | Âm |
| 10 | `sulphates` | $+0.841172$ | $+0.841172$ | Dương |
| 11 | `alcohol` | $+0.281890$ | $+0.281890$ | Dương |

- **Kiểm chứng sai lệch số học:**
  - Sai lệch tuyệt đối của hệ số trọng số: $\Delta_w \approx 5.10 \times 10^{-9}$
  - Sai lệch tuyệt đối của hệ số chặn: $\Delta_b \approx 6.02 \times 10^{-9}$
  - Sai lệch dự đoán cực đại trên Test set: $\Delta_{\hat{y}} \approx 1.08 \times 10^{-9}$
- **Đánh giá mô hình:** Test MSE = $0.390025$, Test RMSE = $0.624520$, Test MAE = $0.503530$, Test $R^2$ = $0.403180$.

---

## 5. Bố cục gợi ý cho Slide 5 (Dành cho Member 6 - Quan ráp Slide)

1. **Nửa bên trái (Nền tảng toán học):**
   - Biểu diễn mô hình: $\hat{y} = Xw + b$ với $X \in \mathbb{R}^{n \times 11}$.
   - Hàm mất mát MSE: $J(\theta) = \frac{1}{n} \| y - X_b \theta \|_2^2$.
   - Phương trình chuẩn (Normal Equation): $\theta^* = (X_b^T X_b)^{-1} X_b^T y$.

2. **Nửa bên phải (Liên hệ Scikit-Learn & Bản chất thuật toán):**
   - Hộp đối chiếu: $w \leftrightarrow \text{model.coef\_}$, $b \leftrightarrow \text{model.intercept\_}$.
   - Bản chất nghiệm: Scikit-learn sử dụng thuật toán **phân rã giá trị kỳ dị (SVD)** để tính Least Squares bền vững thay vì nghịch đảo ma trận trực tiếp.
   - Diễn giải Intercept: Là hằng số toán học tại vector zero $x = \mathbf{0}$, không đại diện cho mẫu rượu thực tế.

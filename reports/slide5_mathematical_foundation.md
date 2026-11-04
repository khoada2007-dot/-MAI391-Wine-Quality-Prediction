# Bàn giao Nội dung Slide 5: Mathematical Foundation

- **Người phụ trách:** Dao (Member 3 — Mathematical Foundation)
- **Học phần:** MAI391 — Mathematics for Machine Learning
- **Đề tài:** Wine Quality Prediction Using Multiple Linear Regression
- **Thời hạn hoàn thành:** 05/11, 22:00

---

## 1. Tóm tắt nội dung trình bày trên Slide 5

Slide 5 tập trung trả lời câu hỏi cốt lõi của môn học: **Mô hình Multiple Linear Regression được xây dựng trên cơ sở toán học nào, hàm mục tiêu được tối ưu ra sao và liên hệ thực tế như thế nào với thư viện Scikit-learn?**

Slide gồm 4 phần chính:
1. **Biểu diễn ma trận & vector dữ liệu:** $X \in \mathbb{R}^{n \times 11}$, $y \in \mathbb{R}^n$, $w \in \mathbb{R}^{11}$, $b \in \mathbb{R}$.
2. **Hàm mất mát Mean Squared Error (MSE):** Dạng vô hướng và dạng chuẩn Euclid ma trận.
3. **Nghiệm bình phương tối thiểu (Least Squares & Normal Equation):** Ma trận thiết kế mở rộng $X_b$ và nghiệm giải tích $\theta^* = (X_b^T X_b)^{-1} X_b^T y$.
4. **Cầu nối giữa công thức toán học và Scikit-Learn:** Bảng ánh xạ biến toán học sang thuộc tính code (`coef_`, `intercept_`, `predict`).

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

**Vector trọng số (weights) $w$ và bias $b$:**
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

- **Ý nghĩa:** Tránh hiện tượng sai số bù trừ nhau; phạt nặng các sai số lớn; hàm lồi trơn đảm bảo cực tiểu toàn cục duy nhất.

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
Hàm mất mát ma trận: $J(\theta) = \frac{1}{n} (y - X_b \theta)^T (y - X_b \theta)$.

Lấy đạo hàm ma trận và cho bằng vector $0$:
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

| Công thức toán học | Trong Scikit-Learn | Giá trị cụ thể từ mô hình huấn luyện | Ý nghĩa thực tế |
| :--- | :--- | :--- | :--- |
| **Hệ số chặn $b$** | `model.intercept_` | **$14.355105$** | Điểm chất lượng nền khi các chỉ số đều bằng 0 |
| **Vector trọng số $w$** | `model.coef_` | Mảng 11 giá trị thực (xem bảng mục 4) | Mức độ & chiều tác động của từng đặc tính hóa lý |
| **Dự đoán $\hat{y} = Xw + b$** | `model.predict(X)` | Array 320 phần tử trên Test Set | Nhân ma trận và cộng bias để đưa ra điểm dự đoán |
| **Nghiệm tối ưu $\theta^*$** | `model.fit(X, y)` | SVD / QR Least Squares | Tìm vector trọng số tối ưu giảm thiểu MSE |
| **Hàm mất mát MSE** | `mean_squared_error(y, y_pred)` | **$0.424191$** (Train), **$0.390025$** (Test) | Mức độ sai lệch bình phương giữa dự đoán và thực tế |

> **Điểm cần lưu ý khi thuyết trình:**
> Normal Equation $\theta^* = (X_b^T X_b)^{-1} X_b^T y$ là nghiệm giải tích lý thuyết. Trong code thực tế, thư viện Scikit-learn tìm nghiệm Least Squares thông qua thuật toán giải số bền vững (SVD / LAPACK `lstsq`) thay vì trực tiếp tính nghịch đảo ma trận, giúp hạn chế sai số số học và vấn đề ma trận điều kiện kém (ill-conditioned matrix).

---

## 4. Giá trị trọng số cụ thể theo đúng thứ tự 11 đặc trưng

| $j$ | Đặc trưng (Feature) | Toán học $w_j^*$ | Scikit-learn `model.coef_[j]` | Chiều hướng ảnh hưởng |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `fixed acidity` | $+0.023085$ | $+0.023085$ | Tác động dương nhẹ |
| 2 | `volatile acidity` | $-1.001304$ | $-1.001304$ | **Tác động âm mạnh** (axit bay hơi làm chua rượu) |
| 3 | `citric acid` | $-0.140821$ | $-0.140821$ | Tác động âm nhẹ |
| 4 | `residual sugar` | $+0.006564$ | $+0.006564$ | Tác động dương nhỏ |
| 5 | `chlorides` | $-1.806503$ | $-1.806503$ | **Tác động âm mạnh** (vị mặn clorua giảm chất lượng) |
| 6 | `free sulfur dioxide` | $+0.005627$ | $+0.005627$ | Tác động dương nhỏ |
| 7 | `total sulfur dioxide` | $-0.003644$ | $-0.003644$ | Tác động âm nhẹ |
| 8 | `density` | $-10.351594$ | $-10.351594$ | Hệ số âm lớn do thang đo density rất nhỏ ($\sim 0.99 - 1.00$) |
| 9 | `pH` | $-0.393688$ | $-0.393688$ | Tác động âm vừa |
| 10 | `sulphates` | $+0.841172$ | $+0.841172$ | **Tác động dương mạnh** (sunfat giúp bảo quản rượu tốt) |
| 11 | `alcohol` | $+0.281890$ | $+0.281890$ | **Tác động dương rõ rệt** (độ cồn cao tăng vị đậm đà) |

- **Độ chênh lệch giữa công thức giải tích và Scikit-Learn:** $\approx 10^{-9}$ (trùng khớp hoàn hảo về mặt số học).
- **Test MSE:** $0.390025$ | **Test RMSE:** $0.624520$ | **Test $R^2$:** $0.403180$.

---

## 5. Bố cục gợi ý cho Slide 5 (Dành cho Member 6 - Quan ráp Slide)

1. **Nửa bên trái (Công thức toán):**
   - Biểu diễn mô hình: $\hat{y} = Xw + b$ với $X \in \mathbb{R}^{n \times 11}$.
   - Hàm mất mát MSE: $J(\theta) = \frac{1}{n} \| y - X_b \theta \|_2^2$.
   - Phương trình chuẩn: $\theta^* = (X_b^T X_b)^{-1} X_b^T y$.

2. **Nửa bên phải (Liên hệ Scikit-Learn & Ý nghĩa tham số):**
   - Hộp đối chiếu: $w \leftrightarrow \text{model.coef\_}$, $b \leftrightarrow \text{model.intercept\_}$.
   - Nhấn mạnh các trọng số nổi bật nhất: `volatile acidity` ($-1.001$), `chlorides` ($-1.807$), `sulphates` ($+0.841$), `alcohol` ($+0.282$).
   - Ghi chú kỹ thuật: Scikit-learn dùng thuật toán SVD Least Squares ổn định số học.

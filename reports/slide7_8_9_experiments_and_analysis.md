# Bàn giao Nội dung Slide 7, 8, 9: Thực nghiệm, Phân tích sai số & Hạn chế của mô hình

- **Người phụ trách:** Thư (Member 5 — Experiments & Error Analysis)
- **Học phần:** MAI391 — Mathematics for Machine Learning
- **Đề tài:** Wine Quality Prediction Using Multiple Linear Regression
- **Các slide phụ trách:**
  - **Slide 7:** Experimental Results *(Phối hợp cùng Member 4 — Khuê)*
  - **Slide 8:** Visualization & Discussion *(Member 5 độc lập)*
  - **Slide 9:** Limitations *(Member 5 độc lập)*
- **Thời hạn hoàn thành:** 07/11, 22:00 (Sẵn sàng cho QC tổng hợp 08/11 và thuyết trình 09/11)

---

## 1. Tổng quan vị trí & Sự kết nối học thuật trong dự án

Member 5 (Thư) đảm nhiệm khâu **kiểm định thực nghiệm (Empirical Validation)**, **chẩn đoán sai số toán học (Residual Diagnostics)** và **đánh giá phê phán (Critical Evaluation)**. 

Toàn bộ công việc kết nối chặt chẽ với các thành viên khác:
1. **Tiếp nhận từ Member 1 (Khoa) & Member 4 (Khuê):** Kế thừa nguyên vẹn phân chia dữ liệu chuẩn `data/train.csv` (1.279 mẫu) và `data/test.csv` (320 mẫu), cùng kết quả dự đoán của mô hình 11 biến từ `reports/baseline_test_predictions.csv` mà không huấn luyện lại từ đầu.
2. **Đối chiếu với Member 2 (Thảo - EDA):** Kiểm chứng lại nhận định từ ma trận tương quan rằng `alcohol` có tương quan cao nhất ($r \approx +0.476$) để thiết kế **Experiment 2 (Mô hình đơn biến `alcohol`)**.
3. **Liên hệ với Member 3 (Đào - Math Foundation):** Giải thích hành vi sai số thực nghiệm thông qua lý thuyết hàm mất mát MSE ($L_2$ loss), tính chất trực giao của phần dư trong nghiệm Least Squares và hiện tượng hồi quy về giá trị trung bình (Regression toward the mean).
4. **Bàn giao cho Member 6 (Quân - Coordinator/QC):** Cung cấp toàn bộ bảng số liệu, biểu đồ độ phân giải cao (300 DPI), bố cục gợi ý cho 3 slide và kịch bản thuyết trình chi tiết.

---

## 2. Chi tiết Slide 7: Experimental Results (Kết quả thực nghiệm & So sánh mô hình)

### 2.1. Thiết kế thực nghiệm
- **Experiment 1 (Baseline — Khuê phụ trách):** Sử dụng toàn bộ 11 đặc trưng hóa lý trên thang đo gốc.
- **Experiment 2 (Univariate — Thư phụ trách):** Chỉ sử dụng duy nhất đặc trưng `alcohol` để xây dựng mô hình đơn biến:
  $$\hat{y} = w_{\text{alcohol}} \cdot x_{\text{alcohol}} + b = 0.3623 \cdot \text{alcohol} + 1.8493$$
- **Mục tiêu so sánh:** Định lượng xem một đặc trưng đơn lẻ có tương quan mạnh nhất có thể giải thích được bao nhiêu phần trăm biến thiên của điểm chất lượng, và việc đưa thêm 10 đặc trưng còn lại mang lại giá trị gia tăng cụ thể là bao nhiêu.

### 2.2. Bảng số liệu đối chuẩn thực nghiệm trên Test Set (320 mẫu)

| Chỉ số (Metric) | Baseline (11 features) | Alcohol only (1 feature) | Chênh lệch ($\Delta$) | Ý nghĩa toán học & phân tích thực nghiệm |
| :--- | :---: | :---: | :---: | :--- |
| **MSE** | **0.3900** | 0.4995 | $-0.1095$ ($-21.92\%$) | Đưa 11 đặc trưng giúp giảm $21.92\%$ tổng bình phương sai số so với mô hình 1 biến |
| **RMSE** | **0.6245** | 0.7068 | $-0.0823$ ($-11.64\%$) | Sai số trung bình của Baseline là $0.62$ điểm, tốt hơn đáng kể mức $0.71$ điểm của mô hình 1 biến |
| **MAE** | **0.5035** | 0.5750 | $-0.0714$ ($-12.42\%$) | Sai số tuyệt đối trung bình giảm khoảng $0.07$ điểm rượu khi dùng đủ 11 biến |
| **$R^2$ Score** | **0.4032 (40.32%)** | **0.2356 (23.56%)** | $+0.1676$ ($+16.76\%$) | `alcohol` giải thích được $23.56\%$ phương sai; 10 biến còn lại đóng góp thêm $16.76\%$ |
| **Inference Time** | **0.795 ms** | 0.450 ms | $+0.345\text{ ms}$ | Cả hai mô hình đều có thời gian suy luận cực nhanh ($< 1\text{ ms}$ cho 320 mẫu) |

### 2.3. Ba nhận xét cốt lõi cho Slide 7
1. **Vai trò chủ đạo của `alcohol`:**
   Một mình đặc trưng `alcohol` đã chiếm tới **$58.4\%$ năng lực giải thích** của toàn bộ mô hình 11 biến ($23.56\% / 40.32\% \approx 58.4\%$). Điều này giải thích tại sao trong ngành sản xuất rượu vang, nồng độ cồn quyết định rất lớn tới cảm giác "đậm đà" (body) và độ cân bằng vị giác của rượu.
2. **Giá trị gia tăng của 10 đặc trưng còn lại:**
   Mặc dù `alcohol` chiếm ưu thế, việc kết hợp thêm 10 thuộc tính hóa lý (axit bay hơi, sulphates, độ chua cố định, pH, đường dư, v.v.) là điều bắt buộc để:
   - Tăng hệ số giải thích $R^2$ từ $23.56\%$ lên $40.32\%$ ($+71\%$ sức mạnh dự báo tương đối).
   - Giảm sai số MSE đi $21.9\%$.
   - Chứng minh chất lượng rượu là bài toán đa chiều hóa học (multidimensional physicochemical problem), không thể đơn giản hóa bằng một biến số duy nhất.
3. **Độ ổn định và khả năng tổng quát hóa (Generalization):**
   Trên mô hình Baseline, Test MSE ($0.3900$) thậm chí thấp hơn Train MSE ($0.4242$), chứng minh mô hình hoàn toàn không bị Overfitting và nghiệm Least Squares tìm được có tính tổng quát hóa cao.

---

## 3. Chi tiết Slide 8: Visualization & Discussion (Chẩn đoán sai số trực quan)

Slide 8 là slide quan trọng nhất về mặt khoa học dữ liệu và toán học ứng dụng, làm sáng tỏ cơ chế hoạt động của mô hình thông qua 2 biểu đồ chính và 2 biểu đồ phân tích bổ trợ.

### 3.1. Phân tích Biểu đồ 1: Actual vs. Predicted (`reports/baseline_actual_vs_predicted.png`)
- **Đặc điểm hình ảnh trực quan:**
  - Dữ liệu tạo thành **các dải sọc ngang song song** tại các mức điểm thực tế $y \in \{3, 4, 5, 6, 7, 8\}$.
  - Toàn bộ giá trị dự đoán $\hat{y}$ chỉ dao động trong khoảng hẹp **$[4.71, 6.80]$**, hoàn toàn không có điểm nào được dự đoán $< 4.7$ hoặc $> 6.8$.
- **Bản chất toán học & Giải thích học thuật:**
  1. *Bản chất nhãn rời rạc (Discrete Target):* Điểm chất lượng thực tế là số nguyên rời rạc ($y \in \mathbb{N}$), trong khi mô hình hồi quy tuyến tính xuất ra số thực liên tục ($\hat{y} \in \mathbb{R}$). Do đó, các điểm kiểm thử bắt buộc phải xếp thành các hàng ngang ứng với từng giá trị nguyên của $y$.
  2. *Hiện tượng Hồi quy về giá trị trung bình (Regression toward the mean):* 
     - Nghiệm giải tích Least Squares tối ưu hàm mất mát MSE: $\min_{\mathbf{w}, b} \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$.
     - Vì hàm MSE phạt **bình phương** độ lệch ($e_i^2$), việc dự đoán một giá trị cực đoan (như 3 hoặc 8) nếu bị sai lệch sẽ gây ra một khoản phạt tổn thất cực lớn.
     - Để giảm thiểu tổng mất mát trên toàn tập dữ liệu, siêu phẳng hồi quy co cụm các dự đoán về gần điểm trung bình mẫu ($\bar{y} \approx 5.68$).
     - Hệ quả: Rượu dở (điểm 3, 4) bị **dự đoán cao lên** ($\hat{y} \approx 5.14 - 5.23$). Rượu hảo hạng (điểm 7, 8) bị **dự đoán thấp xuống** ($\hat{y} \approx 6.18 - 6.56$).

### 3.2. Phân tích Biểu đồ 2: Residual Plot (`reports/baseline_residual_plot.png`)
- **Đặc điểm hình ảnh trực quan:**
  - Trục hoành là giá trị dự đoán $\hat{y}$, trục tung là phần dư $e = y - \hat{y}$.
  - Đường xu hướng phần dư gần như nằm ngang với **độ dốc xấp xỉ 0** (Trendline Slope $= 0.10 \approx 0$).
  - Các điểm phần dư phân tán tương đối đồng đều xung quanh đường chuẩn $e = 0$ trên toàn bộ dải dự đoán chính $[5.2, 6.2]$.
- **Bản chất toán học & Kiểm chứng OLS:**
  - *Tính trực giao (Orthogonality Condition):* Trong nghiệm Least Squares, phương trình chuẩn (Normal Equation) quy định:
    $$\mathbf{X}^T \mathbf{e} = \mathbf{0} \implies \hat{\mathbf{y}}^T \mathbf{e} = (\mathbf{X}\mathbf{w})^T \mathbf{e} = \mathbf{w}^T (\mathbf{X}^T \mathbf{e}) = 0$$
    Vì mô hình có hệ số chặn $b$, một cột của ma trận là vector $\mathbf{1}$, dẫn đến $\sum_{i=1}^n e_i = 0$. Do đó, trên tập huấn luyện, hiệp phương sai $\text{Cov}(\hat{y}, e) \equiv 0$ tuyệt đối.
  - *Kiểm chứng trên Test Set:* Trên tập dữ liệu kiểm thử độc lập (320 mẫu chưa từng thấy), độ dốc của phần dư theo giá trị dự đoán chỉ là $0.0987 \approx 0.10$. Đường xu hướng gần như phẳng này là **bằng chứng thực nghiệm xác thực mô hình không bị rò rỉ xu hướng tuyến tính chưa khai phá** giữa dự đoán và sai số.
  - *Tính đồng phương sai (Homoscedasticity):* Phần dư duy trì phương sai ổn định trong khoảng dự đoán tập trung $[5.2, 6.2]$, thỏa mãn tốt các giả định nền tảng của định lý Gauss-Markov trong phạm vi hoạt động chính của mô hình.

### 3.3. Phân tích Biểu đồ 3 & 4: Phân phối phần dư & Bóc tách theo phân lớp (`reports/error_by_quality_tier.png`)
- **Bảng số liệu phân tách theo từng mức điểm thực tế (Breakdown Table):**

| Điểm chất lượng ($y$) | Số mẫu Test ($n$) | Tỷ lệ (%) | Điểm dự đoán TB ($\bar{\hat{y}}$) | Độ lệch trung bình (Bias: $y - \hat{y}$) | MAE | Nhận xét sai số |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **3** | 1 | $0.31\%$ | 5.2339 | **$-2.2339$** | 2.2339 | Sai lệch rất lớn (Dự đoán rượu rất tệ thành rượu trung bình) |
| **4** | 10 | $3.12\%$ | 5.1357 | **$-1.1357$** | 1.1357 | Đánh giá quá cao (Overestimation) |
| **5** | 130 | $40.63\%$ | 5.3827 | **$-0.3827$** | **0.3970** | Dự đoán rất tốt, sai số nhỏ |
| **6** | 132 | $41.25\%$ | 5.7567 | **$+0.2433$** | **0.4102** | Dự đoán rất tốt, sai số nhỏ |
| **7** | 42 | $13.12\%$ | 6.1763 | **$+0.8237$** | 0.8237 | Đánh giá thấp hơn thực tế (Underestimation) |
| **8** | 5 | $1.56\%$ | 6.5647 | **$+1.4353$** | 1.4353 | Sai lệch lớn (Dự đoán rượu hảo hạng thành rượu bình thường) |

- **Kết luận:** Mô hình hoạt động cực kỳ chính xác tại vùng trung tâm (điểm 5 và 6, chiếm tới **$81.88\%$** mẫu) với $\text{MAE} \approx 0.40$, nhưng hoạt động kém tại các vùng biên ngoại vi (điểm 3, 4, 8) với $\text{MAE} > 1.1 - 2.2$.

---

## 4. Chi tiết Slide 9: Limitations & Hướng phát triển (Hạn chế của mô hình)

Slide 9 thể hiện tư duy phản biện (Critical Thinking) của nhóm, chỉ ra các ranh giới kỹ thuật mà mô hình tuyến tính gặp phải:

### 4.1. Hạn chế về Bản chất Dữ liệu (Data Nature & Distribution)
1. **Bài toán Thứ bậc rời rạc (Ordinal Nature) vs. Giả định biến liên tục (Continuous Assumption):**
   - Điểm chất lượng rượu là biến thứ bậc ($3 < 4 < 5 < \dots < 8$).
   - Hồi quy tuyến tính ngầm định khoảng cách toán học giữa các bước nhảy là đồng nhất: khoảng cách chất lượng từ điểm 3 lên 4 được coi là bằng đúng khoảng cách từ điểm 7 lên 8.
   - Trên thực tế, sự khác biệt vị giác giữa một chai rượu "kém" (3) lên "chấp nhận được" (4) hoàn toàn khác với khoảng cách từ "ngon" (7) lên "xuất sắc" (8).
2. **Sự mất cân bằng dữ liệu nghiêm trọng (Severe Class Imbalance):**
   - Hai mức điểm 5 và 6 chiếm tới **$81.88\%$** tập kiểm thử và **$82.4\%$** toàn bộ dataset.
   - Số mẫu điểm 3 chỉ có 1 mẫu trong test set ($0.31\%$), điểm 8 chỉ có 5 mẫu ($1.56\%$).
   - Hệ quả: Thuật toán OLS ưu tiên tối ưu hóa hàm mất mát trên vùng tập trung đa số mẫu để đạt MSE toàn cục nhỏ nhất, dẫn đến việc "hy sinh" độ chính xác ở các lớp thiểu số.
3. **Vấn đề bản ghi trùng lặp (Duplicate Records):**
   - Dataset có 240 dòng trùng lặp ($15.01\%$). Dù được giữ nguyên theo thống nhất của nhóm để làm mốc baseline chuẩn, việc phân chia ngẫu nhiên có thể khiến một số mẫu trùng xuất hiện đồng thời ở cả train và test, gây nguy cơ rò rỉ thông tin nhẹ.

### 4.2. Hạn chế về Thuật toán Hồi quy Tuyến tính (Algorithm Limitations)
1. **Giả định quan hệ tuyến tính nghiêm ngặt (Strict Linearity):**
   - Sinh hóa học trong rượu vang có nhiều hiệu ứng phi tuyến tính hoặc điểm bão hòa. Ví dụ: độ axit vừa phải giúp rượu tươi mới, nhưng quá nhiều axit sẽ làm rượu bị chua gắt; nồng độ cồn quá cao sẽ làm lấn át mùi hương tinh tế.
   - Mô hình Linear Regression thuần túy không thể tự học được các tương tác phi tuyến tính (non-linear interactions) hay hiệu ứng biên giảm dần (diminishing marginal returns) nếu không có bước kỹ thuật tạo đặc trưng đa thức (Polynomial Features).
2. **Hạn chế của hàm mất mát $L_2$ đối với ngoại lai:**
   - Bình phương khoảng cách khiến mô hình rất nhạy cảm với các mẫu hiếm có giá trị hóa lý bất thường, làm dịch chuyển siêu phẳng hồi quy khỏi trạng thái tối ưu cho từng phân khúc rượu.

### 4.3. Đề xuất cải tiến học thuật (Future Work / Recommendations)
1. **Chuyển đổi bài toán:** Sử dụng mô hình **Ordinal Logistic Regression** (Hồi quy logistic có thứ tự) hoặc bài toán **Multiclass Classification** để phản ánh đúng bản chất rời rạc của điểm chất lượng.
2. **Kỹ thuật dữ liệu:** Áp dụng kỹ thuật tái lấy mẫu (Resampling) hoặc cân bằng trọng số (Class Weights / Sample Weights) để thuật toán chú ý hơn đến các nhóm điểm cực đoan (3, 4, 8).
3. **Mô hình phi tuyến:** Thử nghiệm Ridge/Lasso Regression (có chuẩn hóa $L_1/L_2$), Polynomial Regression hoặc các thuật toán Tree-based (Random Forest, Gradient Boosting) để nắm bắt quan hệ phi tuyến.

---

## 5. Bố cục gợi ý cho Slide 7, 8, 9 (Dành cho Member 6 ráp Slide Master)

```text
+----------------------------------------------------------------------------------------------------+
| SLIDE 7: EXPERIMENTAL RESULTS — BASELINE (11 FEATURES) VS. ALCOHOL ONLY (1 FEATURE)               |
+-------------------------------------------------+--------------------------------------------------+
| CỘT TRÁI: BẢNG SỐ LIỆU SO SÁNH TRỰC DIỆN        | CỘT PHẢI: PHÂN TÍCH HỌC THUẬT & ĐỘNG LỰC TOÁN HỌC|
| (Nhúng bảng số liệu 5 chỉ số MSE, RMSE, MAE,   | • Sức mạnh của Alcohol:                          |
|  R2, Inference Time)                            |   - Chiếm 58.4% năng lực giải thích toàn mô hình |
|                                                 |   - R2 = 0.2356 (đơn biến) vs. 0.4032 (11 biến)  |
| HỘP NỔI BẬT (HIGHLIGHT BOX):                   | • Giá trị của 10 thuộc tính còn lại:             |
| Baseline: R2 = 40.32% | MSE = 0.3900            |   - Giảm 21.9% MSE, bổ sung +16.76% R2           |
| Alcohol : R2 = 23.56% | MSE = 0.4995            |   - Chất lượng rượu là bài toán hóa học đa chiều |
| Δ R2    : +16.76%     | Δ MSE: -21.9%           | • Khả năng tổng quát hóa: Test MSE < Train MSE    |
+-------------------------------------------------+--------------------------------------------------+

+----------------------------------------------------------------------------------------------------+
| SLIDE 8: VISUALIZATION & DISCUSSION — MODEL DIAGNOSTICS & ERROR BEHAVIOR                           |
+-------------------------------------------------+--------------------------------------------------+
| HÌNH TRÁI: Actual vs. Predicted Quality         | HÌNH PHẢI: Residual Plot (Error Diagnostics)     |
| (File: reports/baseline_actual_vs_predicted.png)| (File: reports/baseline_residual_plot.png)       |
+-------------------------------------------------+--------------------------------------------------+
| 3 ĐIỂM NHẤN PHÂN TÍCH TOÁN HỌC:                                                                    |
| 1. Hiện tượng sọc ngang: Do y là số nguyên rời rạc (3-8) trong khi y_hat là số thực liên tục.      |
| 2. Hồi quy về trung bình (Regression to mean): Dự đoán bị co cụm trong dải [4.71, 6.80] do tối ưu  |
|    hàm mất mát MSE (L2 norm) nhằm tránh phạt nặng sai số bình phương ở vùng biên.                 |
| 3. Tính trực giao OLS & Đồng phương sai: Trendline phẳng (Slope ≈ 0.10), phần dư phân tán đều quanh 0. |
+----------------------------------------------------------------------------------------------------+

+----------------------------------------------------------------------------------------------------+
| SLIDE 9: LIMITATIONS & RECOMMENDATIONS                                                             |
+-------------------------------------------------+--------------------------------------------------+
| CỘT TRÁI: CÁC GIỚI HẠN KỸ THUẬT CỐT LÕI         | CỘT PHẢI: BẰNG CHỨNG THỰC NGHIỆM & ĐỀ XUẤT       |
| 1. Bản chất nhãn Ordinal:                       | • Bằng chứng mất cân bằng (Imbalance):           |
|    - Xem thứ bậc như liên tục, ngầm định các     |   - 81.9% mẫu tập trung ở điểm 5 & 6 (MAE ~ 0.40)|
|      khoảng cách vị giác đều bằng nhau.          |   - Điểm 3, 4, 8 chỉ chiếm 5% (MAE > 1.1 - 2.2)  |
| 2. Mất cân bằng lớp trầm trọng (Class Imbalance)| • Bằng chứng phi tuyến: Tương tác hóa học phức tạp|
| 3. Giả định quan hệ tuyến tính thuần túy        | • Hướng phát triển:                              |
|    - Bỏ qua các tương tác phi tuyến và bão hòa. |   - Ordinal Logistic Regression / Multiclass     |
| 4. 240 dòng dữ liệu trùng lặp trong bộ dữ liệu  |   - Polynomial Features & Mô hình Tree-based     |
+-------------------------------------------------+--------------------------------------------------+
```

---

## 6. Kịch bản thuyết trình mẫu dành cho Thư (Speech Transcript)

> **Lời mở đầu & Tiếp nối từ Khuê (Slide 6):**  
> *"Cảm ơn Khuê đã trình bày quy trình huấn luyện và kết quả chuẩn của mô hình Baseline. Tiếp theo, em xin đại diện nhóm trình bày phần Thực nghiệm mở rộng, Phân tích sai số và các Hạn chế cốt lõi của mô hình trên các Slide 7, 8 và 9."*

### Kịch bản Slide 7 (Thực nghiệm so sánh):
*"Thưa thầy cô và các bạn, bước vào Slide 7, nhóm đặt ra một câu hỏi thực nghiệm tự nhiên: Trong phân tích EDA của bạn Thảo, nồng độ cồn (`alcohol`) là biến có tương quan mạnh nhất với điểm chất lượng. Vậy nếu nhóm chỉ sử dụng duy nhất biến này để xây dựng mô hình hồi quy đơn biến, kết quả sẽ ra sao?*

*Nhìn vào bảng số liệu trên màn hình: Một mình đặc trưng `alcohol` đã đem lại hệ số $R^2 = 23.56\%$. Khi so sánh với mô hình Baseline dùng cả 11 đặc trưng có $R^2 = 40.32\%$, ta nhận thấy nồng độ cồn giải thích tới hơn $58\%$ năng lực dự báo của toàn bộ mô hình! Tuy nhiên, việc bổ sung 10 đặc trưng hóa học còn lại vẫn vô cùng giá trị: nó giúp giảm tới $21.9\%$ sai số bình phương MSE và đẩy $R^2$ tăng thêm gần $17\%$. Điều này khẳng định chất lượng rượu vang là một trải nghiệm cảm quan đa chiều và phụ thuộc vào sự cân bằng giữa nhiều yếu tố hóa lý."*

### Kịch bản Slide 8 (Phân tích sai số trực quan):
*"Chuyển sang Slide 8, đây là phần chẩn đoán sai số trực quan để hiểu sâu sắc bản chất toán học của mô hình:*

*Ở biểu đồ bên trái — Actual vs. Predicted: Ta lập tức quan sát thấy hai hiện tượng thú vị. Thứ nhất là các dải điểm xếp thành hàng ngang, điều này rất tự nhiên vì điểm đánh giá thực tế của chuyên gia là số nguyên từ 3 đến 8, trong khi mô hình hồi quy xuất ra số thực liên tục. Thứ hai, toàn bộ dự đoán chỉ dao động trong dải hẹp từ 4.71 đến 6.80. Mô hình không bao giờ dám dự đoán một chai rượu đạt điểm 3 hay điểm 8.*

*Nguyên nhân toán học nằm ở hiện tượng 'Hồi quy về giá trị trung bình' (Regression toward the mean) khi tối ưu hóa hàm mất mát MSE. Vì MSE phạt bình phương sai số, nên siêu phẳng Least Squares chấp nhận kéo dự đoán về vùng mật độ tập trung đông nhất quanh điểm 5.6 để hạn chế rủi ro sai số lớn. Trên biểu đồ phần dư bên phải, đường xu hướng gần như nằm ngang phẳng (Slope = 0.10 ≈ 0) và phân tán đều quanh trục 0, là bằng chứng thực nghiệm xác thực tính trực giao Cov(ŷ, e) ≈ 0 và tính đồng phương sai (Homoscedasticity) của nghiệm OLS trên dữ liệu kiểm thử. Sự thiên lệch có hệ thống (rượu dở bị đoán tăng điểm, rượu hảo hạng bị đoán giảm điểm) được bóc tách rõ nét khi ta nhóm sai số theo từng mức điểm thực tế."*

### Kịch bản Slide 9 (Hạn chế & Hướng phát triển):
*"Từ những phân tích thực nghiệm trên, ở Slide 9, nhóm thẳng thắn nhìn nhận 3 hạn chế cốt lõi của mô hình:*

*Thứ nhất, về mặt toán học, hồi quy tuyến tính giả định biến mục tiêu liên tục và khoảng cách giữa các bậc là đều nhau, trong khi điểm rượu là thang đo thứ bậc (ordinal) dựa trên cảm quan chủ quan của con người. Thứ hai, dữ liệu bị mất cân bằng trầm trọng: hơn 81% mẫu thử nghiệm dồn vào điểm 5 và 6, khiến mô hình bị 'thiên vị' và học rất kém ở các phân lớp hiếm như 3 hay 8. Thứ ba, quan hệ giữa nồng độ hóa chất và vị giác vốn mang tính phi tuyến tính cao.*

*Vì vậy, trong tương lai, nhóm đề xuất chuyển sang mô hình Ordinal Logistic Regression hoặc sử dụng các thuật toán phi tuyến tính như Random Forest, Gradient Boosting để khắc phục triệt để các hạn chế này. Sau đây, xin mời bạn Quân tổng kết lại dự án và điều phối phần Q&A."*

---

## 7. Bộ câu hỏi & câu trả lời phản biện (Q&A Defense) dành cho Giảng viên MAI391

**Câu hỏi 1 của Giảng viên:** *Trên biểu đồ Residual Plot (phần dư theo giá trị dự đoán ŷ), ta thấy đường trendline có độ dốc xấp xỉ 0 (Slope ≈ 0.10) và phân tán đều quanh trục 0. Điều này mang ý nghĩa toán học gì trong OLS?*
> **Gợi ý trả lời cho Thư:**  
> *"Dạ thưa thầy/cô, theo phương trình chuẩn Normal Equation của OLS, vector phần dư $\mathbf{e}$ về mặt toán học trực giao với không gian các cột của ma trận đặc trưng $X$, tức $\mathbf{X}^T \mathbf{e} = \mathbf{0}$. Do $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w}$, ta có $(\hat{\mathbf{y}})^T \mathbf{e} = 0$, nghĩa là trên tập train hiệp phương sai $\text{Cov}(\hat{y}, e) \equiv 0$ tuyệt đối. Trên tập kiểm thử độc lập 320 mẫu, việc đường xu hướng duy trì độ dốc phẳng (Slope = 0.10 ≈ 0) và phần dư phân tán đều xung quanh trục $e=0$ trên dải dự đoán chính [5.2, 6.2] là bằng chứng thực nghiệm xác nhận mô hình đã trích xuất hết thông tin tương quan tuyến tính và thỏa mãn tốt giả định đồng phương sai (Homoscedasticity) của định lý Gauss-Markov."*

**Câu hỏi 2 của Giảng viên:** *Mô hình đạt $R^2 = 40.32\%$. Con số này là cao hay thấp, và tại sao nó lại có giá trị thực tiễn trong môn học?*
> **Gợi ý trả lời cho Thư:**  
> *"Dạ thưa thầy/cô, nếu so với các bài toán mô phỏng vật lý thuần túy, $R^2 = 40.32\%$ có vẻ khiêm tốn. Tuy nhiên, trong bài toán đánh giá cảm quan thực phẩm và con người, $40.32\%$ là một con số rất có ý nghĩa. Nghiên cứu gốc của Cortez et al. (2009) khi công bố bộ dữ liệu này cũng chỉ ra rằng các mô hình tuyến tính đạt sai số quanh mức này vì cảm nhận hương vị con người có tính chủ quan cao và chịu ảnh hưởng của các hợp chất vi lượng chưa được đo đạc trong 11 chỉ số hóa lý. Việc đạt $R^2 \approx 40\%$ trên tập Test mà không bị overfitting chứng minh Multiple Linear Regression đã nắm bắt được cấu trúc tuyến tính chính yếu của bộ dữ liệu."*

**Câu hỏi 3 của Giảng viên:** *Tại sao nhóm không loại bỏ 240 dòng dữ liệu trùng lặp ngay từ đầu?*
> **Gợi ý trả lời cho Thư:**  
> *"Dạ thưa thầy/cô, nhóm đã kiểm tra và ghi nhận chính xác 240 dòng trùng lặp ($15\%$). Tuy nhiên, trong sản xuất rượu vang, các mẻ rượu từ cùng một thùng ủ hoặc cùng một quy trình đo lường hoàn toàn có thể cho ra các thông số hóa lý y hệt nhau và nhận cùng mức đánh giá cảm quan. Để giữ tính nguyên bản của tập dữ liệu UCI và đảm bảo tính nhất quán đối chuẩn giữa 6 thành viên, nhóm quyết định giữ lại các dòng này cho mô hình Baseline, đồng thời thảo luận thẳng thắn về nguy cơ rò rỉ thông tin nhẹ như một trong những hạn chế của dự án ở Slide 9."*

---

## 8. Danh mục sản phẩm bàn giao (Deliverables Checklist)

| STT | Tên file / Sản phẩm | Đường dẫn | Mô tả chi tiết | Trạng thái |
| :---: | :--- | :--- | :--- | :---: |
| 1 | **Notebook thực nghiệm hoàn chỉnh** | [`notebooks/Thu_Experiments.ipynb`](../notebooks/Thu_Experiments.ipynb) | Chứa đầy đủ code, markdown giải thích toán học, render sẵn toàn bộ biểu đồ và output bảng số liệu | ✅ Đã hoàn tất |
| 2 | **Bảng so sánh 2 mô hình** | [`reports/model_comparison.csv`](model_comparison.csv) | So sánh MSE, RMSE, MAE, $R^2$ giữa Baseline 11 biến và Alcohol 1 biến | ✅ Đã hoàn tất |
| 3 | **Bảng số liệu đối chuẩn Slide 7** | [`reports/alcohol_experiment_metrics.csv`](alcohol_experiment_metrics.csv) | Số liệu chênh lệch phục vụ trực tiếp cho Slide 7 | ✅ Đã hoàn tất |
| 4 | **Bảng dự đoán mô hình Alcohol** | [`reports/alcohol_test_predictions.csv`](alcohol_test_predictions.csv) | Dữ liệu dự đoán và phần dư của mô hình 1 biến | ✅ Đã hoàn tất |
| 5 | **Bảng phân tích sai số theo phân lớp** | [`reports/error_by_quality_metrics.csv`](error_by_quality_metrics.csv) | Chi tiết số lượng mẫu, MAE, Bias theo từng mức điểm từ 3 đến 8 | ✅ Đã hoàn tất |
| 6 | **Biểu đồ Actual vs. Predicted** | [`reports/baseline_actual_vs_predicted.png`](baseline_actual_vs_predicted.png) | Biểu đồ 300 DPI có đường $y=\hat{y}$, chú thích sọc ngang và co về trung bình | ✅ Đã hoàn tất |
| 7 | **Biểu đồ Residual Plot** | [`reports/baseline_residual_plot.png`](baseline_residual_plot.png) | Biểu đồ 300 DPI thể hiện đường $e=0$, trendline dốc âm và phân tích phương sai | ✅ Đã hoàn tất |
| 8 | **Biểu đồ Phân phối phần dư** | [`reports/baseline_residual_distribution.png`](baseline_residual_distribution.png) | Histogram và đường khớp chuẩn kiểm tra giả định Gauss-Markov | ✅ Đã hoàn tất |
| 9 | **Biểu đồ Sai số theo phân lớp** | [`reports/error_by_quality_tier.png`](error_by_quality_tier.png) | Biểu đồ kép thể hiện sự mất cân bằng dữ liệu và sai số vượt trội ở vùng biên | ✅ Đã hoàn tất |
| 10 | **Tài liệu bàn giao Slide 7, 8, 9** | [`reports/slide7_8_9_experiments_and_analysis.md`](slide7_8_9_experiments_and_analysis.md) | Tài liệu hướng dẫn ráp slide, kịch bản thuyết trình và bộ câu hỏi phản biện | ✅ Đã hoàn tất |

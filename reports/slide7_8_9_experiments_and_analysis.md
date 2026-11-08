# Bàn giao Nội dung Slide 7, 8, 9: Thực nghiệm, Chẩn đoán sai số & Hạn chế của mô hình

- **Người phụ trách:** Thư (Member 5 — Experiments & Error Analysis)
- **Học phần:** MAI391 — Mathematics for Machine Learning
- **Đề tài:** Wine Quality Prediction Using Multiple Linear Regression
- **Các slide phụ trách:**
  - **Slide 7:** Experimental Results *(Phối hợp cùng Member 4 — Khuê)*
  - **Slide 8:** Visualization & Discussion *(Member 5 độc lập)*
  - **Slide 9:** Limitations *(Member 5 độc lập)*
- **Thời hạn hoàn thành:** 07/11, 22:00 (Đã hoàn thiện nghiệm thu và kiểm định học thuật)

---

## 1. Tổng quan vị trí & Sự kết nối học thuật trong dự án

Member 5 (Thư) đảm nhiệm khâu **kiểm định thực nghiệm (Empirical Validation)**, **chẩn đoán sai số mô hình (Residual Diagnostics)** và **đánh giá phê phán dưới góc nhìn kinh tế lượng (Econometric & Critical Evaluation)**.

Toàn bộ công việc kết nối chặt chẽ và nhất quán với các thành viên trong nhóm:
1. **Tiếp nhận từ Member 1 (Khoa) & Member 4 (Khuê):** Kế thừa nguyên vẹn tập dữ liệu chuẩn `data/train.csv` (1.279 mẫu) và `data/test.csv` (320 mẫu). Nạp trực tiếp mô hình đã huấn luyện từ `models/baseline_linear_regression.joblib` và kết quả dự đoán từ `reports/baseline_test_predictions.csv`, **tuyệt đối không huấn luyện lại mô hình 11 biến từ đầu**.
2. **Đối chiếu với Member 2 (Thảo - EDA):** Kiểm chứng lại nhận định từ ma trận tương quan rằng `alcohol` có tương quan tuyến tính mẫu lớn nhất với điểm chất lượng ($r \approx +0.476$) để thiết lập **Experiment 2 (Mô hình đơn biến `alcohol`)**.
3. **Liên hệ với Member 3 (Đào - Math Foundation):** Kết nối nghiệm giải tích Ordinary Least Squares (OLS) với các thuộc tính của hàm mất mát Mean Squared Error ($L_2$ loss), cơ chế hồi quy về giá trị trung bình (Regression toward the mean) và phân biệt rạch ròi giữa **tính trực giao giải tích trên tập Train** với **chẩn đoán thực nghiệm ngoài mẫu trên tập Test**.
4. **Bàn giao cho Member 6 (Quân - Coordinator/QC):** Cung cấp toàn bộ bảng số liệu chuẩn xác, 4 biểu đồ 300 DPI, bố cục slide gợi ý và kịch bản thuyết trình khoa học, thận trọng về mặt thống kê.

---

## 2. Chi tiết Slide 7: Experimental Results (Kết quả thực nghiệm & So sánh năng lực dự báo)

### 2.1. Thiết kế thực nghiệm
- **Mô hình đa biến Baseline (11 đặc trưng — Khuê phụ trách):** Sử dụng toàn bộ 11 đặc trưng hóa lý trên thang đo tự nhiên (unscaled).
- **Mô hình đơn biến Experiment 2 (1 đặc trưng — Thư phụ trách):** Chỉ sử dụng đặc trưng `alcohol` để xây dựng mô hình hồi quy đơn biến:
  $$\hat{y} = w_{\text{alcohol}} \cdot x_{\text{alcohol}} + b = 0.3623 \cdot \text{alcohol} + 1.8493$$
- **Mục tiêu so sánh:** Đánh giá năng lực dự báo tuyến tính giữa một mô hình tối giản chỉ dùng thuộc tính có tương quan đơn biến cao nhất với mô hình đa biến kết hợp đầy đủ 11 chỉ số hóa lý.

### 2.2. Bảng số liệu đối chuẩn thực nghiệm trên Test Set (320 mẫu)

| Chỉ số (Metric) | Baseline (11 features) | Alcohol only (1 feature) | Chênh lệch ($\Delta = \text{Baseline} - \text{Alcohol}$) | Phân tích học thuật & Ý nghĩa thực nghiệm |
| :--- | :---: | :---: | :---: | :--- |
| **MSE** | **0.390025** | 0.499528 | $-0.109503$ ($-21.92\%$) | Mô hình đa biến giảm $21.92\%$ tổng bình phương sai số so với mô hình đơn biến |
| **RMSE** | **0.624520** | 0.706773 | $-0.082253$ ($-11.64\%$) | Sai số trung bình của Baseline là $0.62$ điểm, thấp hơn mức $0.71$ điểm của mô hình 1 biến |
| **MAE** | **0.503530** | 0.574966 | $-0.071436$ ($-12.42\%$) | Sai số tuyệt đối trung bình giảm $\approx 0.07$ điểm rượu khi dùng đủ 11 biến |
| **$R^2$ Score** | **0.403180 (40.32%)** | **0.235618 (23.56%)** | $+0.167562$ ($+16.76\%$) | Mô hình đa biến đạt $R^2$ cao hơn $16.76$ điểm phần trăm so với mô hình đơn biến |
| **Inference Time\*** | $\approx 0.80\text{ ms}$ | $\approx 0.45\text{ ms}$ | $+0.35\text{ ms}$ | Cả hai mô hình đều có chi phí tính toán cực thấp ($< 1\text{ ms}$ cho 320 mẫu) |

*\*Ghi chú về thời gian thực thi (Timing Benchmark): Thời gian huấn luyện ($\approx 15-85\text{ ms}$) và thời gian suy luận ($\approx 0.5-1.4\text{ ms}$) được đo bằng `time.perf_counter()` mang tính chất tham khảo phần cứng trên môi trường máy tính cụ thể; không dùng để kết luận tuyệt đối về độ phức tạp tính toán giữa các thuật toán.*

### 2.3. Ba nhận xét cốt lõi dưới góc nhìn Kinh tế lượng cho Slide 7
1. **So sánh năng lực giải thích phương sai tuyến tính của `alcohol`:**
   Mô hình đơn biến `alcohol` đạt $R^2 = 23.56\%$, tương đương **$58.44\%$ tỷ lệ $R^2$** của mô hình 11 biến ($0.2356 / 0.4032$). Điều này cho thấy nồng độ cồn mang lượng thông tin tuyến tính lớn nhất về biến động điểm số trong mẫu dữ liệu. 
   > *Lưu ý quan trọng:* Đây là **so sánh tương quan thống kê trong dự báo (predictive association)**, tuyệt đối không suy diễn quan hệ nhân quả (causality) hay cho rằng cồn "quyết định" chất lượng rượu.
2. **Năng lực dự báo gia tăng của mô hình đa biến (Incremental Predictive Power):**
   Khi đưa thêm 10 thuộc tính hóa lý vào mô hình, hệ số $R^2$ tăng thêm **$16.76$ điểm phần trăm** (từ $23.56\%$ lên $40.32\%$), đồng thời MSE giảm **$21.92\%$**. 
   > *Góc nhìn kinh tế lượng:* Vì giữa các đặc trưng có hiện tượng đa cộng tuyến (ví dụ: `alcohol` tương quan âm với `density` $r \approx -0.496$), ta không thể phân rã $16.76\%$ này thành đóng góp cộng dồn độc lập của từng biến. Con số $16.76$ điểm phần trăm phản ánh **giá trị gia tăng tổng thể** khi mô hình khai thác đồng thời cấu trúc đa biến của các hợp chất hóa lý (độ axit, sulfate, đường dư...).
3. **Đánh giá về độ ổn định ngoài mẫu (Out-of-sample Generalization):**
   Trên mô hình Baseline, Test MSE ($0.3900$) có giá trị xấp xỉ và thấp hơn nhẹ so với Train MSE ($0.4242$). Điều này cho thấy mô hình kiểm soát tốt sai số và không xuất hiện biểu hiện quá khớp nghiêm trọng (severe overfitting) trên tập kiểm thử này. Sự chênh lệch nhẹ này phản ánh biến thiên chọn mẫu ngẫu nhiên (sampling variability) của tập test 320 mẫu, không dùng để tuyên bố "chứng minh mô hình hoàn toàn không bị overfitting".

---

## 3. Chi tiết Slide 8: Visualization & Discussion (Chẩn đoán sai số trực quan)

Slide 8 làm sáng tỏ bản chất toán học của mô hình OLS thông qua 4 biểu đồ chẩn đoán trực quan chuẩn 300 DPI:

### 3.1. Phân tích Biểu đồ 1: Actual vs. Predicted (`reports/baseline_actual_vs_predicted.png`)
- **Hiện tượng sọc ngang (Horizontal Bands):**
  - Các điểm kiểm thử xếp thành các hàng ngang song song ứng với các mức điểm nguyên thực tế $y \in \{3, 4, 5, 6, 7, 8\}$, trong khi mô hình hồi quy tuyến tính OLS dự đoán trên miền số thực liên tục $\hat{y} \in \mathbb{R}$.
- **Hiện tượng co về giá trị trung bình (Regression toward the mean):**
  - Dải giá trị dự đoán của mô hình bị co cụm trong khoảng hẹp **$[4.7063, 6.8040]$**. Mô hình không đưa ra bất kỳ dự đoán nào $< 4.7$ hoặc $> 6.8$.
  - *Cơ chế tối ưu hóa $L_2$:* Hàm mất mát $MSE = \frac{1}{n}\sum(y_i - \hat{y}_i)^2$ phạt theo bình phương sai số ($e_i^2$). Do đó, siêu phẳng OLS kéo các dự đoán về gần điểm trung bình mẫu ($\bar{y} \approx 5.68$) để hạn chế các khoản phạt tổn thất lớn ở vùng biên.
  - *Hệ quả phân lớp:* Rượu chất lượng thấp ($y=3, 4$) bị **dự đoán cao hơn thực tế** ($\hat{y} \approx 5.14 - 5.23$, Mean Residual âm). Rượu hảo hạng ($y=7, 8$) bị **dự đoán thấp hơn thực tế** ($\hat{y} \approx 6.18 - 6.56$, Mean Residual dương).

### 3.2. Phân tích Biểu đồ 2: Residual Plot (`reports/baseline_residual_plot.png`)
- **Đặc điểm trực quan:** Trục hoành là giá trị dự đoán $\hat{y}$, trục tung là phần dư kiểm thử $e = y - \hat{y}$. Đường xu hướng có độ dốc mẫu: $\text{Slope} = \mathbf{+0.0987 \approx +0.10}$.
- **Phân định rạch ròi giữa lý thuyết Train và chẩn đoán Test:**
  1. *Tính trực giao OLS trên tập Train (Algebraic Identity):* Theo phương trình chuẩn OLS:
     $$\mathbf{X}_{\text{train}}^T \mathbf{e}_{\text{train}} = \mathbf{0} \implies \hat{\mathbf{y}}_{\text{train}}^T \mathbf{e}_{\text{train}} = 0 \implies \text{Cov}(\hat{y}_{\text{train}}, e_{\text{train}}) \equiv 0$$
     Đây là một đẳng thức đại số bắt buộc trên tập huấn luyện do cơ chế tìm nghiệm Least Squares.
  2. *Chẩn đoán thực nghiệm ngoài mẫu trên tập Test (Empirical Diagnostic):* Trên tập test độc lập, các phần dư không bị ràng buộc toán học phải trực giao với $\hat{\mathbf{y}}$. Độ dốc mẫu $+0.0987 \approx +0.10$ là một **chỉ báo chẩn đoán thực nghiệm ngoài mẫu**, cho thấy phần dư kiểm thử chỉ xuất hiện độ lệch tuyến tính rất nhẹ theo giá trị dự đoán.
  3. *Thực tế về giả định Gauss-Markov:* Trong dải dự đoán trung tâm $[5.2, 6.2]$, phần dư phân tán tương đối đồng đều quanh trục $e=0$. Tuy nhiên, trên toàn cục, do biến mục tiêu $y$ bị chặn trên và chặn dưới trong miền $[3, 8]$, phần dư ở vùng biên chịu ràng buộc giới hạn (boundary constraints), gây ra hiện tượng phương sai thay đổi có điều kiện (conditional heteroscedasticity) tại các mức điểm cực đoan.

### 3.3. Phân tích Biểu đồ 3 & 4: Phân phối phần dư & Bóc tách sai số theo phân lớp (`reports/error_by_quality_tier.png`)
- **Bảng số liệu phân tách theo từng mức điểm thực tế (Breakdown Table):**

| Điểm chất lượng ($y$) | Số mẫu Test ($n$) | Tỷ lệ (%) | Điểm dự đoán TB ($\bar{\hat{y}}$) | Độ lệch trung bình (Bias: $y - \hat{y}$) | MAE | Nhận xét kinh tế lượng |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **3** | 1 | $0.31\%$ | 5.2339 | **$-2.2339$** | 2.2339 | Đánh giá quá cao rượu kém (Overestimation) |
| **4** | 10 | $3.12\%$ | 5.1357 | **$-1.1357$** | 1.1357 | Đánh giá quá cao rượu kém (Overestimation) |
| **5** | 130 | $40.63\%$ | 5.3827 | **$-0.3827$** | **0.3970** | Dự đoán rất tốt, sai số nhỏ |
| **6** | 132 | $41.25\%$ | 5.7567 | **$+0.2433$** | **0.4102** | Dự đoán rất tốt, sai số nhỏ |
| **7** | 42 | $13.12\%$ | 6.1763 | **$+0.8237$** | 0.8237 | Đánh giá thấp hơn thực tế (Underestimation) |
| **8** | 5 | $1.56\%$ | 6.5647 | **$+1.4353$** | 1.4353 | Đánh giá thấp hơn thực tế (Underestimation) |

- **Kết luận:** Mô hình OLS dự đoán chính xác nhất ở vùng trung tâm (điểm 5 và 6 chiếm tới **$81.88\%$** mẫu) với $\text{MAE} \approx 0.40$, nhưng sai số tăng vọt ở các phân lớp hiếm ngoại vi (điểm 3, 4, 8) với $\text{MAE} > 1.1 - 2.2$.

---

## 4. Chi tiết Slide 9: Limitations & Hướng phát triển (Hạn chế dưới góc nhìn Kinh tế lượng)

Slide 9 thể hiện tư duy phản biện khoa học (Critical Thinking) của nhóm, chỉ ra các ranh giới phương pháp luận của mô hình hồi quy tuyến tính cổ điển:

### 4.1. Hạn chế về Bản chất Dữ liệu (Data Distribution & Structure)
1. **Bản chất nhãn Thứ bậc rời rạc (Ordinal Nature) vs. Giả định biến liên tục (Continuous Linear Assumption):**
   - Điểm chất lượng là biến thứ bậc rời rạc ($y \in \{3, 4, 5, 6, 7, 8\}$).
   - Mô hình Linear Regression ngầm định khoảng cách toán học giữa các bước nhảy là đồng nhất: khoảng cách vị giác từ điểm 3 lên 4 được giả định bằng đúng từ 7 lên 8. Trên thực tế cảm quan con người, sự khác biệt giữa một chai rượu "hỏng/kém" lên "uống được" hoàn toàn khác với từ "ngon" lên "hảo hạng".
2. **Mất cân bằng dữ liệu nghiêm trọng (Severe Class Imbalance):**
   - Hai mức điểm 5 và 6 chiếm tới **$81.88\%$** tập kiểm thử (262 / 320 mẫu) và **$82.49\%$** toàn bộ dataset (1.319 / 1.599 mẫu).
   - Nhóm điểm cực đoan 3 và 8 chỉ chiếm lần lượt $0.31\%$ và $1.56\%$.
   - Thuật toán OLS tối thiểu hóa tổng bình phương phần dư trên toàn bộ mẫu, do đó siêu phẳng hồi quy ưu tiên tối ưu hóa cho vùng đa số mẫu để giảm tổng MSE, dẫn tới việc dự đoán kém ở các lớp thiểu số.
3. **Dữ liệu quan sát (Observational Data) vs. Suy diễn can thiệp nhân quả (Causal Inference):**
   - Dataset là dữ liệu quan sát thụ động, không phải thử nghiệm ngẫu nhiên có đối chứng (Randomized Controlled Trial).
   - Các hệ số hồi quy $\hat{\beta}_j$ chỉ phản ánh **liên hệ tương quan từng phần (partial correlation)** có điều kiện, không phản ánh tác động can thiệp nhân quả (ví dụ: không thể kết luận tăng nhân tạo nồng độ cồn sẽ làm tăng điểm chất lượng rượu).
4. **Vấn đề bản ghi trùng lặp (Duplicate Records):**
   - Dataset chứa 240 dòng trùng lặp ($15.01\%$). Dù được giữ lại để làm mốc baseline chuẩn theo dataset UCI gốc, việc phân chia ngẫu nhiên có thể dẫn tới sự trùng lặp bản ghi giữa train và test, tiềm ẩn nguy cơ rò rỉ thông tin nhẹ.

### 4.2. Hạn chế về Thuật toán Hồi quy Tuyến tính (Algorithm Limitations)
1. **Giả định quan hệ tuyến tính nghiêm ngặt (Strict Linearity):**
   - Quá trình hóa sinh trong rượu vang có nhiều hiệu ứng phi tuyến tính và điểm bão hòa (ví dụ: độ axit vừa phải giúp cân bằng vị giác, nhưng quá nhiều axit sẽ làm rượu bị chua gắt; nồng độ cồn quá cao sẽ lấn át mùi hương tinh tế).
   - Mô hình OLS thuần túy không thể tự học các tương tác phi tuyến tính (non-linear interactions) nếu không có bước kỹ thuật tạo đặc trưng đa thức (Polynomial Features).

### 4.3. Đề xuất cải tiến học thuật (Future Recommendations)
1. **Chuyển đổi bài toán:** Sử dụng mô hình **Ordinal Logistic Regression** (Hồi quy logistic có thứ tự) để phản ánh đúng bản chất rời rạc thứ bậc của biến mục tiêu.
2. **Kỹ thuật xử lý mất cân bằng:** Áp dụng kỹ thuật cân bằng trọng số (Sample Weights / Cost-sensitive learning) để phạt nặng sai số ở các phân lớp hiếm (3, 4, 8).
3. **Mô hình phi tuyến:** Thử nghiệm Ridge/Lasso Regression (có chuẩn hóa tham số để kiểm soát đa cộng tuyến), Polynomial Features hoặc các thuật toán Tree-based (Random Forest, Gradient Boosting) để nắm bắt các quan hệ phi tuyến tính.

---

## 5. Bố cục gợi ý cho Slide 7, 8, 9 (Dành cho Member 6 ráp Slide Master)

```text
+----------------------------------------------------------------------------------------------------+
| SLIDE 7: EXPERIMENTAL RESULTS — BASELINE (11 FEATURES) VS. ALCOHOL ONLY (1 FEATURE)               |
+-------------------------------------------------+--------------------------------------------------+
| CỘT TRÁI: BẢNG SỐ LIỆU SO SÁNH TRỰC DIỆN        | CỘT PHẢI: PHÂN TÍCH KINH TẾ LƯỢNG & ĐỘNG LỰC TOÁN|
| (Nhúng bảng số liệu 5 chỉ số MSE, RMSE, MAE,   | • So sánh tương quan của Alcohol:                |
|  R2, Inference Time kèm ghi chú benchmark)      |   - Đạt 58.44% mức R2 của mô hình 11 biến        |
|                                                 |   - R2 = 0.2356 (1 biến) vs. 0.4032 (11 biến)    |
| HỘP NỔI BẬT (HIGHLIGHT BOX):                   | • Năng lực dự báo gia tăng của 11 biến:          |
| Baseline: R2 = 40.32% | MSE = 0.3900            |   - Giảm 21.92% MSE, R2 tăng +16.76 điểm %       |
| Alcohol : R2 = 23.56% | MSE = 0.4995            |   - Đa cộng tuyến khiến R2 không thể phân rã cộng|
| Δ R2    : +16.76% pts | Δ MSE: -21.92%          | • Độ ổn định ngoài mẫu: Test MSE ~ Train MSE     |
+-------------------------------------------------+--------------------------------------------------+

+----------------------------------------------------------------------------------------------------+
| SLIDE 8: VISUALIZATION & DISCUSSION — MODEL DIAGNOSTICS & ERROR BEHAVIOR                           |
+-------------------------------------------------+--------------------------------------------------+
| HÌNH TRÁI: Actual vs. Predicted Quality         | HÌNH PHẢI: Residual Plot (Error Diagnostics)     |
| (File: reports/baseline_actual_vs_predicted.png)| (File: reports/baseline_residual_plot.png)       |
+-------------------------------------------------+--------------------------------------------------+
| 3 ĐIỂM NHẤN PHÂN TÍCH TOÁN HỌC & KINH TẾ LƯỢNG:                                                    |
| 1. Hiện tượng sọc ngang: Do y là số nguyên rời rạc (3-8) trong khi y_hat là số thực liên tục.      |
| 2. Hồi quy về trung bình (Regression to mean): Dự đoán bị co cụm trong dải [4.71, 6.80] do hàm mất |
|    mát L2 phạt bình phương sai số ở các vùng biên cực đoan.                                        |
| 3. Chẩn đoán phần dư ngoài mẫu: Trendline thực nghiệm phẳng (Slope ≈ +0.10), phần dư phân tán      |
|    tương đối đều ở dải trung tâm, nhưng có phương sai thay đổi có điều kiện ở các mức điểm biên.   |
+----------------------------------------------------------------------------------------------------+

+----------------------------------------------------------------------------------------------------+
| SLIDE 9: LIMITATIONS & RECOMMENDATIONS                                                             |
+-------------------------------------------------+--------------------------------------------------+
| CỘT TRÁI: GIỚI HẠN PHƯƠNG PHÁP LUẬN & DỮ LIỆU   | CỘT PHẢI: BẰNG CHỨNG THỰC NGHIỆM & HƯỚNG PHÁT TRIỂN|
| 1. Bản chất nhãn Ordinal:                       | • Bằng chứng mất cân bằng (Imbalance):           |
|    - Giả định khoảng cách vị giác đều nhau.     |   - 81.88% mẫu tập trung ở điểm 5 & 6 (MAE ~ 0.40|
| 2. Mất cân bằng lớp nghiêm trọng (Imbalance)    |   - Điểm 3, 4, 8 chỉ chiếm ~5% (MAE > 1.1 - 2.2) |
| 3. Dữ liệu quan sát vs Suy diễn nhân quả        | • Bằng chứng phi tuyến: Tương tác hóa học phức tạp|
|    - Tương quan từng phần, không phải can thiệp.| • Hướng phát triển:                              |
| 4. Giả định tuyến tính thuần túy                |   - Ordinal Logistic Regression                  |
| 5. 240 dòng trùng lặp trong dữ liệu gốc         |   - Cost-sensitive learning & Tree-based models  |
+-------------------------------------------------+--------------------------------------------------+
```

---

## 6. Kịch bản thuyết trình mẫu dành cho Thư (Speech Transcript)

> **Lời mở đầu & Tiếp nối từ Khuê (Slide 6):**  
> *"Cảm ơn Khuê đã trình bày quy trình huấn luyện và kết quả chuẩn của mô hình Baseline. Tiếp theo, em xin đại diện nhóm trình bày phần Thực nghiệm so sánh, Chẩn đoán sai số và Hạn chế của mô hình trên các Slide 7, 8 và 9 dưới góc nhìn phương pháp luận và kiểm định toán học."*

### Kịch bản Slide 7 (Thực nghiệm so sánh năng lực dự báo):
*"Thưa thầy cô và các bạn, bước vào Slide 7, nhóm đặt ra một câu hỏi thực nghiệm tự nhiên: Trong phân tích EDA của bạn Thảo, nồng độ cồn (`alcohol`) là thuộc tính có tương quan tuyến tính đơn biến mạnh nhất với chất lượng rượu ($r \approx +0.48$). Vậy nếu nhóm xây dựng một mô hình hồi quy đơn biến chỉ dựa vào nồng độ cồn, năng lực dự báo sẽ ra sao so với mô hình 11 biến?*

*Nhìn vào bảng số liệu trên màn hình: Một mình đặc trưng `alcohol` đã đem lại hệ số $R^2 = 23.56\%$ trên tập test độc lập, tương đương hơn $58\%$ mức $R^2$ của mô hình 11 biến. Tuy nhiên, nhóm nhấn mạnh đây là **so sánh tương quan dự báo tuyến tính**, không phải quan hệ nhân quả. Khi mở rộng sang mô hình đa biến kết hợp đủ 11 thuộc tính hóa lý, hệ số $R^2$ tăng thêm $16.76$ điểm phần trăm lên mức $40.32\%$ và sai số MSE giảm tới $21.92\%$. Do giữa các thuộc tính hóa học tồn tại hiện tượng đa cộng tuyến, mức tăng $16.76$ điểm phần trăm này phản ánh giá trị gia tăng tổng thể của hệ thống hóa lý đa chiều chứ không thể phân rã cộng dồn độc lập. Cuối cùng, mức sai số giữa Train MSE ($0.4242$) và Test MSE ($0.3900$) là rất nhỏ, cho thấy mô hình kiểm soát tốt sai số kiểm thử và không có biểu hiện quá khớp nghiêm trọng."*

### Kịch bản Slide 8 (Chẩn đoán sai số trực quan):
*"Chuyển sang Slide 8, đây là phần chẩn đoán sai số trực quan để hiểu sâu sắc cơ chế hoạt động của nghiệm OLS:*

*Ở biểu đồ bên trái — Actual vs. Predicted: Ta quan sát thấy các dải điểm xếp thành hàng ngang do biến mục tiêu thực tế là số nguyên rời rạc từ 3 đến 8, trong khi mô hình hồi quy dự đoán trên miền số thực liên tục. Đáng chú ý, toàn bộ dự đoán chỉ dao động trong dải hẹp từ 4.71 đến 6.80. Hiện tượng này được gọi là **'Hồi quy về giá trị trung bình' (Regression toward the mean)**. Do hàm mất mát MSE phạt theo bình phương sai số, siêu phẳng OLS kéo các dự đoán về gần điểm trung bình mẫu quanh 5.68 để hạn chế các khoản phạt tổn thất lớn ở vùng biên.*

*Chuyển sang biểu đồ phần dư bên phải: Về mặt giải tích, tính trực giao OLS $\text{Cov}(\hat{y}, e) = 0$ là đẳng thức toán học trên tập huấn luyện. Trên tập kiểm thử độc lập 320 mẫu này, đường xu hướng thực nghiệm có độ dốc rất nhỏ (Slope $\approx +0.10$), cho thấy phần dư kiểm thử chỉ có độ lệch tuyến tính rất nhẹ theo giá trị dự đoán. Khi bóc tách sai số theo từng phân lớp điểm thực tế ở bảng bên dưới, ta thấy rõ: ở hai phân lớp đa số là 5 và 6 (chiếm gần $82\%$ mẫu test), sai số tuyệt đối MAE chỉ khoảng $0.40$ điểm; nhưng ở các phân lớp hiếm như 3, 4 hay 8, sai số MAE tăng vọt lên từ $1.1$ đến hơn $2.2$ điểm."*

### Kịch bản Slide 9 (Hạn chế & Hướng phát triển):
*"Từ những phân tích thực nghiệm trên, ở Slide 9, nhóm thẳng thắn chỉ ra các giới hạn phương pháp luận của mô hình:*

*Thứ nhất, hồi quy tuyến tính giả định biến mục tiêu liên tục và khoảng cách giữa các bước nhảy là đồng nhất, trong khi điểm rượu là biến thứ bậc (ordinal) dựa trên cảm quan chủ quan của con người. Thứ hai, sự mất cân bằng dữ liệu nghiêm trọng — với gần $82\%$ mẫu tập trung ở điểm 5 và 6 — khiến thuật toán OLS ưu tiên tối ưu hóa vùng trung tâm để giảm tổng sai số toàn cục, dẫn tới dự đoán kém ở các phân lớp hiếm. Thứ ba, đây là dữ liệu quan sát (observational data), các hệ số hồi quy chỉ phản ánh liên hệ tương quan tuyến tính từng phần chứ không chứng minh tác động can thiệp nhân quả. Thứ tư, các phản ứng hóa sinh trong rượu vang vốn mang tính phi tuyến tính cao.*

*Vì vậy, trong tương lai, nhóm đề xuất chuyển đổi bài toán sang Ordinal Logistic Regression, áp dụng kỹ thuật cân bằng trọng số hoặc sử dụng các mô hình Tree-based phi tuyến tính như Random Forest, Gradient Boosting để cải thiện độ chính xác. Sau đây, xin mời bạn Quân tổng kết lại dự án và điều phối phần Q&A."*

---

## 7. Bộ câu hỏi & câu trả lời phản biện (Q&A Defense) dành cho Giảng viên MAI391

**Câu hỏi 1 của Giảng viên:** *Độ dốc của đường xu hướng trên biểu đồ Residual Plot ở tập test là $+0.0987 \approx +0.10$. Con số này có chứng minh được phần dư trực giao với giá trị dự đoán hay không?*
> **Gợi ý trả lời cho Thư:**  
> *"Dạ thưa thầy/cô, về mặt giải tích toán học, tính trực giao $\mathbf{X}^T \mathbf{e} = \mathbf{0} \implies \hat{\mathbf{y}}^T \mathbf{e} = 0 \implies \text{Cov}(\hat{y}, e) \equiv 0$ là một đẳng thức đại số bắt buộc trên tập huấn luyện (training set identity) do cơ chế giải phương trình chuẩn OLS. Tuy nhiên, trên tập kiểm thử ngoài mẫu (out-of-sample test set), phần dư không bị ràng buộc toán học phải trực giao với giá trị dự đoán. Con số độ dốc thực nghiệm $+0.0987 \approx +0.10$ trên tập test chỉ là một chỉ báo chẩn đoán thực nghiệm (empirical diagnostic), cho thấy phần dư kiểm thử chỉ xuất hiện độ lệch tuyến tính rất nhẹ theo giá trị dự đoán trên mẫu dữ liệu cụ thể này, chứ không phải là một chứng minh toán học cho tính trực giao ngoài mẫu."*

**Câu hỏi 2 của Giảng viên:** *Mô hình đơn biến alcohol đạt $R^2 = 23.56\%$ và mô hình 11 biến đạt $R^2 = 40.32\%$. Ta có thể nói 10 biến còn lại đóng góp độc lập $16.76\%$ vào $R^2$ hay không?*
> **Gợi ý trả lời cho Thư:**  
> *"Dạ thưa thầy/cô, về mặt kinh tế lượng, ta không thể phân rã $16.76\%$ đó thành đóng góp cộng dồn độc lập của 10 biến còn lại. Lý do là vì các thuộc tính hóa lý trong rượu vang có mối tương quan lẫn nhau (hiện tượng đa cộng tuyến — ví dụ `alcohol` tương quan âm với `density` $r \approx -0.496$). Khi các biến phụ thuộc tuyến tính vào nhau, không gian cột của chúng bị chồng lấn một phần, nên $R^2$ không có tính chất cộng dồn trực giao. Con số $+16.76$ điểm phần trăm phản ánh giá trị gia tăng của mô hình đa biến nói chung trong việc giải thích phương sai ngoài mẫu so với mô hình đơn biến."*

**Câu hỏi 3 của Giảng viên:** *Tại sao mô hình OLS lại bị co cụm dự đoán trong dải [4.71, 6.80] và không dự đoán được các điểm cực đoan 3 hay 8?*
> **Gợi ý trả lời cho Thư:**  
> *"Dạ thưa thầy/cô, nguyên nhân cốt lõi bắt nguồn từ hàm mất mát $MSE = \frac{1}{n}\sum(y_i - \hat{y}_i)^2$ ($L_2$ loss) kết hợp với sự mất cân bằng dữ liệu (gần $82\%$ mẫu tập trung ở điểm 5 và 6). Hàm $L_2$ phạt sai số theo hàm bậc hai ($e_i^2$), nên việc dự đoán một giá trị cực đoan như 3 hay 8 nếu bị sai lệch sẽ gây ra tổn thất cực lớn. Để giảm thiểu tổng tổn thất trên toàn bộ tập dữ liệu, siêu phẳng OLS kéo các dự đoán co cụm về gần giá trị trung bình mẫu ($\approx 5.68$). Đây chính là hiện tượng hồi quy về giá trị trung bình (Regression toward the mean) điển hình trong thống kê."*

---

## 8. Danh mục sản phẩm bàn giao (Deliverables Checklist)

| STT | Tên file / Sản phẩm | Đường dẫn | Mô tả chi tiết | Trạng thái |
| :---: | :--- | :--- | :--- | :---: |
| 1 | **Notebook thực nghiệm hoàn chỉnh** | [`notebooks/Thu_Experiments.ipynb`](../notebooks/Thu_Experiments.ipynb) | Tự thân thực thi 100% (Restart & Run All clean), có công thức toán và đồ thị chuẩn | ✅ Đã kiểm định |
| 2 | **Bảng so sánh 2 mô hình** | [`reports/model_comparison.csv`](model_comparison.csv) | So sánh MSE, RMSE, MAE, $R^2$ giữa Baseline 11 biến và Alcohol 1 biến | ✅ Đã kiểm định |
| 3 | **Bảng số liệu đối chuẩn Slide 7** | [`reports/alcohol_experiment_metrics.csv`](alcohol_experiment_metrics.csv) | Số liệu chênh lệch phục vụ trực tiếp cho Slide 7 | ✅ Đã kiểm định |
| 4 | **Bảng dự đoán mô hình Alcohol** | [`reports/alcohol_test_predictions.csv`](alcohol_test_predictions.csv) | Dữ liệu dự đoán và phần dư của mô hình 1 biến | ✅ Đã kiểm định |
| 5 | **Bảng phân tích sai số theo phân lớp** | [`reports/error_by_quality_metrics.csv`](error_by_quality_metrics.csv) | Chi tiết số lượng mẫu, MAE, Bias theo từng mức điểm từ 3 đến 8 | ✅ Đã kiểm định |
| 6 | **Biểu đồ Actual vs. Predicted** | [`reports/baseline_actual_vs_predicted.png`](baseline_actual_vs_predicted.png) | Biểu đồ 300 DPI có đường $y=\hat{y}$, chú thích sọc ngang và co về trung bình | ✅ Đã kiểm định |
| 7 | **Biểu đồ Residual Plot** | [`reports/baseline_residual_plot.png`](baseline_residual_plot.png) | Biểu đồ 300 DPI thể hiện đường $e=0$, trendline thực nghiệm (Slope = +0.10) và phân tích phương sai | ✅ Đã kiểm định |
| 8 | **Biểu đồ Phân phối phần dư** | [`reports/baseline_residual_distribution.png`](baseline_residual_distribution.png) | Histogram và đường khớp chuẩn kiểm tra giả định Gauss-Markov | ✅ Đã kiểm định |
| 9 | **Biểu đồ Sai số theo phân lớp** | [`reports/error_by_quality_tier.png`](error_by_quality_tier.png) | Biểu đồ kép thể hiện sự mất cân bằng dữ liệu và sai số vượt trội ở vùng biên | ✅ Đã kiểm định |
| 10 | **Tài liệu bàn giao Slide 7, 8, 9** | [`reports/slide7_8_9_experiments_and_analysis.md`](slide7_8_9_experiments_and_analysis.md) | Tài liệu hướng dẫn ráp slide, kịch bản thuyết trình và bộ câu hỏi phản biện chuẩn kinh tế lượng | ✅ Đã kiểm định |
| 11 | **Bộ kiểm thử tự động toàn diện** | [`test_thu_deliverables.py`](../test_thu_deliverables.py) | Bộ 50/50 tests kiểm định cả 2 lớp Kỹ thuật & Học thuật, bao gồm Restart & Run All check | ✅ Đã kiểm định |

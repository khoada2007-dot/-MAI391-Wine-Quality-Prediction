import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="wide")

st.title("🍷 Red Wine Quality Prediction")
st.markdown("Ứng dụng dự đoán điểm chất lượng rượu vang đỏ bằng **Multiple Linear Regression** (MAI391).")

@st.cache_resource
def load_model():
    return joblib.load("wine_quality_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Chưa tìm thấy file wine_quality_model.pkl! Lỗi: {e}")
    st.stop()

st.subheader("Nhập thông số hóa lý của mẫu rượu:")
col1, col2, col3 = st.columns(3)

with col1:
    fixed_acidity = st.slider("Fixed Acidity (g/dm³)", 4.0, 16.0, 8.3, step=0.1)
    volatile_acidity = st.slider("Volatile Acidity (g/dm³)", 0.10, 1.60, 0.52, step=0.01)
    citric_acid = st.slider("Citric Acid (g/dm³)", 0.0, 1.0, 0.26, step=0.01)
    residual_sugar = st.slider("Residual Sugar (g/dm³)", 0.5, 16.0, 2.5, step=0.1)

with col2:
    chlorides = st.slider("Chlorides (g/dm³)", 0.010, 0.600, 0.080, step=0.005)
    free_sulfur_dioxide = st.slider("Free Sulfur Dioxide (mg/dm³)", 1.0, 72.0, 14.0, step=1.0)
    total_sulfur_dioxide = st.slider("Total Sulfur Dioxide (mg/dm³)", 6.0, 290.0, 46.0, step=1.0)
    density = st.slider("Density (g/cm³)", 0.9900, 1.0040, 0.9967, step=0.0005)

with col3:
    pH = st.slider("pH", 2.70, 4.00, 3.31, step=0.01)
    sulphates = st.slider("Sulphates (g/dm³)", 0.30, 2.00, 0.65, step=0.01)
    alcohol = st.slider("Alcohol (% vol)", 8.0, 15.0, 10.4, step=0.1)

input_data = pd.DataFrame([[
    fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
    chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
    pH, sulphates, alcohol
]], columns=[
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol"
])

st.divider()

if st.button("🚀 Dự đoán điểm chất lượng", type="primary", use_container_width=True):
  pred = float(model.predict(input_data)[0])

  # Giới hạn hiển thị trong dải [3, 8] của dataset
  display_score = max(3.0, min(8.0, pred))

  st.markdown(f"### Điểm chất lượng dự đoán: **{pred:.2f}**")
  st.caption("Thang điểm chất lượng rượu vang đỏ: từ **3** (kém nhất) đến **8** (tốt nhất)")

  # Thanh tiến trình chuẩn hóa theo thang từ 3 đến 8
  # (3 điểm tương ứng 0%, 8 điểm tương ứng 100%)
  percent = (display_score - 3.0) / (8.0 - 3.0)
  st.progress(percent)

  # Đánh giá đúng theo phân bố của nhóm
  if pred >= 6.5:
    st.success("🌟 Đánh giá: **Chất lượng tốt (High Quality - Thuộc top điểm 7-8)**")
  elif pred >= 5.0:
    st.info("⚖️ Đánh giá: **Chất lượng trung bình / Đạt chuẩn (Medium Quality - Điểm 5-6)**")
  else:
    st.warning("⚠️ Đánh giá: **Chất lượng kém (Low Quality - Điểm 3-4)**")
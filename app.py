import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 設定 matplotlib 支援中文顯示 (Windows 通常使用微軟正黑體)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False # 確保負號顯示正常

# 頁面設定為寬版，讓圖表和指標更具空間感
st.set_page_config(layout="wide", page_title="Linear Regression 互動版")

st.title("Linear Regression Model (CRISP-DM 流程) 互動版")

# 初始化 Session State (給隨機化按鈕使用)
if 'a_val' not in st.session_state:
    st.session_state.a_val = 30
if 'b_val' not in st.session_state:
    st.session_state.b_val = 100
if 'var_val' not in st.session_state:
    st.session_state.var_val = 100

def randomize_params():
    st.session_state.a_val = int(np.random.uniform(-50, 50))
    st.session_state.b_val = int(np.random.uniform(0, 100))
    st.session_state.var_val = int(np.random.uniform(0, 300))

# 側邊欄控制項
st.sidebar.header("⚙️ 互動控制項")
st.sidebar.button("🎲 隨機化切換", on_click=randomize_params)

a = st.sidebar.slider("a（斜率）", -50, 50, key="a_val", help="真實模型斜率")
b = st.sidebar.slider("b（截距）", 0, 100, key="b_val", help="真實模型截距")
var = st.sidebar.slider("var（雜訊變異數）", 0, 300, key="var_val", help="高斯雜訊 ε ~ N(0, var) 的變異數")
n = st.sidebar.slider("n（樣本數）", 20, 2000, 200, help="合成資料點數量")
outlier_k = st.sidebar.slider("離群值數量", 1, 30, 20, help="依絕對殘差標示前 K 個離群點")
seed = st.sidebar.number_input("隨機種子", min_value=0, max_value=9999, value=42, step=1, help="控制結果可重現性")

# ==========================================
# 資料準備與模型訓練
# ==========================================
np.random.seed(seed)
x = np.random.uniform(0, 100, n)

std_dev = np.sqrt(var)
noise = np.random.normal(0, std_dev, n)
y_true = a * x + b
y = y_true + noise

coefficients = np.polyfit(x, y, 1)
m, c = coefficients
y_pred = m * x + c

errors = np.abs(y - y_pred)
mse = np.mean(errors**2)
top_outlier_indices = np.argsort(errors)[-outlier_k:][::-1]

# ==========================================
# UI 介面呈現
# ==========================================

# 1. KPI 指標卡
st.write("### 📊 模型表現與參數 (KPI)")
col1, col2, col3 = st.columns(3)
# st.metric 的 delta 呈現與真實參數的差異
with col1:
    st.metric("估計斜率 (Slope)", f"{m:.2f}", f"{(m - a):.2f} (與真實差異)", delta_color="inverse")
with col2:
    st.metric("估計截距 (Intercept)", f"{c:.2f}", f"{(c - b):.2f} (與真實差異)", delta_color="inverse")
with col3:
    st.metric("均方誤差 (MSE)", f"{mse:.2f}")

st.divider()

# 2. 圖表區：散佈圖 & 殘差長條圖
col_plot1, col_plot2 = st.columns([3, 2])

with col_plot1:
    st.write("### 📈 模型散佈圖")
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    
    # 資料點 (藍)
    ax1.scatter(x, y, color='blue', label='Data points (藍)', alpha=0.5)
    # 離群值 (橘)
    ax1.scatter(x[top_outlier_indices], y[top_outlier_indices], color='orange', label=f'Top {outlier_k} Outliers (橘)', zorder=5)
    # 真實線 (綠虛線)
    ax1.plot(x, y_true, color='green', linestyle='--', label='True Line (綠虛線)', linewidth=2)
    # 擬合線 (紅)
    ax1.plot(x, y_pred, color='red', label='Fitted Line (紅)', linewidth=2)
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)

with col_plot2:
    st.write(f"### 📉 Top-{outlier_k} 絕對殘差長條圖")
    top_errors = errors[top_outlier_indices]
    
    # 建立用來畫水平長條圖的 DataFrame
    df_err = pd.DataFrame({
        'Index': top_outlier_indices,
        'Absolute Error': top_errors
    }).sort_values(by='Absolute Error', ascending=True) # 從小到大排，讓最大的在最上面
    
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.barh(range(len(df_err)), df_err['Absolute Error'], color='salmon', edgecolor='black')
    ax2.set_yticks(range(len(df_err)))
    ax2.set_yticklabels([f"idx {idx}" for idx in df_err['Index']])
    ax2.set_xlabel('Absolute Residual')
    ax2.grid(axis='x', alpha=0.3)
    st.pyplot(fig2)

st.divider()

# 3. 離群值表格
st.write("### ⚠️ 離群值明細 (Outliers Table)")
df_outliers = pd.DataFrame({
    'Data Index': top_outlier_indices,
    'x': x[top_outlier_indices],
    'y_true (真實)': y_true[top_outlier_indices],
    'y_actual (觀測)': y[top_outlier_indices],
    'y_pred (預測)': y_pred[top_outlier_indices],
    'Absolute Error': errors[top_outlier_indices]
})
# 將表格加入漸層色顯示
st.dataframe(
    df_outliers.style.background_gradient(subset=['Absolute Error'], cmap='OrRd'),
    width='stretch'
)

st.divider()

# 4. 資料統計面板
st.write("### 📋 資料與模型統計摘要")
with st.expander("展開查看詳細 x/y 統計摘要與模型 JSON", expanded=False):
    col_stat1, col_stat2 = st.columns(2)
    
    with col_stat1:
        st.write("**資料統計摘要 (Describe)**")
        df_all = pd.DataFrame({'x': x, 'y': y, 'y_pred': y_pred, 'Error': errors})
        st.dataframe(df_all.describe(), width='stretch')
    
    with col_stat2:
        st.write("**模型 JSON (Model Parameters)**")
        st.json({
            "true_model": {
                "slope": a,
                "intercept": b,
                "variance": var
            },
            "fitted_model": {
                "slope": round(m, 4),
                "intercept": round(c, 4)
            },
            "metrics": {
                "mse": round(mse, 4),
                "n_samples": n,
                "outliers_count": outlier_k
            }
        })

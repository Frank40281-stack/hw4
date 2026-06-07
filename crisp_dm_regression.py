import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 設定 matplotlib 支援中文顯示 (Windows 通常使用微軟正黑體)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False # 確保負號顯示正常

# 預設參數
a = 30
b = 100
var = 100
n = 200
outlier_k = 20
seed = 42

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

fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.scatter(x, y, color='blue', label='Data points (藍)', alpha=0.5)
ax1.scatter(x[top_outlier_indices], y[top_outlier_indices], color='orange', label=f'Top {outlier_k} Outliers (橘)', zorder=5)
ax1.plot(x, y_true, color='green', linestyle='--', label='True Line (綠虛線)', linewidth=2)
ax1.plot(x, y_pred, color='red', label='Fitted Line (紅)', linewidth=2)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title(f'Linear Regression (a={a}, b={b}, var={var}, n={n})')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 將圖表存為圖片
plt.savefig('crisp_dm_linear_regression.png', dpi=300, bbox_inches='tight')
print("Image successfully saved as crisp_dm_linear_regression.png")

# CRISP-DM Linear Regression Project

此專案展示如何運用 CRISP-DM（Cross-Industry Standard Process for Data Mining）流程來進行線性回歸分析，並透過 Streamlit 打造一個互動式的 Dashboard 讓使用者動態調整參數與觀察離群值。

## 專案檔案結構

- `app.py`: Streamlit 網頁應用程式（包含 KPI、長條圖、散佈圖與漸層表格）。
- `crisp_dm_regression.py`: 純 Python 腳本，用於執行基本的線性回歸分析並匯出靜態圖表。
- `crisp_dm_linear_regression.png`: 透過上述腳本匯出的模型散佈圖範例。
- `requirements.txt`: 專案所需的 Python 套件清單。
- `.gitignore`: Git 忽略清單。

## 快速啟動

1. **安裝套件**：
   ```bash
   pip install -r requirements.txt
   ```

2. **執行互動式 Dashboard (Streamlit)**：
   ```bash
   streamlit run app.py
   ```
   執行後會自動開啟瀏覽器，展示互動式介面。

3. **執行靜態腳本並匯出圖表**：
   ```bash
   python crisp_dm_regression.py
   ```
   執行後將會在當前目錄生成 `crisp_dm_linear_regression.png` 圖片。

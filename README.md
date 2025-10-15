# ⚡ Fast Track Bareilly — Hourly Load Forecast

### 📄 Overview
This project forecasts the next **24 hours of electricity demand** for Bareilly using past 7 days of smart meter data.  
The pipeline cleans, resamples, and models hourly energy readings using a **Ridge Regression** model and a **Seasonal Naive** baseline.

---

### 🧠 Features
- Converts 3-min readings to hourly totals.
- Fills small gaps (≤2 hours) using conservative forward fill.
- Caps outliers using 1st–99th percentile clipping.
- Generates next 24-hour forecast (T+1 to T+24).
- Computes **MAE, WAPE, sMAPE** metrics.
- Saves plots & results to reproducible folders.

---

### ⚙️ Setup

```bash
git clone https://github.com/Snehalatha1015/fast_track_bareilly.git
cd fast_track_bareilly
python -m venv venv
source venv/bin/activate        # or venv\Scripts\activate on Windows
pip install -r requirements.txt

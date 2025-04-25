# 🚛 Otto Dörner Forecast Dashboard

A Streamlit dashboard for forecasting container activity (delivery, pickup, and net change) based on Otto Dörner logistics data. Designed for intuitive planning, interactive filtering, and automated PDF report generation.

---

## 📊 Features

- 📈 Forecast container deliveries, pickups, or net remaining containers
- 🗓 Filter by dispatch center, container types, and forecast duration (1–30 days)
- 📎 Download a professional PDF report with forecast charts
- 🔍 Interactive, real-time charts powered by Plotly
- 🧠 Uses historical data from 2021–2025 Q1 for model training

---

## 🏁 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/sharmaD91/otto_doerner_challenge.git
cd otto-doerner-forecast
```
### 2. Install dependencies

```bash
pip install -r requirements.txt
```
### 3. Run the Streamlit app

```bash
streamlit run dashboard.py
```
### 4. Access the dashboard

#### Open your web browser and navigate to `http://localhost:8501` to access the dashboard.

---
## 📂 Directory Structure

```
otto-doerner-forecast/
│
├── dashboard.py               # Main Streamlit app
├── otto_forecast_model.py     # OttoForecaster model logic
├── otto_files/                # Input data files (.CSV / .XLSX)
├── requirements.txt           # Required packages
└── README.md                  # This file
```

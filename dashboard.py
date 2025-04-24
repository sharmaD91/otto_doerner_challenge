import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from fpdf import FPDF
from PIL import Image
import io
from otto_forecast_model import OttoForecaster

st.set_page_config(page_title="Otto Dörner - Forecast Dashboard", layout="wide")

logo_url = "https://upload.wikimedia.org/wikipedia/commons/b/b3/Otto_Dörner_Logo_2021.png"
st.sidebar.markdown(f"<div style='text-align: center;'><img src='{logo_url}' width='180'></div>", unsafe_allow_html=True)

page = st.sidebar.radio("Select View", ["📈 Forecast"])

forecaster = OttoForecaster()

def load_data_and_train():
    df = forecaster.load_and_prepare_data([
        "otto_files/MMX_Hackathon2025_year2021.CSV",
        "otto_files/MMX_Hackathon2025_year2022.xlsx",
        "otto_files/MMX_Hackathon2025_year2023.xlsx",
        "otto_files/MMX_Hackathon2025_year2024.xlsx",
        "otto_files/MMX_Hackathon2025_year2025Q1.CSV",
    ])
    forecaster.train_model(df)
    return df

df = load_data_and_train()

# UI Filter with display name mapping
dispatch_display_map = {
    "HH": "Hamburg (HH)",
    "SME": "Itzehoe (SME)",
    "KIE": "Kiel (KIE)"
}
dispatch_centers = sorted(df['DspZenKz'].dropna().str.strip().unique())
dispatch_display_list = [dispatch_display_map.get(code, code) for code in dispatch_centers]
display_to_code = {v: k for k, v in dispatch_display_map.items()}
selected_display = st.sidebar.selectbox("Select Dispatch Center", dispatch_display_list)
dspzen_selected = display_to_code.get(selected_display, selected_display)

container_options = ["Mulde", "Roll-off container"]
container_selected = st.sidebar.multiselect("Select Container Type(s)", container_options, default=container_options)

forecast_days = st.sidebar.slider("Select Number of Days to Forecast", min_value=1, max_value=30, value=7)

if page == "📈 Forecast":
    st.title(f"🔮 Forecast for Next {forecast_days} Days in {selected_display}")

    forecast_data = forecaster.predict(start_date=datetime.today().strftime("%Y-%m-%d"), location=dspzen_selected, days=forecast_days)
    forecast_data.rename(columns={"container_M": "Mulde", "container_C": "Roll-off container"}, inplace=True)

    # Diagramm
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    if "Mulde" in container_selected:
        ax2.plot(forecast_data["date"], forecast_data["Mulde"], marker='o', label="Mulde")
    if "Roll-off container" in container_selected:
        ax2.plot(forecast_data["date"], forecast_data["Roll-off container"], marker='o', label="Roll-off container")
    ax2.set_title(f"Forecasted Container Demand - {selected_display}")
    ax2.set_ylabel("Containers")
    ax2.legend()
    st.pyplot(fig2)

    with st.expander("📋 Forecast Data Table"):
        st.dataframe(forecast_data[["date"] + container_selected])

    def create_pdf_report(df, location, chart_figure):
        img_buffer = io.BytesIO()
        chart_figure.savefig(img_buffer, format='PNG')
        img_buffer.seek(0)
        img_path = "chart_temp.png"
        with open(img_path, "wb") as f:
            f.write(img_buffer.read())

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Otto Dörner Forecast Report - {location}", ln=True, align='C')
        pdf.ln(10)
        pdf.image(img_path, w=180)
        pdf.ln(5)
        pdf.set_font("Arial", 'B', size=10)
        for col in ["date"] + container_selected:
            pdf.cell(45, 8, txt=str(col), border=1)
        pdf.ln()
        pdf.set_font("Arial", size=10)
        for index, row in df.iterrows():
            for col in ["date"] + container_selected:
                pdf.cell(45, 8, txt=str(row[col]), border=1)
            pdf.ln()

        pdf_output = io.BytesIO()
        pdf_bytes = pdf.output(dest='S').encode('latin1')
        pdf_output.write(pdf_bytes)
        pdf_output.seek(0)
        return pdf_output

    pdf_report = create_pdf_report(forecast_data, selected_display, fig2)
    st.download_button("📄 Download PDF Report", data=pdf_report, file_name="forecast_report.pdf")
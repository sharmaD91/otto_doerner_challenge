import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from fpdf import FPDF
from PIL import Image
import io
from otto_forecast_model import OttoForecaster
import plotly.express as px
from io import BytesIO
from PIL import Image


st.set_page_config(page_title="Otto Dörner - Forecast Dashboard", layout="wide")

logo_url = "https://upload.wikimedia.org/wikipedia/commons/b/b3/Otto_Dörner_Logo_2021.png"
# st.sidebar.markdown(f"<div style='text-align: center;'><img src='{logo_url}' width='180'></div>", unsafe_allow_html=True)
st.markdown(
    f"""
    <h1 style='text-align: center; color: #003366;'> Otto Dörner - Forecast Dashboard</h1>
    """,
    unsafe_allow_html=True
)
with st.sidebar:
    st.image(logo_url, width=180)
    st.markdown("---")

page = st.sidebar.radio("Select View", ["📈 Forecast - Containers Delivered", "📈 Forecast - Containers Picked Up", "📈 Forecast - Net Containers remaining"])

def create_pdf_report(df, location, chart_figure):
    img_path = "chart_temp.png"
    with open(img_path, "wb") as f:
        f.write(chart_figure)

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

forecaster = OttoForecaster()

df = forecaster.load_and_prepare_data([
    "otto_files/MMX_Hackathon2025_year2021.CSV",
    "otto_files/MMX_Hackathon2025_year2022.xlsx",
    "otto_files/MMX_Hackathon2025_year2023.xlsx",
    "otto_files/MMX_Hackathon2025_year2024.xlsx",
    "otto_files/MMX_Hackathon2025_year2025Q1.CSV",
])

if page == "📈 Forecast - Containers Delivered":
    containers = "pick"
elif page == "📈 Forecast - Containers Picked Up":
    containers = "put"
elif page == "📈 Forecast - Net Containers remaining":
    containers = "difference"

forecaster.difference_flag = False
forecaster.train_model(df, containers)

# UI Filter with display name mapping
dispatch_display_map = {
    "HH": "Hamburg (HH)",
    "SME": "Itzehoe (SME)",
    "KIE": "Kiel (KIE)"
}
dispatch_centers = sorted(df['DspZenKz'].dropna().str.strip().unique())
dispatch_display_list = [dispatch_display_map.get(code, code) for code in dispatch_centers]
display_to_code = {v: k for k, v in dispatch_display_map.items()}
# selected_display = st.sidebar.selectbox("Select Dispatch Center", dispatch_display_list

container_options = ["Skip", "Roll-off", 'C07', 'C10', 'C15', 'C20', 'C25', 'C30', 'C35', 'CP20', 'M03', 'M05', 'M07', 'M10', 'MP12', 'C12', 'MP10', 'C23', 'C18', 'C37', 'CP16', 'C34', 'M12', 'C13', 'C33', 'C24', 'C27', 'C17', 'CP18', 'C08', 'C26', 'C28', 'MP08', 'C22', 'C29', 'C36', 'C05', 'C40', 'C21', 'CB35', 'CB17']

# container_selected = st.sidebar.multiselect("Select Container Type(s)", container_options, default=["Skip", "Roll-off"])

# forecast_days = st.sidebar.slider("Select Number of Days to Forecast", min_value=1, max_value=30, value=7)

with st.sidebar.expander("🔧 Filter Options", expanded=True):
    selected_display = st.selectbox("Dispatch Center", dispatch_display_list)
    container_selected = st.multiselect("Container Type(s)", container_options, default=["Skip", "Roll-off"])
    forecast_days = st.slider("Forecast Days", 1, 30, 7)
    dspzen_selected = display_to_code.get(selected_display, selected_display)
    
# st.title(f"🔮 Forecast for Next {forecast_days} Days in {selected_display}")
st.markdown(
    f"""
    <h3 style='text-align: center; color: #666;'>Forecast for Next {forecast_days} Days in {selected_display}</h3>
    """,
    unsafe_allow_html=True
)

forecast_data = forecaster.predict(start_date=datetime.today().strftime("%Y-%m-%d"), location=dspzen_selected, days=forecast_days)

forecast_data.rename(columns={"container_M": "Skip", "container_C": "Roll-off"}, inplace=True)

# Diagramm
# fig2, ax2 = plt.subplots(figsize=(10, 4))
# for container in container_selected:
#     if container in forecast_data.columns:
#         ax2.plot(
#             forecast_data["date"],
#             forecast_data[container],
#             marker='o',
#             label=container + " Containers"
#         )

fig = px.line(
    forecast_data,
    x="date",
    y=container_selected,
    labels={"value": "Containers", "variable": "Container Type"},
    title=f"Forecasted Container Demand - {selected_display}",
)
fig.update_traces(mode='lines+markers')
st.plotly_chart(fig, use_container_width=True)

# ax2.set_title(f"Forecasted Container Demand - {selected_display}")
# ax2.set_ylabel("Containers")
# ax2.legend()
# st.pyplot(fig2)

st.caption(f"Forecast starting from **{datetime.today().strftime('%B %d, %Y')}**")

st.sidebar.info("🧠 Forecast Model: Trained with last 4 years of data\n📅 Last updated: April 2025")


with st.expander("📋 Forecast Data Table"):
    st.dataframe(forecast_data[["date"] + container_selected])

img_bytes = fig.to_image(format="png")  # Plotly figure to PNG bytes
pdf_report = create_pdf_report(forecast_data, selected_display, img_bytes)

st.download_button("📄 Download PDF Report", data=pdf_report, file_name="forecast_report.pdf")

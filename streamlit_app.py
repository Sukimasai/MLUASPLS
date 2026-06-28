import streamlit as st
import pandas as pd
from ml_service import load_model_bundle, build_dashboard_payload
from datasyn import generate_mock_data

EXPECTED_COLUMNS = [
    'Airline Name', 'Overall_Rating', 'Review_Title', 'Review Date', 'Verified', 
    'Review', 'Aircraft', 'Type Of Traveller', 'Seat Type', 'Route', 
    'Date Flown', 'Seat Comfort', 'Cabin Staff Service', 'Food & Beverages', 
    'Ground Service', 'Inflight Entertainment', 'Wifi & Connectivity', 
    'Value For Money', 'Recommended'
]

st.set_page_config(page_title="EagleInsight Dashboard", layout="wide")
st.title("EagleInsight: Airline Review Analyzer")

st.sidebar.header("Testing & Tools")
if st.sidebar.button("Generate Test Dataset"):
    st.session_state['mock_data'] = generate_mock_data()
    st.sidebar.success("Data generated! Use the download button below.")

if 'mock_data' in st.session_state:
    st.sidebar.download_button(
        "Download Generated CSV",
        data=st.session_state['mock_data'].to_csv(index=False),
        file_name="mock_airline_reviews.csv"
    )

# Main Section
st.write("Upload a CSV or use generated data to see AI insights.")
template_df = pd.DataFrame(columns=EXPECTED_COLUMNS)
st.download_button("Download Template", template_df.to_csv(index=False), "template.csv")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
df = None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
elif 'mock_data' in st.session_state:
    df = st.session_state['mock_data']

if df is not None:
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns: {', '.join(missing_cols)}")
    else:
        bundle = load_model_bundle()
        with st.spinner("Analyzing..."):
            payload = build_dashboard_payload(df, bundle)
            st.metric("Total Reviews", payload["summary"]["total_reviews"])
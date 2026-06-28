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
    with st.spinner("Generating synthetic data..."):
        st.session_state['mock_data'] = generate_mock_data()
        st.sidebar.success("Data generated successfully!")

if 'mock_data' in st.session_state:
    st.sidebar.download_button(
        label="📥 Download Generated CSV",
        data=st.session_state['mock_data'].to_csv(index=False),
        file_name="mock_airline_reviews.csv",
        mime="text/csv"
    )

st.subheader("Data Upload")
col1, col2 = st.columns(2)
with col1:
    template_df = pd.DataFrame(columns=EXPECTED_COLUMNS)
    st.download_button("📥 Download Template", template_df.to_csv(index=False), "template.csv")

uploaded_file = st.file_uploader("Upload your Airline_review.csv", type=["csv"])

df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
elif 'mock_data' in st.session_state:
    if st.checkbox("Use generated mock data instead?"):
        df = st.session_state['mock_data']

if df is not None:
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    
    if df.empty:
        st.error("The uploaded CSV is empty.")
    elif missing_cols:
        st.error(f"Invalid file format. Missing columns: {', '.join(missing_cols)}")
    else:
        st.success("File format verified! Processing...")
        
        bundle = load_model_bundle()
        with st.spinner("Analyzing dataset..."):
            payload = build_dashboard_payload(df, bundle)
            
            st.header("Executive Summary")
            summary = payload["summary"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Reviews", summary["total_reviews"])
            c2.metric("Recommended", summary["recommended_reviews"])
            c3.metric("Not Recommended", summary["not_recommended_reviews"])
            c4.metric("Avg Confidence", f"{summary['average_confidence']}%")

            st.divider()
            
            st.header("Extracted Insights")
            if payload["insights"]:
                st.dataframe(pd.DataFrame(payload["insights"]), use_container_width=True)
            
            st.divider()
            
            st.header("Priority Matrix")
            m1, m2 = st.columns(2)
            with m1:
                st.subheader("Priority Targets")
                st.json(payload["matrix"]["priority_target"])
            with m2:
                st.subheader("Major Projects")
                st.json(payload["matrix"]["major_projects"])
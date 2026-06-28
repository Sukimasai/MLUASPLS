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
col1, col2 = st.columns([1, 4])
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
    # 1. Validation Logic
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    
    if df.empty:
        st.error("The uploaded CSV is empty.")
    elif missing_cols:
        st.error(f"Invalid file format. Missing columns: {', '.join(missing_cols)}")
    else:
        st.success("File format verified! Processing...")
        
        try:
            bundle = load_model_bundle()
            with st.spinner("Analyzing dataset..."):
                payload = build_dashboard_payload(df, bundle)
                
                st.header("Executive Summary")
                summary = payload["summary"]
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Reviews", summary["total_reviews"])
                col2.metric("Recommended", summary["recommended_reviews"])
                col3.metric("Not Recommended", summary["not_recommended_reviews"])
                col4.metric("Average Confidence", f"{summary['average_confidence']}%")

                st.divider()

                st.header("Extracted Insights")
                insights = payload["insights"]
                if insights:
                    st.dataframe(pd.DataFrame(insights), use_container_width=True)
                else:
                    st.info("No actionable insights found based on current rules.")

                st.divider()

                st.header("Priority Matrix")
                matrix = payload["matrix"]
                
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.subheader("Priority Targets (High Impact, Easy)")
                    st.json(matrix["priority_target"])
                    
                    st.subheader("Major Projects (High Impact, Hard)")
                    st.json(matrix["major_projects"])
                
                with m_col2:
                    st.subheader("Fill-ins (Low Impact, Easy)")
                    st.json(matrix["fill_ins"])
                    
                    st.subheader("Not Important (Low Impact, Hard)")
                    st.json(matrix["not_important"])

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
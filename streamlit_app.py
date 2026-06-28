import streamlit as st
import pandas as pd
from ml_service import load_model_bundle, build_dashboard_payload

EXPECTED_COLUMNS = [
    'Airline Name', 'Overall_Rating', 'Review_Title', 'Review Date', 'Verified', 
    'Review', 'Aircraft', 'Type Of Traveller', 'Seat Type', 'Route', 
    'Date Flown', 'Seat Comfort', 'Cabin Staff Service', 'Food & Beverages', 
    'Ground Service', 'Inflight Entertainment', 'Wifi & Connectivity', 
    'Value For Money', 'Recommended'
]

st.set_page_config(page_title="EagleInsight Dashboard", layout="wide")
st.title("EagleInsight: Airline Review Analyzer")
st.markdown("Upload a CSV of airline reviews to generate AI-driven insights and recommendations.")

st.subheader("Data Upload")
st.write("If you don't have your data file ready, you can download the template below:")

template_df = pd.DataFrame(columns=EXPECTED_COLUMNS)
csv_template = template_df.to_csv(index=False)

st.download_button(
    label="📥 Download Data Template",
    data=csv_template,
    file_name="airline_reviews_template.csv",
    mime="text/csv"
)

@st.cache_resource
def get_model():
    return load_model_bundle()

bundle = get_model()

uploaded_file = st.file_uploader("Upload Airline_review.csv", type=["csv"])

if uploaded_file is not None:
    try:
        dataframe = pd.read_csv(uploaded_file)
        
        missing_cols = [col for col in EXPECTED_COLUMNS if col not in dataframe.columns]
        
        if dataframe.empty:
            st.error("The uploaded CSV is empty.")
        elif missing_cols:
            st.error(f"Invalid file format. Missing columns: {', '.join(missing_cols)}")
        else:
            with st.spinner("Analyzing dataset..."):
                payload = build_dashboard_payload(dataframe, bundle)
                
                st.header("Executive Summary")
                summary = payload["summary"]
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Reviews", summary["total_reviews"])
                col2.metric("Recommended", summary["recommended_reviews"])
                col3.metric("Not Recommended", summary["not_recommended_reviews"])
                col4.metric("Avg Confidence", f"{summary['average_confidence']}%")

                st.divider()

                st.header("Extracted Insights")
                insights = payload["insights"]
                if insights:
                    st.dataframe(pd.DataFrame(insights), use_container_width=True)
                else:
                    st.info("No actionable insights found.")

                st.divider()
                
                st.header("Priority Matrix")
                matrix = payload["matrix"]
                
                m1, m2 = st.columns(2)
                with m1:
                    st.subheader("Priority Targets")
                    st.json(matrix["priority_target"])
                    st.subheader("Major Projects")
                    st.json(matrix["major_projects"])
                with m2:
                    st.subheader("Fill-ins")
                    st.json(matrix["fill_ins"])
                    st.subheader("Not Important")
                    st.json(matrix["not_important"])

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
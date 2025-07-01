import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import json
import os
from real_time_crime_extractor import RealTimeCrimeDataExtractor  # Import your extractor class

# Set up the Streamlit page
st.set_page_config(page_title="Real-Time Crime Tracker", layout="wide")
st.title("🌍 Real-Time Crime Data Tracker")

# Initialize Extractor with API Key (Replace with your actual API Key)
extractor = RealTimeCrimeDataExtractor('your_news_api_key')

# Sidebar Controls
st.sidebar.header("Crime Data Controls")
days = st.sidebar.slider("Select Days of Data", min_value=1, max_value=30, value=7)

if st.sidebar.button("Fetch Crime Data"):
    with st.spinner("Fetching latest crime data..."):
        crime_data = extractor.fetch_real_time_crime_data(days_back=days)
        if not crime_data.empty:
            crime_data.to_csv("crime_data.csv", index=False)  # Save data
            st.success(f"Data collected: {len(crime_data)} incidents")
        else:
            st.warning("No crime data found.")

# Load crime data if available
try:
    crime_data = pd.read_csv("crime_data.csv")
    st.subheader(f"Crime Data for the Last {days} Days")
    st.dataframe(crime_data)
except FileNotFoundError:
    st.warning("No crime data available. Fetch data first.")

# Generate and Display Heatmap
if 'crime_data' in locals() and not crime_data.empty:
    st.subheader("📍 Crime Hotspots Map")
    heatmap = extractor.generate_crime_heatmap(crime_data)
    if heatmap:
        folium_static(heatmap)

# Load and display crime analysis
try:
    st.write("Attempting to load crime_analysis.json...")
    json_path = os.path.join(os.getcwd(), "crime_analysis.json")
    st.write(f"Full path: {json_path}")
    st.write(f"File exists: {os.path.exists(json_path)}")

    analysis = {}
    file_content = ""

    if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
        # Try different encodings
        for encoding in ["utf-8", "utf-16", "latin-1"]:
            try:
                with open(json_path, "r", encoding=encoding, errors="ignore") as f:
                    file_content = f.read()
                analysis = json.loads(file_content)
                st.success(f"Loaded using encoding: {encoding}")
                break
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                st.warning(f"Failed to load with {encoding}: {e}")

    if analysis:
        st.subheader("📊 Crime Analysis")
        st.write(f"**Total Incidents:** {analysis['total_incidents']}")

        st.write("### Incidents by Category")
        st.bar_chart(pd.Series(analysis["by_category"]))

        st.write("### Top Locations")
        st.bar_chart(pd.Series(analysis["top_locations"]))

        st.write("### Hourly Distribution")
        st.line_chart(pd.Series(analysis["hourly_distribution"]))

        st.write("### Daily Distribution")
        st.bar_chart(pd.Series(analysis["daily_distribution"]))
    else:
        st.warning("Failed to load crime analysis data.")

except FileNotFoundError:
    st.warning("No analysis data available. Fetch data first.")

st.sidebar.info("Developed using Streamlit & Folium")

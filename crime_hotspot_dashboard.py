import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium import plugins
from streamlit_folium import folium_static
import json
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Import custom modules
from real_time_crime_extractor import RealTimeCrimeDataExtractor
from crime_predictor import CrimePredictor
from intervention_engine import InterventionEngine
from feedback_validator import FeedbackValidator

# Set page configuration
st.set_page_config(
    page_title="Real-Time Crime Hotspot Prediction",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E3A8A;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .high-risk {
        color: #DC2626;
        font-weight: bold;
    }
    .medium-risk {
        color: #F59E0B;
        font-weight: bold;
    }
    .low-risk {
        color: #10B981;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for persistent data
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'crime_data' not in st.session_state:
    st.session_state.crime_data = None
if 'hotspots' not in st.session_state:
    st.session_state.hotspots = None
if 'risk_amplifiers' not in st.session_state:
    st.session_state.risk_amplifiers = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'validation_metrics' not in st.session_state:
    st.session_state.validation_metrics = None

# Helper functions for loading data
def load_crime_data():
    """Load crime data from CSV file"""
    try:
        data = pd.read_csv('crime_data.csv')
        return data
    except FileNotFoundError:
        return None

def load_crime_analysis():
    """Load crime analysis from JSON file"""
    try:
        with open('crime_analysis.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        st.error("Error decoding crime analysis JSON file. The file may be corrupted.")
        return None

def load_crime_hotspots():
    """Load predicted crime hotspots"""
    try:
        with open('crime_hotspots.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        st.error("Error decoding hotspots JSON file. The file may be corrupted.")
        return None

def load_risk_amplifiers():
    """Load risk amplifiers data"""
    try:
        with open('risk_amplifiers.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        st.error("Error decoding risk amplifiers JSON file. The file may be corrupted.")
        return None

def load_intervention_recommendations():
    """Load intervention recommendations"""
    try:
        with open('intervention_recommendations.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        st.error("Error decoding recommendations JSON file. The file may be corrupted.")
        return None

def load_validation_metrics():
    """Load model validation metrics"""
    try:
        with open('validation_metrics.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        st.error("Error decoding validation metrics JSON file. The file may be corrupted.")
        return None

# Function to create a hotspot map
def create_hotspot_map(crime_data, hotspots=None):
    """Create an interactive map with crime incidents and predicted hotspots"""
    if crime_data is None or crime_data.empty:
        return None
    
    # Calculate map center based on data
    center_lat = crime_data['latitude'].mean()
    center_lon = crime_data['longitude'].mean()
    
    # Create base map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles="CartoDB positron")
    
    # Add crime incidents
    incidents = folium.FeatureGroup(name="Crime Incidents")
    
    # Create a marker cluster for better performance with many points
    marker_cluster = plugins.MarkerCluster()
    
    # Add markers for each crime incident
    for idx, row in crime_data.iterrows():
        if pd.notna(row['latitude']) and pd.notna(row['longitude']):
            popup_text = f"<b>{row['category']}</b><br>{row['title']}<br>{row['date']}<br>{row['location_name']}"
            
            # Choose icon color based on crime category
            icon_color = 'red'
            if row['category'].lower() == 'theft':
                icon_color = 'blue'
            elif row['category'].lower() == 'assault':
                icon_color = 'red'
            elif row['category'].lower() == 'robbery':
                icon_color = 'orange'
            elif row['category'].lower() == 'shooting':
                icon_color = 'darkred'
            elif row['category'].lower() == 'murder':
                icon_color = 'black'
            
            marker = folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=folium.Popup(popup_text, max_width=300),
                icon=folium.Icon(color=icon_color, icon="info-sign")
            )
            marker_cluster.add_child(marker)
    
    incidents.add_child(marker_cluster)
    m.add_child(incidents)
    
    # Add heatmap layer
    heat_data = [[row['latitude'], row['longitude']] for idx, row in crime_data.iterrows() 
                if pd.notna(row['latitude']) and pd.notna(row['longitude'])]
    
    if heat_data:
        m.add_child(plugins.HeatMap(heat_data, radius=15, blur=10, gradient={0.4: 'blue', 0.65: 'yellow', 1: 'red'}))
    
    # Add predicted hotspots if available
    if hotspots:
        hotspot_group = folium.FeatureGroup(name="Predicted Hotspots")
        
        for hotspot_id, data in hotspots.items():
            if 'coordinates' in data and 'lat' in data['coordinates'] and 'lon' in data['coordinates']:
                lat = data['coordinates']['lat']
                lon = data['coordinates']['lon']
                risk_score = data.get('prediction', {}).get('risk_score', 0)
                crime_types = data.get('prediction', {}).get('crime_types', {})
                
                # Determine risk level color
                if risk_score >= 0.7:
                    color = 'red'
                    risk_level = 'High'
                elif risk_score >= 0.4:
                    color = 'orange'
                    risk_level = 'Medium'
                else:
                    color = 'green'
                    risk_level = 'Low'
                
                # Create popup content
                popup_content = f"""
                <div style='width: 200px'>
                    <h4>Hotspot #{hotspot_id}</h4>
                    <p><b>Risk Level:</b> {risk_level} ({risk_score:.2f})</p>
                    <p><b>Location:</b> {data.get('location', 'Unknown')}</p>
                    <p><b>Predicted Crime Types:</b></p>
                    <ul>
                """
                
                for crime_type, probability in crime_types.items():
                    popup_content += f"<li>{crime_type}: {probability:.2f}</li>"
                
                popup_content += "</ul></div>"
                
                # Add circle marker for hotspot
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=risk_score * 20,  # Size based on risk score
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.4,
                    popup=folium.Popup(popup_content, max_width=300)
                ).add_to(hotspot_group)
        
        m.add_child(hotspot_group)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

# Function to create risk amplifier visualization
def visualize_risk_amplifiers(risk_amplifiers):
    """Create visualizations for risk amplifiers"""
    if not risk_amplifiers:
        st.warning("No risk amplifier data available.")
        return
    
    # Extract temporal patterns
    if 'temporal_patterns' in risk_amplifiers:
        temporal = risk_amplifiers['temporal_patterns']
        
        # Time of day patterns
        if 'time_of_day' in temporal:
            st.subheader("Crime Risk by Time of Day")
            time_data = pd.DataFrame({
                'Hour': [int(h) for h in temporal['time_of_day'].keys()],
                'Risk Score': list(temporal['time_of_day'].values())
            })
            time_data = time_data.sort_values('Hour')
            
            fig = px.line(time_data, x='Hour', y='Risk Score', 
                         title="Crime Risk by Hour of Day",
                         labels={'Risk Score': 'Risk Amplification Factor'},
                         line_shape='spline')
            fig.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=2))
            st.plotly_chart(fig, use_container_width=True)
        
        # Day of week patterns
        if 'day_of_week' in temporal:
            st.subheader("Crime Risk by Day of Week")
            # Define order of days
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_data = pd.DataFrame({
                'Day': list(temporal['day_of_week'].keys()),
                'Risk Score': list(temporal['day_of_week'].values())
            })
            # Reorder days
            day_data['Day'] = pd.Categorical(day_data['Day'], categories=day_order, ordered=True)
            day_data = day_data.sort_values('Day')
            
            fig = px.bar(day_data, x='Day', y='Risk Score',
                        title="Crime Risk by Day of Week",
                        labels={'Risk Score': 'Risk Amplification Factor'},
                        color='Risk Score',
                        color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
    
    # Environmental factors
    if 'environmental_factors' in risk_amplifiers:
        st.subheader("Environmental Risk Factors")
        env_factors = risk_amplifiers['environmental_factors']
        
        env_data = pd.DataFrame({
            'Factor': list(env_factors.keys()),
            'Impact': list(env_factors.values())
        })
        
        # Sort by impact for better visualization
        env_data = env_data.sort_values('Impact', ascending=False)
        
        fig = px.bar(env_data, x='Factor', y='Impact',
                    title="Environmental Risk Factors Impact",
                    labels={'Impact': 'Risk Amplification Factor'},
                    color='Impact',
                    color_continuous_scale='RdYlGn_r')
        st.plotly_chart(fig, use_container_width=True)

# Function to display intervention recommendations
def display_recommendations(recommendations):
    """Display intervention recommendations in an organized way"""
    if not recommendations:
        st.warning("No intervention recommendations available.")
        return
    
    st.markdown("<h2 class='sub-header'>Intervention Recommendations</h2>", unsafe_allow_html=True)
    
    # Group recommendations by risk level
    high_risk = {}
    medium_risk = {}
    low_risk = {}
    
    for hotspot_id, data in recommendations.items():
        risk_level = data.get('risk_level', 'unknown')
        if risk_level == 'high':
            high_risk[hotspot_id] = data
        elif risk_level == 'medium':
            medium_risk[hotspot_id] = data
        elif risk_level == 'low':
            low_risk[hotspot_id] = data
    
    # Display recommendations by risk level
    if high_risk:
        st.markdown("<h3 class='high-risk'>High Risk Areas</h3>", unsafe_allow_html=True)
        for hotspot_id, data in high_risk.items():
            with st.expander(f"Hotspot #{hotspot_id} - {data.get('location', 'Unknown')} - {data.get('primary_crime_type', 'Unknown')} Risk"):
                st.write(f"**Primary Crime Type:** {data.get('primary_crime_type', 'Unknown')}")
                st.write("**Recommended Interventions:**")
                
                for intervention in data.get('interventions', []):
                    st.markdown(f"**{intervention.get('type', 'Unknown')}** - *Effectiveness: {intervention.get('effectiveness', 0):.2f}*")
                    st.write(intervention.get('description', ''))
                    
                    # Display specific actions
                    if 'actions' in intervention and intervention['actions']:
                        st.write("Specific Actions:")
                        for action in intervention['actions']:
                            st.markdown(f"- {action}")
    
    if medium_risk:
        st.markdown("<h3 class='medium-risk'>Medium Risk Areas</h3>", unsafe_allow_html=True)
        for hotspot_id, data in medium_risk.items():
            with st.expander(f"Hotspot #{hotspot_id} - {data.get('location', 'Unknown')} - {data.get('primary_crime_type', 'Unknown')} Risk"):
                st.write(f"**Primary Crime Type:** {data.get('primary_crime_type', 'Unknown')}")
                st.write("**Recommended Interventions:**")
                
                for intervention in data.get('interventions', []):
                    st.markdown(f"**{intervention.get('type', 'Unknown')}** - *Effectiveness: {intervention.get('effectiveness', 0):.2f}*")
                    st.write(intervention.get('description', ''))
                    
                    # Display specific actions
                    if 'actions' in intervention and intervention['actions']:
                        st.write("Specific Actions:")
                        for action in intervention['actions']:
                            st.markdown(f"- {action}")
    
    if low_risk:
        st.markdown("<h3 class='low-risk'>Low Risk Areas</h3>", unsafe_allow_html=True)
        for hotspot_id, data in low_risk.items():
            with st.expander(f"Hotspot #{hotspot_id} - {data.get('location', 'Unknown')} - {data.get('primary_crime_type', 'Unknown')} Risk"):
                st.write(f"**Primary Crime Type:** {data.get('primary_crime_type', 'Unknown')}")
                st.write("**Recommended Interventions:**")
                
                for intervention in data.get('interventions', []):
                    st.markdown(f"**{intervention.get('type', 'Unknown')}** - *Effectiveness: {intervention.get('effectiveness', 0):.2f}*")
                    st.write(intervention.get('description', ''))
                    
                    # Display specific actions
                    if 'actions' in intervention and intervention['actions']:
                        st.write("Specific Actions:")
                        for action in intervention['actions']:
                            st.markdown(f"- {action}")

# Function to display validation metrics
def display_validation_metrics(validation_metrics):
    """Display model validation metrics and performance trends"""
    if not validation_metrics:
        st.warning("No validation metrics available.")
        return
    
    st.markdown("<h2 class='sub-header'>Model Performance Metrics</h2>", unsafe_allow_html=True)
    
    # Display overall accuracy trend
    if 'prediction_accuracy' in validation_metrics:
        accuracy_data = validation_metrics['prediction_accuracy']
        if accuracy_data:
            st.subheader("Prediction Accuracy Trend")
            
            # Create dataframe for plotting
            df_accuracy = pd.DataFrame({
                'Validation Run': list(range(1, len(accuracy_data) + 1)),
                'Accuracy': accuracy_data
            })
            
            fig = px.line(df_accuracy, x='Validation Run', y='Accuracy',
                         title="Prediction Accuracy Over Time",
                         labels={'Accuracy': 'Overall Accuracy Score'},
                         markers=True)
            
            # Add threshold line
            if 'validation_thresholds' in validation_metrics:
                threshold = validation_metrics['validation_thresholds'].get('prediction_accuracy', 0.7)
                fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                             annotation_text="Minimum Acceptable Accuracy",
                             annotation_position="bottom right")
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Display intervention effectiveness
    if 'intervention_effectiveness' in validation_metrics and validation_metrics['intervention_effectiveness']:
        st.subheader("Intervention Effectiveness")
        
        effectiveness = validation_metrics['intervention_effectiveness']
        df_interventions = pd.DataFrame({
            'Intervention Type': list(effectiveness.keys()),
            'Effectiveness Score': list(effectiveness.values())
        })
        
        # Sort by effectiveness
        df_interventions = df_interventions.sort_values('Effectiveness Score', ascending=False)
        
        fig = px.bar(df_interventions, x='Intervention Type', y='Effectiveness Score',
                    title="Intervention Effectiveness Comparison",
                    color='Effectiveness Score',
                    color_continuous_scale='Blues')
        
        # Add threshold line
        if 'validation_thresholds' in validation_metrics:
            threshold = validation_metrics['validation_thresholds'].get('intervention_effectiveness', 0.3)
            fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                         annotation_text="Minimum Acceptable Effectiveness",
                         annotation_position="bottom right")
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Display detailed metrics if available
    if 'feedback_history' in validation_metrics and validation_metrics['feedback_history']:
        st.subheader("Detailed Validation History")
        
        history = validation_metrics['feedback_history']
        for i, validation in enumerate(history):
            with st.expander(f"Validation Run #{i+1} - {validation.get('timestamp', 'Unknown date')}"): 
                if 'metrics' in validation:
                    metrics = validation['metrics']
                    cols = st.columns(4)
                    
                    if 'spatial_accuracy' in metrics:
                        cols[0].metric("Spatial Accuracy", f"{metrics['spatial_accuracy']:.2f}")
                    
                    if 'type_accuracy' in metrics:
                        cols[1].metric("Crime Type Accuracy", f"{metrics['type_accuracy']:.2f}")
                    
                    if 'risk_correlation' in metrics:
                        cols[2].metric("Risk Score Correlation", f"{metrics['risk_correlation']:.2f}")
                    
                    if 'false_positive_rate' in metrics:
                        cols[3].metric("False Positive Rate", f"{metrics['false_positive_rate']:.2f}")
                    
                    if 'overall_accuracy' in metrics:
                        st.metric("Overall Accuracy", f"{metrics['overall_accuracy']:.2f}")
                
                if 'details' in validation and validation['details']:
                    st.write("**Detailed Findings:**")
                    st.json(validation['details'])

# Main dashboard function
def main():
    # Display header
    st.markdown("<h1 class='main-header'>🔍 Real-Time Crime Hotspot Prediction</h1>", unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", [
        "Dashboard Overview",
        "Crime Hotspot Map",
        "Risk Amplifier Analysis",
        "Intervention Recommendations",
        "Model Performance"
    ])
    
    # Initialize components
    extractor = RealTimeCrimeDataExtractor("b070622b14b340efb5e9a0585811ea02")  # Replace with your API key
    predictor = CrimePredictor()
    intervention_engine = InterventionEngine()
    validator = FeedbackValidator()
    
    # Data controls in sidebar
    st.sidebar.title("Data Controls")
    days_back = st.sidebar.slider("Days of Data to Analyze", 1, 30, 7)
    
    # Refresh data button
    if st.sidebar.button("Fetch New Crime Data"):
        with st.spinner("Fetching real-time crime data..."):
            crime_data = extractor.fetch_real_time_crime_data(days_back=days_back)
            if not crime_data.empty:
                # Save data to CSV
                crime_data.to_csv('crime_data.csv', index=False)
                
                # Generate and save crime analysis
                analysis = extractor.analyze_crime_data(crime_data)
                with open('crime_analysis.json', 'w') as f:
                    json.dump(analysis, f, indent=4)
                
                st.session_state.crime_data = crime_data
                st.session_state.last_update = datetime.now()
                
                st.sidebar.success(f"Data updated successfully! {len(crime_data)} incidents retrieved.")
            else:
                st.sidebar.error("No crime data found. Please try again later.")
    
    # Generate predictions button
    if st.sidebar.button("Generate Predictions & Recommendations"):
        crime_data = load_crime_data()
        
        if crime_data is not None and not crime_data.empty:
            with st.spinner("Generating crime predictions and recommendations..."):
                # Load regional data for transfer learning
                predictor.load_regional_data()
                
                # Identify risk amplifiers
                risk_amplifiers = predictor.identify_risk_amplifiers(crime_data)
                with open('risk_amplifiers.json', 'w') as f:
                    json.dump(risk_amplifiers, f, indent=4)
                
                # Predict crime hotspots
                hotspots = predictor.predict_hotspots(crime_data)
                with open('crime_hotspots.json', 'w') as f:
                    json.dump(hotspots, f, indent=4)
                
                # Generate intervention recommendations
                recommendations = intervention_engine.generate_recommendations(hotspots, risk_amplifiers)
                with open('intervention_recommendations.json', 'w') as f:
                    json.dump(recommendations, f, indent=4)
                
                # Save the models
                predictor.save_models()
                
                # Update session state
                st.session_state.hotspots = hotspots
                st.session_state.risk_amplifiers = risk_amplifiers
                st.session_state.recommendations = recommendations
                
                st.sidebar.success("Predictions and recommendations generated!")
        else:
            st.sidebar.error("No crime data available. Please fetch data first.")
    
    # Validate model button
    if st.sidebar.button("Validate Model Performance"):
        crime_data = load_crime_data()
        hotspots = load_crime_hotspots()
        
        if crime_data is not None and hotspots is not None:
            with st.spinner("Validating model predictions..."):
                valid = validator.validate_predictions(hotspots, crime_data)
                
                # Load updated validation metrics
                validation_metrics = load_validation_metrics()
                st.session_state.validation_metrics = validation_metrics
                
                if valid:
                    st.sidebar.success("Model validation passed!")
                else:
                    st.sidebar.warning("Model validation below threshold. Check feedback.")
        else:
            st.sidebar.error("Missing data for validation. Generate predictions first.")
    
    # Display last update time if available
    if st.session_state.last_update:
        st.sidebar.info(f"Last data update: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data if not in session state
    if st.session_state.crime_data is None:
        st.session_state.crime_data = load_crime_data()
    if st.session_state.hotspots is None:
        st.session_state.hotspots = load_crime_hotspots()
    if st.session_state.risk_amplifiers is None:
        st.session_state.risk_amplifiers = load_risk_amplifiers()
    if st.session_state.recommendations is None:
        st.session_state.recommendations = load_intervention_recommendations()
    if st.session_state.validation_metrics is None:
        st.session_state.validation_metrics = load_validation_metrics()
    
    # Get data from session state
    crime_data = st.session_state.crime_data
    hotspots = st.session_state.hotspots
    risk_amplifiers = st.session_state.risk_amplifiers
    recommendations = st.session_state.recommendations
    validation_metrics = st.session_state.validation_metrics
    
    # Also try to load crime analysis
    analysis = load_crime_analysis()
    
    # Display appropriate page content
    if page == "Dashboard Overview":
        # Display overview metrics
        if crime_data is not None and not crime_data.empty and analysis is not None:
            # Key metrics in a row
            st.markdown("<h2 class='sub-header'>Crime Overview</h2>", unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Incidents", analysis['total_incidents'])
            
            with col2:
                top_category = max(analysis['by_category'].items(), key=lambda x: x[1])
                st.metric("Most Common Crime", f"{top_category[0]} ({top_category[1]})")
            
            with col3:
                top_location = max(analysis['top_locations'].items(), key=lambda x: x[1])
                st.metric("Most Affected Location", f"{top_location[0]} ({top_location[1]})")
            
            with col4:
                if hotspots:
                    num_high_risk = sum(1 for h in hotspots.values() 
                                      if h.get('prediction', {}).get('risk_score', 0) >= 0.7)
                    st.metric("High Risk Hotspots", num_high_risk)
                else:
                    st.metric("High Risk Hotspots", "N/A")
            
            # Crime distribution by category
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Crime Distribution by Category")
                fig_category = px.pie(
                    values=list(analysis['by_category'].values()),
                    names=list(analysis['by_category'].keys()),
                    title="Crime Categories Distribution",
                    hole=0.4
                )
                st.plotly_chart(fig_category, use_container_width=True)
            
            with col2:
                st.subheader("Crime Distribution by Location")
                fig_location = px.bar(
                    x=list(analysis['top_locations'].keys()),
                    y=list(analysis['top_locations'].values()),
                    title="Crime Incidents by Location",
                    labels={'x': 'Location', 'y': 'Number of Incidents'},
                    color=list(analysis['top_locations'].values()),
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_location, use_container_width=True)
            
            # Time-based analysis
            st.subheader("Temporal Crime Patterns")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Hourly Distribution**")
                fig_hourly = px.bar(
                    x=list(analysis['hourly_distribution'].keys()),
                    y=list(analysis['hourly_distribution'].values()),
                    title="Crime Incidents by Hour of Day",
                    labels={'x': 'Hour', 'y': 'Number of Incidents'}
                )
                st.plotly_chart(fig_hourly, use_container_width=True)
            
            with col2:
                st.write("**Daily Distribution**")
                # Define order of days
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                
                # Create dataframe for daily distribution
                daily_data = pd.DataFrame({
                    'Day': list(analysis['daily_distribution'].keys()),
                    'Count': list(analysis['daily_distribution'].values())
                })
                
                # Reorder days
                daily_data['Day'] = pd.Categorical(daily_data['Day'], categories=day_order, ordered=True)
                daily_data = daily_data.sort_values('Day')
                
                fig_daily = px.bar(
                    daily_data,
                    x='Day',
                    y='Count',
                    title="Crime Incidents by Day of Week",
                    labels={'Count': 'Number of Incidents'},
                    color='Count',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_daily, use_container_width=True)
            
            # Display crime data table
            with st.expander("View Raw Crime Data"):
                st.dataframe(crime_data)
        else:
            st.warning("No crime data available. Please fetch data first.")
    
    elif page == "Crime Hotspot Map":
        st.markdown("<h2 class='sub-header'>Crime Hotspot Map</h2>", unsafe_allow_html=True)
        
        if crime_data is not None and not crime_data.empty:
            # Create map
            hotspot_map = create_hotspot_map(crime_data, hotspots)
            if hotspot_map:
                st.write("This map shows both historical crime incidents and predicted crime hotspots. Use the layer control in the top right to toggle different views.")
                folium_static(hotspot_map, width=1200, height=600)
                
                # Display hotspot details if available
                if hotspots:
                    st.subheader("Predicted Hotspot Details")
                    
                    # Create a dataframe for hotspots
                    hotspot_data = []
                    for hotspot_id, data in hotspots.items():
                        risk_score = data.get('prediction', {}).get('risk_score', 0)
                        location = data.get('location', 'Unknown')
                        
                        # Determine primary crime type
                        crime_types = data.get('prediction', {}).get('crime_types', {})
                        primary_crime = max(crime_types.items(), key=lambda x: x[1])[0] if crime_types else 'Unknown'
                        
                        # Determine risk level
                        if risk_score >= 0.7:
                            risk_level = 'High'
                        elif risk_score >= 0.4:
                            risk_level = 'Medium'
                        else:
                            risk_level = 'Low'
                        
                        hotspot_data.append({
                            'Hotspot ID': hotspot_id,
                            'Location': location,
                            'Risk Score': risk_score,
                            'Risk Level': risk_level,
                            'Primary Crime Type': primary_crime
                        })
                    
                    # Create dataframe and display
                    df_hotspots = pd.DataFrame(hotspot_data)
                    
                    # Add styling
                    def highlight_risk(val):
                        if val == 'High':
                            return 'background-color: #FFCCCC; color: #DC2626; font-weight: bold'
                        elif val == 'Medium':
                            return 'background-color: #FFF3CD; color: #F59E0B; font-weight: bold'
                        elif val == 'Low':
                            return 'background-color: #D1FAE5; color: #10B981; font-weight: bold'
                        return ''
                    
                    # Sort by risk score
                    df_hotspots = df_hotspots.sort_values('Risk Score', ascending=False)
                    
                    # Display styled dataframe
                    st.dataframe(df_hotspots.style.applymap(highlight_risk, subset=['Risk Level']))
            else:
                st.error("Failed to create map. Check if crime data contains valid coordinates.")
        else:
            st.warning("No crime data available. Please fetch data first.")
    
    elif page == "Risk Amplifier Analysis":
        st.markdown("<h2 class='sub-header'>Risk Amplifier Analysis</h2>", unsafe_allow_html=True)
        
        if risk_amplifiers:
            st.write("Risk amplifiers are factors that increase the probability of crime in specific areas or time periods.")
            
            # Visualize risk amplifiers
            visualize_risk_amplifiers(risk_amplifiers)
            
            # Display raw risk amplifier data
            with st.expander("View Raw Risk Amplifier Data"):
                st.json(risk_amplifiers)
        else:
            st.warning("No risk amplifier data available. Generate predictions first.")
    
    elif page == "Intervention Recommendations":
        st.markdown("<h2 class='sub-header'>Intervention Recommendations</h2>", unsafe_allow_html=True)
        
        if recommendations:
            st.write("Based on predicted crime hotspots and risk factors, the system recommends the following interventions:")
            
            # Display recommendations
            display_recommendations(recommendations)
            
            # Display raw recommendations data
            with st.expander("View Raw Recommendations Data"):
                st.json(recommendations)
        else:
            st.warning("No intervention recommendations available. Generate predictions first.")
    
    elif page == "Model Performance":
        st.markdown("<h2 class='sub-header'>Model Performance Metrics</h2>", unsafe_allow_html=True)
        
        if validation_metrics:
            st.write("These metrics show how well the prediction model is performing and how effective the recommended interventions are.")
            
            # Display validation metrics
            display_validation_metrics(validation_metrics)
            
            # Display raw validation data
            with st.expander("View Raw Validation Data"):
                st.json(validation_metrics)
        else:
            st.warning("No validation metrics available. Validate model performance first.")

# Run the main dashboard function
if __name__ == "__main__":
    main()
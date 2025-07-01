import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import folium
from streamlit_folium import folium_static
import os
from real_time_crime_extractor import RealTimeCrimeDataExtractor
from crime_predictor import CrimePredictor
from intervention_engine import InterventionEngine
from feedback_validator import FeedbackValidator

def load_data():
    """Load crime data from CSV file"""
    try:
        return pd.read_csv('crime_data.csv')
    except FileNotFoundError:
        return None

def load_analysis():
    """Load crime analysis from JSON file"""
    try:
        with open('crime_analysis.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def load_hotspots():
    """Load predicted crime hotspots"""
    try:
        with open('crime_hotspots.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def load_risk_amplifiers():
    """Load risk amplifiers data"""
    try:
        with open('risk_amplifiers.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
        
def load_recommendations():
    """Load intervention recommendations"""
    try:
        with open('intervention_recommendations.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def load_validation_metrics():
    """Load model validation metrics"""
    try:
        with open('validation_metrics.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def create_dashboard():
    st.set_page_config(page_title="Advanced Crime Analysis Dashboard", layout="wide")
    st.title("🌎 Predictive Crime Analysis Dashboard")
    
    # Initialize components
    predictor = CrimePredictor()
    intervention_engine = InterventionEngine()
    validator = FeedbackValidator()
    
    # Load data
    crime_data = load_data()
    analysis = load_analysis()
    hotspots = load_hotspots()
    risk_amplifiers = load_risk_amplifiers()
    recommendations = load_recommendations()
    
    # Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", [
        "Dashboard Overview", 
        "Crime Prediction", 
        "Risk Amplifiers", 
        "Intervention Recommendations",
        "Feedback Validation"
    ])
    
    # Sidebar controls
    st.sidebar.header("Data Controls")
    
    if st.sidebar.button("Refresh Crime Data"):
        with st.spinner("Fetching new crime data..."):
            api_key = st.secrets["news_api_key"] if "news_api_key" in st.secrets else "2a6116afec2140c6b9e6c073cf5950cd"
            extractor = RealTimeCrimeDataExtractor(api_key)
            crime_data = extractor.fetch_real_time_crime_data(days_back=7)
            if not crime_data.empty:
                crime_data.to_csv('crime_data.csv', index=False)
                
                # Generate and save crime analysis
                analysis = extractor.analyze_crime_data(crime_data)
                with open('crime_analysis.json', 'w') as f:
                    json.dump(analysis, f, indent=4)
                    
                st.success("Data refreshed successfully!")
            else:
                st.error("No new data found")
                
    # Run prediction models and generate recommendations
    if crime_data is not None and st.sidebar.button("Generate Predictions"):
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
            
            st.success("Predictions and recommendations generated!")
    
    if crime_data is not None and st.sidebar.button("Validate Model Performance"):
        with st.spinner("Validating model predictions..."):
            if hotspots:
                valid = validator.validate_predictions(hotspots, crime_data)
                if valid:
                    st.sidebar.success("Model validation passed!")
                else:
                    st.sidebar.warning("Model validation below threshold. Check feedback.")
                    
                # Generate improvement recommendations
                recommendations = validator.get_improvement_recommendations()
                
                # Visualize trends
                validator.visualize_feedback_trends()
    
    # Display appropriate content based on selected page
    if page == "Dashboard Overview":
        display_dashboard_overview(crime_data, analysis)
    elif page == "Crime Prediction":
        display_crime_prediction(crime_data, hotspots, predictor)
    elif page == "Risk Amplifiers":
        display_risk_amplifiers(risk_amplifiers)
    elif page == "Intervention Recommendations":
        display_intervention_recommendations(hotspots, recommendations, intervention_engine)
    elif page == "Feedback Validation":
        display_feedback_validation(validator)
    
def display_dashboard_overview(crime_data, analysis):
    """Display main dashboard overview"""
    st.header("Crime Analysis Overview")
    
    if crime_data is None or analysis is None:
        st.warning("No data available. Please refresh the data first.")
        return
    
    # Key metrics in a row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Incidents", analysis['total_incidents'])
    
    with col2:
        top_category = max(analysis['by_category'].items(), key=lambda x: x[1])
        st.metric("Most Common Crime", f"{top_category[0]} ({top_category[1]})")
    
    with col3:
        top_location = max(analysis['top_locations'].items(), key=lambda x: x[1])
        st.metric("Most Affected Location", f"{top_location[0]} ({top_location[1]})")

    # Crime Distribution by Category
    st.subheader("Crime Distribution by Category")
    fig_category = px.pie(
        values=list(analysis['by_category'].values()),
        names=list(analysis['by_category'].keys()),
        title="Crime Categories Distribution"
    )
    st.plotly_chart(fig_category, use_container_width=True)

    # Time-based Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Hourly Crime Distribution")
        fig_hourly = px.bar(
            x=list(analysis['hourly_distribution'].keys()),
            y=list(analysis['hourly_distribution'].values()),
            title="Crime Incidents by Hour",
            labels={"x": "Hour of Day", "y": "Number of Incidents"}
        )
        st.plotly_chart(fig_hourly, use_container_width=True)
    
    with col2:
        st.subheader("Daily Crime Distribution")
        fig_daily = px.bar(
            x=list(analysis['daily_distribution'].keys()),
            y=list(analysis['daily_distribution'].values()),
            title="Crime Incidents by Day",
            labels={"x": "Day of Week", "y": "Number of Incidents"}
        )
        st.plotly_chart(fig_daily, use_container_width=True)

    # Top Locations
    st.subheader("Top 10 Affected Locations")
    fig_locations = px.bar(
        x=list(analysis['top_locations'].keys()),
        y=list(analysis['top_locations'].values()),
        title="Crime Incidents by Location",
        labels={"x": "Location", "y": "Number of Incidents"}
    )
    st.plotly_chart(fig_locations, use_container_width=True)

    # Detailed Data Table (collapsible)
    with st.expander("View Detailed Crime Data"):
        st.dataframe(crime_data)

def display_crime_prediction(crime_data, hotspots, predictor):
    """Display crime prediction page"""
    st.header("Crime Prediction System")
    
    if crime_data is None:
        st.warning("No crime data available. Please refresh the data first.")
        return
        
    if hotspots is None:
        st.info("No predictions available. Please generate predictions first.")
        return
    
    # Display hotspot map
    st.subheader("🔥 Predicted Crime Hotspots")
    
    # Convert hotspots to map
    if hotspots:
        lats = []
        lons = []
        weights = []
        info = []
        
        for hotspot_id, data in hotspots.items():
            coords = data.get('coordinates', {})
            if 'lat' in coords and 'lon' in coords:
                lats.append(coords['lat'])
                lons.append(coords['lon'])
                weights.append(data.get('prediction', {}).get('risk_score', 0.5) * 100)
                
                # Create popup info
                location = data.get('location', 'Unknown')
                risk_score = data.get('prediction', {}).get('risk_score', 0)
                crime_types = data.get('prediction', {}).get('crime_types', {})
                
                crime_type_info = ""
                for crime_type, prob in crime_types.items():
                    crime_type_info += f"- {crime_type}: {prob*100:.1f}%<br>"
                
                info.append(f"""
                <b>Location:</b> {location}<br>
                <b>Risk Score:</b> {risk_score*100:.1f}%<br>
                <b>Predicted Crime Types:</b><br>
                {crime_type_info}
                """)
        
        if lats and lons:
            # Create center point
            center_lat = sum(lats) / len(lats)
            center_lon = sum(lons) / len(lons)
            
            # Create map
            m = folium.Map(location=[center_lat, center_lon], zoom_start=6)
            
            # Add heatmap
            from folium.plugins import HeatMap
            HeatMap(list(zip(lats, lons, weights))).add_to(m)
            
            # Add markers
            for i in range(len(lats)):
                folium.Marker(
                    location=[lats[i], lons[i]],
                    popup=folium.Popup(info[i], max_width=300),
                    icon=folium.Icon(color='red', icon='info-sign')
                ).add_to(m)
            
            # Display map
            folium_static(m)
    
    # Prediction details
    with st.expander("View Detailed Prediction Data"):
        if hotspots:
            for hotspot_id, data in hotspots.items():
                st.write(f"**Hotspot ID:** {hotspot_id}")
                st.write(f"**Location:** {data.get('location', 'Unknown')}")
                st.write(f"**Risk Score:** {data.get('prediction', {}).get('risk_score', 0)*100:.1f}%")
                st.write("**Predicted Crime Types:**")
                for crime_type, prob in data.get('prediction', {}).get('crime_types', {}).items():
                    st.write(f"- {crime_type}: {prob*100:.1f}%")
                st.write("---")
    
    # Transfer learning configuration
    st.subheader("🔄 Transfer Learning Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Configure similar regions for transfer learning:")
        target_region = st.text_input("Target Region", "Current Data")
        similar_regions = st.text_area("Similar Regions (one per line)", 
                                    "New York\nChicago\nLos Angeles")
        weights = st.text_area("Weights (one per line, between 0-1)", 
                            "0.8\n0.6\n0.5")
    
    with col2:
        st.write("Current Regional Similarities:")
        try:
            if os.path.exists('similar_regions.json'):
                with open('similar_regions.json', 'r') as f:
                    region_data = json.load(f)
                    for region, data in region_data.items():
                        st.write(f"**{region}**")
                        st.write(f"Similar regions: {', '.join(data['similar_regions'])}")
                        st.write(f"Weights: {', '.join([str(w) for w in data['weights']])}")
                        st.write("---")
        except Exception as e:
            st.error(f"Error loading region data: {str(e)}")
    
    if st.button("Update Region Similarities"):
        try:
            # Parse input
            similar_region_list = [r.strip() for r in similar_regions.split('\n') if r.strip()]
            weight_list = [float(w.strip()) for w in weights.split('\n') if w.strip()]
            
            # Validate
            if len(similar_region_list) != len(weight_list):
                st.error("Number of regions and weights must match")
            else:
                # Update region similarity
                predictor.update_region_similarity(target_region, similar_region_list, weight_list)
                st.success(f"Updated similarity data for {target_region}")
        except Exception as e:
            st.error(f"Error updating region similarities: {str(e)}")

def display_risk_amplifiers(risk_amplifiers):
    """Display risk amplifiers page"""
    st.header("Risk Amplification Analysis")
    
    if risk_amplifiers is None:
        st.warning("No risk amplifier data available. Please generate predictions first.")
        return
    
    # Time-based risk patterns
    st.subheader("⏰ Time-Based Risk Patterns")
    
    if 'time_patterns' in risk_amplifiers:
        col1, col2 = st.columns(2)
        
        with col1:
            # Hourly risk patterns
            if 'hourly' in risk_amplifiers['time_patterns']:
                hourly_data = risk_amplifiers['time_patterns']['hourly']
                fig_hourly = px.bar(
                    x=list(hourly_data.keys()),
                    y=list(hourly_data.values()),
                    title="Crime Risk by Hour",
                    labels={"x": "Hour of Day", "y": "Risk Factor"}
                )
                fig_hourly.update_layout(yaxis_range=[0, 1])
                st.plotly_chart(fig_hourly, use_container_width=True)
                
                # Identify peak hours
                peak_hours = [hour for hour, risk in hourly_data.items() if float(risk) > 0.7]
                if peak_hours:
                    st.info(f"🚨 Peak risk hours: {', '.join(peak_hours)}")
        
        with col2:
            # Daily risk patterns
            if 'daily' in risk_amplifiers['time_patterns']:
                daily_data = risk_amplifiers['time_patterns']['daily']
                fig_daily = px.bar(
                    x=list(daily_data.keys()),
                    y=list(daily_data.values()),
                    title="Crime Risk by Day",
                    labels={"x": "Day of Week", "y": "Risk Factor"}
                )
                fig_daily.update_layout(yaxis_range=[0, 1])
                st.plotly_chart(fig_daily, use_container_width=True)
                
                # Identify peak days
                peak_days = [day for day, risk in daily_data.items() if float(risk) > 0.7]
                if peak_days:
                    st.info(f"🚨 Peak risk days: {', '.join(peak_days)}")
    
    # Location-based hotspots
    st.subheader("📍 Location Risk Hotspots")
    
    if 'location_hotspots' in risk_amplifiers:
        hotspot_data = risk_amplifiers['location_hotspots']
        
        # Create a map of hotspots
        if hotspot_data:
            locations = []
            risk_scores = []
            names = []
            
            for location_name, data in hotspot_data.items():
                if 'lat' in data and 'lon' in data:
                    locations.append([data['lat'], data['lon']])
                    risk_scores.append(data['risk_score'] * 100)
                    names.append(location_name)
            
            if locations:
                center_lat = sum(loc[0] for loc in locations) / len(locations)
                center_lon = sum(loc[1] for loc in locations) / len(locations)
                
                m = folium.Map(location=[center_lat, center_lon], zoom_start=6)
                
                for i, loc in enumerate(locations):
                    radius = max(5, risk_scores[i] / 5)
                    folium.CircleMarker(
                        location=loc,
                        radius=radius,
                        color='red',
                        fill=True,
                        fill_color='red',
                        fill_opacity=0.7,
                        popup=f"{names[i]}: Risk {risk_scores[i]:.1f}%"
                    ).add_to(m)
                
                folium_static(m)
                
                # Show top risk locations
                st.subheader("Top High-Risk Locations")
                
                # Convert to dataframe for display
                risk_df = pd.DataFrame([
                    {
                        "Location": location,
                        "Risk Score": data['risk_score'] * 100,
                        "Incident Count": data.get('incident_count', 0)
                    }
                    for location, data in hotspot_data.items()
                ])
                
                if not risk_df.empty:
                    risk_df = risk_df.sort_values("Risk Score", ascending=False).reset_index(drop=True)
                    st.dataframe(risk_df)
    
    # Crime category correlations
    st.subheader("🔍 Crime Category Correlations")
    
    if 'category_correlations' in risk_amplifiers:
        category_data = risk_amplifiers['category_correlations']
        
        if category_data:
            fig_categories = px.pie(
                values=list(category_data.values()),
                names=list(category_data.keys()),
                title="Crime Categories Risk Distribution"
            )
            st.plotly_chart(fig_categories, use_container_width=True)
            
            # Show in table form as well
            st.write("Risk factor by crime category:")
            cat_df = pd.DataFrame([
                {"Category": cat, "Risk Factor": risk * 100}
                for cat, risk in category_data.items()
            ]).sort_values("Risk Factor", ascending=False).reset_index(drop=True)
            
            st.dataframe(cat_df)
    
    # Environmental factors
    if 'environmental_factors' in risk_amplifiers and risk_amplifiers['environmental_factors']:
        st.subheader("🌍 Environmental Risk Factors")
        
        env_factors = risk_amplifiers['environmental_factors']
        
        for factor, value in env_factors.items():
            st.write(f"**{factor}:** {value}")

def display_intervention_recommendations(hotspots, recommendations, intervention_engine):
    """Display intervention recommendations page"""
    st.header("Intervention Recommendation Engine")
    
    if hotspots is None or recommendations is None:
        st.warning("No recommendations available. Please generate predictions first.")
        return
    
    # Display intervention map
    st.subheader("🗺️ Intervention Priority Map")
    
    intervention_map = intervention_engine.generate_intervention_map(hotspots, recommendations)
    if intervention_map:
        folium_static(intervention_map)
    
    # Display recommendations by location
    st.subheader("📋 Recommended Interventions by Location")
    
    for hotspot_id, rec_data in recommendations.items():
        location = rec_data.get('location', 'Unknown')
        risk_level = rec_data.get('risk_level', 'medium')
        crime_type = rec_data.get('primary_crime_type', 'Unknown')
        
        # Color based on risk level
        if risk_level == 'high':
            risk_color = "🔴"
        elif risk_level == 'medium':
            risk_color = "🟠"
        else:
            risk_color = "🟢"
        
        # Create an expander for each location
        with st.expander(f"{risk_color} {location} ({crime_type.capitalize()})"):
            st.write(f"**Risk Level:** {risk_level.upper()}")
            st.write(f"**Primary Crime Type:** {crime_type}")
            
            # Display interventions
            st.write("### Recommended Interventions")
            
            for intervention in rec_data.get('interventions', []):
                int_type = intervention.get('type', 'Unknown')
                priority = intervention.get('priority', 'medium')
                effectiveness = intervention.get('effectiveness', 0) * 100
                
                # Priority indicator
                if priority == 'high':
                    priority_indicator = "🔴 HIGH"
                elif priority == 'medium':
                    priority_indicator = "🟠 MEDIUM"
                else:
                    priority_indicator = "🟢 LOW"
                
                st.write(f"#### {int_type.replace('_', ' ').title()} ({priority_indicator})")
                st.write(f"**Effectiveness:** {effectiveness:.1f}%")
                st.write(f"**Description:** {intervention.get('description', '')}")
                
                # Display specific actions
                st.write("**Specific Actions:**")
                for action in intervention.get('specific_actions', []):
                    st.write(f"- {action}")
    
    # Feedback tracking
    st.subheader("📊 Intervention Feedback Tracking")
    
    with st.form("intervention_feedback"):
        st.write("Track intervention effectiveness")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Select hotspot and intervention
            hotspot_options = [rec_data.get('location', hotspot_id) for hotspot_id, rec_data in recommendations.items()]
            selected_hotspot = st.selectbox("Select Location", hotspot_options)
            
            # Find the hotspot_id for the selected location
            selected_hotspot_id = None
            for hotspot_id, rec_data in recommendations.items():
                if rec_data.get('location', hotspot_id) == selected_hotspot:
                    selected_hotspot_id = hotspot_id
                    break
            
            if selected_hotspot_id:
                intervention_options = [int_data.get('type', 'unknown') 
                                       for int_data in recommendations[selected_hotspot_id].get('interventions', [])]
                selected_intervention = st.selectbox("Select Intervention", intervention_options)
            else:
                selected_intervention = None
        
        with col2:
            # Implementation details
            implementation_date = st.date_input("Implementation Date", datetime.now())
            before_incidents = st.number_input("Incidents Before Implementation", min_value=0, value=10)
            after_incidents = st.number_input("Incidents After Implementation", min_value=0, value=5)
        
        # Additional notes
        notes = st.text_area("Additional Notes", "")
        
        # Submit button
        submit = st.form_submit_button("Record Feedback")
        
        if submit and selected_hotspot_id and selected_intervention:
            # Create feedback data
            follow_up_data = {
                'before_incidents': before_incidents,
                'after_incidents': after_incidents,
                'notes': notes
            }
            
            # Record the feedback
            intervention_engine.track_intervention_effectiveness(
                selected_hotspot_id,
                selected_intervention,
                implementation_date,
                follow_up_data
            )
            
            st.success("Feedback recorded successfully!")

def display_feedback_validation(validator):
    """Display feedback validation page"""
    st.header("Feedback Loop Validation System")
    
    # Check if we have validation metrics
    validation_metrics = load_validation_metrics()
    if validation_metrics is None:
        st.info("No validation metrics available yet. Please validate model performance first.")
        return
    
    # Display overall accuracy trend
    st.subheader("📈 Prediction Accuracy Trend")
    
    if 'prediction_accuracy' in validation_metrics and validation_metrics['prediction_accuracy']:
        accuracy_data = validation_metrics['prediction_accuracy']
        
        fig = px.line(
            x=list(range(len(accuracy_data))),
            y=accuracy_data,
            markers=True,
            title="Prediction Accuracy Over Time",
            labels={"x": "Validation Round", "y": "Accuracy Score"}
        )
        
        # Add threshold line
        fig.add_hline(y=0.7, line_dash="dash", line_color="red", 
                     annotation_text="Minimum Threshold", annotation_position="bottom right")
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Display intervention effectiveness
    st.subheader("🎯 Intervention Effectiveness")
    
    if 'intervention_effectiveness' in validation_metrics and validation_metrics['intervention_effectiveness']:
        int_data = validation_metrics['intervention_effectiveness']
        
        # Convert to dataframe for visualization
        rows = []
        for int_type, values in int_data.items():
            for value in values:
                rows.append({
                    'Intervention Type': int_type,
                    'Effectiveness': value
                })
        
        if rows:
            df = pd.DataFrame(rows)
            
            # Calculate average effectiveness by type
            avg_df = df.groupby('Intervention Type')['Effectiveness'].mean().reset_index()
            avg_df['Effectiveness'] = avg_df['Effectiveness'] * 100  # Convert to percentage
            
            fig = px.bar(
                avg_df,
                x='Intervention Type',
                y='Effectiveness',
                title="Average Intervention Effectiveness",
                labels={"Effectiveness": "Effectiveness (%)"}
            )
            
            # Add threshold line
            fig.add_hline(y=30, line_dash="dash", line_color="red", 
                         annotation_text="Minimum Threshold", annotation_position="bottom right")
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Improvement recommendations
    st.subheader("💡 System Improvement Recommendations")
    
    recommendations = validator.get_improvement_recommendations()
    
    if recommendations:
        # Model improvements
        if recommendations.get('model_improvements', []):
            st.write("#### Model Improvement Suggestions")
            for rec in recommendations['model_improvements']:
                st.write(f"- {rec}")
        
        # Intervention adjustments
        if recommendations.get('intervention_adjustments', []):
            st.write("#### Intervention Adjustment Suggestions")
            for rec in recommendations['intervention_adjustments']:
                st.write(f"- {rec}")
    else:
        st.info("No improvement recommendations available at this time.")
    
    # Manual validation metrics entry
    st.subheader("✅ Manual Validation")
    
    with st.form("manual_validation"):
        st.write("Add manual validation metrics for model performance:")
        
        model_name = st.text_input("Model Name", "crime_predictor")
        accuracy = st.slider("Accuracy", 0.0, 1.0, 0.75)
        precision = st.slider("Precision", 0.0, 1.0, 0.70)
        recall = st.slider("Recall", 0.0, 1.0, 0.65)
        
        submit = st.form_submit_button("Add Validation Data")
        
        if submit:
            performance_metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall
            }
            
            validator.validate_model_performance(model_name, performance_metrics)
            st.success("Validation metrics added successfully!")
            
            # Visualize updated metrics
            validator.visualize_feedback_trends()
            
            # Show generated visualization files
            if os.path.exists('prediction_accuracy_trend.png'):
                st.image('prediction_accuracy_trend.png')
            
            if os.path.exists('intervention_effectiveness.png'):
                st.image('intervention_effectiveness.png')
                
            if os.path.exists('false_positive_trend.png'):
                st.image('false_positive_trend.png')

if __name__ == "__main__":
    create_dashboard() 
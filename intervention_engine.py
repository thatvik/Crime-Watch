import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import folium
from folium import plugins
from folium.plugins import MarkerCluster

class InterventionEngine:
    def __init__(self):
        """Initialize the intervention recommendation engine"""
        # Define intervention types and their base effectiveness
        self.intervention_types = {
            'police_patrol': {
                'description': 'Strategic police patrol deployment',
                'base_effectiveness': 0.75
            },
            'community_outreach': {
                'description': 'Community engagement and education programs',
                'base_effectiveness': 0.65
            },
            'environmental_modification': {
                'description': 'Physical environment improvements (lighting, cameras, etc.)',
                'base_effectiveness': 0.70
            },
            'public_awareness': {
                'description': 'Targeted public awareness campaigns',
                'base_effectiveness': 0.60
            },
            'targeted_enforcement': {
                'description': 'Focused enforcement on specific crime types',
                'base_effectiveness': 0.80
            }
        }
        
        # Load intervention effectiveness tracking data
        self.load_effectiveness_data()
        
    def load_effectiveness_data(self):
        """Load historical intervention effectiveness data"""
        try:
            if os.path.exists('intervention_effectiveness.json'):
                with open('intervention_effectiveness.json', 'r') as f:
                    self.effectiveness_data = json.load(f)
            else:
                self.effectiveness_data = {}
        except Exception as e:
            print(f"Error loading effectiveness data: {str(e)}")
            self.effectiveness_data = {}
    
    def generate_recommendations(self, hotspots, risk_amplifiers):
        """Generate intervention recommendations based on predictions and risk factors"""
        recommendations = {}
        
        for hotspot_id, data in hotspots.items():
            location = data.get('location', 'Unknown')
            risk_score = data.get('prediction', {}).get('risk_score', 0)
            
            # Determine risk level
            if risk_score >= 0.7:
                risk_level = 'high'
            elif risk_score >= 0.4:
                risk_level = 'medium'
            else:
                risk_level = 'low'
                
            # Determine primary crime type
            crime_types = data.get('prediction', {}).get('crime_types', {})
            primary_crime_type = max(crime_types.items(), key=lambda x: x[1])[0] if crime_types else 'unknown'
            
            # Generate appropriate interventions
            interventions = self._select_interventions(risk_level, primary_crime_type, risk_amplifiers)
            
            # Create recommendation
            recommendations[hotspot_id] = {
                'location': location,
                'risk_level': risk_level,
                'primary_crime_type': primary_crime_type,
                'interventions': interventions
            }
            
        return recommendations
        
    def _select_interventions(self, risk_level, crime_type, risk_amplifiers):
        """Select appropriate interventions based on risk level and crime type"""
        interventions = []
        
        # Select intervention types based on crime type
        if crime_type == 'theft':
            interventions.append(self._create_intervention('police_patrol', risk_level, 
                ['Increase visible patrol during peak hours', 
                 'Deploy plainclothes officers in high-risk areas',
                 'Establish police substations in hotspot areas']))
                 
            interventions.append(self._create_intervention('environmental_modification', risk_level,
                ['Improve lighting in vulnerable areas',
                 'Install security cameras at key locations',
                 'Use crime prevention through environmental design (CPTED) principles']))
                 
        elif crime_type == 'assault' or crime_type == 'violence':
            interventions.append(self._create_intervention('police_patrol', risk_level, 
                ['Deploy officers during nightlife hours',
                 'Focus patrol near bars and entertainment venues',
                 'Implement rapid response protocols for violent incidents']))
                 
            interventions.append(self._create_intervention('community_outreach', risk_level,
                ['Establish violence prevention programs',
                 'Conduct conflict resolution workshops',
                 'Partner with local businesses for safety initiatives']))
                 
        elif crime_type == 'burglary':
            interventions.append(self._create_intervention('environmental_modification', risk_level,
                ['Improve property security measures',
                 'Install neighborhood watch signage',
                 'Address abandoned properties and urban blight']))
                 
            interventions.append(self._create_intervention('public_awareness', risk_level,
                ['Distribute home security checklists',
                 'Hold community meetings on burglary prevention',
                 'Create a neighborhood alert system']))
                 
        else:
            # Default interventions for other crime types
            interventions.append(self._create_intervention('police_patrol', risk_level,
                ['Increase patrol frequency in hotspot areas',
                 'Coordinate with neighboring jurisdictions',
                 'Use data-driven deployment strategies']))
                 
            interventions.append(self._create_intervention('community_outreach', risk_level,
                ['Hold regular community safety meetings',
                 'Create neighborhood watch programs',
                 'Implement anonymous tip reporting systems']))
        
        # Add environmental modification based on time patterns if available
        if 'time_patterns' in risk_amplifiers:
            time_strategies = []
            
            if 'hourly' in risk_amplifiers['time_patterns']:
                # Find peak hours
                hourly = risk_amplifiers['time_patterns']['hourly']
                peak_hours = [hour for hour, risk in hourly.items() if float(risk) > 0.7]
                
                if peak_hours:
                    peak_hour_str = ', '.join(peak_hours)
                    time_strategies.append(f"Focus resources during peak hours: {peak_hour_str}")
            
            if 'daily' in risk_amplifiers['time_patterns']:
                # Find peak days
                daily = risk_amplifiers['time_patterns']['daily']
                peak_days = [day for day, risk in daily.items() if float(risk) > 0.7]
                
                if peak_days:
                    peak_day_str = ', '.join(peak_days)
                    time_strategies.append(f"Increase coverage on high-risk days: {peak_day_str}")
            
            if time_strategies:
                interventions.append(self._create_intervention('targeted_enforcement', risk_level, time_strategies))
        
        return interventions
    
    def _create_intervention(self, int_type, risk_level, specific_actions):
        """Create an intervention recommendation"""
        # Get base data
        base_data = self.intervention_types.get(int_type, {
            'description': 'Custom intervention',
            'base_effectiveness': 0.5
        })
        
        # Calculate priority based on risk level
        if risk_level == 'high':
            priority = 'high'
            effectiveness_factor = 1.0
        elif risk_level == 'medium':
            priority = 'medium'
            effectiveness_factor = 0.8
        else:
            priority = 'low'
            effectiveness_factor = 0.6
            
        # Calculate adjusted effectiveness
        effectiveness = base_data['base_effectiveness'] * effectiveness_factor
        
        # Adjust based on historical effectiveness if available
        if int_type in self.effectiveness_data:
            hist_effectiveness = self.effectiveness_data[int_type]['recent_effectiveness']
            effectiveness = (effectiveness + hist_effectiveness) / 2
        
        return {
            'type': int_type,
            'description': base_data['description'],
            'priority': priority,
            'effectiveness': effectiveness,
            'specific_actions': specific_actions
        }
        
    def generate_intervention_map(self, hotspots, recommendations):
        """Generate a map visualization of intervention recommendations"""
        if not hotspots or not recommendations:
            return None
            
        # Create map centered on the average location
        lats = []
        lons = []
        
        for hotspot_id, data in hotspots.items():
            coords = data.get('coordinates', {})
            if 'lat' in coords and 'lon' in coords:
                lats.append(coords['lat'])
                lons.append(coords['lon'])
        
        if not lats or not lons:
            return None
            
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        
        # Add marker cluster
        marker_cluster = MarkerCluster().add_to(m)
        
        # Add markers for each hotspot
        for hotspot_id, data in hotspots.items():
            coords = data.get('coordinates', {})
            if 'lat' in coords and 'lon' in coords:
                location = [coords['lat'], coords['lon']]
                
                # Get recommendation data
                rec_data = recommendations.get(hotspot_id, {})
                risk_level = rec_data.get('risk_level', 'low')
                location_name = rec_data.get('location', data.get('location', 'Unknown Location'))
                
                # Format popup content
                popup_content = f"""
                <h4>{location_name}</h4>
                <p><b>Risk Level:</b> {risk_level.upper()}</p>
                <p><b>Primary Crime Type:</b> {rec_data.get('primary_crime_type', 'unknown')}</p>
                <h5>Key Interventions:</h5>
                <ul>
                """
                
                for intervention in rec_data.get('interventions', [])[:2]:  # Only show top 2
                    int_type = intervention.get('type', 'unknown').replace('_', ' ').title()
                    effectiveness = intervention.get('effectiveness', 0) * 100
                    popup_content += f"<li><b>{int_type}</b> (Effectiveness: {effectiveness:.1f}%)</li>"
                
                popup_content += "</ul>"
                
                # Set marker color based on risk level
                if risk_level == 'high':
                    color = 'red'
                elif risk_level == 'medium':
                    color = 'orange'
                else:
                    color = 'green'
                
                # Create marker
                folium.Marker(
                    location=location,
                    popup=folium.Popup(popup_content, max_width=300),
                    tooltip=f"{location_name} ({risk_level.capitalize()})",
                    icon=folium.Icon(color=color, icon='info-sign')
                ).add_to(marker_cluster)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 180px; height: 120px; 
                    border:2px solid grey; z-index:9999; font-size:14px;
                    background-color:white; padding: 10px">
          <p><i class="fa fa-circle" style="color:red"></i> High Risk</p>
          <p><i class="fa fa-circle" style="color:orange"></i> Medium Risk</p>
          <p><i class="fa fa-circle" style="color:green"></i> Low Risk</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        return m
    
    def track_intervention_effectiveness(self, hotspot_id, intervention_type, implementation_date, follow_up_data):
        """Track the effectiveness of implemented interventions"""
        try:
            # Convert implementation date to string if it's a datetime object
            if isinstance(implementation_date, datetime):
                implementation_date = implementation_date.strftime('%Y-%m-%d')
                
            # Calculate effectiveness
            before_incidents = follow_up_data.get('before_incidents', 0)
            after_incidents = follow_up_data.get('after_incidents', 0)
            
            if before_incidents > 0:
                effectiveness = max(0, 1 - (after_incidents / before_incidents))
            else:
                effectiveness = 0.5  # Default if no before data
                
            # Create tracking record
            tracking_record = {
                'hotspot_id': hotspot_id,
                'intervention_type': intervention_type,
                'implementation_date': implementation_date,
                'effectiveness': effectiveness,
                'before_incidents': before_incidents,
                'after_incidents': after_incidents,
                'notes': follow_up_data.get('notes', '')
            }
            
            # Load existing tracking data
            if os.path.exists('intervention_tracking.json'):
                with open('intervention_tracking.json', 'r') as f:
                    tracking_data = json.load(f)
            else:
                tracking_data = []
                
            # Add new record
            tracking_data.append(tracking_record)
            
            # Save updated tracking data
            with open('intervention_tracking.json', 'w') as f:
                json.dump(tracking_data, f, indent=4)
                
            # Update effectiveness data for this intervention type
            if intervention_type not in self.effectiveness_data:
                self.effectiveness_data[intervention_type] = {
                    'all_effectiveness': [],
                    'recent_effectiveness': 0
                }
                
            self.effectiveness_data[intervention_type]['all_effectiveness'].append(effectiveness)
            
            # Update recent effectiveness (average of last 5 implementations)
            all_values = self.effectiveness_data[intervention_type]['all_effectiveness']
            recent = all_values[-5:] if len(all_values) >= 5 else all_values
            self.effectiveness_data[intervention_type]['recent_effectiveness'] = sum(recent) / len(recent)
            
            # Save updated effectiveness data
            with open('intervention_effectiveness.json', 'w') as f:
                json.dump(self.effectiveness_data, f, indent=4)
                
            return True
            
        except Exception as e:
            print(f"Error tracking intervention effectiveness: {str(e)}")
            return False 
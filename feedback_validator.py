import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
import json
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

class FeedbackValidator:
    def __init__(self):
        """Initialize the feedback validation system"""
        self.validation_metrics = {
            'prediction_accuracy': [],
            'intervention_effectiveness': {},
            'model_performance': {},
            'feedback_history': []
        }
        self.validation_thresholds = {
            'prediction_accuracy': 0.7,  # Minimum acceptable accuracy
            'intervention_effectiveness': 0.3,  # Minimum acceptable effectiveness
            'false_positive_rate': 0.2   # Maximum acceptable false positive rate
        }
    
    def validate_predictions(self, predictions, actual_data):
        """Validate the accuracy of crime hotspot predictions"""
        if not predictions or not isinstance(actual_data, pd.DataFrame) or actual_data.empty:
            return False
            
        # Prepare validation results
        validation_result = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': {},
            'details': {}
        }
        
        # 1. Spatial accuracy validation
        spatial_accuracy = self._validate_spatial_accuracy(predictions, actual_data)
        validation_result['metrics']['spatial_accuracy'] = spatial_accuracy
        
        # 2. Crime type prediction accuracy
        type_accuracy = self._validate_crime_type_accuracy(predictions, actual_data)
        validation_result['metrics']['type_accuracy'] = type_accuracy
        
        # 3. Risk score correlation
        risk_correlation = self._validate_risk_score_correlation(predictions, actual_data)
        validation_result['metrics']['risk_correlation'] = risk_correlation
        
        # 4. False positive rate
        false_positive_rate = self._calculate_false_positive_rate(predictions, actual_data)
        validation_result['metrics']['false_positive_rate'] = false_positive_rate
        
        # Overall accuracy score (weighted combination of different metrics)
        overall_accuracy = (
            0.4 * spatial_accuracy +
            0.3 * type_accuracy +
            0.2 * risk_correlation +
            0.1 * (1 - false_positive_rate)  # Lower false positive is better
        )
        validation_result['metrics']['overall_accuracy'] = overall_accuracy
        
        # Add to history
        self.validation_metrics['prediction_accuracy'].append(overall_accuracy)
        self.validation_metrics['feedback_history'].append(validation_result)
        
        # Save validation metrics
        self._save_validation_metrics()
        
        return overall_accuracy > self.validation_thresholds['prediction_accuracy']
    
    def _validate_spatial_accuracy(self, predictions, actual_data):
        """Validate how accurately the model predicted crime locations"""
        if 'latitude' not in actual_data.columns or 'longitude' not in actual_data.columns:
            return 0.0
            
        # Extract predicted hotspot coordinates
        predicted_coords = []
        for hotspot_id, hotspot_data in predictions.items():
            coords = hotspot_data.get('coordinates', {})
            if 'lat' in coords and 'lon' in coords:
                predicted_coords.append((coords['lat'], coords['lon']))
        
        if not predicted_coords:
            return 0.0
            
        # Calculate distance from each actual crime to nearest predicted hotspot
        actual_coords = actual_data[['latitude', 'longitude']].values
        
        # For each actual crime, find the minimum distance to any predicted hotspot
        correct_predictions = 0
        threshold_distance = 0.05  # ~5km at equator
        
        for actual_lat, actual_lon in actual_coords:
            min_dist = float('inf')
            for pred_lat, pred_lon in predicted_coords:
                # Simple Euclidean distance (could be replaced with haversine for more accuracy)
                dist = np.sqrt((actual_lat - pred_lat)**2 + (actual_lon - pred_lon)**2)
                min_dist = min(min_dist, dist)
            
            # Count as correct if within threshold
            if min_dist < threshold_distance:
                correct_predictions += 1
        
        # Calculate spatial accuracy
        return correct_predictions / len(actual_coords) if len(actual_coords) > 0 else 0.0
    
    def _validate_crime_type_accuracy(self, predictions, actual_data):
        """Validate how accurately the model predicted crime types"""
        if 'category' not in actual_data.columns:
            return 0.0
            
        # Count actual crimes by type
        actual_types = actual_data['category'].value_counts().to_dict()
        
        # Count predicted crimes by type
        predicted_types = {}
        for hotspot_id, hotspot_data in predictions.items():
            crime_types = hotspot_data.get('prediction', {}).get('crime_types', {})
            for crime_type, likelihood in crime_types.items():
                if crime_type not in predicted_types:
                    predicted_types[crime_type] = 0
                predicted_types[crime_type] += likelihood  # Weight by likelihood
        
        # Normalize the predictions
        total_pred = sum(predicted_types.values())
        if total_pred > 0:
            for crime_type in predicted_types:
                predicted_types[crime_type] /= total_pred
                
        # Calculate accuracy using cosine similarity between distribution vectors
        all_types = set(list(actual_types.keys()) + list(predicted_types.keys()))
        
        actual_vec = np.array([actual_types.get(t, 0) for t in all_types])
        predicted_vec = np.array([predicted_types.get(t, 0) for t in all_types])
        
        # Normalize the vectors
        actual_norm = np.linalg.norm(actual_vec)
        predicted_norm = np.linalg.norm(predicted_vec)
        
        if actual_norm > 0 and predicted_norm > 0:
            similarity = np.dot(actual_vec, predicted_vec) / (actual_norm * predicted_norm)
            return float(similarity)
        
        return 0.0
    
    def _validate_risk_score_correlation(self, predictions, actual_data):
        """Validate correlation between predicted risk scores and actual crime frequency"""
        if 'latitude' not in actual_data.columns or 'longitude' not in actual_data.columns:
            return 0.0
            
        # Create grid cells
        lat_bin_size = 0.01  # About 1km
        lon_bin_size = 0.01
        
        # Create actual crime density grid
        actual_data['lat_bin'] = (actual_data['latitude'] / lat_bin_size).astype(int)
        actual_data['lon_bin'] = (actual_data['longitude'] / lon_bin_size).astype(int)
        
        # Count crimes in each grid cell
        actual_grid = actual_data.groupby(['lat_bin', 'lon_bin']).size().reset_index(name='count')
        
        # Create predicted risk grid
        predicted_grid = {}
        for hotspot_id, hotspot_data in predictions.items():
            coords = hotspot_data.get('coordinates', {})
            risk_score = hotspot_data.get('prediction', {}).get('risk_score', 0)
            
            if 'lat' in coords and 'lon' in coords:
                lat_bin = int(coords['lat'] / lat_bin_size)
                lon_bin = int(coords['lon'] / lon_bin_size)
                
                grid_key = (lat_bin, lon_bin)
                predicted_grid[grid_key] = risk_score
        
        # Merge actual and predicted data
        comparison_data = []
        for idx, row in actual_grid.iterrows():
            grid_key = (row['lat_bin'], row['lon_bin'])
            predicted_risk = predicted_grid.get(grid_key, 0)
            comparison_data.append({
                'actual_count': row['count'],
                'predicted_risk': predicted_risk
            })
        
        if not comparison_data:
            return 0.0
            
        # Calculate correlation
        df = pd.DataFrame(comparison_data)
        if len(df) < 2:
            return 0.0
            
        correlation = df['actual_count'].corr(df['predicted_risk'])
        return max(0, correlation)  # Return non-negative correlation
    
    def _calculate_false_positive_rate(self, predictions, actual_data):
        """Calculate the rate of false positive predictions"""
        if 'latitude' not in actual_data.columns or 'longitude' not in actual_data.columns:
            return 1.0  # Worst case
            
        # Create grid cells
        lat_bin_size = 0.01
        lon_bin_size = 0.01
        
        # Identify cells with actual crimes
        actual_data['lat_bin'] = (actual_data['latitude'] / lat_bin_size).astype(int)
        actual_data['lon_bin'] = (actual_data['longitude'] / lon_bin_size).astype(int)
        actual_cells = set(zip(actual_data['lat_bin'], actual_data['lon_bin']))
        
        # Identify cells with predicted crimes
        predicted_cells = set()
        for hotspot_id, hotspot_data in predictions.items():
            coords = hotspot_data.get('coordinates', {})
            if 'lat' in coords and 'lon' in coords:
                lat_bin = int(coords['lat'] / lat_bin_size)
                lon_bin = int(coords['lon'] / lon_bin_size)
                predicted_cells.add((lat_bin, lon_bin))
        
        if not predicted_cells:
            return 1.0
            
        # Calculate true positives and false positives
        true_positives = len(actual_cells.intersection(predicted_cells))
        false_positives = len(predicted_cells - actual_cells)
        
        # Calculate false positive rate
        if true_positives + false_positives > 0:
            false_positive_rate = false_positives / (true_positives + false_positives)
        else:
            false_positive_rate = 1.0
            
        return false_positive_rate
    
    def validate_intervention_effectiveness(self, intervention_data):
        """Validate the effectiveness of intervention strategies"""
        if not intervention_data:
            return {}
            
        # Prepare results dictionary
        effectiveness_results = {}
        
        for hotspot_id, interventions in intervention_data.items():
            hotspot_results = {}
            
            for intervention_id, intervention_details in interventions.items():
                intervention_type = intervention_details.get('type', 'unknown')
                follow_up_data = intervention_details.get('follow_up_data', {})
                
                # Calculate effectiveness if we have before/after data
                effectiveness = 0.0
                if 'before_incidents' in follow_up_data and 'after_incidents' in follow_up_data:
                    before = follow_up_data['before_incidents']
                    after = follow_up_data['after_incidents']
                    
                    if before > 0:
                        effectiveness = (before - after) / before
                        # Ensure valid range
                        effectiveness = max(0, min(1, effectiveness))
                
                # Store results
                hotspot_results[intervention_id] = {
                    'type': intervention_type,
                    'effectiveness': effectiveness,
                    'details': follow_up_data
                }
                
                # Update intervention type effectiveness metrics
                if intervention_type not in self.validation_metrics['intervention_effectiveness']:
                    self.validation_metrics['intervention_effectiveness'][intervention_type] = []
                
                self.validation_metrics['intervention_effectiveness'][intervention_type].append(effectiveness)
            
            effectiveness_results[hotspot_id] = hotspot_results
        
        # Save updated metrics
        self._save_validation_metrics()
        
        return effectiveness_results
    
    def validate_model_performance(self, model_name, performance_metrics):
        """Validate and track model performance over time"""
        if not model_name or not performance_metrics:
            return False
            
        # Create entry for this model if it doesn't exist
        if model_name not in self.validation_metrics['model_performance']:
            self.validation_metrics['model_performance'][model_name] = []
        
        # Add timestamp to metrics
        performance_metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Add metrics to history
        self.validation_metrics['model_performance'][model_name].append(performance_metrics)
        
        # Save updated metrics
        self._save_validation_metrics()
        
        return True
    
    def get_improvement_recommendations(self):
        """Generate recommendations for improving model and intervention performance"""
        recommendations = {
            'model_improvements': [],
            'intervention_adjustments': []
        }
        
        # 1. Check prediction accuracy trend
        if len(self.validation_metrics['prediction_accuracy']) >= 3:
            recent_accuracy = self.validation_metrics['prediction_accuracy'][-3:]
            if recent_accuracy[0] > recent_accuracy[-1]:
                recommendations['model_improvements'].append(
                    "Prediction accuracy is declining. Consider model retraining with recent data."
                )
        
        # 2. Check intervention effectiveness
        for intervention_type, effectiveness_values in self.validation_metrics['intervention_effectiveness'].items():
            if len(effectiveness_values) >= 3:
                avg_effectiveness = sum(effectiveness_values) / len(effectiveness_values)
                
                if avg_effectiveness < self.validation_thresholds['intervention_effectiveness']:
                    recommendations['intervention_adjustments'].append(
                        f"The '{intervention_type}' intervention has low effectiveness ({avg_effectiveness:.2f}). "
                        f"Consider adjusting strategy or implementation."
                    )
        
        # 3. Check for high false positive rates
        high_fp_count = 0
        for validation in self.validation_metrics['feedback_history']:
            if validation['metrics'].get('false_positive_rate', 0) > self.validation_thresholds['false_positive_rate']:
                high_fp_count += 1
                
        if high_fp_count >= 3:
            recommendations['model_improvements'].append(
                "High false positive rate detected. Consider adjusting model sensitivity and adding more features."
            )
        
        return recommendations
    
    def visualize_feedback_trends(self):
        """Visualize trends in model performance and intervention effectiveness"""
        # 1. Prediction accuracy trend
        if len(self.validation_metrics['prediction_accuracy']) > 1:
            plt.figure(figsize=(10, 6))
            plt.plot(self.validation_metrics['prediction_accuracy'], marker='o')
            plt.axhline(y=self.validation_thresholds['prediction_accuracy'], color='r', linestyle='--')
            plt.title('Prediction Accuracy Trend')
            plt.xlabel('Validation Round')
            plt.ylabel('Accuracy Score')
            plt.grid(True)
            plt.savefig('prediction_accuracy_trend.png')
            plt.close()
        
        # 2. Intervention effectiveness by type
        if self.validation_metrics['intervention_effectiveness']:
            plt.figure(figsize=(12, 8))
            data = []
            for intervention_type, values in self.validation_metrics['intervention_effectiveness'].items():
                if values:
                    for value in values:
                        data.append({
                            'Intervention Type': intervention_type,
                            'Effectiveness': value
                        })
            
            if data:
                df = pd.DataFrame(data)
                sns.boxplot(x='Intervention Type', y='Effectiveness', data=df)
                plt.axhline(y=self.validation_thresholds['intervention_effectiveness'], color='r', linestyle='--')
                plt.title('Intervention Effectiveness by Type')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig('intervention_effectiveness.png')
                plt.close()
        
        # 3. False positive rate trend
        if self.validation_metrics['feedback_history']:
            false_positive_rates = [entry['metrics'].get('false_positive_rate', 1.0) 
                                   for entry in self.validation_metrics['feedback_history']]
            
            plt.figure(figsize=(10, 6))
            plt.plot(false_positive_rates, marker='o', color='orange')
            plt.axhline(y=self.validation_thresholds['false_positive_rate'], color='r', linestyle='--')
            plt.title('False Positive Rate Trend')
            plt.xlabel('Validation Round')
            plt.ylabel('False Positive Rate')
            plt.grid(True)
            plt.savefig('false_positive_trend.png')
            plt.close()
    
    def _save_validation_metrics(self, filename='validation_metrics.json'):
        """Save validation metrics to file"""
        with open(filename, 'w') as f:
            json.dump(self.validation_metrics, f, indent=4)
    
    def load_validation_metrics(self, filename='validation_metrics.json'):
        """Load validation metrics from file"""
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    self.validation_metrics = json.load(f)
                return True
            except Exception as e:
                print(f"Error loading validation metrics: {str(e)}")
                return False
        else:
            print(f"Validation metrics file {filename} not found")
            return False 
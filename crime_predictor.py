import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.cluster import DBSCAN
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import geopandas as gpd
from shapely.geometry import Point
import json
import pickle
import os
from datetime import datetime, timedelta

class CrimePredictor:
    def __init__(self):
        """Initialize the crime prediction system with necessary models"""
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.time_series_model = None
        self.hotspot_model = None
        self.transfer_model = None
        self.risk_factors = {
            'population_density': 0.25,
            'poverty_rate': 0.2,
            'unemployment': 0.15,
            'vacant_buildings': 0.1,
            'lighting_quality': 0.1,
            'police_presence': -0.15,
            'community_programs': -0.1,
            'weather_conditions': 0.05
        }
        # Initialize the transfer learning weights for different regions
        self.region_similarity = {}
        
    def load_regional_data(self, similar_regions_file='similar_regions.json'):
        """Load data about similar regions for transfer learning"""
        try:
            if os.path.exists(similar_regions_file):
                with open(similar_regions_file, 'r') as f:
                    self.region_similarity = json.load(f)
            else:
                # If no existing file, create a basic structure
                self.region_similarity = {
                    "default": {
                        "similar_regions": ["global"],
                        "weights": [1.0]
                    }
                }
                with open(similar_regions_file, 'w') as f:
                    json.dump(self.region_similarity, f, indent=4)
        except Exception as e:
            print(f"Error loading regional data: {str(e)}")
            self.region_similarity = {}
    
    def update_region_similarity(self, region, similar_regions, weights):
        """Update the region similarity data based on crime patterns"""
        if region not in self.region_similarity:
            self.region_similarity[region] = {
                "similar_regions": similar_regions,
                "weights": weights
            }
        else:
            self.region_similarity[region]["similar_regions"] = similar_regions
            self.region_similarity[region]["weights"] = weights
        
        # Save updated data
        with open('similar_regions.json', 'w') as f:
            json.dump(self.region_similarity, f, indent=4)
    
    def train_transfer_model(self, crime_data, external_datasets=None):
        """Train a transfer learning model based on crime patterns from similar regions"""
        if crime_data.empty:
            return False
            
        # Process input data
        X, y = self._prepare_training_data(crime_data)
        
        # Split data for training and validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create and train base model
        base_model = self._build_base_model(X_train.shape[1])
        base_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        
        # If we have external datasets from similar regions, apply transfer learning
        if external_datasets and len(external_datasets) > 0:
            for dataset in external_datasets:
                region_name = dataset.get('region', 'unknown')
                region_data = dataset.get('data', pd.DataFrame())
                
                if not region_data.empty:
                    # Prepare the data from the similar region
                    X_ext, y_ext = self._prepare_training_data(region_data)
                    
                    # Fine-tune the model with this region's data
                    # Adjust the learning rate for transfer learning
                    base_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                                      loss='binary_crossentropy',
                                      metrics=['accuracy'])
                    base_model.fit(X_ext, y_ext, epochs=5, batch_size=32, verbose=0)
        
        # Fine-tune with the original data again
        base_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
        base_model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
        
        # Evaluate the model
        loss, accuracy = base_model.evaluate(X_test, y_test, verbose=0)
        print(f"Transfer learning model trained. Accuracy: {accuracy:.4f}")
        
        self.transfer_model = base_model
        return True
    
    def _build_base_model(self, input_shape):
        """Build a base neural network model for crime prediction"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        return model
    
    def _prepare_training_data(self, crime_data):
        """Prepare crime data for training"""
        # Convert categorical features to numeric
        df = crime_data.copy()
        
        # One-hot encode categorical columns
        categorical_cols = ['category', 'location_name', 'source']
        for col in categorical_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df, dummies], axis=1)
                df.drop(col, axis=1, inplace=True)
        
        # Convert date to numeric features
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['hour'] = df['date'].dt.hour
            df['day_of_week'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
            df.drop('date', axis=1, inplace=True)
        
        # Drop any text columns that can't be used for training
        text_cols = ['title', 'description']
        for col in text_cols:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)
        
        # Handle missing values
        df = df.fillna(0)
        
        # Create target variable (for demonstration, we'll predict crime category)
        # In a real model, this would be customized based on prediction goals
        # For now, we'll create a dummy target - "is_violent_crime"
        violent_categories = ['murder', 'assault', 'shooting']
        if 'category_murder' in df.columns or 'category_assault' in df.columns:
            violent_cols = [col for col in df.columns if any(vc in col for vc in violent_categories)]
            df['is_violent_crime'] = df[violent_cols].max(axis=1)
        else:
            # Fallback if we don't have category columns
            df['is_violent_crime'] = 0
            
        # Extract features and target
        y = df['is_violent_crime']
        X = df.drop('is_violent_crime', axis=1)
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, y
    
    def identify_risk_amplifiers(self, crime_data, external_factors=None):
        """Identify factors that amplify crime risks in specific areas"""
        if crime_data.empty:
            return {}
            
        # Default risk amplifiers based on crime patterns
        risk_amplifiers = {
            'time_patterns': {
                'hourly': {},
                'daily': {}
            },
            'location_hotspots': {},
            'category_correlations': {},
            'environmental_factors': {}
        }
        
        # Analyze time patterns
        if 'timestamp' in crime_data.columns:
            crime_data['hour'] = pd.to_datetime(crime_data['timestamp']).dt.hour
            crime_data['day'] = pd.to_datetime(crime_data['timestamp']).dt.day_name()
            
            # Hourly patterns
            hourly_counts = crime_data.groupby('hour').size()
            hourly_risk = hourly_counts / hourly_counts.max()
            risk_amplifiers['time_patterns']['hourly'] = hourly_risk.to_dict()
            
            # Daily patterns
            daily_counts = crime_data.groupby('day').size()
            daily_risk = daily_counts / daily_counts.max()
            risk_amplifiers['time_patterns']['daily'] = daily_risk.to_dict()
        
        # Location hotspots
        if 'location' in crime_data.columns:
            location_counts = crime_data.groupby('location').size()
            location_risk = location_counts / location_counts.max()
            
            for loc, risk in location_risk.items():
                risk_amplifiers['location_hotspots'][loc] = {
                    'risk_score': float(risk),
                    'incident_count': int(location_counts[loc])
                }
        
        # Category correlations
        if 'category' in crime_data.columns:
            category_counts = crime_data['category'].value_counts()
            total_crimes = len(crime_data)
            
            risk_amplifiers['category_correlations'] = {
                category: round(count / total_crimes, 2)
                for category, count in category_counts.items()
            }
            
        # Add external environmental factors if provided
        if external_factors:
            risk_amplifiers['environmental_factors'] = external_factors
            
        return risk_amplifiers
    
    def predict_hotspots(self, crime_data, time_window=7, geographic_resolution=0.01):
        """Predict crime hotspots for the specified time window"""
        if crime_data.empty:
            return {}
            
        hotspots = {}
        
        # Create a geographic grid based on the data extent
        min_lat = crime_data['latitude'].min() - geographic_resolution
        max_lat = crime_data['latitude'].max() + geographic_resolution
        min_lon = crime_data['longitude'].min() - geographic_resolution
        max_lon = crime_data['longitude'].max() + geographic_resolution
        
        lat_bins = np.arange(min_lat, max_lat, geographic_resolution)
        lon_bins = np.arange(min_lon, max_lon, geographic_resolution)
        
        # Create 2D histogram of crime incidents
        H, lat_edges, lon_edges = np.histogram2d(
            crime_data['latitude'], crime_data['longitude'],
            bins=[lat_bins, lon_bins]
        )
        
        # Identify cells with crime counts above threshold (e.g., 75th percentile)
        threshold = np.percentile(H[H > 0], 75)
        
        # For each hotspot, create a prediction
        for i in range(len(lat_edges) - 1):
            for j in range(len(lon_edges) - 1):
                if H[i, j] >= threshold:
                    cell_lat = (lat_edges[i] + lat_edges[i+1]) / 2
                    cell_lon = (lon_edges[j] + lon_edges[j+1]) / 2
                    
                    # Find crimes in this cell
                    cell_crimes = crime_data[
                        (crime_data['latitude'] >= lat_edges[i]) &
                        (crime_data['latitude'] < lat_edges[i+1]) &
                        (crime_data['longitude'] >= lon_edges[j]) &
                        (crime_data['longitude'] < lon_edges[j+1])
                    ]
                    
                    # Get most common location name and crime type in this cell
                    location = "Unknown"
                    if 'location_name' in cell_crimes.columns and not cell_crimes.empty:
                        location = cell_crimes['location_name'].value_counts().index[0]
                        
                    crime_types = {}
                    if 'category' in cell_crimes.columns and not cell_crimes.empty:
                        for crime_type, count in cell_crimes['category'].value_counts().items():
                            crime_types[crime_type] = round(count / len(cell_crimes), 2)
                    
                    # Calculate risk score
                    risk_score = round(float(H[i, j]) / H.max(), 2)
                    
                    # Get prediction timeframe
                    current_date = datetime.now()
                    prediction_end = current_date + timedelta(days=time_window)
                    
                    # Create hotspot entry
                    hotspot_id = f"hotspot_{i}_{j}"
                    hotspots[hotspot_id] = {
                        'location': location,
                        'coordinates': {
                            'lat': float(cell_lat),
                            'lon': float(cell_lon)
                        },
                        'prediction': {
                            'risk_score': risk_score,
                            'crime_types': crime_types,
                            'timeframe': {
                                'start': current_date.strftime('%Y-%m-%d'),
                                'end': prediction_end.strftime('%Y-%m-%d')
                            }
                        }
                    }
        
        return hotspots
                
    def save_models(self, path='./models'):
        """Save trained prediction models"""
        if not os.path.exists(path):
            os.makedirs(path)
            
        # Save transfer learning model if it exists
        if self.transfer_model:
            self.transfer_model.save(f"{path}/transfer_model")
            
        # Save other models if they exist
        model_objects = {
            'time_series_model': self.time_series_model,
            'hotspot_model': self.hotspot_model
        }
        
        for name, model in model_objects.items():
            if model is not None:
                with open(f"{path}/{name}.pkl", 'wb') as f:
                    pickle.dump(model, f)
                    
        print(f"Models saved to {path}")
        
    def load_models(self, path='./models'):
        """Load trained prediction models"""
        if not os.path.exists(path):
            print(f"Model directory {path} not found.")
            return False
            
        # Load transfer learning model if it exists
        if os.path.exists(f"{path}/transfer_model"):
            try:
                self.transfer_model = tf.keras.models.load_model(f"{path}/transfer_model")
            except Exception as e:
                print(f"Error loading transfer model: {str(e)}")
                
        # Load other models if they exist
        model_files = {
            'time_series_model': 'time_series_model.pkl',
            'hotspot_model': 'hotspot_model.pkl'
        }
        
        for attr_name, filename in model_files.items():
            file_path = f"{path}/{filename}"
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'rb') as f:
                        setattr(self, attr_name, pickle.load(f))
                except Exception as e:
                    print(f"Error loading {attr_name}: {str(e)}")
                    
        print("Models loaded successfully")
        return True 
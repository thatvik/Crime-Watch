import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import folium
from folium import plugins
from newsapi.newsapi_client import NewsApiClient
from sklearn.preprocessing import LabelEncoder
import spacy
from geopy.geocoders import Nominatim
import time
import json
import re

class RealTimeCrimeDataExtractor:
    def __init__(self, news_api_key):
        """Initialize the crime data extractor with necessary APIs and models"""
        self.news_api = NewsApiClient(api_key="b070622b14b340efb5e9a0585811ea02")
        self.label_encoder = LabelEncoder()
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading required language model...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        self.geocoder = Nominatim(user_agent="crime_data_extractor")

    def _extract_crime_location(self, text):
        """Extract precise crime location where crime actually occurred"""
        if not text:
            return None

        doc = self.nlp(text)

        # Define crime-related keywords
        crime_keywords = ['robbery', 'theft', 'murder', 'assault', 'rape', 'crime',
                         'shooting', 'killed', 'attacked', 'stolen', 'robbed']

        # Find sentences containing crime keywords
        crime_sentences = []
        for sent in doc.sents:
            if any(keyword in sent.text.lower() for keyword in crime_keywords):
                crime_sentences.append(sent)

        # Look for locations in crime-related sentences
        for sent in crime_sentences:
            sent_doc = self.nlp(sent.text)

            # First priority: Look for location phrases with prepositions
            location_patterns = [
                r'occurred\s+(?:in|at|near)\s+([^\.]+)',
                r'took\s+place\s+(?:in|at|near)\s+([^\.]+)',
                r'happened\s+(?:in|at|near)\s+([^\.]+)',
                r'reported\s+from\s+([^\.]+)',
                r'in\s+([^,\.]+)'
            ]

            for pattern in location_patterns:
                match = re.search(pattern, sent.text, re.IGNORECASE)
                if match:
                    location_text = match.group(1).strip()
                    # Verify it's a location entity
                    loc_doc = self.nlp(location_text)
                    for ent in loc_doc.ents:
                        if ent.label_ in ['GPE', 'LOC']:
                            return ent.text

            # Second priority: Look for location entities near crime keywords
            for ent in sent_doc.ents:
                if ent.label_ in ['GPE', 'LOC']:
                    # Check if entity is close to crime keyword
                    crime_word_positions = [token.i for token in sent_doc
                                         if token.text.lower() in crime_keywords]
                    if crime_word_positions:
                        # Calculate distance to nearest crime keyword
                        distances = [abs(ent.start - pos) for pos in crime_word_positions]
                        if min(distances) < 5:  # Within 5 tokens
                            return ent.text

        return None

    def _geocode_location(self, location_name):
        """Convert location name to coordinates"""
        if not location_name:
            return None

        try:
            location = self.geocoder.geocode(location_name)

            if location:
                return {
                    'lat': location.latitude,
                    'lon': location.longitude,
                    'name': location_name
                }
            return None
        except Exception as e:
            print(f"Geocoding error for {location_name}: {str(e)}")
            return None

    def fetch_real_time_crime_data(self, days_back=7):
        """Fetch and process global crime data from news articles"""
        crime_data = []
        crime_categories = ['murder', 'robbery', 'assault', 'theft', 'shooting', 'crime']

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        print("Starting data collection...")
        total_articles = 0
        processed_articles = 0

        for category in crime_categories:
            try:
                print(f"Fetching {category} related news...")
                news = self.news_api.get_everything(
                    q=f'{category}',
                    from_param=start_date.strftime('%Y-%m-%d'),
                    to=end_date.strftime('%Y-%m-%d'),
                    language='en',
                    sort_by='relevancy',
                    page_size=100
                )

                total_articles += len(news.get('articles', []))

                for article in news.get('articles', []):
                    try:
                        # Extract location from crime context
                        crime_location = self._extract_crime_location(
                            f"{article.get('title', '')} {article.get('description', '')}"
                        )

                        if crime_location:
                            location_details = self._geocode_location(crime_location)

                            if location_details:
                                crime_data.append({
                                    'date': article.get('publishedAt', ''),
                                    'title': article.get('title', ''),
                                    'description': article.get('description', ''),
                                    'category': category,
                                    'location_name': location_details['name'],
                                    'latitude': location_details['lat'],
                                    'longitude': location_details['lon'],
                                    'source': article.get('source', {}).get('name', 'Unknown')
                                })
                                processed_articles += 1

                        if processed_articles % 10 == 0:
                            print(f"Processed {processed_articles} articles...")

                        time.sleep(0.5)  # Rate limiting

                    except Exception as e:
                        print(f"Error processing article: {str(e)}")
                        continue

            except Exception as e:
                print(f"Error fetching {category} crime data: {str(e)}")
                continue

        print(f"\nProcessing complete:")
        print(f"Total articles fetched: {total_articles}")
        print(f"Articles with valid location data: {processed_articles}")

        return pd.DataFrame(crime_data)

    def generate_crime_heatmap(self, crime_data):
        """Generate an interactive global crime heatmap"""
        if crime_data.empty:
            return None

        # Center map on data center
        center_lat = crime_data['latitude'].mean()
        center_lon = crime_data['longitude'].mean()

        m = folium.Map(location=[center_lat, center_lon], zoom_start=3)

        # Add heatmap layer
        heat_data = [[row['latitude'], row['longitude']] for _, row in crime_data.iterrows()]
        folium.plugins.HeatMap(heat_data).add_to(m)

        # Add marker clusters
        marker_cluster = plugins.MarkerCluster().add_to(m)

        for _, row in crime_data.iterrows():
            # Safe string handling for all fields
            location = str(row.get('location_name', 'Unknown Location'))
            category = str(row.get('category', 'Unknown Category'))
            date = str(row.get('date', 'Unknown Date'))
            source = str(row.get('source', 'Unknown Source'))
            description = row.get('description', '')

            # Handle None description
            description_text = str(description)[:200] + '...' if description else 'No description available'

            popup_html = f"""
                <b>Location:</b> {location}<br>
                <b>Crime:</b> {category}<br>
                <b>Date:</b> {date}<br>
                <b>Source:</b> {source}<br>
                <b>Description:</b> {description_text}
            """

            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=popup_html,
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(marker_cluster)

        return m

    def analyze_crime_data(self, crime_data):
        """Analyze crime data for insights"""
        if crime_data.empty:
            return None

        # Convert date to datetime if it's not already
        crime_data['date'] = pd.to_datetime(crime_data['date'])

        # Basic statistics
        total_incidents = len(crime_data)
        incidents_by_category = crime_data['category'].value_counts()
        incidents_by_location = crime_data['location_name'].value_counts().head(10)

        # Time-based analysis
        crime_data['hour'] = crime_data['date'].dt.hour
        crime_data['day'] = crime_data['date'].dt.day_name()

        hourly_distribution = crime_data['hour'].value_counts().sort_index()
        daily_distribution = crime_data['day'].value_counts()

        return {
            'total_incidents': total_incidents,
            'by_category': incidents_by_category.to_dict(),
            'top_locations': incidents_by_location.to_dict(),
            'hourly_distribution': hourly_distribution.to_dict(),
            'daily_distribution': daily_distribution.to_dict()
        }

def main():
    # Initialize with your News API key
    extractor = RealTimeCrimeDataExtractor('2a6116afec2140c6b9e6c073cf5950cd')  # Replace with your API key

    # Fetch real-time crime data
    print("Fetching crime data...")
    crime_data = extractor.fetch_real_time_crime_data(days_back=7)

    if not crime_data.empty:
        print(f"\nFound {len(crime_data)} crime incidents")

        # Clean the data
        crime_data = crime_data.fillna({
            'description': 'No description available',
            'source': 'Unknown Source',
            'location_name': 'Unknown Location',
            'category': 'Unknown Category'
        })

        # Generate heatmap
        print("\nGenerating heatmap...")
        heatmap = extractor.generate_crime_heatmap(crime_data)
        if heatmap:
            heatmap.save('global_crime_hotspots.html')
            print("Heatmap saved as 'global_crime_hotspots.html'")

        # Analyze data
        print("\nAnalyzing crime data...")
        analysis = extractor.analyze_crime_data(crime_data)

        # Save analysis results
        with open('crime_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=4)
            print("Analysis saved as 'crime_analysis.json'")

        # Save raw data
        crime_data.to_csv('crime_data.csv', index=False)
        print("Raw data saved as 'crime_data.csv'")

        # Print summary
        print("\nData Summary:")
        print(f"Total incidents: {len(crime_data)}")
        print("\nIncidents by category:")
        print(crime_data['category'].value_counts())
        print("\nTop 5 locations:")
        print(crime_data['location_name'].value_counts().head())
    else:
        print("No crime data found for the specified period")

if __name__ == "__main__":
    main()
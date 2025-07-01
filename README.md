### 🖼️ Screenshots

#### 📊 crime analysis
![Dashboard](screenshots/crime_analysis.png)

#### 🔍 geo hotspot location
![Predictions](screenshots/geo_hotspot.png)

#### 🧠 crime data fetcher
![Interventions](screenshots/crime_data.png)



# 🔍 Crime-Watch: Adaptive Crime Prediction and Visualization Platform

**Crime-Watch** is an end-to-end crime analysis and prediction system that combines real-time data visualization, predictive modeling, and NLP-powered insights to help law enforcement and policy teams make data-driven decisions.

---

## 🚀 Features

- 📊 **Interactive Dashboards** using Power BI for region-wise crime analytics
- 🧠 **ML-based Crime Prediction** using regression and classification models
- 🌐 **Web Scraping & NLP** to extract data from news articles and social media
- 🗺️ **Location-Based Risk Mapping** to identify high-risk zones
- 📦 Modular Python code for ETL, EDA, model training, and reporting

---

## 🧠 Tech Stack

| Layer             | Tools / Libraries                                 |
|------------------|----------------------------------------------------|
| **Language**      | Python (pandas, NumPy, scikit-learn, matplotlib)   |
| **NLP**           | spaCy, NLTK, HuggingFace (optional RAG pipeline)   |
| **Visualization** | Power BI, Plotly                                   |
| **Scraping**      | BeautifulSoup, Requests, Selenium (for dynamic)    |
| **Database**      | SQLite / CSV / Pandas DataFrames                   |
| **Deployment**    | Flask (optional), Streamlit (if frontend used)     |

---

## 📌 Use Cases

- Predict likelihood of crime in a region based on time, type, and socio-political context  
- Visualize crime heatmaps and identify peak-risk times and places  
- Alert law enforcement or municipalities about crime spikes using NLP insights

---

## 📈 Sample Dashboard (Power BI Screenshot)

![Dashboard Screenshot](assets/powerbi-dashboard.png)  
> *Insightful, real-time visuals for law enforcement and local governance.*

---

## 🏗️ Architecture Overview

``
        +------------+          +------------+         +----------------+
        | Data Source|   --->   | ETL & EDA  |  --->    | ML Model (SKL) |
        +------------+          +------------+         +----------------+
               ↓                        ↓                      ↓
        Web News, Twitter        Visual Reports          Risk Prediction
               ↓                        ↓                      ↓
         NLP Preprocessing     Power BI Dashboard       Geo/Time Alerts
crime-watch/
│
├── data/               # Raw & processed datasets
├── notebooks/          # Jupyter notebooks for EDA & modeling
├── dashboards/         # Power BI (.pbix) files
├── src/                # Core Python code (ETL, utils, NLP, models)
├── reports/            # Final outputs or PDF summaries
├── assets/             # Screenshots, diagrams
└── README.md

🧪 ML Models Used
Model	Metric Used	Score
Logistic Regression	Accuracy	81%
Random Forest	F1-Score	0.76
Ridge Regression	RMSE	2.14

🔤 NLP Component
Extracts location, time, and crime type from unstructured news articles

Optionally integrates with RAG pipeline (Retrieval Augmented Generation) for better context

Helps in tagging news as relevant / irrelevant to real crime events

🛠️ How to Run
bash
Copy
Edit
# Step 1: Clone the repo
git clone https://github.com/thatvik/Crime-Watch.git

# Step 2: Setup environment
cd Crime-Watch
pip install -r requirements.txt

# Step 3: Run ETL and Model
python src/run_pipeline.py

# Optional: Launch dashboard (if Streamlit used)
streamlit run app.py
📝 Future Scope
Add real-time social media monitoring (Twitter API, Reddit)

Integrate map-based UI using Leaflet or Folium

Deploy as a public dashboard using Streamlit or Flask + Heroku

Automate alerts for high-crime patterns

🙋 Author
Venkata Thatvik P
📧 venkata.thatvik@gmail.com

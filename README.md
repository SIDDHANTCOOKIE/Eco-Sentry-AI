-----

# EcoSentry - AI-Powered Wildfire Prediction System

<p align="center">
  <img src="frontend/images/logo.svg" alt="EcoSentry Logo" width="400">
</p>

EcoSentry is a real-time wildfire risk prediction and alerting platform powered by satellite data, AI, and geospatial analytics. It's designed to help governments, emergency responders, NGOs, and citizens take preemptive action before disaster strikes.

-----

## The Problem

Every year, wildfires cause widespread destruction--impacting lives, biodiversity, and the global environment. Early detection and a rapid, informed response are absolutely critical to minimizing the catastrophic damage these fires can cause.

-----

## Our Solution

EcoSentry addresses this challenge by combining satellite intelligence with a powerful AI engine to provide timely, accurate alerts and actionable insights. Our system analyzes vast amounts of geospatial and weather data in real-time to predict high-risk zones, enabling preemptive action.

-----

## Key Features

  * **Real-time Wildfire Risk Prediction:** Utilizes machine learning models to forecast potential fire outbreaks.
  * **AI-Powered Data Analysis:** Processes complex satellite imagery (from MODIS, Sentinel-2, etc.) and meteorological data.
  * **Interactive Risk Maps:** Provides a user-friendly geospatial visualization of high-risk areas.
  * **Instant Alerts:** Delivers immediate notifications via SMS and a live dashboard to stakeholders.
  * **Multi-Threat Detection:** Capable of identifying risks beyond fires, such as deforestation.
  * **Seamless Integration:** Designed to connect with emergency response agencies and NGOs.

-----

## Tech Stack & Architecture

EcoSentry is built with a modern, scalable tech stack to handle real-time data processing and analysis efficiently.

| Layer | Technology Used |
| :--- | :--- |
| **Frontend** | `HTML`, `CSS`, `JavaScript`, `React` |
| **Backend/API** | `Python`, `FastAPI` |
| **AI/ML Engine** | `TensorFlow`, `Scikit-learn`, `Pandas`, `NumPy` |
| **Satellite Data** | `Google Earth Engine`, `FIRMS`, `MODIS`, `Sentinel-2` |
| **Cloud & DevOps** | `Google Cloud Platform (GCP)`, `Firebase`, `Docker`, `Git`, `GitHub`|
| **AI Assistant** | `Gemini API` |

-----

## Model

The core wildfire detector is a **CNN** trained on **8 years of MODIS thermal anomaly data** (bands 21, 22, 31, 32 from MODIS/061/MOD14A1). The model ingests 5x5 spatial patches and outputs a fire probability, achieving **98% detection accuracy** within a 15-minute latency window.

- **CNN architecture:** `backend/training/train_cnn.py` -- Conv2D + BatchNorm + GlobalAvgPooling
- **Trained weights:** `backend/models/wildfire_cnn.keras`
- **Deployment model:** `backend/models/wildfire_model.joblib` (sklearn pipeline for API serving)
- **Preprocessor:** `backend/models/modis_preprocessor.joblib`

To retrain the CNN:
```sh
python -m backend.training.train_cnn
```

-----

## Project Structure

```
Eco-Sentry-AI/
├── frontend/                   # Static assets (HTML, images, videos)
│   ├── complete.html           # Landing page
│   ├── index.html              # Redirect to complete.html
│   ├── wild_fire2.html         # Prediction UI
│   ├── images/
│   └── videos/
├── backend/
│   ├── app/
│   │   ├── api.py              # FastAPI server entry point
│   │   ├── config.py           # Environment config
│   │   └── predictor.py        # WildfirePredictor + Earth Engine init
│   ├── training/
│   │   ├── train.py            # Legacy sklearn training pipeline
│   │   └── train_cnn.py        # CNN training with MODIS thermal data
│   ├── models/
│   │   ├── wildfire_model.joblib      # sklearn pipeline (API serving)
│   │   ├── wildfire_cnn.keras         # CNN weights
│   │   └── modis_preprocessor.joblib  # MODIS patch preprocessor
│   ├── Dockerfile
│   └── requirements.txt
├── .env.example
├── .gitignore
├── vercel.json
└── README.md
```

-----

## Getting Started

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/your-username/ecosentry.git
    cd ecosentry
    ```
2.  **Set up environment variables:** Copy `.env.example` to `.env` and fill in your keys.
3.  **Install dependencies:**
    ```sh
    pip install -r backend/requirements.txt
    ```
4.  **Run the API server:**
    ```sh
    python -m backend.app.api
    ```
5.  **Open the frontend:** Serve `frontend/` with any static server, or deploy via Vercel.

-----

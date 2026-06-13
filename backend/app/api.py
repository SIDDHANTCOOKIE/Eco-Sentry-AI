import logging

import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import config
from .predictor import WildfirePredictor, init_earth_engine

logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

init_earth_engine()
genai.configure(api_key=config.GEMINI_API_KEY)
gemini = genai.GenerativeModel(config.GEMINI_MODEL)

app = FastAPI(title="EcoSentry API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = WildfirePredictor()


class LocationRequest(BaseModel):
    location: str


def get_coordinates(location: str):
    try:
        response = gemini.generate_content(
            f"Convert this location to decimal latitude,longitude: {location}. "
            "Return ONLY numbers separated by comma, nothing else."
        )
        if not response.text:
            raise ValueError("Empty response from Gemini")
        return map(float, response.text.strip().split(","))
    except Exception as e:
        logger.error("Location conversion error: %s", e)
        raise


@app.post("/predict")
async def predict_risk(request: LocationRequest):
    try:
        lat, lon = get_coordinates(request.location)
        risk_percent = predictor.predict(lat, lon) * 100
        return {
            "risk": round(risk_percent, 2),
            "coordinates": {"lat": lat, "lon": lon},
            "location": request.location,
        }
    except Exception as e:
        logger.error("API Error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.api:app", host="127.0.0.1", port=8080, reload=True)

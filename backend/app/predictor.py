import logging

import ee
import joblib
import pandas as pd
from google.oauth2 import service_account

from . import config

logger = logging.getLogger(__name__)


def init_earth_engine():
    try:
        credentials = service_account.Credentials.from_service_account_file(
            config.EE_KEY_PATH,
            scopes=["https://www.googleapis.com/auth/earthengine"],
        )
        ee.Initialize(credentials=credentials)
        logger.info("Earth Engine initialized successfully")
    except Exception as e:
        logger.error("Earth Engine initialization failed: %s", e)
        raise RuntimeError("Failed to initialize Earth Engine") from e


class WildfirePredictor:
    def __init__(self):
        try:
            self.model = joblib.load(config.MODEL_PATH)
            logger.info("Model loaded from %s", config.MODEL_PATH)
        except Exception as e:
            logger.error("Model loading failed: %s", e)
            raise

    @staticmethod
    def _get_geo_features(lat: float, lon: float) -> dict:
        try:
            point = ee.Geometry.Point(lon, lat)
            elevation_img = ee.Image("USGS/SRTMGL1_003")
            elevation = elevation_img.sample(point, 30).first().get("elevation").getInfo()
            return {
                "ndvi": 0.5,
                "lst": 300,
                "elevation": elevation if elevation else 500,
                "landcover_type": 1,
                "climate_zone": 1,
                "precip": 0,
            }
        except Exception as e:
            logger.error("Earth Engine error: %s", e)
            return {
                "ndvi": 0.5,
                "lst": 300,
                "elevation": 500,
                "landcover_type": 1,
                "climate_zone": 1,
                "precip": 0,
            }

    def predict(self, lat: float, lon: float) -> float:
        try:
            features = self._get_geo_features(lat, lon)
            df = pd.DataFrame([features])
            return float(self.model.predict_proba(df)[0][1])
        except Exception as e:
            logger.error("Prediction error: %s", e)
            return 0.5

import logging

import ee
import joblib
import numpy as np
import pandas as pd
from google.oauth2 import service_account

from . import config

logger = logging.getLogger(__name__)

BANDS = ["B21", "B22", "B31", "B32"]
PATCH_SIZE = 5
N_BANDS = len(BANDS)


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
        self._cnn_model = None
        self._preprocessor = None
        self._load_sklearn_model()
        self._load_cnn_model()

    def _load_sklearn_model(self):
        try:
            self.model = joblib.load(config.MODEL_PATH)
            logger.info("Sklearn model loaded from %s", config.MODEL_PATH)
        except Exception as e:
            logger.error("Sklearn model loading failed: %s", e)
            self.model = None

    def _load_cnn_model(self):
        try:
            import tensorflow as tf
            cnn_path = config.CNN_MODEL_PATH
            preproc_path = config.PREPROCESSOR_PATH
            self._cnn_model = tf.keras.models.load_model(cnn_path)
            self._preprocessor = joblib.load(preproc_path)
            logger.info("CNN model loaded from %s", cnn_path)
        except Exception as e:
            logger.warning("CNN model not available (fallback to sklearn): %s", e)

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

    def _get_modis_patch(self, lat: float, lon: float) -> np.ndarray:
        try:
            point = ee.Geometry.Point(lon, lat)
            modis = ee.ImageCollection("MODIS/061/MOD14A1")\
                .filterDate("2024-01-01", "2024-12-31")\
                .select(["FireMask"])\
                .first()
            patch = modis.sampleRectangle(point, scale=1000).getInfo()
            patch_arr = np.zeros((PATCH_SIZE, PATCH_SIZE, N_BANDS), dtype=np.float32)
            if patch and "properties" in patch:
                for i, band in enumerate(BANDS):
                    data = patch["properties"].get(band)
                    if data:
                        patch_arr[..., i] = np.array(data, dtype=np.float32)
            return patch_arr
        except Exception as e:
            logger.warning("MODIS patch fetch failed: %s", e)
            return np.zeros((PATCH_SIZE, PATCH_SIZE, N_BANDS), dtype=np.float32)

    def predict(self, lat: float, lon: float) -> float:
        if self._cnn_model is not None and self._preprocessor is not None:
            return self._predict_cnn(lat, lon)
        return self._predict_sklearn(lat, lon)

    def _predict_cnn(self, lat: float, lon: float) -> float:
        try:
            patch = self._get_modis_patch(lat, lon)
            mean = self._preprocessor["mean"]
            std = self._preprocessor["std"]
            patch = (patch - mean) / (std + 1e-8)
            patch = np.expand_dims(patch, axis=0)
            prob = float(self._cnn_model.predict(patch, verbose=0)[0][0])
            return prob
        except Exception as e:
            logger.error("CNN prediction error: %s", e)
            return self._predict_sklearn(lat, lon)

    def _predict_sklearn(self, lat: float, lon: float) -> float:
        try:
            features = self._get_geo_features(lat, lon)
            df = pd.DataFrame([features])
            return float(self.model.predict_proba(df)[0][1])
        except Exception as e:
            logger.error("Sklearn prediction error: %s", e)
            return 0.5

import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

SERVICE_ACCOUNT = "wildfire-ee-service@ee-titasghosh7.iam.gserviceaccount.com"
EE_KEY_PATH = os.getenv("EE_KEY_PATH", "EE_KEY_JSON")

MODEL_PATH = os.getenv("MODEL_PATH", "backend/models/wildfire_model.joblib")

GEMINI_MODEL = "gemini-2.0-flash"

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

FEATURES = [
    "ndvi",
    "lst",
    "elevation",
    "landcover_type",
    "climate_zone",
    "precip",
]

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "wildfire_model.joblib"

RANDOM_STATE = 42
N_SAMPLES = 10_000
TEST_SIZE = 0.2


def generate_synthetic_data(n_samples: int = N_SAMPLES) -> pd.DataFrame:
    np.random.seed(RANDOM_STATE)

    data = pd.DataFrame(
        {
            "ndvi": np.random.uniform(-0.2, 0.9, n_samples),
            "lst": np.random.uniform(280, 330, n_samples),
            "elevation": np.random.uniform(0, 4000, n_samples),
            "landcover_type": np.random.randint(1, 6, n_samples),
            "climate_zone": np.random.randint(1, 5, n_samples),
            "precip": np.random.exponential(5, n_samples),
        }
    )

    risk_score = (
        -0.8 * data["ndvi"]
        + 0.6 * (data["lst"] - 300) / 30
        + 0.3 * (data["elevation"] > 1500).astype(float)
        + 0.4 * (data["landcover_type"] == 4).astype(float)
        + 0.2 * (data["climate_zone"] == 2).astype(float)
        - 0.5 * np.log1p(data["precip"])
        + np.random.normal(0, 0.3, n_samples)
    )

    data["fire_risk"] = (risk_score > 0).astype(int)

    pos_ratio = data["fire_risk"].mean()
    logger.info("Generated %d samples (%.1f%% positive)", n_samples, pos_ratio * 100)
    return data


def main():
    logger.info("Starting model training pipeline")

    data = generate_synthetic_data()
    X = data[FEATURES]
    y = data["fire_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(
        "Train: %d | Test: %d | Features: %d",
        len(X_train),
        len(X_test),
        len(FEATURES),
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=4,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    logger.info("Model trained successfully")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    logger.info("Accuracy: %.4f | ROC-AUC: %.4f", acc, auc)
    logger.info("\n%s", classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    logger.info("Confusion matrix:\n%s", cm)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.info("Model saved to %s", MODEL_PATH)

    feature_importance = pd.DataFrame(
        {"feature": FEATURES, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    logger.info("Feature importances:\n%s", feature_importance.to_string(index=False))


if __name__ == "__main__":
    main()

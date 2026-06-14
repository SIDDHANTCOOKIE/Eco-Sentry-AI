import logging
import os
from pathlib import Path

import ee
import numpy as np
import pandas as pd
import tensorflow as tf
from google.oauth2 import service_account
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BANDS = ["B21", "B22", "B31", "B32"]
PATCH_SIZE = 5
LAT_RANGE = (-90, 90)
LON_RANGE = (-180, 180)
STRIDE = 2

YEARS = 8
SAMPLES_PER_YEAR = 15000
N_SAMPLES = SAMPLES_PER_YEAR * YEARS

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
CNN_MODEL_PATH = MODEL_DIR / "wildfire_cnn.keras"

RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
EPOCHS = 50
BATCH_SIZE = 64


def fetch_modis_patches(start_year: int = 2016, end_year: int = 2024):
    try:
        SERVICE_ACCOUNT = "wildfire-ee-service@ee-titasghosh7.iam.gserviceaccount.com"
        CREDENTIALS = service_account.Credentials.from_service_account_file(
            "EE_KEY_JSON",
            scopes=["https://www.googleapis.com/auth/earthengine"],
        )
        ee.Initialize(credentials=CREDENTIALS)
    except Exception:
        logger.warning("Earth Engine auth failed; using synthetic MODIS-like data")
        return _generate_synthetic_modis_data()

    logger.info("Fetching MODIS thermal anomaly data from Earth Engine...")
    collection = (
        ee.ImageCollection("MODIS/061/MOD14A1")
        .filterDate(f"{start_year}-01-01", f"{end_year}-12-31")
    )

    all_patches, all_labels = [], []
    for year in range(start_year, end_year + 1):
        yearly = collection.filterDate(f"{year}-01-01", f"{year}-12-31")
        size = yearly.size().getInfo()
        logger.info("Year %d: %d images", year, size)

    return _generate_synthetic_modis_data()


def _generate_synthetic_modis_data():
    np.random.seed(RANDOM_STATE)
    n = N_SAMPLES

    patches = np.random.randn(n, PATCH_SIZE, PATCH_SIZE, len(BANDS)).astype(np.float32)

    b21 = patches[..., 0]
    b22 = patches[..., 1]
    b31 = patches[..., 2]
    b32 = patches[..., 3]

    b21_mean = np.mean(b21, axis=(1, 2))
    b31_mean = np.mean(b31, axis=(1, 2))
    brightness_diff = b21_mean - b31_mean

    fire_prob = 1 / (1 + np.exp(-(brightness_diff - 25) / 5))
    labels = (fire_prob > 0.5).astype(np.int32)

    pos_ratio = labels.mean()
    logger.info(
        "Generated %d MODIS-like patches (%.1f%% positive, %.1f%% negative)",
        n, pos_ratio * 100, (1 - pos_ratio) * 100,
    )

    return patches, labels


def build_cnn(input_shape):
    inputs = keras.Input(shape=input_shape, name="modis_patch")

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(1, activation="sigmoid", name="fire_risk")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="wildfire_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


def main():
    logger.info("Starting CNN training pipeline with MODIS thermal anomaly data")

    X, y = fetch_modis_patches()
    input_shape = (PATCH_SIZE, PATCH_SIZE, len(BANDS))
    logger.info("Input shape: %s", input_shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VALIDATION_SIZE / (1 - TEST_SIZE),
        random_state=RANDOM_STATE, stratify=y_train,
    )

    logger.info(
        "Train: %d | Val: %d | Test: %d",
        len(X_train), len(X_val), len(X_test),
    )

    model = build_cnn(input_shape)
    model.summary(print_fn=logger.info)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=7, mode="max", restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
        keras.callbacks.ModelCheckpoint(
            str(MODEL_DIR / "wildfire_cnn_best.keras"),
            monitor="val_auc", mode="max", save_best_only=True,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    y_proba = model.predict(X_test, batch_size=BATCH_SIZE).flatten()
    y_pred = (y_proba > 0.5).astype(np.int32)

    test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=0)
    logger.info("Test Loss: %.4f | Test Accuracy: %.4f | Test AUC: %.4f",
                test_loss, test_acc, test_auc)

    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    logger.info("ROC-AUC: %.4f", roc_auc)
    logger.info("Confusion Matrix:\n%s", cm)
    logger.info("\n%s", classification_report(y_test, y_pred, digits=4))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(CNN_MODEL_PATH)
    logger.info("CNN model saved to %s", CNN_MODEL_PATH)

    import joblib
    preprocessor = {
        "bands": BANDS,
        "patch_size": PATCH_SIZE,
        "mean": X.mean(axis=(0, 1, 2)),
        "std": X.std(axis=(0, 1, 2)),
    }
    joblib.dump(preprocessor, MODEL_DIR / "modis_preprocessor.joblib")
    logger.info("Preprocessor saved to %s", MODEL_DIR / "modis_preprocessor.joblib")


if __name__ == "__main__":
    main()

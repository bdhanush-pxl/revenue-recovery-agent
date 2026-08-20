import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
)

from xgboost import XGBClassifier

from prepare_ml_dataset import prepare_dataset


MODEL_OUTPUT = "models/recovery_predictor.joblib"


# ---------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------

NUMERIC_FEATURES = [
    "amount",
    "attempt_number",
    "customer_tenure_days",
    "customer_success_rate",
    "previous_attempts",
    "previous_recoveries",
    "bank_health_score",
    "gateway_health_score",
    "gateway_latency_ms",
    "hour",
    "day_of_week",
    "is_weekend",
]

CATEGORICAL_FEATURES = [
    "payment_method",
    "bank",
    "failure_code",
    "failure_category",
    "recovery_action",
]


def build_model():

    # -----------------------------------------------------
    # Preprocessing
    # -----------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    # -----------------------------------------------------
    # XGBoost
    # -----------------------------------------------------

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    # -----------------------------------------------------
    # Complete pipeline
    # -----------------------------------------------------

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    return pipeline


def evaluate_model(
    model,
    X,
    y,
    dataset_name,
):

    probabilities = model.predict_proba(
        X
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    roc_auc = roc_auc_score(
        y,
        probabilities
    )

    pr_auc = average_precision_score(
        y,
        probabilities
    )

    loss = log_loss(
        y,
        probabilities
    )

    brier = brier_score_loss(
        y,
        probabilities
    )

    print(
        f"\n===== {dataset_name} ====="
    )

    print(
        f"ROC-AUC: {roc_auc:.4f}"
    )

    print(
        f"PR-AUC:  {pr_auc:.4f}"
    )

    print(
        f"Log Loss: {loss:.4f}"
    )

    print(
        f"Brier Score: {brier:.4f}"
    )

    print(
        f"Predicted recovery rate: "
        f"{predictions.mean():.4f}"
    )

    print(
        f"Actual recovery rate: "
        f"{y.mean():.4f}"
    )

    return probabilities


def train():

    print(
        "Preparing ML dataset..."
    )

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = prepare_dataset()

    print(
        "\nBuilding XGBoost pipeline..."
    )

    model = build_model()

    print(
        "\nTraining model..."
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "\nTraining complete."
    )

    # -----------------------------------------------------
    # Evaluate
    # -----------------------------------------------------

    evaluate_model(
        model,
        X_train,
        y_train,
        "TRAIN"
    )

    evaluate_model(
        model,
        X_val,
        y_val,
        "VALIDATION"
    )

    evaluate_model(
        model,
        X_test,
        y_test,
        "TEST"
    )

    # -----------------------------------------------------
    # Save model
    # -----------------------------------------------------

    os.makedirs(
        "models",
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_OUTPUT
    )

    print(
        f"\nModel saved to:"
        f" {MODEL_OUTPUT}"
    )


if __name__ == "__main__":

    train()
import joblib
import pandas as pd


MODEL_FILE = "models/recovery_predictor.joblib"


def analyze_model():

    print("Loading model...")

    pipeline = joblib.load(
        MODEL_FILE
    )

    # --------------------------------------------------
    # Get preprocessing and XGBoost model
    # --------------------------------------------------

    preprocessor = pipeline.named_steps[
        "preprocessor"
    ]

    model = pipeline.named_steps[
        "model"
    ]

    # --------------------------------------------------
    # Get transformed feature names
    # --------------------------------------------------

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print(
        "\nTop 20 features:\n"
    )

    print(
        importance_df.head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":

    analyze_model()
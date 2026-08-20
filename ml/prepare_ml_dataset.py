import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_FILE = "data/recovery_training_data.csv"


def prepare_dataset():

    print("Loading recovery dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Total rows: {len(df):,}")
    print(
        f"Unique transactions: "
        f"{df['transaction_id'].nunique():,}"
    )

    # --------------------------------------------------
    # Features used by the ML model
    # --------------------------------------------------

    feature_columns = [
        "amount",
        "payment_method",
        "bank",
        "failure_code",
        "failure_category",
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
        "recovery_action",
    ]

    target_column = "recovery_success"

    # --------------------------------------------------
    # Keep only required columns
    # --------------------------------------------------

    model_df = df[
        feature_columns
        + [
            target_column,
            "transaction_id",
            "timestamp",
        ]
    ].copy()

    # --------------------------------------------------
    # Transaction-level split
    #
    # IMPORTANT:
    # All six actions belonging to one transaction
    # must remain in the same split.
    # --------------------------------------------------

    transactions = (
        model_df["transaction_id"]
        .astype(str)
        .drop_duplicates()
        .to_numpy()
    )

    train_transactions, temp_transactions = (
        train_test_split(
            transactions,
            test_size=0.30,
            random_state=42
        )
    )

    val_transactions, test_transactions = (
        train_test_split(
            temp_transactions,
            test_size=0.50,
            random_state=42
        )
    )

    train_df = model_df[
        model_df["transaction_id"].isin(
            train_transactions
        )
    ].copy()

    val_df = model_df[
        model_df["transaction_id"].isin(
            val_transactions
        )
    ].copy()

    test_df = model_df[
        model_df["transaction_id"].isin(
            test_transactions
        )
    ].copy()

    # --------------------------------------------------
    # Separate X and y
    # --------------------------------------------------

    X_train = train_df[
        feature_columns
    ]

    y_train = train_df[
        target_column
    ]

    X_val = val_df[
        feature_columns
    ]

    y_val = val_df[
        target_column
    ]

    X_test = test_df[
        feature_columns
    ]

    y_test = test_df[
        target_column
    ]

    # --------------------------------------------------
    # Print dataset information
    # --------------------------------------------------

    print("\nDataset split:")
    print(
        f"Train transactions: "
        f"{len(train_transactions):,}"
    )

    print(
        f"Validation transactions: "
        f"{len(val_transactions):,}"
    )

    print(
        f"Test transactions: "
        f"{len(test_transactions):,}"
    )

    print("\nRows:")
    print(
        f"Train rows: "
        f"{len(train_df):,}"
    )

    print(
        f"Validation rows: "
        f"{len(val_df):,}"
    )

    print(
        f"Test rows: "
        f"{len(test_df):,}"
    )

    print("\nTarget distribution:")

    print(
        "Train:"
    )

    print(
        y_train.value_counts(
            normalize=True
        )
    )

    print(
        "\nValidation:"
    )

    print(
        y_val.value_counts(
            normalize=True
        )
    )

    print(
        "\nTest:"
    )

    print(
        y_test.value_counts(
            normalize=True
        )
    )

    # --------------------------------------------------
    # Verify transaction leakage
    # --------------------------------------------------

    train_ids = set(
        train_df["transaction_id"]
    )

    val_ids = set(
        val_df["transaction_id"]
    )

    test_ids = set(
        test_df["transaction_id"]
    )

    print("\nLeakage checks:")

    print(
        "Train ∩ Validation:",
        len(train_ids & val_ids)
    )

    print(
        "Train ∩ Test:",
        len(train_ids & test_ids)
    )

    print(
        "Validation ∩ Test:",
        len(val_ids & test_ids)
    )

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    )


if __name__ == "__main__":

    prepare_dataset()
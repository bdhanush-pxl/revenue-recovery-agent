import joblib
import pandas as pd

from prepare_ml_dataset import prepare_dataset
from recovery_decision_engine import (
    decide,
)


MODEL_FILE = "models/recovery_predictor.joblib"


# ---------------------------------------------------------
# Load the original recovery dataset
# ---------------------------------------------------------

def load_test_data():

    df = pd.read_csv(
        "data/recovery_training_data.csv"
    )

    # ---------------------------------------------
    # Recreate the same transaction split
    # ---------------------------------------------

    transactions = (
        df["transaction_id"]
        .astype(str)
        .drop_duplicates()
        .to_numpy()
    )

    from sklearn.model_selection import train_test_split

    train_transactions, temp_transactions = (
        train_test_split(
            transactions,
            test_size=0.30,
            random_state=42,
        )
    )

    _, test_transactions = (
        train_test_split(
            temp_transactions,
            test_size=0.50,
            random_state=42,
        )
    )

    test_df = df[
        df["transaction_id"].isin(
            test_transactions
        )
    ].copy()

    return test_df


# ---------------------------------------------------------
# Create one payment context from the dataset
# ---------------------------------------------------------

def extract_payment(row):

    return {
        "amount": row["amount"],
        "payment_method": row["payment_method"],
        "bank": row["bank"],
        "failure_code": row["failure_code"],
        "failure_category": row["failure_category"],
        "attempt_number": row["attempt_number"],
        "customer_tenure_days": row["customer_tenure_days"],
        "customer_success_rate": row["customer_success_rate"],
        "previous_attempts": row["previous_attempts"],
        "previous_recoveries": row["previous_recoveries"],
        "bank_health_score": row["bank_health_score"],
        "gateway_health_score": row["gateway_health_score"],
        "gateway_latency_ms": row["gateway_latency_ms"],
        "hour": row["hour"],
        "day_of_week": row["day_of_week"],
        "is_weekend": row["is_weekend"],
    }


# ---------------------------------------------------------
# Evaluate strategies
# ---------------------------------------------------------

def evaluate():

    print(
        "Loading model..."
    )

    model = joblib.load(
        MODEL_FILE
    )

    print(
        "Loading test transactions..."
    )

    test_df = load_test_data()

    # ---------------------------------------------
    # We only need one row per transaction
    # because the six rows represent six actions.
    # ---------------------------------------------

    transactions = (
        test_df
        .drop_duplicates(
            subset=["transaction_id"]
        )
    )

    print(
        f"Test transactions: "
        f"{len(transactions):,}"
    )

    # ---------------------------------------------
    # Lookup actual simulator outcomes
    # ---------------------------------------------

    outcome_lookup = (
        test_df
        .set_index(
            [
                "transaction_id",
                "recovery_action",
            ]
        )
    )

    strategies = {
        "ALWAYS_RETRY_NOW":
            lambda row: "RETRY_NOW",

        "ALWAYS_RETRY_LATER":
            lambda row: "RETRY_LATER",

        "RULE_BASED":
            lambda row: rule_based_action(row),
    }

    # ---------------------------------------------
    # Add ML strategy separately
    # ---------------------------------------------

    results = {}

    for strategy_name, strategy_function in strategies.items():

        recovered_amount = 0.0
        successful_transactions = 0

        for _, row in transactions.iterrows():

            action = strategy_function(
                row
            )

            try:

                outcome = outcome_lookup.loc[
                    (
                        row["transaction_id"],
                        action,
                    )
                ]

                success = int(
                    outcome[
                        "recovery_success"
                    ]
                )

                amount = float(
                    outcome[
                        "recovered_amount"
                    ]
                )

            except KeyError:

                success = 0
                amount = 0.0

            recovered_amount += amount

            successful_transactions += success

        results[
            strategy_name
        ] = {
            "recovered_amount":
                recovered_amount,

            "successful_transactions":
                successful_transactions,
        }

    # ---------------------------------------------
    # ML strategy
    # ---------------------------------------------

    ml_recovered = 0.0
    ml_successes = 0

    for _, row in transactions.iterrows():

        payment = extract_payment(
            row
        )

        _, best = decide(
            model,
            payment
        )

        action = best[
            "recovery_action"
        ]

        outcome = outcome_lookup.loc[
            (
                row["transaction_id"],
                action,
            )
        ]

        ml_recovered += float(
            outcome[
                "recovered_amount"
            ]
        )

        ml_successes += int(
            outcome[
                "recovery_success"
            ]
        )

    results[
        "ML_DECISION_ENGINE"
    ] = {
        "recovered_amount":
            ml_recovered,

        "successful_transactions":
            ml_successes,
    }

    # ---------------------------------------------
    # Print results
    # ---------------------------------------------

    print(
        "\n======================================"
    )

    print(
        "RECOVERY STRATEGY EVALUATION"
    )

    print(
        "======================================"
    )

    for name, result in results.items():

        print(
            f"\n{name}"
        )

        print(
            "Recovered amount: ₹",
            round(
                result[
                    "recovered_amount"
                ],
                2
            )
        )

        print(
            "Successful recoveries:",
            result[
                "successful_transactions"
            ]
        )

    print(
        "\n======================================"
    )


# ---------------------------------------------------------
# Rule-based baseline
# ---------------------------------------------------------

def rule_based_action(row):

    failure = row[
        "failure_code"
    ]

    if failure == "TIMEOUT":

        return "RETRY_LATER"

    if failure == "BANK_UNAVAILABLE":

        return "RETRY_LATER"

    if failure == "GATEWAY_ERROR":

        return "RETRY_LATER"

    if failure == "EXPIRED_CARD":

        return "CHANGE_PAYMENT_METHOD"

    if failure == "INSUFFICIENT_FUNDS":

        return "CUSTOMER_NUDGE"

    if failure == "INVALID_DETAILS":

        return "CUSTOMER_NUDGE"

    if failure == "HARD_DECLINE":

        return "ESCALATE"

    return "DO_NOT_RETRY"


if __name__ == "__main__":

    evaluate()
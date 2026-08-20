import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from recovery_decision_engine import decide


MODEL_FILE = "models/recovery_predictor.joblib"
DATA_FILE = "data/recovery_training_data.csv"


def get_test_transactions(df):
    """
    Recreate the exact transaction-level test split
    used during model training.
    """

    transactions = (
        df["transaction_id"]
        .astype(str)
        .drop_duplicates()
        .to_numpy()
    )

    _, temp_transactions = train_test_split(
        transactions,
        test_size=0.30,
        random_state=42,
    )

    _, test_transactions = train_test_split(
        temp_transactions,
        test_size=0.50,
        random_state=42,
    )

    return test_transactions


def extract_payment(row):
    """
    Extract only information available at decision time.
    """

    return {
        "amount": float(row["amount"]),
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


def rule_based_action(row):
    """
    Static business-rule baseline.
    """

    failure = row["failure_code"]

    if failure in [
        "TIMEOUT",
        "BANK_UNAVAILABLE",
        "GATEWAY_ERROR",
    ]:
        return "RETRY_LATER"

    if failure == "EXPIRED_CARD":
        return "CHANGE_PAYMENT_METHOD"

    if failure in [
        "INSUFFICIENT_FUNDS",
        "INVALID_DETAILS",
    ]:
        return "CUSTOMER_NUDGE"

    if failure == "HARD_DECLINE":
        return "ESCALATE"

    return "DO_NOT_RETRY"


def evaluate():

    print("Loading data...")

    df = pd.read_csv(
        DATA_FILE
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    test_transactions = get_test_transactions(
        df
    )

    test_df = df[
        df["transaction_id"].isin(
            test_transactions
        )
    ].copy()

    print(
        f"Test transactions: "
        f"{test_df['transaction_id'].nunique():,}"
    )

    model = joblib.load(
        MODEL_FILE
    )

    # -----------------------------------------------------
    # Evaluate one transaction at a time
    # -----------------------------------------------------

    results = []

    for transaction_id, group in test_df.groupby(
        "transaction_id"
    ):

        # One row is enough to obtain transaction context
        base_row = group.iloc[0]

        payment = extract_payment(
            base_row
        )

        # ---------------------------------------------
        # ML decision
        # ---------------------------------------------

        scored, best = decide(
            model,
            payment
        )

        ml_action = best[
            "recovery_action"
        ]

        ml_probability = float(
            best[
                "recovery_probability"
            ]
        )

        ml_expected_value = float(
            best[
                "expected_recovery_value"
            ]
        )

        # ---------------------------------------------
        # Simulator's theoretical best action
        # ---------------------------------------------

        oracle_row = group.loc[
            group[
                "recovery_probability"
            ].idxmax()
        ]

        oracle_action = oracle_row[
            "recovery_action"
        ]

        oracle_probability = float(
            oracle_row[
                "recovery_probability"
            ]
        )

        oracle_expected_value = float(
            oracle_row[
                "amount"
            ]
            * oracle_row[
                "recovery_probability"
            ]
        )

        # ---------------------------------------------
        # Actual outcome of ML-selected action
        # ---------------------------------------------

        ml_outcome = group[
            group[
                "recovery_action"
            ] == ml_action
        ].iloc[0]

        ml_recovered_amount = float(
            ml_outcome[
                "recovered_amount"
            ]
        )

        ml_success = int(
            ml_outcome[
                "recovery_success"
            ]
        )

        # ---------------------------------------------
        # Actual outcome of oracle action
        # ---------------------------------------------

        oracle_outcome = group[
            group[
                "recovery_action"
            ] == oracle_action
        ].iloc[0]

        oracle_recovered_amount = float(
            oracle_outcome[
                "recovered_amount"
            ]
        )

        oracle_success = int(
            oracle_outcome[
                "recovery_success"
            ]
        )

        # ---------------------------------------------
        # Rule-based action
        # ---------------------------------------------

        rule_action = rule_based_action(
            base_row
        )

        rule_outcome = group[
            group[
                "recovery_action"
            ] == rule_action
        ].iloc[0]

        rule_recovered_amount = float(
            rule_outcome[
                "recovered_amount"
            ]
        )

        results.append({

            "transaction_id":
                transaction_id,

            "failure_code":
                base_row["failure_code"],

            "failure_category":
                base_row["failure_category"],

            "bank":
                base_row["bank"],

            "amount":
                float(base_row["amount"]),

            "timestamp":
                base_row["timestamp"],

            "ml_action":
                ml_action,

            "ml_probability":
                ml_probability,

            "ml_expected_value":
                ml_expected_value,

            "ml_recovered_amount":
                ml_recovered_amount,

            "ml_success":
                ml_success,

            "oracle_action":
                oracle_action,

            "oracle_probability":
                oracle_probability,

            "oracle_expected_value":
                oracle_expected_value,

            "oracle_recovered_amount":
                oracle_recovered_amount,

            "oracle_success":
                oracle_success,

            "rule_action":
                rule_action,

            "rule_recovered_amount":
                rule_recovered_amount,

            "ml_matches_oracle":
                ml_action == oracle_action,

        })

    results_df = pd.DataFrame(
        results
    )

    # -----------------------------------------------------
    # Overall policy agreement
    # -----------------------------------------------------

    agreement = (
        results_df[
            "ml_matches_oracle"
        ].mean()
    )

    print(
        "\n=========================================="
    )

    print(
        "POLICY EVALUATION"
    )

    print(
        "=========================================="
    )

    print(
        f"\nML / Oracle action agreement: "
        f"{agreement:.2%}"
    )

    # -----------------------------------------------------
    # Expected recovery comparison
    # -----------------------------------------------------

    total_ml_expected = (
        results_df[
            "ml_expected_value"
        ].sum()
    )

    total_oracle_expected = (
        results_df[
            "oracle_expected_value"
        ].sum()
    )

    total_rule_recovered = (
        results_df[
            "rule_recovered_amount"
        ].sum()
    )

    total_ml_recovered = (
        results_df[
            "ml_recovered_amount"
        ].sum()
    )

    total_oracle_recovered = (
        results_df[
            "oracle_recovered_amount"
        ].sum()
    )

    print(
        "\nExpected recovery:"
    )

    print(
        "ML expected recovery: ₹",
        round(
            total_ml_expected,
            2
        )
    )

    print(
        "Oracle expected recovery: ₹",
        round(
            total_oracle_expected,
            2
        )
    )

    print(
        "\nActual simulated recovery:"
    )

    print(
        "ML recovered: ₹",
        round(
            total_ml_recovered,
            2
        )
    )

    print(
        "Rule-based recovered: ₹",
        round(
            total_rule_recovered,
            2
        )
    )

    print(
        "Oracle recovered: ₹",
        round(
            total_oracle_recovered,
            2
        )
    )

    # -----------------------------------------------------
    # Failure-type analysis
    # -----------------------------------------------------

    print(
        "\n=========================================="
    )

    print(
        "AGREEMENT BY FAILURE TYPE"
    )

    print(
        "=========================================="
    )

    failure_summary = (
        results_df
        .groupby(
            "failure_code"
        )
        .agg(
            transactions=(
                "transaction_id",
                "count"
            ),
            agreement=(
                "ml_matches_oracle",
                "mean"
            ),
            ml_recovered=(
                "ml_recovered_amount",
                "sum"
            ),
            oracle_recovered=(
                "oracle_recovered_amount",
                "sum"
            ),
        )
        .sort_values(
            "agreement",
            ascending=False
        )
    )

    print(
        failure_summary.to_string()
    )

    # -----------------------------------------------------
    # Action distribution
    # -----------------------------------------------------

    print(
        "\n=========================================="
    )

    print(
        "ML ACTION DISTRIBUTION"
    )

    print(
        "=========================================="
    )

    print(
        results_df[
            "ml_action"
        ].value_counts()
    )

    # -----------------------------------------------------
    # HDFC incident analysis
    # -----------------------------------------------------

    incident_start = pd.Timestamp(
        "2026-08-12 08:00"
    )

    incident_end = pd.Timestamp(
        "2026-08-12 20:00"
    )

    incident_mask = (
        (results_df["bank"] == "HDFC")
        &
        (
            results_df["timestamp"]
            >= incident_start
        )
        &
        (
            results_df["timestamp"]
            <= incident_end
        )
    )

    incident_df = results_df[
        incident_mask
    ]

    normal_hdfc_df = results_df[
        (results_df["bank"] == "HDFC")
        &
        ~incident_mask
    ]

    print(
        "\n=========================================="
    )

    print(
        "HDFC INCIDENT ANALYSIS"
    )

    print(
        "=========================================="
    )

    print(
        f"\nIncident transactions: "
        f"{len(incident_df)}"
    )

    print(
        f"Normal HDFC transactions: "
        f"{len(normal_hdfc_df)}"
    )

    if len(incident_df) > 0:

        print(
            "\nIncident ML actions:"
        )

        print(
            incident_df[
                "ml_action"
            ].value_counts()
        )

        print(
            "\nIncident ML/oracle agreement:",
            f"{incident_df['ml_matches_oracle'].mean():.2%}"
        )

    if len(normal_hdfc_df) > 0:

        print(
            "\nNormal HDFC ML actions:"
        )

        print(
            normal_hdfc_df[
                "ml_action"
            ].value_counts()
        )

        print(
            "\nNormal HDFC ML/oracle agreement:",
            f"{normal_hdfc_df['ml_matches_oracle'].mean():.2%}"
        )

    # -----------------------------------------------------
    # Save detailed results
    # -----------------------------------------------------

    output_file = (
        "data/policy_evaluation_results.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nDetailed results saved to:"
        f" {output_file}"
    )


if __name__ == "__main__":

    evaluate()
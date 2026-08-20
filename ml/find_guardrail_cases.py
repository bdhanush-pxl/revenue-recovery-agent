import pandas as pd


PAYMENTS_FILE = "data/payments.csv"
RECOVERY_FILE = "data/recovery_training_data.csv"


def main():

    payments = pd.read_csv(
        PAYMENTS_FILE
    )

    recovery = pd.read_csv(
        RECOVERY_FILE
    )

    # ==================================================
    # 1. PERMANENT FAILURE
    # ==================================================

    print("\n==========================================")
    print("PERMANENT FAILURE CASES")
    print("==========================================")

    permanent = payments[
        payments["failure_category"]
        == "PERMANENT"
    ]

    print(
        permanent[
            [
                "transaction_id",
                "amount",
                "failure_code",
                "bank",
                "attempt_number",
                "bank_health_score",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    # ==================================================
    # 2. DEGRADED BANK CASES
    # ==================================================

    print("\n==========================================")
    print("DEGRADED BANK CASES")
    print("==========================================")

    degraded = payments[
        payments["bank_health_score"] < 0.70
    ]

    print(
        degraded[
            [
                "transaction_id",
                "amount",
                "failure_code",
                "failure_category",
                "bank",
                "bank_health_score",
                "attempt_number",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    # ==================================================
    # 3. VERY LOW PROBABILITY CASES
    # ==================================================

    print("\n==========================================")
    print("VERY LOW RECOVERY PROBABILITY CASES")
    print("==========================================")

    low_probability = recovery[
        recovery["recovery_probability"] < 0.10
    ]

    # Show only unique transactions
    low_probability = (
        low_probability
        .sort_values(
            "recovery_probability"
        )
        .drop_duplicates(
            "transaction_id"
        )
    )

    print(
        low_probability[
            [
                "transaction_id",
                "amount",
                "failure_code",
                "failure_category",
                "recovery_probability",
                "recovery_action",
                "attempt_number",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
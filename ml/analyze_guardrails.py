import pandas as pd


PAYMENTS_FILE = "data/payments.csv"
RECOVERY_FILE = "data/recovery_training_data.csv"


def main():

    print("Loading datasets...")

    payments = pd.read_csv(
        PAYMENTS_FILE
    )

    recovery = pd.read_csv(
        RECOVERY_FILE
    )

    print(
        f"Payments: {len(payments):,}"
    )

    print(
        f"Recovery rows: {len(recovery):,}"
    )

    # ==================================================
    # 1. PAYMENT AMOUNT DISTRIBUTION
    # ==================================================

    print(
        "\n=========================================="
    )

    print(
        "PAYMENT AMOUNT DISTRIBUTION"
    )

    print(
        "=========================================="
    )

    print(
        payments["amount"].describe(
            percentiles=[
                0.50,
                0.75,
                0.90,
                0.95,
                0.99
            ]
        )
    )

    # ==================================================
    # 2. ATTEMPT DISTRIBUTION
    # ==================================================

    print(
        "\n=========================================="
    )

    print(
        "ATTEMPT DISTRIBUTION"
    )

    print(
        "=========================================="
    )

    print(
        payments[
            "attempt_number"
        ].value_counts()
        .sort_index()
    )

    # ==================================================
    # 3. RECOVERY BY ATTEMPT
    # ==================================================

    print(
        "\n=========================================="
    )

    print(
        "RECOVERY BY ATTEMPT"
    )

    print(
        "=========================================="
    )

    attempt_summary = (
        recovery
        .groupby(
            "attempt_number"
        )
        .agg(
            transactions=(
                "transaction_id",
                "nunique"
            ),
            recovery_probability=(
                "recovery_success",
                "mean"
            ),
            avg_recovery_probability=(
                "recovery_probability",
                "mean"
            ),
            recovered_amount=(
                "recovered_amount",
                "sum"
            ),
        )
    )

    print(
        attempt_summary.to_string()
    )

    # ==================================================
    # 4. RECOVERY BY BANK HEALTH
    # ==================================================

    print(
        "\n=========================================="
    )

    print(
        "RECOVERY BY BANK HEALTH"
    )

    print(
        "=========================================="
    )

    payments["health_bucket"] = pd.cut(
        payments[
            "bank_health_score"
        ],
        bins=[
            0,
            0.60,
            0.70,
            0.80,
            0.90,
            1.00
        ],
        labels=[
            "<0.60",
            "0.60-0.70",
            "0.70-0.80",
            "0.80-0.90",
            "0.90-1.00"
        ],
        include_lowest=True
    )

    health_summary = (
        payments
        .groupby(
            "health_bucket",
            observed=True
        )
        .agg(
            transactions=(
                "transaction_id",
                "count"
            ),
            failure_rate=(
                "payment_status",
                lambda x:
                (x == "FAILED").mean()
            ),
            avg_health=(
                "bank_health_score",
                "mean"
            )
        )
    )

    print(
        health_summary.to_string()
    )

    # ==================================================
    # 5. RECOVERY BY FAILURE CATEGORY
    # ==================================================

    print(
        "\n=========================================="
    )

    print(
        "RECOVERY BY FAILURE CATEGORY"
    )

    print(
        "=========================================="
    )

    category_summary = (
        recovery
        .groupby(
            "failure_category"
        )
        .agg(
            transactions=(
                "transaction_id",
                "nunique"
            ),
            recovery_rate=(
                "recovery_success",
                "mean"
            ),
            recovered_amount=(
                "recovered_amount",
                "sum"
            )
        )
        .sort_values(
            "recovery_rate",
            ascending=False
        )
    )

    print(
        category_summary.to_string()
    )

    # ==================================================
    # 6. RECOVERY BY AMOUNT BAND
    # ==================================================

    print(
        "\n=========================================="
    )

    print(
        "RECOVERY BY AMOUNT BAND"
    )

    print(
        "=========================================="
    )

    recovery_amounts = (
        recovery[
            [
                "transaction_id",
                "amount",
                "recovery_success",
                "recovered_amount"
            ]
        ]
        .drop_duplicates(
            "transaction_id"
        )
    )

    recovery_amounts[
        "amount_band"
    ] = pd.cut(
        recovery_amounts[
            "amount"
        ],
        bins=[
            0,
            1000,
            5000,
            10000,
            25000,
            50000,
            float("inf")
        ],
        labels=[
            "<1K",
            "1K-5K",
            "5K-10K",
            "10K-25K",
            "25K-50K",
            "50K+"
        ],
        include_lowest=True
    )

    amount_summary = (
        recovery_amounts
        .groupby(
            "amount_band",
            observed=True
        )
        .agg(
            transactions=(
                "transaction_id",
                "count"
            ),
            recovery_rate=(
                "recovery_success",
                "mean"
            ),
            recovered_amount=(
                "recovered_amount",
                "sum"
            )
        )
    )

    print(
        amount_summary.to_string()
    )


if __name__ == "__main__":

    main()
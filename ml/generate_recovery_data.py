import pandas as pd
import sys
import os

# ---------------------------------------------------------
# Make sure Python can find recovery_simulator.py
# ---------------------------------------------------------

sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)

from recovery_simulator import (
    RECOVERY_ACTIONS,
    calculate_recovery_probability,
    simulate_recovery
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

INPUT_FILE = "data/payments.csv"
OUTPUT_FILE = "data/recovery_training_data.csv"


# ---------------------------------------------------------
# Generate recovery dataset
# ---------------------------------------------------------

def generate_recovery_dataset():

    print("Loading payment data...")

    df = pd.read_csv(
        INPUT_FILE
    )

    # -----------------------------------------------------
    # Only failed payments need recovery
    # -----------------------------------------------------

    failed_df = df[
        df["payment_status"] == "FAILED"
    ].copy()

    print(
        f"Failed payments found: "
        f"{len(failed_df):,}"
    )

    records = []

    # -----------------------------------------------------
    # For every failed payment
    # evaluate every possible action
    # -----------------------------------------------------

    for _, row in failed_df.iterrows():

        for action in RECOVERY_ACTIONS:

            result = simulate_recovery(
                row,
                action
            )

            record = {

                # -----------------------------
                # Transaction information
                # -----------------------------

                "transaction_id":
                    row["transaction_id"],

                "customer_id":
                    row["customer_id"],

                "merchant_id":
                    row["merchant_id"],

                "timestamp":
                    row["timestamp"],

                "amount":
                    row["amount"],

                "payment_method":
                    row["payment_method"],

                "bank":
                    row["bank"],

                # -----------------------------
                # Failure information
                # -----------------------------

                "failure_code":
                    row["failure_code"],

                "failure_category":
                    row["failure_category"],

                "attempt_number":
                    row["attempt_number"],

                # -----------------------------
                # Customer information
                # -----------------------------

                "customer_tenure_days":
                    row["customer_tenure_days"],

                "customer_success_rate":
                    row["customer_success_rate"],

                "previous_attempts":
                    row["previous_attempts"],

                "previous_recoveries":
                    row["previous_recoveries"],

                # -----------------------------
                # Infrastructure
                # -----------------------------

                "bank_health_score":
                    row["bank_health_score"],

                "gateway_health_score":
                    row["gateway_health_score"],

                "gateway_latency_ms":
                    row["gateway_latency_ms"],

                # -----------------------------
                # Time
                # -----------------------------

                "hour":
                    row["hour"],

                "day_of_week":
                    row["day_of_week"],

                "is_weekend":
                    row["is_weekend"],

                # -----------------------------
                # Recovery decision
                # -----------------------------

                "recovery_action":
                    action,

                # -----------------------------
                # Ground truth
                # -----------------------------

                "recovery_probability":
                    result[
                        "recovery_probability"
                    ],

                "recovery_success":
                    result[
                        "recovery_success"
                    ],

                "recovered_amount":
                    result[
                        "recovered_amount"
                    ]
            }

            records.append(
                record
            )

    # -----------------------------------------------------
    # Create dataframe
    # -----------------------------------------------------

    recovery_df = pd.DataFrame(
        records
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    recovery_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nGenerated "
        f"{len(recovery_df):,} recovery records."
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )

    # -----------------------------------------------------
    # Basic statistics
    # -----------------------------------------------------

    print(
        "\nRecovery actions:"
    )

    print(
        recovery_df[
            "recovery_action"
        ].value_counts()
    )

    print(
        "\nRecovery success:"
    )

    print(
        recovery_df[
            "recovery_success"
        ].value_counts()
    )

    print(
        "\nAverage recovery probability:"
    )

    print(
        recovery_df[
            "recovery_probability"
        ].mean()
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    generate_recovery_dataset()
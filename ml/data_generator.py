import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

N_TRANSACTIONS = 50_000
SEED = 42

np.random.seed(SEED)

OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "payments.csv")


# ---------------------------------------------------------
# Basic entities
# ---------------------------------------------------------

BANKS = [
    "HDFC",
    "ICICI",
    "SBI",
    "AXIS",
    "KOTAK",
    "PNB",
    "BOB",
    "IDFC",
    "YES",
    "INDUSIND"
]

PAYMENT_METHODS = [
    "UPI",
    "CARD",
    "NETBANKING",
    "WALLET"
]

FAILURE_CODES = [
    "TIMEOUT",
    "INSUFFICIENT_FUNDS",
    "EXPIRED_CARD",
    "HARD_DECLINE",
    "INVALID_DETAILS",
    "BANK_UNAVAILABLE",
    "GATEWAY_ERROR"
]

FAILURE_CATEGORIES = {
    "TIMEOUT": "TEMPORARY",
    "INSUFFICIENT_FUNDS": "CUSTOMER_FIXABLE",
    "EXPIRED_CARD": "CUSTOMER_FIXABLE",
    "HARD_DECLINE": "PERMANENT",
    "INVALID_DETAILS": "CUSTOMER_FIXABLE",
    "BANK_UNAVAILABLE": "SYSTEMIC",
    "GATEWAY_ERROR": "SYSTEMIC"
}

RECOVERY_ACTIONS = [
    "RETRY_NOW",
    "RETRY_LATER",
    "CUSTOMER_NUDGE",
    "CHANGE_PAYMENT_METHOD",
    "ESCALATE",
    "DO_NOT_RETRY"
]


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_bank_health(timestamps, banks):
    """
    Generate bank health scores.

    Most banks remain healthy.
    One bank experiences a simulated degradation
    during a specific time window.
    """

    health = np.random.normal(
        loc=0.94,
        scale=0.025,
        size=len(timestamps)
    )

    health = np.clip(
        health,
        0.70,
        1.0
    )

    # ---------------------------------------------
    # Simulated systemic incident
    # ---------------------------------------------
    #
    # HDFC experiences degradation on
    # 12 Aug 2026 between 11 AM and 2 PM.
    #

    degradation_start = pd.Timestamp(
        "2026-08-12 08:00"
    )

    degradation_end = pd.Timestamp(
        "2026-08-12 20:00"
    )

    affected_bank = "HDFC"

    mask = (
        (timestamps >= degradation_start)
        & (timestamps <= degradation_end)
        & (banks == affected_bank)
    )

    # Significantly reduce health for
    # affected transactions.

    health[mask] -= np.random.uniform(
        0.25,
        0.40,
        mask.sum()
    )

    return np.clip(
        health,
        0.30,
        1.0
    )


# ---------------------------------------------------------
# Generate transactions
# ---------------------------------------------------------

def generate_transactions(n):

    # -----------------------------
    # IDs
    # -----------------------------

    transaction_ids = [
        f"TX{i:06d}"
        for i in range(1, n + 1)
    ]

    customer_ids = [
        f"C{np.random.randint(1, 20_001):05d}"
        for _ in range(n)
    ]

    merchant_ids = [
        f"M{np.random.randint(1, 501):04d}"
        for _ in range(n)
    ]

    # -----------------------------
    # Timestamp
    # -----------------------------

    start_date = pd.Timestamp("2026-08-01")
    end_date = pd.Timestamp("2026-08-15")

    timestamps = pd.to_datetime(
        np.random.uniform(
            start_date.value,
            end_date.value,
            n
        )
    )

    timestamps = pd.Series(timestamps)

    # -----------------------------
    # Transaction amount
    # -----------------------------

    # Log-normal distribution creates many
    # small/medium transactions and fewer
    # high-value transactions.

    amounts = np.random.lognormal(
        mean=np.log(1800),
        sigma=1.0,
        size=n
    )

    amounts = np.clip(
        amounts,
        100,
        100_000
    )

    amounts = np.round(amounts, 2)

    # -----------------------------
    # Payment method
    # -----------------------------

    payment_methods = np.random.choice(
        PAYMENT_METHODS,
        size=n,
        p=[0.55, 0.25, 0.15, 0.05]
    )

    # -----------------------------
    # Banks
    # -----------------------------

    banks = np.random.choice(
        BANKS,
        size=n
    )

    # -----------------------------
    # Customer behavior
    # -----------------------------

    customer_tenure = np.random.randint(
        30,
        1500,
        size=n
    )

    customer_success_rate = np.random.beta(
        8,
        2,
        size=n
    )

    previous_attempts = np.random.poisson(
        lam=0.8,
        size=n
    )

    previous_recoveries = np.minimum(
        previous_attempts,
        np.random.poisson(
            lam=0.5,
            size=n
        )
    )

    # -----------------------------
    # Infrastructure
    # -----------------------------

    bank_health = generate_bank_health(
    timestamps,
    banks
    )

    gateway_health = np.random.normal(
        0.95,
        0.025,
        n
    )

    gateway_health = np.clip(
        gateway_health,
        0.70,
        1.0
    )

    gateway_latency = np.random.normal(
        180,
        60,
        n
    )

    gateway_latency = np.clip(
        gateway_latency,
        50,
        1000
    )

    # Make latency worse when bank health is poor.
    gateway_latency += (
        (1 - bank_health) * 600
    )

    # -----------------------------
    # Time features
    # -----------------------------

    hours = timestamps.dt.hour.values

    day_of_week = timestamps.dt.dayofweek.values

    is_weekend = (
        day_of_week >= 5
    ).astype(int)

    # -----------------------------
    # Failure probability
    # -----------------------------

    base_failure_probability = 0.07

    # System degradation increases failures.
    system_penalty = (
        (1 - bank_health) * 0.60
        +
        (1 - gateway_health) * 0.30
    )

    # High latency increases failure probability.
    latency_penalty = np.maximum(
        gateway_latency - 250,
        0
    ) / 3000

    failure_probability = (
        base_failure_probability
        + system_penalty
        + latency_penalty
    )

    failure_probability = np.clip(
        failure_probability,
        0.02,
        0.75
    )

    payment_failed = (
        np.random.random(n)
        < failure_probability
    )

    # -----------------------------
    # Failure codes
    # -----------------------------

    failure_codes = np.full(
        n,
        "NONE",
        dtype=object
    )

    failed_indices = np.where(
        payment_failed
    )[0]

    for i in failed_indices:

        # Systemic failures become more likely
        # when infrastructure health is poor.

        if bank_health[i] < 0.75:
            choices = [
                "TIMEOUT",
                "BANK_UNAVAILABLE",
                "GATEWAY_ERROR"
            ]

            probs = [
                0.45,
                0.40,
                0.15
            ]

        else:
            choices = FAILURE_CODES

            probs = [
                0.25,  # timeout
                0.20,  # insufficient funds
                0.15,  # expired card
                0.15,  # hard decline
                0.10,  # invalid details
                0.10,  # bank unavailable
                0.05   # gateway error
            ]

        failure_codes[i] = np.random.choice(
            choices,
            p=probs
        )

    failure_categories = [
        FAILURE_CATEGORIES.get(
            code,
            "NONE"
        )
        for code in failure_codes
    ]

    # -----------------------------
    # Attempt number
    # -----------------------------

    attempt_number = (
        previous_attempts + 1
    )

    # -----------------------------
    # Create dataframe
    # -----------------------------

    df = pd.DataFrame({
        "transaction_id": transaction_ids,
        "customer_id": customer_ids,
        "merchant_id": merchant_ids,
        "timestamp": timestamps,
        "amount": amounts,
        "payment_method": payment_methods,
        "bank": banks,

        "payment_status": np.where(
            payment_failed,
            "FAILED",
            "SUCCESS"
        ),

        "failure_code": failure_codes,
        "failure_category": failure_categories,

        "attempt_number": attempt_number,

        "customer_tenure_days": customer_tenure,
        "customer_success_rate": np.round(
            customer_success_rate,
            3
        ),

        "previous_attempts": previous_attempts,
        "previous_recoveries": previous_recoveries,

        "bank_health_score": np.round(
            bank_health,
            3
        ),

        "gateway_health_score": np.round(
            gateway_health,
            3
        ),

        "gateway_latency_ms": np.round(
            gateway_latency,
            2
        ),

        "hour": hours,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend
    })

    return df


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("Generating synthetic payment data...")

    df = generate_transactions(
        N_TRANSACTIONS
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nGenerated {len(df):,} transactions."
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )

    print("\nPayment status:")
    print(
        df["payment_status"]
        .value_counts()
    )

    print("\nFailure distribution:")
    print(
        df.loc[
            df["payment_status"] == "FAILED",
            "failure_code"
        ].value_counts()
    )

    print("\nFirst 5 rows:")
    print(
        df.head()
    )
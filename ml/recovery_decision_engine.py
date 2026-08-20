import joblib
import pandas as pd


MODEL_FILE = "models/recovery_predictor.joblib"


ACTIONS = [
    "RETRY_NOW",
    "RETRY_LATER",
    "CUSTOMER_NUDGE",
    "CHANGE_PAYMENT_METHOD",
    "ESCALATE",
    "DO_NOT_RETRY",
]


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

def load_model():

    return joblib.load(
        MODEL_FILE
    )


# ---------------------------------------------------------
# Create candidate action rows
# ---------------------------------------------------------

def create_candidates(
    payment
):

    candidates = []

    for action in ACTIONS:

        row = payment.copy()

        row[
            "recovery_action"
        ] = action

        candidates.append(
            row
        )

    return pd.DataFrame(
        candidates
    )


# ---------------------------------------------------------
# Predict recovery probabilities
# ---------------------------------------------------------

def score_actions(
    model,
    payment
):

    candidates = create_candidates(
        payment
    )

    probabilities = (
        model.predict_proba(
            candidates
        )[:, 1]
    )

    candidates[
        "recovery_probability"
    ] = probabilities

    candidates[
        "expected_recovery_value"
    ] = (
        candidates["amount"]
        * candidates[
            "recovery_probability"
        ]
    )

    return candidates


# ---------------------------------------------------------
# Apply guardrails
# ---------------------------------------------------------

def apply_guardrails(
    candidates,
    payment
):

    candidates = candidates.copy()

    # -----------------------------------------------------
    # Rule 1:
    # DO_NOT_RETRY has no recovery attempt
    # -----------------------------------------------------

    candidates.loc[
        candidates["recovery_action"]
        == "DO_NOT_RETRY",
        "recovery_probability"
    ] = 0.0

    candidates.loc[
        candidates["recovery_action"]
        == "DO_NOT_RETRY",
        "expected_recovery_value"
    ] = 0.0

    # -----------------------------------------------------
    # Rule 2:
    # Permanent failures should not be retried immediately
    # -----------------------------------------------------

    if payment[
        "failure_category"
    ] == "PERMANENT":

        candidates.loc[
            candidates[
                "recovery_action"
            ] == "RETRY_NOW",
            "expected_recovery_value"
        ] = 0.0

    # -----------------------------------------------------
    # Rule 3:
    # Extremely low-value opportunities
    #
    # Don't waste expensive recovery actions.
    # -----------------------------------------------------

    if payment["amount"] < 100:

        candidates.loc[
            candidates[
                "recovery_action"
            ] == "ESCALATE",
            "expected_recovery_value"
        ] = 0.0

    return candidates


# ---------------------------------------------------------
# Select best action
# ---------------------------------------------------------

def select_best_action(
    candidates
):

    best_index = (
        candidates[
            "expected_recovery_value"
        ]
        .idxmax()
    )

    return candidates.loc[
        best_index
    ]


# ---------------------------------------------------------
# Main decision function
# ---------------------------------------------------------

def decide(
    model,
    payment
):

    scored = score_actions(
        model,
        payment
    )

    scored = apply_guardrails(
        scored,
        payment
    )

    best = select_best_action(
        scored
    )

    return (
        scored,
        best
    )


# ---------------------------------------------------------
# Example
# ---------------------------------------------------------

if __name__ == "__main__":

    model = load_model()

    # -----------------------------------------------------
    # Example failed payment
    # -----------------------------------------------------

    payment = {
        "amount": 10000.0,
        "payment_method": "UPI",
        "bank": "HDFC",
        "failure_code": "TIMEOUT",
        "failure_category": "TEMPORARY",
        "attempt_number": 1,
        "customer_tenure_days": 500,
        "customer_success_rate": 0.85,
        "previous_attempts": 5,
        "previous_recoveries": 3,
        "bank_health_score": 0.95,
        "gateway_health_score": 0.92,
        "gateway_latency_ms": 350,
        "hour": 14,
        "day_of_week": 2,
        "is_weekend": 0,
    }

    scored, best = decide(
        model,
        payment
    )

    print(
        "\nCandidate actions:\n"
    )

    print(
        scored[
            [
                "recovery_action",
                "recovery_probability",
                "expected_recovery_value",
            ]
        ]
        .sort_values(
            "expected_recovery_value",
            ascending=False
        )
        .to_string(
            index=False
        )
    )

    print(
        "\n================================"
    )

    print(
        "RECOMMENDED ACTION:",
        best[
            "recovery_action"
        ]
    )

    print(
        "Recovery probability:",
        round(
            best[
                "recovery_probability"
            ],
            4
        )
    )

    print(
        "Expected recovery value: ₹",
        round(
            best[
                "expected_recovery_value"
            ],
            2
        )
    )

    print(
        "================================"
    )
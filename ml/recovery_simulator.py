import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Recovery actions
# ---------------------------------------------------------

RECOVERY_ACTIONS = [
    "RETRY_NOW",
    "RETRY_LATER",
    "CUSTOMER_NUDGE",
    "CHANGE_PAYMENT_METHOD",
    "ESCALATE",
    "DO_NOT_RETRY"
]


# ---------------------------------------------------------
# Base recovery probabilities
# ---------------------------------------------------------
#
# These represent our synthetic world's
# underlying recovery behavior.
#
# They are NOT real Razorpay statistics.
#

BASE_PROBABILITIES = {

    "TIMEOUT": {
        "RETRY_NOW": 0.55,
        "RETRY_LATER": 0.82,
        "CUSTOMER_NUDGE": 0.35,
        "CHANGE_PAYMENT_METHOD": 0.45,
        "ESCALATE": 0.60,
        "DO_NOT_RETRY": 0.00
    },

    "INSUFFICIENT_FUNDS": {
        "RETRY_NOW": 0.15,
        "RETRY_LATER": 0.58,
        "CUSTOMER_NUDGE": 0.68,
        "CHANGE_PAYMENT_METHOD": 0.35,
        "ESCALATE": 0.40,
        "DO_NOT_RETRY": 0.00
    },

    "EXPIRED_CARD": {
        "RETRY_NOW": 0.03,
        "RETRY_LATER": 0.05,
        "CUSTOMER_NUDGE": 0.55,
        "CHANGE_PAYMENT_METHOD": 0.88,
        "ESCALATE": 0.40,
        "DO_NOT_RETRY": 0.00
    },

    "HARD_DECLINE": {
        "RETRY_NOW": 0.02,
        "RETRY_LATER": 0.03,
        "CUSTOMER_NUDGE": 0.08,
        "CHANGE_PAYMENT_METHOD": 0.15,
        "ESCALATE": 0.20,
        "DO_NOT_RETRY": 0.00
    },

    "INVALID_DETAILS": {
        "RETRY_NOW": 0.04,
        "RETRY_LATER": 0.06,
        "CUSTOMER_NUDGE": 0.72,
        "CHANGE_PAYMENT_METHOD": 0.55,
        "ESCALATE": 0.25,
        "DO_NOT_RETRY": 0.00
    },

    "BANK_UNAVAILABLE": {
        "RETRY_NOW": 0.20,
        "RETRY_LATER": 0.78,
        "CUSTOMER_NUDGE": 0.30,
        "CHANGE_PAYMENT_METHOD": 0.45,
        "ESCALATE": 0.65,
        "DO_NOT_RETRY": 0.00
    },

    "GATEWAY_ERROR": {
        "RETRY_NOW": 0.35,
        "RETRY_LATER": 0.75,
        "CUSTOMER_NUDGE": 0.25,
        "CHANGE_PAYMENT_METHOD": 0.40,
        "ESCALATE": 0.65,
        "DO_NOT_RETRY": 0.00
    }
}


# ---------------------------------------------------------
# Recovery probability calculation
# ---------------------------------------------------------

def calculate_recovery_probability(
    row,
    action
):
    """
    Calculate the probability that a failed payment
    will be recovered after a particular action.

    This is our synthetic ground-truth simulator.
    """

    failure_code = row["failure_code"]

        # DO_NOT_RETRY means no recovery attempt
    # and therefore has zero recovery probability.
    if action == "DO_NOT_RETRY":
        return 0.0

    # ---------------------------------------------
    # Get base probability
    # ---------------------------------------------

    probability = BASE_PROBABILITIES.get(
        failure_code,
        {}
    ).get(
        action,
        0.0
    )

    # ---------------------------------------------
    # Customer behavior
    # ---------------------------------------------

    customer_success_rate = (
        row["customer_success_rate"]
    )

    # Good customers are slightly easier
    # to recover.

    customer_adjustment = (
        customer_success_rate - 0.75
    ) * 0.20

    probability += customer_adjustment

    # ---------------------------------------------
    # Previous recovery history
    # ---------------------------------------------

    previous_attempts = (
        row["previous_attempts"]
    )

    previous_recoveries = (
        row["previous_recoveries"]
    )

    if previous_attempts > 0:

        historical_success = (
            previous_recoveries
            / previous_attempts
        )

        probability += (
            historical_success - 0.5
        ) * 0.10

    # ---------------------------------------------
    # Bank health
    # ---------------------------------------------

    bank_health = (
        row["bank_health_score"]
    )

    # Temporary/systemic failures benefit
    # from healthy infrastructure.

    if failure_code in [
        "TIMEOUT",
        "BANK_UNAVAILABLE",
        "GATEWAY_ERROR"
    ]:

        probability += (
            bank_health - 0.8
        ) * 0.30

    # ---------------------------------------------
    # Gateway health
    # ---------------------------------------------

    gateway_health = (
        row["gateway_health_score"]
    )

    if failure_code == "GATEWAY_ERROR":

        probability += (
            gateway_health - 0.8
        ) * 0.25

    # ---------------------------------------------
    # Clamp probability
    # ---------------------------------------------

    probability = np.clip(
        probability,
        0.0,
        0.98
    )

    return float(
        round(probability, 4)
    )


# ---------------------------------------------------------
# Simulate actual outcome
# ---------------------------------------------------------

def simulate_recovery(
    row,
    action
):
    """
    Simulate whether the recovery action
    actually succeeds.
    """

    probability = calculate_recovery_probability(
        row,
        action
    )

    success = (
        np.random.random()
        < probability
    )

    if success:

        recovered_amount = row["amount"]

    else:

        recovered_amount = 0.0

    return {
        "recovery_probability": probability,
        "recovery_success": int(success),
        "recovered_amount": recovered_amount
    }


# ---------------------------------------------------------
# Test the simulator
# ---------------------------------------------------------

if __name__ == "__main__":

    print(
        "\nRecovery probability examples:\n"
    )

    examples = [
        ("TIMEOUT", "RETRY_NOW"),
        ("TIMEOUT", "RETRY_LATER"),
        ("EXPIRED_CARD", "RETRY_NOW"),
        (
            "EXPIRED_CARD",
            "CHANGE_PAYMENT_METHOD"
        ),
        (
            "INSUFFICIENT_FUNDS",
            "CUSTOMER_NUDGE"
        )
    ]

    for failure, action in examples:

        print(
            f"{failure:20s}"
            f" + "
            f"{action:25s}"
        )

        print(
            "Probability:",
            BASE_PROBABILITIES[
                failure
            ][action]
        )

        print()
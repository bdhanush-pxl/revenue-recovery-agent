import pandas as pd


PAYMENTS_FILE = "data/payments.csv"


def get_high_value_threshold():

    payments = pd.read_csv(
        PAYMENTS_FILE
    )

    return float(
        payments["amount"].quantile(
            0.99
        )
    )


def evaluate_action(
    payment_context,
    recovery_result,
    bank_health,
    proposed_action=None,
):
    """
    Evaluate whether the recommended recovery action
    is safe to execute.

    Guardrails are deterministic and cannot be overridden
    by the LLM agent.
    """

    risk_flags = []

    blocking_reasons = []

    recommended_action = (
        recovery_result[
            "recommended_action"
        ]
    )

    if proposed_action is None:
        proposed_action = recommended_action

    recovery_probability = float(
        recovery_result[
            "recovery_probability"
        ]
    )

    amount = float(
        payment_context[
            "amount"
        ]
    )

    failure_category = (
        payment_context[
            "failure_category"
        ]
    )

    attempt_number = int(
        payment_context[
            "attempt_number"
        ]
    )

    bank_health_score = float(
        bank_health[
            "health_score"
        ]
    )

    # ==================================================
    # 1. Very low recovery probability
    # ==================================================

    if recovery_probability < 0.10:

        risk_flags.append(
            "VERY_LOW_RECOVERY_PROBABILITY"
        )

        # ESCALATE is a human-review pathway,
        # not an automatic payment recovery action.
        if proposed_action != "ESCALATE":
            blocking_reasons.append(
                (
                    "Predicted recovery probability "
                    "is below 10%."
                )
            )

    # ==================================================
    # 2. Permanent failure
    # ==================================================

    if failure_category == "PERMANENT":

        risk_flags.append(
            "PERMANENT_FAILURE"
        )

        if proposed_action in {
            "RETRY_NOW",
            "RETRY_LATER",
        }:

            blocking_reasons.append(
                (
                    "Permanent failures should "
                    "not be automatically retried."
                )
            )

    # ==================================================
    # 3. Bank incident
    # ==================================================

    if bank_health_score < 0.70:

        risk_flags.append(
            "SYSTEMIC_BANK_INCIDENT"
        )

        if proposed_action == "RETRY_NOW":

            blocking_reasons.append(
                (
                    "Immediate retry is suppressed "
                    "while bank health is degraded."
                )
            )

    # ==================================================
    # 4. High-value payment
    # ==================================================

    high_value_threshold = (
        get_high_value_threshold()
    )

    if amount >= high_value_threshold:

        risk_flags.append(
            "HIGH_VALUE_PAYMENT"
        )

    # ==================================================
    # 5. Repeated attempts
    # ==================================================

    if attempt_number >= 4:

        risk_flags.append(
            "REPEATED_ATTEMPTS"
        )


    # ==================================================
    # Execution mode
    # ==================================================

    if proposed_action == "ESCALATE":
        execution_mode = "HUMAN_REVIEW"
    else:
        execution_mode = "AUTOMATIC"

    # ==================================================
    # Final decision
    # ==================================================

    allowed = (
        len(blocking_reasons) == 0
    )

    if not allowed:

        risk_level = "HIGH"

    elif len(risk_flags) >= 2:

        risk_level = "MEDIUM"

    elif len(risk_flags) == 1:

        risk_level = "LOW"

    else:

        risk_level = "NORMAL"

    if allowed:

        reason = (
            f"Proposed action {proposed_action} "
            "passed all active guardrails."
        )

    else:

        reason = " ".join(
            blocking_reasons
        )

    return {

        "allowed": allowed,

        "execution_mode": execution_mode,

        "risk_level": risk_level,

        "risk_flags": risk_flags,

        "blocking_reasons":
            blocking_reasons,

        "recommended_action":
            recommended_action,

        "recovery_probability":
            recovery_probability,

        "high_value_threshold":
            round(
                high_value_threshold,
                2
            ),

        "reason": reason,

        "proposed_action": proposed_action,
    }
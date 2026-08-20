from agent.tools import (
    get_payment_context,
    get_bank_health,
    get_recovery_options,
)

from agent.guardrails import (
    evaluate_action,
)


def test_action(
    transaction_id,
    proposed_action
):

    print(
        "\n=========================================="
    )

    print(
        f"TRANSACTION: {transaction_id}"
    )

    print(
        f"PROPOSED ACTION: {proposed_action}"
    )

    print(
        "=========================================="
    )

    payment = get_payment_context(
        transaction_id
    )

    if not payment["success"]:

        print(payment)
        return

    bank = get_bank_health(
        payment["bank"],
        payment["timestamp"]
    )

    recovery = get_recovery_options(
        transaction_id
    )

    if not recovery["success"]:

        print(recovery)
        return

    decision = evaluate_action(
        payment,
        recovery,
        bank,
        proposed_action=proposed_action,
    )

    print(
        "\nPayment:"
    )

    print(
        "Failure:",
        payment["failure_code"]
    )

    print(
        "Category:",
        payment["failure_category"]
    )

    print(
        "Amount:",
        payment["amount"]
    )

    print(
        "Bank health:",
        bank["health_score"]
    )

    print(
        "Model recommendation:",
        recovery[
            "recommended_action"
        ]
    )

    print(
        "Model probability:",
        recovery[
            "recovery_probability"
        ]
    )

    print(
        "\nGUARDRAIL RESULT:"
    )

    print(
        decision
    )


if __name__ == "__main__":

    # ------------------------------------------
    # Test A: Permanent failure
    # ------------------------------------------

    test_action(
        "TX000060",
        "RETRY_NOW"
    )

    # ------------------------------------------
    # Test B: Degraded bank
    # ------------------------------------------

    test_action(
        "TX001487",
        "RETRY_NOW"
    )

    # ------------------------------------------
    # Test C: Normal transaction
    # ------------------------------------------

    test_action(
        "TX000053",
        "CUSTOMER_NUDGE"
    )
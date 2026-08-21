from agent.tools import (
    get_payment_context,
    get_bank_health,
    get_recovery_options,
    evaluate_recovery_action,
    execute_recovery_action,
    verify_payment,
)


def run_test(transaction_id):
    print("=" * 60)
    print(f"END-TO-END TEST: {transaction_id}")
    print("=" * 60)

    # 1. Payment context
    payment = get_payment_context(transaction_id)

    if not payment["success"]:
        print("Payment lookup failed:")
        print(payment)
        return

    print("\n[1] PAYMENT CONTEXT")
    print(payment)

    # 2. Bank health
    bank_health = get_bank_health(
        payment["bank"],
        payment["timestamp"],
    )

    print("\n[2] BANK HEALTH")
    print(bank_health)

    # 3. ML recovery options
    recovery = get_recovery_options(transaction_id)

    print("\n[3] ML RECOVERY OPTIONS")
    print(recovery)

    # 4. Proposed action
    proposed_action = recovery["recommended_action"]

    print("\n[4] PROPOSED ACTION")
    print(proposed_action)

    # 5. Guardrails
    guardrail = evaluate_recovery_action(
        transaction_id,
        proposed_action,
    )

    print("\n[5] GUARDRAIL DECISION")
    print(guardrail)

    execution_mode = guardrail.get(
        "execution_mode",
        "AUTOMATIC",
    )

    # =====================================================
    # BLOCKED
    # =====================================================

    if not guardrail["allowed"]:
        print("\n[6] EXECUTION")
        print("BLOCKED")
        print("No automatic action was executed.")

        print("\nBlocking reasons:")
        for reason in guardrail["blocking_reasons"]:
            print(f"- {reason}")

        return

    # =====================================================
    # HUMAN REVIEW
    # =====================================================

    if execution_mode == "HUMAN_REVIEW":
        print("\n[6] EXECUTION")
        print("HUMAN REVIEW")

        escalation = execute_recovery_action(
            transaction_id,
            "ESCALATE",
        )

        print("Escalation result:")
        print(escalation)

        print("\n[7] VERIFICATION")
        print("N/A - no payment recovery was executed.")

        return

    # =====================================================
    # AUTOMATIC
    # =====================================================

    if execution_mode == "AUTOMATIC":
        print("\n[6] EXECUTION")

        execution = execute_recovery_action(
            transaction_id,
            proposed_action,
        )

        print(execution)

        if not execution.get("success"):
            print("\nExecution failed.")
            return

        print("\n[7] VERIFICATION")

        verification = verify_payment(
            transaction_id,
            proposed_action,
        )

        print(verification)

        print("\n[8] FINAL OUTCOME")

        if verification.get("recovery_success") == 1:
            print("RECOVERED")
            print(
                f"Recovered amount: ₹ "
                f"{verification.get('recovered_amount', 0):,.2f}"
            )
        else:
            print("NOT RECOVERED")


if __name__ == "__main__":
    run_test("TX000053")
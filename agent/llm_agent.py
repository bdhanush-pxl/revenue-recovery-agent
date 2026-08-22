import os

from google import genai

from agent.tools import (
    get_payment_context,
    get_bank_health,
    get_recovery_options,
    evaluate_recovery_action,
    execute_recovery_action,
    verify_payment,
)


# =========================================================
# GEMINI CLIENT
# =========================================================

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not configured."
    )

client = genai.Client(
    api_key=api_key
)


# =========================================================
# TOOL 1 — PAYMENT CONTEXT
# =========================================================

def get_payment_context_tool(
    transaction_id: str
):
    """
    Retrieve detailed information about
    a payment transaction.
    """

    print(
        "\n[TOOL CALL] get_payment_context"
    )

    print(
        f"[TOOL ARGUMENT] transaction_id={transaction_id}"
    )

    result = get_payment_context(
        transaction_id
    )

    print(
        "[TOOL RESULT]"
    )

    print(
        result
    )

    return result


# =========================================================
# TOOL 2 — BANK HEALTH
# =========================================================

def get_bank_health_tool(
    bank: str,
    transaction_timestamp: str
):
    """
    Retrieve bank health around the time
    of a specific payment transaction.
    """

    print(
        "\n[TOOL CALL] get_bank_health"
    )

    print(
        f"[TOOL ARGUMENT] "
        f"bank={bank}, "
        f"transaction_timestamp={transaction_timestamp}"
    )

    result = get_bank_health(
        bank,
        transaction_timestamp
    )

    print(
        "[TOOL RESULT]"
    )

    print(
        result
    )

    return result


# =========================================================
# TOOL 3 — RECOVERY OPTIONS
# =========================================================

def get_recovery_options_tool(
    transaction_id: str
):
    """
    Retrieve ML-powered recovery options,
    probabilities, and expected recovery values.
    """

    print(
        "\n[TOOL CALL] get_recovery_options"
    )

    print(
        f"[TOOL ARGUMENT] transaction_id={transaction_id}"
    )

    result = get_recovery_options(
        transaction_id
    )

    print(
        "[TOOL RESULT]"
    )

    print(
        result
    )

    return result



# =========================================================
# TOOL 4 — GUARDRAIL EVALUATION
# =========================================================

def evaluate_recovery_action_tool(
    transaction_id: str,
    proposed_action: str,
):
    """
    Evaluate whether a proposed recovery action
    passes deterministic safety guardrails.

    The LLM cannot bypass this decision.
    """

    print(
        "\n[TOOL CALL] evaluate_recovery_action"
    )

    print(
        f"[TOOL ARGUMENT] "
        f"transaction_id={transaction_id}, "
        f"proposed_action={proposed_action}"
    )

    result = evaluate_recovery_action(
        transaction_id,
        proposed_action,
    )

    print(
        "[TOOL RESULT]"
    )

    print(
        result
    )

    return result




# =========================================================
# TOOL 5 — RECOVERY ACTION EXECUTION
# =========================================================

def execute_recovery_action_tool(
    transaction_id: str,
    action: str,
):
    """
    Execute an approved recovery action.

    Execution is only allowed after deterministic
    guardrail evaluation.
    """

    print(
        "\n[TOOL CALL] execute_recovery_action"
    )

    print(
        f"[TOOL ARGUMENT] "
        f"transaction_id={transaction_id}, "
        f"action={action}"
    )

    result = execute_recovery_action(
        transaction_id,
        action,
    )

    print(
        "[TOOL RESULT]"
    )

    print(
        result
    )

    return result



# =========================================================
# TOOL 6 — PAYMENT VERIFICATION
# =========================================================

def verify_payment_tool(
    transaction_id: str,
    action: str,
):
    """
    Verify whether the recovery action resulted
    in a successful payment recovery.
    """

    print(
        "\n[TOOL CALL] verify_payment"
    )

    print(
        f"[TOOL ARGUMENT] "
        f"transaction_id={transaction_id}, "
        f"action={action}"
    )

    result = verify_payment(
        transaction_id,
        action,
    )

    print(
        "[TOOL RESULT]"
    )

    print(
        result
    )

    return result


# =========================================================
# AGENT
# =========================================================

def run_agent(
    transaction_id
):

    prompt = f"""
You are a payment recovery decision-support agent.

Analyze transaction {transaction_id}.

You have four tools:

1. get_payment_context
2. get_bank_health
3. get_recovery_options
4. evaluate_recovery_action

WORKFLOW:

1. Retrieve the payment context.

2. Use the bank and timestamp from the payment
   context to retrieve transaction-time bank health.

3. Retrieve ML recovery options.

4. Select the recovery action that appears
   most appropriate based on the tool results.

5. ALWAYS call evaluate_recovery_action before
   executing any recovery action.

6. Treat the deterministic guardrail result as
   authoritative.

7. If:
      allowed == true
      AND
      execution_mode == "AUTOMATIC"

   then call execute_recovery_action.

8. If execution succeeds, call verify_payment
   to determine whether the payment was recovered.

9. If:
      execution_mode == "HUMAN_REVIEW"

   do NOT execute a payment recovery action.
   The case should be escalated for manual review.

10. If:
      allowed == false

    do NOT execute the proposed action.

    Use the guardrail's recommended_action as the
    safe alternative and explain why the original
    action was blocked.

IMPORTANT SAFETY RULES:

- The ML model provides predictions only.

- You provide reasoning and orchestration.

- Deterministic guardrails have final authority.

- NEVER override a guardrail BLOCK decision.

- NEVER execute an action unless the guardrail
  explicitly allows it.

- "allowed == true" is not sufficient by itself.
  For automatic execution, execution_mode must
  equal "AUTOMATIC".

- If execution_mode == "HUMAN_REVIEW", do not
  automatically execute a payment recovery action.

- ESCALATE means the case is routed for manual
  review.

- Never retry a permanent failure automatically.

- Never perform RETRY_NOW during a systemic bank
  incident when the guardrail blocks it.

- Never invent transaction facts.

- Clearly distinguish tool-derived facts from
  inference.

- ALL monetary amounts are in Indian Rupees (INR).

- ALWAYS display monetary amounts using the ₹ symbol.

- NEVER use $, USD, or dollar notation.

- Do not convert or alter monetary values returned
  by the tools.

- After an automatic execution, always verify
  whether recovery actually occurred.

- Do not claim recovery succeeded unless
  verify_payment confirms it.


EXECUTION RULES:

- Guardrails are authoritative.
- NEVER execute an action unless the guardrail result allows it.
- If execution_mode == "AUTOMATIC":
    1. Execute the approved action.
    2. Verify the payment outcome.
    3. Report execution status.
    4. Report verification status.
    5. Report recovered amount if available.

- If execution_mode == "HUMAN_REVIEW":
    1. Do NOT execute the recovery action as an automatic payment action.
    2. Escalate the case using ESCALATE.
    3. State clearly that the case was escalated.
    4. Do NOT claim that payment recovery occurred.
    5. Do NOT call verify_payment.

- If the guardrail action is BLOCKED:
    1. Do NOT execute any recovery action.
    2. State that no automatic action was executed.
    3. Explain every blocking reason.
    4. Use the guardrail recommended action only as a safe recommendation,
       not as proof that it was executed.

IMPORTANT:
The LLM must never infer execution success.
Execution status and verification status must come from tools.


OUTPUT FORMAT RULES:

- This is an India-based payment recovery system.
- All monetary values must be displayed in INR.
- Always use the ₹ symbol.
- Example: ₹10,641.08
- Never write $10,641.08 or USD 10,641.08.
- Use the exact monetary value returned by the tools.


Your final response must contain:

1. Payment summary
2. Bank health
3. ML recommendation
4. Proposed action
5. Guardrail decision
6. Final safe action
7. Reasoning
8. Execution decision
9. Verification result
10. Final outcome

For AUTOMATIC actions, report:
- Whether execution was permitted
- Execution status
- Verification status
- Recovered amount if available

For HUMAN_REVIEW:
- State that the case was escalated
- Do not claim payment recovery occurred

For BLOCKED actions:
- State that no automatic action was executed
- Explain the blocking reasons

"""

    # -----------------------------------------------------
    # Gemini chat with tools
    # -----------------------------------------------------

    chat = client.chats.create(

        model="gemini-3.6-flash",

        config={
            "tools": [
                get_payment_context_tool,
                get_bank_health_tool,
                get_recovery_options_tool,
                evaluate_recovery_action_tool,
                execute_recovery_action_tool,
                verify_payment_tool,
            ]
        }
    )

    # -----------------------------------------------------
    # Send request
    # -----------------------------------------------------

    response = chat.send_message(
        prompt
    )

    # -----------------------------------------------------
    # Final response
    # -----------------------------------------------------

    print(
        "\n=========================================="
    )

    print(
        "AGENT RESPONSE"
    )

    print(
        "=========================================="
    )

    print(
        response.text
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    run_agent(
        "TX000053"
    )
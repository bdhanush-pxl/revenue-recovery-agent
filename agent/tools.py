import pandas as pd
import joblib


PAYMENTS_FILE = "data/payments.csv"
MODEL_FILE = "models/recovery_predictor.joblib"
RECOVERY_FILE = "data/recovery_training_data.csv"


def get_payment_context(
    transaction_id: str
):
    """
    Retrieve the context of a payment.

    This represents a tool that our AI agent
    can call when it needs information about
    a failed payment.
    """

    df = pd.read_csv(
        PAYMENTS_FILE
    )

    transaction = df[
        df["transaction_id"]
        == transaction_id
    ]

    if transaction.empty:

        return {
            "success": False,
            "error": (
                f"Transaction "
                f"{transaction_id} "
                f"not found."
            )
        }

    row = transaction.iloc[0]

    return {
        "success": True,

        "transaction_id":
            row["transaction_id"],

        "customer_id":
            row["customer_id"],

        "merchant_id":
            row["merchant_id"],

        "amount":
            float(row["amount"]),

        "payment_method":
            row["payment_method"],

        "bank":
            row["bank"],

        "payment_status":
            row["payment_status"],

        "failure_code":
            row["failure_code"],

        "failure_category":
            row["failure_category"],

        "timestamp":
            row["timestamp"],

        "attempt_number":
            int(row["attempt_number"]),

        "customer_tenure_days":
            int(row["customer_tenure_days"]),

        "customer_success_rate":
            float(
                row[
                    "customer_success_rate"
                ]
            ),

        "previous_attempts":
            int(row["previous_attempts"]),

        "previous_recoveries":
            int(row["previous_recoveries"]),

        "bank_health_score":
            float(
                row[
                    "bank_health_score"
                ]
            ),

        "gateway_health_score":
            float(
                row[
                    "gateway_health_score"
                ]
            ),

        "gateway_latency_ms":
            float(
                row[
                    "gateway_latency_ms"
                ]
            ),

        "hour":
            int(row["hour"]),

        "day_of_week":
            int(row["day_of_week"]),

        "is_weekend":
            int(row["is_weekend"]),
    }


def get_bank_health(
    bank,
    transaction_timestamp=None,
):
    """
    Return bank health information.

    If transaction_timestamp is supplied, the function
    also evaluates the bank's health around that transaction.

    This prevents the agent from using a current/overall
    bank score when making a decision about a historical
    transaction.
    """

    try:

        df = pd.read_csv(
            PAYMENTS_FILE
        )

    except Exception as e:

        return {
            "success": False,
            "error": (
                f"Unable to load payment data: {str(e)}"
            )
        }

    bank_df = df[
        df["bank"] == bank
    ].copy()

    if bank_df.empty:

        return {
            "success": False,
            "error": (
                f"Bank {bank} not found."
            )
        }

    # --------------------------------------------------
    # Overall bank health
    # --------------------------------------------------

    overall_failure_rate = (
        bank_df["payment_status"]
        .eq("FAILED")
        .mean()
    )

    overall_health_score = (
        bank_df["bank_health_score"]
        .mean()
    )

    # --------------------------------------------------
    # Transaction-time health
    # --------------------------------------------------

    transaction_health_score = None
    recent_failure_rate = None
    recent_transactions = 0

    if transaction_timestamp is not None:

        df["timestamp"] = pd.to_datetime(
            df["timestamp"]
        )

        transaction_timestamp = pd.to_datetime(
            transaction_timestamp
        )

        # Look at a 6-hour window around the transaction.
        window_start = (
            transaction_timestamp
            - pd.Timedelta(hours=3)
        )

        window_end = (
            transaction_timestamp
            + pd.Timedelta(hours=3)
        )

        recent = df[
            (df["bank"] == bank)
            &
            (df["timestamp"] >= window_start)
            &
            (df["timestamp"] <= window_end)
        ].copy()

        recent_transactions = len(
            recent
        )

        if not recent.empty:

            recent_failure_rate = (
                recent["payment_status"]
                .eq("FAILED")
                .mean()
            )

            transaction_health_score = (
                recent["bank_health_score"]
                .mean()
            )

    # --------------------------------------------------
    # Select health score used for decision-making
    # --------------------------------------------------

    if transaction_health_score is not None:

        decision_health_score = (
            transaction_health_score
        )

    else:

        decision_health_score = (
            overall_health_score
        )

    if decision_health_score >= 0.90:

        status = "HEALTHY"

    elif decision_health_score >= 0.70:

        status = "DEGRADED"

    else:

        status = "CRITICAL"

    return {

        "success": True,

        "bank": bank,

        "status": status,

        # Health used by the decision engine
        "health_score": round(
            float(
                decision_health_score
            ),
            4
        ),

        "failure_rate": (
            round(
                float(
                    recent_failure_rate
                ),
                4
            )
            if recent_failure_rate is not None
            else round(
                float(
                    overall_failure_rate
                ),
                4
            )
        ),

        "overall_health_score": round(
            float(
                overall_health_score
            ),
            4
        ),

        "overall_failure_rate": round(
            float(
                overall_failure_rate
            ),
            4
        ),

        "transactions_analyzed": len(
            bank_df
        ),

        "recent_transactions":
            recent_transactions,

        "transaction_health_score": (
            round(
                float(
                    transaction_health_score
                ),
                4
            )
            if transaction_health_score is not None
            else None
        ),

        "recent_failure_rate": (
            round(
                float(
                    recent_failure_rate
                ),
                4
            )
            if recent_failure_rate is not None
            else None
        ),
    }



def get_recovery_options(
    transaction_id: str
):
    """
    Use the trained ML model and decision engine
    to evaluate recovery actions for a failed payment.
    """

    # --------------------------------------------------
    # Get payment context
    # --------------------------------------------------

    payment_context = get_payment_context(
        transaction_id
    )

    if not payment_context[
        "success"
    ]:

        return payment_context

    # --------------------------------------------------
    # Verify payment is actually failed
    # --------------------------------------------------

    if payment_context[
        "payment_status"
    ] != "FAILED":

        return {
            "success": False,
            "error": (
                "Recovery analysis is only "
                "available for failed payments."
            )
        }

    # --------------------------------------------------
    # Load trained model
    # --------------------------------------------------

    try:

        model = joblib.load(
            MODEL_FILE
        )

    except Exception as e:

        return {
            "success": False,
            "error": (
                f"Unable to load recovery model: "
                f"{str(e)}"
            )
        }

    # --------------------------------------------------
    # Import decision engine
    # --------------------------------------------------

    from ml.recovery_decision_engine import (
        decide
    )

    # --------------------------------------------------
    # Prepare ML input
    #
    # The decision engine only needs the
    # model features, not IDs/status.
    # --------------------------------------------------

    payment = {

        "amount":
            payment_context["amount"],

        "payment_method":
            payment_context["payment_method"],

        "bank":
            payment_context["bank"],

        "failure_code":
            payment_context["failure_code"],

        "failure_category":
            payment_context["failure_category"],

        "attempt_number":
            payment_context["attempt_number"],

        "customer_tenure_days":
            payment_context[
                "customer_tenure_days"
            ],

        "customer_success_rate":
            payment_context[
                "customer_success_rate"
            ],

        "previous_attempts":
            payment_context[
                "previous_attempts"
            ],

        "previous_recoveries":
            payment_context[
                "previous_recoveries"
            ],

        "bank_health_score":
            payment_context[
                "bank_health_score"
            ],

        "gateway_health_score":
            payment_context[
                "gateway_health_score"
            ],

        "gateway_latency_ms":
            payment_context[
                "gateway_latency_ms"
            ],

        "hour":
            payment_context["hour"],

        "day_of_week":
            payment_context[
                "day_of_week"
            ],

        "is_weekend":
            payment_context[
                "is_weekend"
            ],
    }

    # --------------------------------------------------
    # Run decision engine
    # --------------------------------------------------

    try:

        scored, best = decide(
            model,
            payment
        )

    except Exception as e:

        return {
            "success": False,
            "error": (
                f"Recovery prediction failed: "
                f"{str(e)}"
            )
        }

    # --------------------------------------------------
    # Format alternatives
    # --------------------------------------------------

    alternatives = []

    for _, row in (
        scored
        .sort_values(
            "expected_recovery_value",
            ascending=False
        )
        .iterrows()
    ):

        alternatives.append({

            "action":
                row[
                    "recovery_action"
                ],

            "recovery_probability":
                round(
                    float(
                        row[
                            "recovery_probability"
                        ]
                    ),
                    4
                ),

            "expected_recovery_value":
                round(
                    float(
                        row[
                            "expected_recovery_value"
                        ]
                    ),
                    2
                ),
        })

    # --------------------------------------------------
    # Final structured result
    # --------------------------------------------------

    return {

        "success": True,

        "transaction_id":
            transaction_id,

        "failure_code":
            payment_context[
                "failure_code"
            ],

        "failure_category":
            payment_context[
                "failure_category"
            ],

        "amount":
            payment_context[
                "amount"
            ],

        "recommended_action":
            best[
                "recovery_action"
            ],

        "recovery_probability":
            round(
                float(
                    best[
                        "recovery_probability"
                    ]
                ),
                4
            ),

        "expected_recovery_value":
            round(
                float(
                    best[
                        "expected_recovery_value"
                    ]
                ),
                2
            ),

        "alternatives":
            alternatives,
    }


def execute_recovery_action(
    transaction_id: str,
    action: str
):
    """
    Simulate execution of a recovery action.

    In a production system these operations would
    call payment APIs, notification systems,
    retry schedulers, or case-management systems.
    """

    # --------------------------------------------------
    # Validate action
    # --------------------------------------------------

    valid_actions = {
        "RETRY_NOW",
        "RETRY_LATER",
        "CUSTOMER_NUDGE",
        "CHANGE_PAYMENT_METHOD",
        "ESCALATE",
        "DO_NOT_RETRY",
    }

    if action not in valid_actions:

        return {
            "success": False,
            "error": (
                f"Unknown recovery action: "
                f"{action}"
            )
        }

    # --------------------------------------------------
    # Verify transaction exists
    # --------------------------------------------------

    payment = get_payment_context(
        transaction_id
    )

    if not payment["success"]:

        return payment

    # --------------------------------------------------
    # Only failed payments should enter recovery
    # --------------------------------------------------

    if payment[
        "payment_status"
    ] != "FAILED":

        return {
            "success": False,
            "error": (
                "Recovery action cannot be "
                "executed because the payment "
                "is not failed."
            )
        }

    # --------------------------------------------------
    # Simulate action execution
    # --------------------------------------------------

    if action == "RETRY_NOW":

        return {
            "success": True,
            "transaction_id":
                transaction_id,
            "action":
                action,
            "execution_status":
                "RETRY_INITIATED",
            "message":
                (
                    "Immediate payment retry "
                    "has been initiated."
                ),
        }

    if action == "RETRY_LATER":

        return {
            "success": True,
            "transaction_id":
                transaction_id,
            "action":
                action,
            "execution_status":
                "RETRY_SCHEDULED",
            "retry_after_minutes":
                30,
            "message":
                (
                    "Payment retry scheduled "
                    "for 30 minutes later."
                ),
        }

    if action == "CUSTOMER_NUDGE":

        return {
            "success": True,
            "transaction_id":
                transaction_id,
            "action":
                action,
            "execution_status":
                "CUSTOMER_NOTIFIED",
            "message":
                (
                    "Customer notification sent "
                    "with instructions to resolve "
                    "the payment issue."
                ),
        }

    if action == "CHANGE_PAYMENT_METHOD":

        return {
            "success": True,
            "transaction_id":
                transaction_id,
            "action":
                action,
            "execution_status":
                "PAYMENT_METHOD_CHANGE_REQUESTED",
            "message":
                (
                    "Customer prompted to choose "
                    "an alternative payment method."
                ),
        }

    if action == "ESCALATE":

        return {
            "success": True,
            "transaction_id":
                transaction_id,
            "action":
                action,
            "execution_status":
                "CASE_ESCALATED",
            "message":
                (
                    "Recovery case escalated "
                    "for manual review."
                ),
        }

    if action == "DO_NOT_RETRY":

        return {
            "success": True,
            "transaction_id":
                transaction_id,
            "action":
                action,
            "execution_status":
                "RECOVERY_SUPPRESSED",
            "message":
                (
                    "No automated recovery action "
                    "will be attempted."
                ),
        }



def verify_payment(
    transaction_id: str,
    action: str
):
    """
    Verify the simulated outcome of a recovery action.

    In a production system this would query the payment
    processor / payment database rather than the synthetic
    recovery dataset.
    """

    valid_actions = {
        "RETRY_NOW",
        "RETRY_LATER",
        "CUSTOMER_NUDGE",
        "CHANGE_PAYMENT_METHOD",
        "ESCALATE",
        "DO_NOT_RETRY",
    }

    if action not in valid_actions:

        return {
            "success": False,
            "error": (
                f"Unknown recovery action: "
                f"{action}"
            )
        }

    try:

        df = pd.read_csv(
            RECOVERY_FILE
        )

    except Exception as e:

        return {
            "success": False,
            "error": (
                f"Unable to load recovery "
                f"data: {str(e)}"
            )
        }

    result = df[
        (df["transaction_id"] == transaction_id)
        &
        (df["recovery_action"] == action)
    ]

    if result.empty:

        return {
            "success": False,
            "error": (
                "No recovery outcome found "
                f"for {transaction_id} "
                f"with action {action}."
            )
        }

    row = result.iloc[0]

    recovery_success = int(
        row["recovery_success"]
    )

    recovered_amount = float(
        row["recovered_amount"]
    )

    if recovery_success == 1:

        status = "RECOVERED"

        message = (
            "Payment recovery was successful."
        )

    else:

        status = "NOT_RECOVERED"

        message = (
            "Recovery action did not recover "
            "the payment."
        )

    return {

        "success": True,

        "transaction_id":
            transaction_id,

        "action":
            action,

        "verification_status":
            status,

        "recovery_success":
            recovery_success,

        "recovered_amount":
            round(
                recovered_amount,
                2
            ),

        "message":
            message,
    }


if __name__ == "__main__":

    print(
        "\n=============================="
    )

    print(
        "PAYMENT CONTEXT"
    )

    print(
        "==============================\n"
    )

    payment = get_payment_context(
        "TX000053"
    )

    for key, value in payment.items():

        print(
            f"{key}: {value}"
        )

    print(
        "\n=============================="
    )

    print(
        "BANK HEALTH"
    )

    print(
        "==============================\n"
    )

    health = get_bank_health(
        "HDFC"
    )

    for key, value in health.items():

        print(
            f"{key}: {value}"
        )

    print(
        "\n=============================="
    )

    print(
        "RECOVERY OPTIONS"
    )

    print(
        "==============================\n"
    )

    recovery = get_recovery_options(
        "TX000053"
    )

    print(
        recovery
    )

    print(
        "\n=============================="
    )

    print(
        "ACTION EXECUTION"
    )

    print(
        "==============================\n"
    )

    execution = execute_recovery_action(
        "TX000053",
        "CUSTOMER_NUDGE"
    )

    print(
        execution
    )

    print(
        "\n=============================="
    )

    print(
        "VERIFICATION"
    )

    print(
        "==============================\n"
    )

    verification = verify_payment(
        "TX000053",
        "CUSTOMER_NUDGE"
    )

    print(
        verification
    )
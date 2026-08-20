import pandas as pd


FILE = "data/recovery_training_data.csv"


def main():

    df = pd.read_csv(FILE)

    print(
        "Recovery probability distribution:"
    )

    print(
        df[
            "recovery_probability"
        ].describe(
            percentiles=[
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99
            ]
        )
    )

    # ---------------------------------------------
    # Probability bands
    # ---------------------------------------------

    df["probability_band"] = pd.cut(
        df[
            "recovery_probability"
        ],
        bins=[
            0.0,
            0.10,
            0.20,
            0.30,
            0.40,
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
            1.00
        ],
        labels=[
            "0-10%",
            "10-20%",
            "20-30%",
            "30-40%",
            "40-50%",
            "50-60%",
            "60-70%",
            "70-80%",
            "80-90%",
            "90-100%"
        ],
        include_lowest=True
    )

    summary = (
        df
        .groupby(
            "probability_band",
            observed=True
        )
        .agg(
            rows=(
                "recovery_success",
                "count"
            ),

            actual_success_rate=(
                "recovery_success",
                "mean"
            ),

            average_predicted_probability=(
                "recovery_probability",
                "mean"
            ),

            recovered_amount=(
                "recovered_amount",
                "sum"
            )
        )
    )

    print(
        "\n=========================================="
    )

    print(
        "RECOVERY BY PREDICTED PROBABILITY"
    )

    print(
        "=========================================="
    )

    print(
        summary.to_string()
    )


if __name__ == "__main__":

    main()
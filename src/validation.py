
import pandas as pd


REQUIRED_COLUMNS = [
    "transaction_id",
    "amount",
    "transaction_hour",
    "transactions_last_hour",
]


def validate_transactions(df):
    """Separate valid transactions from invalid transactions."""

    data = df.copy()

    # Check required columns.
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    errors = pd.Series("", index=data.index, dtype="object")

    # Check transaction ID.
    missing_id = (
        data["transaction_id"].isna()
        | data["transaction_id"].astype(str).str.strip().eq("")
    )

    errors.loc[missing_id] += "Missing transaction_id; "

    # Convert numeric fields.
    numeric_columns = [
        "amount",
        "transaction_hour",
        "transactions_last_hour",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

        invalid_number = (
            data[column].isna()
            | ~data[column].map(
                lambda value: pd.notna(value)
                and abs(value) != float("inf")
            )
        )

        errors.loc[invalid_number] += (
            f"Invalid {column}; "
        )

    # Check amount.
    errors.loc[data["amount"] < 0] += (
        "Negative amount; "
    )

    # Check transaction hour: must be an integer from 0 to 23.
    invalid_hour = (
        data["transaction_hour"].notna()
        & (
            ~data["transaction_hour"].between(0, 23)
            | (data["transaction_hour"] % 1 != 0)
        )
    )

    errors.loc[invalid_hour] += (
        "transaction_hour must be an integer from 0 to 23; "
    )

    # Check transaction frequency.
    invalid_frequency = (
        data["transactions_last_hour"].notna()
        & (
            (data["transactions_last_hour"] < 0)
            | (
                data["transactions_last_hour"] % 1 != 0
            )
        )
    )

    errors.loc[invalid_frequency] += (
        "transactions_last_hour must be a non-negative integer; "
    )

    # If fraud labels exist, they must be 0 or 1.
    if "is_fraud" in data.columns:
        data["is_fraud"] = pd.to_numeric(
            data["is_fraud"],
            errors="coerce",
        )

        invalid_label = (
            data["is_fraud"].isna()
            | ~data["is_fraud"].isin([0, 1])
        )

        errors.loc[invalid_label] += (
            "is_fraud must contain 0 or 1; "
        )

    data["validation_error"] = errors.str.rstrip("; ")

    valid = data[errors.eq("")].copy()
    invalid = data[~errors.eq("")].copy()

    return valid, invalid


import pandas as pd


def apply_fraud_rules(df):
    """Apply rules and explain why transactions were flagged."""

    result = df.copy()

    required_columns = [
        "amount",
        "transaction_hour",
        "transactions_last_hour",
    ]

    missing = set(required_columns) - set(result.columns)

    if missing:
        raise ValueError(
            "Missing rule columns: "
            + ", ".join(sorted(missing))
        )

    result["rule_risk_score"] = 0
    reasons = []

    for _, row in result.iterrows():
        score = 0
        transaction_reasons = []

        if row["amount"] >= 100000:
            score += 40
            transaction_reasons.append(
                "High transaction amount"
            )

        if row["transactions_last_hour"] >= 8:
            score += 30
            transaction_reasons.append(
                "High transaction frequency"
            )

        if (
            row["transaction_hour"] <= 4
            or row["transaction_hour"] == 23
        ):
            score += 20
            transaction_reasons.append(
                "Unusual transaction hour"
            )

        reasons.append(
            "; ".join(transaction_reasons)
            if transaction_reasons
            else "No configured rules triggered"
        )

        result.loc[
            row.name, "rule_risk_score"
        ] = min(score, 100)

    result["rule_risk_score"] = (
        result["rule_risk_score"].astype(int)
    )

    result["rule_reasons"] = reasons

    result["rule_flag"] = (
        result["rule_risk_score"] >= 40
    )

    return result

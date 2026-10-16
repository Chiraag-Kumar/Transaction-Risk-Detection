
import pandas as pd

from src.rules import apply_fraud_rules


def test_high_risk_transaction():
    df = pd.DataFrame([{
        "transaction_id": "T0001",
        "amount": 150000,
        "transaction_hour": 2,
        "transactions_last_hour": 10,
    }])

    result = apply_fraud_rules(df)

    assert result.loc[0, "rule_risk_score"] == 90
    assert bool(result.loc[0, "rule_flag"]) is True


def test_low_risk_transaction():
    df = pd.DataFrame([{
        "transaction_id": "T0002",
        "amount": 500,
        "transaction_hour": 10,
        "transactions_last_hour": 1,
    }])

    result = apply_fraud_rules(df)

    assert result.loc[0, "rule_risk_score"] == 0
    assert bool(result.loc[0, "rule_flag"]) is False

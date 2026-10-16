
import pandas as pd
import pytest

from src.validation import validate_transactions


def test_valid_transaction():
    df = pd.DataFrame([{
        "transaction_id": "T0001",
        "amount": 500,
        "transaction_hour": 10,
        "transactions_last_hour": 1,
        "is_fraud": 0,
    }])

    valid, invalid = validate_transactions(df)

    assert len(valid) == 1
    assert len(invalid) == 0


def test_negative_amount():
    df = pd.DataFrame([{
        "transaction_id": "T0002",
        "amount": -100,
        "transaction_hour": 10,
        "transactions_last_hour": 1,
        "is_fraud": 0,
    }])

    valid, invalid = validate_transactions(df)

    assert len(valid) == 0
    assert len(invalid) == 1


def test_missing_column():
    df = pd.DataFrame([{
        "transaction_id": "T0003",
        "amount": 100,
    }])

    with pytest.raises(ValueError):
        validate_transactions(df)

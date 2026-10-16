
import pandas as pd
import pytest

from src.model import train_model, predict_anomalies


def create_sample_data():
    return pd.DataFrame({
        "transaction_id": [f"T{i:03d}" for i in range(1, 101)],
        "amount": [500 + i * 100 for i in range(100)],
        "transaction_hour": [i % 24 for i in range(100)],
        "transactions_last_hour": [i % 10 for i in range(100)],
    })


def test_model_trains_successfully():
    df = create_sample_data()

    model = train_model(df)

    assert model is not None


def test_model_returns_anomaly_predictions():
    df = create_sample_data()

    model = train_model(df)
    results = predict_anomalies(model, df)

    assert "anomaly_flag" in results.columns
    assert "anomaly_score" in results.columns
    assert "ml_risk_score" in results.columns
    assert len(results) == len(df)


def test_risk_scores_are_within_range():
    df = create_sample_data()

    model = train_model(df)
    results = predict_anomalies(model, df)

    assert results["ml_risk_score"].between(0, 100).all()


def test_empty_training_data_raises_error():
    df = pd.DataFrame(columns=[
        "amount",
        "transaction_hour",
        "transactions_last_hour",
    ])

    with pytest.raises(ValueError):
        train_model(df)

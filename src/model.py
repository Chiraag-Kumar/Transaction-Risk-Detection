
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "amount",
    "transaction_hour",
    "transactions_last_hour",
]


def train_model(df, contamination=0.05, random_state=42):
    """Train an Isolation Forest anomaly detector."""

    missing = set(FEATURE_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            "Missing model features: "
            + ", ".join(sorted(missing))
        )

    if df.empty:
        raise ValueError("Cannot train on empty data.")

    X = df[FEATURE_COLUMNS].copy()

    if X.isna().any().any():
        raise ValueError(
            "Model features contain missing values."
        )

    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "detector",
            IsolationForest(
                contamination=contamination,
                random_state=random_state,
            ),
        ),
    ])

    model.fit(X)

    # Save training anomaly scores for risk-score comparison.
    model.reference_scores_ = (
        -model.decision_function(X)
    )

    return model


def predict_anomalies(model, df):
    """Predict anomalies and calculate ML risk scores."""

    result = df.copy()

    X = result[FEATURE_COLUMNS].copy()

    if X.isna().any().any():
        raise ValueError(
            "Prediction data contains missing values."
        )

    predictions = model.predict(X)

    # Lower decision scores mean more unusual transactions.
    anomaly_scores = -model.decision_function(X)

    result["anomaly_flag"] = predictions == -1
    result["anomaly_score"] = anomaly_scores

    # Convert scores into a 0-100 percentile-based risk score.
    reference = model.reference_scores_

    result["ml_risk_score"] = [
        round(
            100 * (reference <= score).mean(),
            2,
        )
        for score in anomaly_scores
    ]

    return result

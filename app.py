
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split

from src.validation import validate_transactions
from src.rules import apply_fraud_rules
from src.model import train_model, predict_anomalies
from src.evaluation import evaluate_model



st.set_page_config(
    page_title="Transaction Risk Detection",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Transaction Risk Detection Platform")
st.write(
    "Upload transaction data to validate records, detect unusual activity, "
    "calculate risk scores, and review suspicious transactions."
)



def add_combined_risk(results):
    results = results.copy()

    results["combined_risk_score"] = (
        0.5 * results["rule_risk_score"]
        + 0.5 * results["ml_risk_score"]
    ).clip(0, 100).round(2)

    results["risk_level"] = pd.cut(
        results["combined_risk_score"],
        bins=[-0.01, 30, 60, 100.01],
        labels=["Low", "Medium", "High"],
        right=False,
    )

    # A transaction is flagged if either detector identifies
    # a concern or its combined score reaches 60.
    results["review_flag"] = (
        results["rule_flag"].astype(bool)
        | results["anomaly_flag"].astype(bool)
        | (results["combined_risk_score"] >= 60)
    )

    return results



def show_evaluation(title, evaluation_df, prediction_column):
    st.subheader(title)

    metrics_df = evaluation_df.copy()
    metrics_df["anomaly_flag"] = (
        metrics_df[prediction_column].astype(int)
    )

    metrics = evaluate_model(metrics_df)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Precision", f"{metrics['precision']:.3f}")
    col2.metric("Recall", f"{metrics['recall']:.3f}")
    col3.metric("F1 Score", f"{metrics['f1_score']:.3f}")
    col4.metric("Accuracy", f"{metrics['accuracy']:.3f}")

    st.markdown("**Confusion Matrix**")

    matrix_df = pd.DataFrame(
        metrics["confusion_matrix"],
        index=["Actual Normal", "Actual Fraud"],
        columns=["Predicted Normal", "Predicted Fraud"],
    )
    st.dataframe(matrix_df, use_container_width=True)

    col1, col2 = st.columns(2)
    col1.metric("False Positives", metrics["false_positives"])
    col2.metric("False Negatives", metrics["false_negatives"])

    with st.expander("View classification report"):
        st.text(metrics["classification_report"])



uploaded_file = st.file_uploader(
    "Upload your transactions CSV",
    type=["csv"],
)

if uploaded_file is None:
    st.info("Upload a CSV file to begin transaction risk analysis.")
    st.stop()



try:
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file)

    if df.empty:
        st.error("The uploaded CSV file is empty.")
        st.stop()

    st.subheader("Uploaded Data")
    st.dataframe(df.head(10), use_container_width=True)

except Exception as error:
    st.error(f"Could not read the CSV file: {error}")
    st.stop()



st.header("Data Validation")

try:
    valid_df, invalid_df = validate_transactions(df)

except Exception as error:
    st.error(f"Data validation failed: {error}")
    st.stop()

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", len(df))
col2.metric("Valid Records", len(valid_df))
col3.metric("Invalid Records", len(invalid_df))

if not invalid_df.empty:
    st.warning(
        f"{len(invalid_df)} invalid records were excluded from analysis."
    )

    st.dataframe(invalid_df, use_container_width=True)

    st.download_button(
        "Download Invalid Records",
        data=invalid_df.to_csv(index=False).encode("utf-8"),
        file_name="invalid_transactions.csv",
        mime="text/csv",
    )

if valid_df.empty:
    st.error("No valid transactions are available for analysis.")
    st.stop()



st.header("Model Evaluation")

has_labels = (
    "is_fraud" in valid_df.columns
    and valid_df["is_fraud"].nunique() == 2
    and valid_df["is_fraud"].value_counts().min() >= 2
)

if has_labels:
    try:
        train_df, test_df = train_test_split(
            valid_df,
            test_size=0.20,
            random_state=42,
            stratify=valid_df["is_fraud"],
        )

        st.write(
            f"Training records: {len(train_df)} | "
            f"Test records: {len(test_df)}. "
            "The test records are held out from model fitting."
        )

        # Fit the model only on the training data.
        evaluation_model = train_model(train_df)

        # Evaluate the ML detector on unseen test records.
        ml_test_results = predict_anomalies(
            evaluation_model, test_df
        )

        # Apply rules to the same held-out records.
        rule_test_results = apply_fraud_rules(ml_test_results)

        combined_test_results = add_combined_risk(
            rule_test_results
        )

        show_evaluation(
            "Machine-Learning Anomaly Detector — Test Set",
            ml_test_results,
            "anomaly_flag",
        )

        show_evaluation(
            "Combined Rule-Based + ML System — Test Set",
            combined_test_results,
            "review_flag",
        )

    except Exception as error:
        st.warning(f"Could not complete test-set evaluation: {error}")
        st.info(
            "Check the dataset size, label distribution, and model inputs."
        )

else:
    st.info(
        "Test-set evaluation requires an is_fraud column containing "
        "both 0 (normal) and 1 (fraud), with at least two examples "
        "of each class. Risk detection can still run without labels."
    )


# --------------------------------------------------
# FULL-DATA RISK ANALYSIS
# --------------------------------------------------
st.header("Risk Detection")

try:
    # Fit a fresh model for analysis of all valid uploaded rows.
    # Evaluation metrics above use a separate held-out test set.
    full_model = train_model(valid_df)

    full_ml_results = predict_anomalies(full_model, valid_df)
    full_rule_results = apply_fraud_rules(full_ml_results)
    results = add_combined_risk(full_rule_results)

except Exception as error:
    st.error(f"Risk analysis failed: {error}")
    st.stop()


# --------------------------------------------------
# DASHBOARD METRICS
# --------------------------------------------------
total_transactions = len(results)
flagged_count = int(results["review_flag"].sum())
high_risk_count = int((results["risk_level"] == "High").sum())

col1, col2, col3 = st.columns(3)

col1.metric("Transactions Analysed", total_transactions)
col2.metric("Flagged for Review", flagged_count)
col3.metric("High-Risk Transactions", high_risk_count)

st.caption(
    "Risk scores prioritise transactions for review; they are not "
    "calibrated probabilities or confirmation of fraud."
)


# --------------------------------------------------
# RISK DISTRIBUTION
# --------------------------------------------------
st.subheader("Risk Distribution")

risk_order = ["Low", "Medium", "High"]

risk_counts = (
    results["risk_level"]
    .value_counts()
    .reindex(risk_order, fill_value=0)
)

risk_chart_df = risk_counts.rename_axis("Risk Level").reset_index(
    name="Transactions"
)

st.bar_chart(
    risk_chart_df.set_index("Risk Level"),
    y="Transactions",
)

st.dataframe(
    risk_chart_df,
    use_container_width=True,
    hide_index=True,
)


# --------------------------------------------------
# TRANSACTION RESULTS
# --------------------------------------------------
st.header("Transaction Results")

display_columns = [
    "transaction_id",
    "amount",
    "transaction_hour",
    "transactions_last_hour",
    "rule_risk_score",
    "rule_reasons",
    "rule_flag",
    "anomaly_flag",
    "ml_risk_score",
    "combined_risk_score",
    "risk_level",
    "review_flag",
]

# Show the ground-truth label for analysis when present.
if "is_fraud" in results.columns:
    display_columns.insert(4, "is_fraud")

display_columns = [
    column for column in display_columns
    if column in results.columns
]

filter_option = st.selectbox(
    "Filter transactions",
    [
        "All Transactions",
        "Flagged for Review",
        "High Risk",
        "Medium Risk",
        "Low Risk",
    ],
)

if filter_option == "Flagged for Review":
    filtered_results = results[results["review_flag"]]

elif filter_option == "High Risk":
    filtered_results = results[results["risk_level"] == "High"]

elif filter_option == "Medium Risk":
    filtered_results = results[results["risk_level"] == "Medium"]

elif filter_option == "Low Risk":
    filtered_results = results[results["risk_level"] == "Low"]

else:
    filtered_results = results

st.dataframe(
    filtered_results[display_columns],
    use_container_width=True,
    hide_index=True,
)


# --------------------------------------------------
# SUSPICIOUS TRANSACTION REPORT
# --------------------------------------------------
st.header("Suspicious Transaction Report")

suspicious_df = results[results["review_flag"]].copy()

st.write(
    f"{len(suspicious_df)} transactions require review."
)

report_columns = [
    "transaction_id",
    "amount",
    "transaction_hour",
    "transactions_last_hour",
    "rule_risk_score",
    "rule_reasons",
    "rule_flag",
    "anomaly_flag",
    "ml_risk_score",
    "combined_risk_score",
    "risk_level",
    "review_flag",
]

if "is_fraud" in suspicious_df.columns:
    report_columns.insert(4, "is_fraud")

report_columns = [
    column for column in report_columns
    if column in suspicious_df.columns
]

st.dataframe(
    suspicious_df[report_columns],
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "Download Suspicious Transactions CSV",
    data=suspicious_df.to_csv(index=False).encode("utf-8"),
    file_name="suspicious_transactions.csv",
    mime="text/csv",
)

st.download_button(
    "Download Complete Risk Report CSV",
    data=results.to_csv(index=False).encode("utf-8"),
    file_name="transaction_risk_report.csv",
    mime="text/csv",
)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()

st.caption(
    "This platform supports transaction screening using configured "
    "rules and anomaly detection. Flagged transactions must be "
    "investigated before being classified as fraudulent."
)

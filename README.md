# Real-Time Transaction Risk Detection Platform

## 1. Project Overview

The Real-Time Transaction Risk Detection Platform is a Python-based application that identifies potentially suspicious financial transactions using rule-based detection and machine-learning anomaly detection.

Users upload a CSV file containing multiple transactions. The system validates the data, calculates risk scores, identifies suspicious transactions, explains why transactions are flagged, and generates downloadable reports.

## 2. Key Features

- **CSV Upload:** Upload transaction datasets for batch analysis.
- **Data Validation:** Detect missing values, invalid amounts, duplicate transaction IDs, incorrect data formats, and out-of-range values.
- **Rule-Based Detection:** Identify suspicious patterns based on transaction amount, transaction frequency, and transaction hour.
- **Machine Learning:** Use Isolation Forest to identify unusual transaction patterns.
- **Risk Scoring:** Combine rule-based and machine-learning scores to prioritize transactions for review.
- **Risk Explanations:** Display reasons why transactions are flagged.
- **Risk Classification:** Categorize transactions as Low, Medium, or High risk.
- **Model Evaluation:** Calculate precision, recall, F1 score, and a confusion matrix when ground-truth fraud labels are available.
- **Downloadable Reports:** Export analysed transactions and invalid records as CSV files.
- **Interactive Dashboard:** Review transaction results and filter records using Streamlit.

## 3. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly
- Pytest

## 4. Project Structure

```text
Transaction-Risk-Detection/
├── data/
│   └── transactions.csv
├── src/
│   ├── validation.py
│   ├── rules.py
│   ├── model.py
│   └── evaluation.py
├── tests/
│   ├── test_validation.py
│   ├── test_rules.py
│   └── test_model.py
├── app.py
├── requirements.txt
└── README.md
```

## 5. How the System Works

### Step 1: Upload Transaction Data

The user uploads a CSV file through the Streamlit dashboard.

### Step 2: Validate the Data

The validation module checks for missing values, invalid numeric values, duplicate transaction IDs, and incorrect ranges.

Invalid records are separated from valid records. The user can view and download the invalid-record report.

### Step 3: Apply Rule-Based Detection

The rule-based detection module assigns risk points according to configured conditions.

| Condition | Risk Points |
|---|---:|
| Transaction amount is at least 100,000 | 40 |
| At least 8 transactions in the last hour | 30 |
| Transaction occurs between midnight and 4 AM, or at 11 PM | 20 |

The rule-based risk score is capped at 100. Transactions with a rule score of 40 or more are flagged by the rule-based detector.

These thresholds are illustrative and should be calibrated using representative transaction data.

### Step 4: Apply Machine Learning

The Isolation Forest algorithm identifies unusual transaction patterns using:

- Transaction amount
- Transaction hour
- Number of transactions in the last hour

The `is_fraud` column is excluded from model training to prevent label leakage.

### Step 5: Calculate the Combined Risk Score

The application combines the rule-based score and machine-learning score:

`Combined Risk Score = (0.5 × Rule Risk Score) + (0.5 × ML Risk Score)`

The application categorizes transactions as:

- **Low Risk:** Score below 30
- **Medium Risk:** Score from 30 to below 60
- **High Risk:** Score of 60 or above

A transaction is flagged for review if either the rule-based detector or the machine-learning detector flags it.

The score is an indicative risk indicator, not a calibrated probability of fraud.

### Step 6: View and Download Results

The dashboard displays validation summaries, transaction risk scores, risk levels, review flags, and explanations. Users can filter the results and download the analysis report.

## 6. Dataset Format

The input CSV should contain the following columns:

| Column | Description |
|---|---|
| `transaction_id` | Unique transaction identifier |
| `amount` | Transaction amount |
| `transaction_hour` | Transaction hour, from 0 to 23 |
| `transactions_last_hour` | Number of transactions in the last hour |
| `is_fraud` | Optional ground-truth label: 0 for normal, 1 for fraud |

The `is_fraud` column is optional for scoring. If provided, it is used for evaluation only and must not be used as a model feature.

## 7. Installation and Setup

### Prerequisites

- Python 3.10 or a compatible Python version
- Visual Studio Code or another Python editor

### Step 1: Open the Project Folder

Open the `Transaction-Risk-Detection` folder in VS Code.

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 3: Activate the Virtual Environment

For Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 4: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The `requirements.txt` file should contain:

```text
pandas
numpy
scikit-learn
streamlit
pytest
```

## 8. Run the Application

From the project root directory, run:

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal, usually:

`http://localhost:8501`

### Using the Application

1. Upload a transaction CSV file.
2. Review the uploaded data and validation summary.
3. Inspect invalid records and their validation errors.
4. Analyse valid transactions using rule-based and machine-learning detection.
5. Filter transactions by risk level or review status.
6. Review risk scores and explanations.
7. Download the transaction risk report and invalid-record report.

## 9. Run Tests

Run the test suite from the project root:

```bash
python -m pytest -v
```

The tests cover data validation, rule-based scoring, and machine-learning prediction functionality.

## 10. Model Evaluation

When the uploaded dataset contains ground-truth fraud labels, the application can calculate:

- **Precision:** The proportion of flagged transactions that are actually fraud.
- **Recall:** The proportion of actual fraud transactions detected.
- **F1 Score:** A combined measure of precision and recall.
- **Confusion Matrix:** Counts of correct and incorrect predictions.

For meaningful performance estimates, evaluation should use a separate, unseen test dataset. Results from training and evaluating on the same records can be misleading.

## 11. Limitations

- A flagged transaction is not necessarily fraudulent.
- Transaction amount alone cannot establish fraud.
- The detection thresholds are illustrative.
- Anomaly scores are not calibrated fraud probabilities.
- Synthetic data does not establish real-world fraud detection performance.
- Production use requires representative historical data, careful testing, threshold tuning, monitoring, and security controls.

## 12. Future Improvements

- Evaluate on a separate test dataset.
- Improve explanations for machine-learning anomalies.
- Tune thresholds to reduce false positives.
- Add database integration and real-time transaction streaming.
- Implement model monitoring and audit logging.

## 13. Conclusion

The Real-Time Transaction Risk Detection Platform demonstrates how data validation, rule-based detection, and machine-learning anomaly detection can be combined in an interactive dashboard for batch transaction analysis.

The application supports CSV uploads, data-quality checks, suspicious transaction identification, risk scoring, model evaluation, and downloadable reports.

This project is intended for educational and technical evaluation purposes and is not a production-ready financial fraud detection system.

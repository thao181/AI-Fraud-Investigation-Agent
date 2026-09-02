"""
Sinh dữ liệu giao dịch thẻ giả lập (synthetic) và train một model XGBoost
để phát hiện gian lận. Model này sẽ thay thế node "Prototype Fraud Risk
Scoring" (rule-based) trong workflow n8n bằng một model ML thật.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix
)
from xgboost import XGBClassifier
import joblib

RNG = np.random.default_rng(42)
N = 20000

# ---------- Sinh feature ----------
customer_avg_amount = RNG.lognormal(mean=np.log(120), sigma=0.5, size=N)
amount = RNG.lognormal(mean=np.log(90), sigma=0.9, size=N)
ratio = amount / np.maximum(customer_avg_amount, 1)

cross_border = RNG.binomial(1, 0.12, size=N)
merchant_risk_score = np.clip(RNG.normal(30, 20, size=N), 0, 100)
transaction_hour = RNG.integers(0, 24, size=N)
is_night = ((transaction_hour >= 0) & (transaction_hour <= 5)).astype(int)
transactions_last_24h = RNG.poisson(1.5, size=N)
recent_failed_attempts = RNG.poisson(0.3, size=N)
is_new_merchant = RNG.binomial(1, 0.25, size=N)
card_not_present = RNG.binomial(1, 0.35, size=N)

# ---------- Sinh nhãn fraud (xác suất phụ thuộc phi tuyến vào feature + nhiễu) ----------
logit = (
    -6.6
    + 1.3 * np.log1p(ratio)
    + 1.9 * cross_border
    + 0.032 * merchant_risk_score
    + 1.05 * is_night
    + 0.4 * transactions_last_24h
    + 0.65 * recent_failed_attempts
    + 0.8 * is_new_merchant
    + 0.45 * card_not_present
    + RNG.normal(0, 0.45, size=N)  # nhiễu để model không "học vẹt" 1 rule tuyến tính
)
fraud_prob = 1 / (1 + np.exp(-logit))
is_fraud = RNG.binomial(1, fraud_prob)

df = pd.DataFrame({
    "amount": amount,
    "customer_avg_transaction_amount": customer_avg_amount,
    "amount_to_average_ratio": ratio,
    "cross_border": cross_border,
    "merchant_risk_score": merchant_risk_score,
    "transaction_hour": transaction_hour,
    "is_night": is_night,
    "transactions_last_24h": transactions_last_24h,
    "recent_failed_attempts": recent_failed_attempts,
    "is_new_merchant": is_new_merchant,
    "card_not_present": card_not_present,
    "is_fraud": is_fraud,
})

print(f"Tỷ lệ fraud trong dữ liệu: {df['is_fraud'].mean():.2%}")

FEATURES = [
    "amount", "customer_avg_transaction_amount", "amount_to_average_ratio",
    "cross_border", "merchant_risk_score", "transaction_hour", "is_night",
    "transactions_last_24h", "recent_failed_attempts", "is_new_merchant",
    "card_not_present",
]

X = df[FEATURES]
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.08,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="aucpr",
    scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
    random_state=42,
)
model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

print("\n--- Kết quả đánh giá trên tập test ---")
print(classification_report(y_test, y_pred, digits=3))
print("ROC-AUC:", round(roc_auc_score(y_test, y_pred_proba), 4))
print("PR-AUC (Average Precision):", round(average_precision_score(y_test, y_pred_proba), 4))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

print("\n--- Feature importance ---")
importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(importances)

joblib.dump({"model": model, "features": FEATURES}, "model.joblib")
print("\nĐã lưu model vào model.joblib")

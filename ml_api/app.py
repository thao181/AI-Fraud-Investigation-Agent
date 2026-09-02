
from datetime import datetime, timezone
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Fraud Risk Scoring API")

bundle = joblib.load("model.joblib")
MODEL = bundle["model"]
FEATURES = bundle["features"]


class Transaction(BaseModel):
    transaction_id: str
    card_id: str
    customer_id: Optional[str] = None
    amount: float
    currency: str
    merchant: str
    merchant_category: Optional[str] = None
    merchant_country: str
    customer_country: str
    location: Optional[str] = None
    timestamp: Optional[str] = None
    transaction_hour: Optional[int] = None
    customer_avg_transaction_amount: Optional[float] = 120.0
    is_new_merchant: Optional[bool] = False
    card_not_present: Optional[bool] = False
    merchant_risk_score: Optional[float] = 25.0
    transactions_last_24h: Optional[int] = 1
    recent_failed_attempts: Optional[int] = 0
    customer_history_summary: Optional[str] = None


def resolve_hour(tx: Transaction) -> int:
    if tx.transaction_hour is not None:
        return tx.transaction_hour
    if tx.timestamp:
        try:
            dt = datetime.fromisoformat(tx.timestamp.replace("Z", "+00:00"))
            return dt.astimezone(timezone.utc).hour
        except ValueError:
            pass
    return 12


@app.post("/predict")
def predict(tx: Transaction):
    avg_amount = max(tx.customer_avg_transaction_amount or 120.0, 1.0)
    ratio = tx.amount / avg_amount
    cross_border = int(tx.merchant_country.upper() != tx.customer_country.upper())
    hour = resolve_hour(tx)
    is_night = int(0 <= hour <= 5)
    merchant_risk = min(max(tx.merchant_risk_score or 25.0, 0), 100)

    features = {
        "amount": tx.amount,
        "customer_avg_transaction_amount": avg_amount,
        "amount_to_average_ratio": ratio,
        "cross_border": cross_border,
        "merchant_risk_score": merchant_risk,
        "transaction_hour": hour,
        "is_night": is_night,
        "transactions_last_24h": tx.transactions_last_24h or 0,
        "recent_failed_attempts": tx.recent_failed_attempts or 0,
        "is_new_merchant": int(tx.is_new_merchant or False),
        "card_not_present": int(tx.card_not_present or False),
    }
    x = np.array([[features[f] for f in FEATURES]])
    proba = float(MODEL.predict_proba(x)[0, 1])
    risk_score = round(proba * 100)
    risk_level = "HIGH" if risk_score >= 70 else "MEDIUM" if risk_score >= 40 else "LOW"

    indicators = []
    if ratio >= 5:
        indicators.append(f"Amount is {ratio:.1f}x the customer's average transaction")
    if cross_border:
        indicators.append("Cross-border transaction")
    if tx.is_new_merchant:
        indicators.append("First transaction with this merchant")
    if is_night:
        indicators.append("Transaction occurred during an unusual overnight period")
    if merchant_risk >= 60:
        indicators.append(f"Merchant risk score is elevated ({merchant_risk:.0f}/100)")
    if (tx.transactions_last_24h or 0) >= 5:
        indicators.append(f"{tx.transactions_last_24h} transactions recorded in the last 24 hours")
    if (tx.recent_failed_attempts or 0) >= 3:
        indicators.append(f"{tx.recent_failed_attempts} recent failed attempts")
    if tx.card_not_present:
        indicators.append("Card-not-present transaction")

    return {
        **tx.model_dump(),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "amount_to_average_ratio": round(ratio, 2),
        "risk_indicators": indicators,
        "requires_investigation": risk_score >= 60,
        "scoring_version": "ml-v1-xgboost",
        "scoring_note": "Score produced by a trained XGBoost model on synthetic data (portfolio demo).",
    }


@app.get("/health")
def health():
    return {"status": "ok"}

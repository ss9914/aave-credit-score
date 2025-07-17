# Wallet Credit Scoring

## Problem Statement
100K user-wallet transactions, the goal is to build a machine learning model that generates a **credit score from 0 to 1000** for each wallet based on historical behavior.

## Features Engineered
- Transaction count per wallet
- Days active on the protocol
- Action types per wallet (deposit, borrow, repay, etc.)

## Model
- `IsolationForest` to detect anomalies.
- `MinMaxScaler` to normalize scores to 0-1000.
- Wallets closer to normal behavior → higher score.
- Riskier or abnormal behavior → lower score.

## Outputs
- `wallet_scores.csv`: CSV of wallets and scores
- `score_distribution.png`: Distribution of scores plot
- `models/model.pkl`: Saved IsolationForest model
- `models/scaler.pkl`: Saved MinMaxScaler

## How to Run
```bash
python score_wallets.py --input data/transactions.json --output wallet_scores.csv
pip install -r requirements.txt
git add README.md
git commit -m "Added proper README.md"
git push

# Analysis of Wallet Credit Scoring

## Score Distribution
The score distribution is visualized in `score_distribution.png`. It clearly shows how most wallets fall into specific ranges, e.g.:
- **0-200**: Very risky behavior, possible bots, or exploit attempts.
- **200-500**: Irregular or inconsistent repayment patterns.
- **500-800**: Healthy but cautious usage.
- **800-1000**: Consistent deposits, repayments, stable activity.

## Key Observations
- **Low Score Wallets (0-200)**  
  These wallets often show signs of:
  - Liquidations
  - Erratic borrowing
  - Unstable deposit/withdrawal behavior

- **Mid Score Wallets (200-500)**  
  These wallets show inconsistent usage patterns or recent entry into the protocol.

- **High Score Wallets (800-1000)**  
  These wallets have:
  - Consistent deposit behavior
  - Repay their borrowings
  - Avoid liquidations
  - Longer protocol history

## Outputs Generated
- `wallet_scores.csv`: Final credit scores for all wallets
- `score_distribution.png`: Visualization of the score spread
- `models/model.pkl`: Saved anomaly detection model
- `models/scaler.pkl`: Scaler for standardizing inputs

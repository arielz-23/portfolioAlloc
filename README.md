# ETF Sector Rotation Portfolio Allocation Model

> **University Final Project** · Machine Learning · Not financial advice.

A reproducible machine-learning workflow that predicts and allocates capital across the 11 major U.S. SPDR sector ETFs using daily price data, FRED macroeconomic features, and XGBoost regression.

---

## ETF Universe

| Ticker | Sector                  |
|--------|-------------------------|
| XLK    | Technology              |
| XLV    | Healthcare              |
| XLF    | Financials              |
| XLI    | Industrials             |
| XLY    | Consumer Discretionary  |
| XLP    | Consumer Staples        |
| XLE    | Energy                  |
| XLU    | Utilities               |
| XLB    | Materials               |
| XLRE   | Real Estate             |
| XLC    | Communication Services  |

Benchmark: **SPY** (SPDR S&P 500 ETF)

---

## Methodology

### Target Variable
For each ETF and date:
```
future_21d_return     = close[t+21] / close[t] - 1
future_21d_volatility = realised volatility of next 21 trading days (annualised)
target_score          = future_21d_return / future_21d_volatility
```

### Two-Model Approach
Two XGBoost regressors are trained:
1. **Return model** → predicts `future_21d_return`
2. **Volatility model** → predicts `future_21d_volatility`

Predicted score = predicted_return / predicted_volatility

### Features
- Momentum returns: 5d, 21d, 63d, 126d, 252d
- Realised volatility: 21d, 63d (annualised)
- Downside volatility: 63d
- Moving-average distance: 50d, 200d
- Relative strength vs equal-weight universe: 21d, 63d
- Rolling correlation with SPY: 63d
- RSI (14d), Bollinger %B (20d)
- ETF one-hot dummies
- **FRED Macro**: 10Y–2Y yield spread, Fed Funds Rate, CPI YoY, unemployment, HY spread, VIX, EUR/USD, industrial production YoY

### Portfolio Construction
At each monthly rebalance:
1. Predict score for all 11 ETFs
2. Select top 5 by predicted score
3. Weight proportional to `max(0, predicted_score)`
4. Constrain: long-only, max 35% per ETF, sum to 100%

### Validation
Strict chronological split — **no data leakage**:
- Train: earliest 70%
- Validation: next 15% (early stopping)
- Test: final 15% (held-out evaluation)

---

## Repository Structure

```
.
├── data/
│   ├── raw/            # Parquet files: etf_prices, spy, macro
│   ├── processed/      # features.csv (panel dataset)
│   └── artifacts/      # trained models, backtest results, allocation
├── reports/
│   ├── insights.md         # manual analysis notes
│   ├── evaluation_report.md # auto-generated after training
│   └── model_card.md        # auto-generated model card
├── src/
│   ├── config.py        # all configurable parameters
│   ├── download_data.py # data acquisition (Excel + yfinance + FRED)
│   ├── build_dataset.py # feature panel construction
│   ├── features.py      # individual feature functions
│   ├── train_model.py   # XGBoost training & inference
│   ├── portfolio.py     # weight optimisation
│   ├── backtest.py      # vectorised backtest engine
│   ├── evaluation.py    # metrics & report generation
│   └── app.py           # Streamlit dashboard
├── notebooks/           # exploratory notebooks (optional)
├── Book3.xlsx           # raw ETF price data (Bloomberg export)
├── requirements.txt
├── README.md
└── main.py              # end-to-end pipeline runner
```

---

## Installation

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt
```

### FRED API Key (optional but recommended)

Macro features (yield curve, VIX, CPI, etc.) require a free FRED API key.

1. Register at https://fred.stlouisfed.org/docs/api/api_key.html
2. Set the environment variable:

```bash
# Windows (PowerShell)
$env:FRED_API_KEY = "your_key_here"

# Windows (Command Prompt)
set FRED_API_KEY=your_key_here

# macOS / Linux
export FRED_API_KEY=your_key_here
```

Without a key, the pipeline still runs using price features only.

---

## How to Run

### Full pipeline (recommended)

```bash
python main.py
```

This runs all 6 steps in sequence:
1. Load ETF prices from `Book3.xlsx`
2. Download SPY from yfinance + macro from FRED
3. Build the feature panel (`data/processed/features.csv`)
4. Train XGBoost models (`data/artifacts/`)
5. Run the vectorised backtest (`data/artifacts/backtest_results.csv`)
6. Compute the latest allocation + write evaluation reports

### Skip steps (using cached data)

```bash
python main.py --skip-download   # use cached raw data
python main.py --skip-train      # use cached models
python main.py --skip-download --skip-train  # only redo backtest + reports
```

### Launch the Streamlit dashboard

```bash
streamlit run src/app.py
```

Then open http://localhost:8501 in your browser.

---

## Expected Outputs

| File | Description |
|------|-------------|
| `data/processed/features.csv` | Long-panel dataset (date × ETF × features) |
| `data/artifacts/xgb_return_model.joblib` | Trained return-prediction model |
| `data/artifacts/xgb_vol_model.joblib` | Trained volatility-prediction model |
| `data/artifacts/backtest_results.csv` | Daily equity curve + returns for all strategies |
| `data/artifacts/latest_allocation.csv` | Current sector weights |
| `data/artifacts/weight_history.csv` | Monthly weight history |
| `reports/evaluation_report.md` | Performance metrics + feature importance |
| `reports/model_card.md` | Full ML model card |

---

## Configuration

All parameters are in `src/config.py`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FORWARD_DAYS` | 21 | Prediction horizon (trading days) |
| `TOP_N` | 5 | Number of ETFs selected per rebalance |
| `MAX_WEIGHT` | 0.35 | Per-ETF weight cap |
| `TRAIN_FRAC` | 0.70 | Fraction of data for training |
| `TWO_MODEL` | True | Use separate return + vol models |
| `TRANSACTION_COST` | 0.0005 | One-way cost per rebalance leg |
| `USE_ALFRED_VINTAGES` | False | Use real-time FRED vintages |

---

## Performance Metrics Calculated

- CAGR (Compound Annual Growth Rate)
- Annualised Volatility
- Sharpe Ratio
- Maximum Drawdown
- Calmar Ratio
- Total Return
- Win Rate (by month)
- Average Monthly Return
- Worst Month
- Average Monthly Turnover

---

## Limitations

1. **Survivorship bias** – ETF universe fixed in hindsight
2. **Short history** – XLRE (Oct 2015), XLC (Jun 2018) reduce effective sample
3. **Simplified costs** – No market-impact model, no bid-ask spread
4. **Regime dependency** – Model may underperform in unseen market regimes
5. **FRED publication lag** – Monthly macro data has 2–4 week release delays
6. **Not live-tradeable** – Academic prototype only

---

## Disclaimer

This project is developed solely for academic purposes as a university final project in Machine Learning. It does not constitute investment advice, financial advice, or a recommendation to buy or sell any security. Past performance is not indicative of future results. The authors assume no responsibility for financial decisions made based on this work.

---

## References

- Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System. *KDD*.
- Mitchell et al. (2019). Model Cards for Model Reporting. *ACM FAccT*.
- Asness et al. (2013). Value and Momentum Everywhere. *Journal of Finance*.
- Moskowitz & Grinblatt (1999). Do Industries Explain Momentum? *Journal of Finance*.

# Insights & Findings

*Generated manually during project analysis*

---

## Data Overview

- **Universe**: 11 SPDR Sector ETFs (XLK, XLV, XLF, XLI, XLY, XLP, XLE, XLU, XLB, XLRE, XLC)
- **Source**: Bloomberg terminal export (Book3.xlsx) + Yahoo Finance (SPY) + FRED API (macro)
- **Date range**: 2015-06-04 → 2026-06-03 (~2,766 trading days)
- **Note**: XLRE was launched in October 2015; XLC in June 2018. Rows before inception are NaN.

---

## Feature Engineering Observations

### Momentum Signals
- Longer-horizon momentum (63d, 126d) tends to capture sector regime trends better than short-term noise (5d).
- Reversal (negative autocorrelation) is observable at the 5d level in some sectors (XLE, XLB).

### Volatility Features
- Realised volatility over 21d captures short-term risk spikes (e.g., COVID March 2020).
- Downside volatility (63d) is more informative than symmetric volatility for return prediction.

### Relative Strength
- Relative strength vs. the equal-weight universe has consistent explanatory power.
- Sectors with the highest relative strength tend to maintain momentum for 1–2 months.

### Macro Features (FRED)
- **Yield spread (T10Y2Y)**: Negative spread (inverted curve) is historically correlated
  with outperformance of defensive sectors (XLU, XLP, XLV) over cyclicals (XLY, XLF).
- **VIX percentile rank**: High VIX regimes favour low-beta sectors (XLU, XLV).
- **HY spread**: Widening credit spreads are negative for financials (XLF) and cyclicals.
- **CPI YoY**: High inflation regimes tend to favour energy (XLE) and materials (XLB).

---

## Model Findings

### Information Coefficient (Rank Correlation)
- Return model test IC: ~0.04–0.08 (typical for cross-sectional equity models)
- Volatility model test IC: ~0.10–0.20 (volatility is more predictable than returns)
- Combined predicted score IC: slightly higher than return model alone

### Feature Importance
- Top features typically include:
  1. Relative strength (21d, 63d)
  2. Rolling 126d and 252d momentum
  3. Downside volatility (63d)
  4. VIX percentile (macro)
  5. Yield spread (macro)
  6. ETF dummy variables (sector identity)

### Overfitting Guard
- Time-series split is strictly enforced — no data leakage
- Early stopping (50 rounds on validation) prevents overfitting
- Max depth of 4 limits tree complexity

---

## Portfolio Construction Notes

- **Top-N selection**: TOP_N=5 gives a balanced concentration/diversification trade-off.
- **Max weight cap (35%)**: Prevents dominance by a single sector;
  historically important because XLK (Technology) would otherwise dominate.
- **Redistribution**: Iterative cap-and-redistribute is numerically stable and correct.
- **Monthly rebalance**: Daily rebalance adds excessive transaction costs with limited benefit.

---

## Backtest Observations

### Performance Attribution
- The model adds value primarily by **avoiding the worst sector** rather than by
  perfectly picking the best one — classic negative selection.
- **Energy sector** (XLE) is the most volatile and shows both the highest max gains and losses.
- During COVID crash (Feb-Mar 2020), the model may over-allocate to recently strong sectors,
  resulting in larger drawdowns.

### Regime Analysis
- **Low-rate regime (2015–2021)**: Growth sectors (XLK, XLC) tend to dominate.
- **Rate-hike regime (2022–2023)**: Defensive and value sectors (XLE, XLP) tend to lead.
- **Post-hike stabilisation (2024–2026)**: Mixed; sector leadership less clear.

---

## Limitations Identified

1. **Short XLRE / XLC history**: Only ~10 years of data for the full 11-ETF universe.
2. **FRED publication lag**: Monthly releases are not available on release day.
   Use `USE_ALFRED_VINTAGES=True` to partially address this.
3. **Transaction costs**: 5 bps (one-way) is optimistic; real spread + market impact
   for a multi-million-dollar fund would be higher.
4. **No regime detection**: The model does not explicitly identify market regimes
   (e.g., bull/bear/sideways) and apply different strategies per regime.

---

## Suggestions for Extension

- Add **macro regime filter**: Use HMM or recession indicators to switch between
  offensive and defensive strategies.
- Add **earnings/fundamental data**: P/E, earnings growth, dividend yield per sector.
- Add **sentiment features**: Put/call ratio, AAII sentiment, Google Trends.
- Implement **Bayesian optimisation** for hyper-parameter tuning.
- Consider **ensemble methods**: Blend XGBoost predictions with a linear model
  for better generalisation across regimes.

# Medication Usage Forecasting from ICD Codes

## Overview
**Plain language:**  
This project focuses on predicting future medication usage in healthcare settings by analyzing historical diagnosis codes (ICD-9/10). By learning patterns in how diseases evolve over time, the model can forecast medication demand before it occurs, helping hospitals and healthcare systems prepare resources more effectively.

**Technical:**  
We model medication usage as a multivariate time-series forecasting problem using ICD code sequences as inputs. Multiple forecasting approaches are evaluated, including traditional statistical baselines and deep learning–based temporal models.

---

## Motivation
**Plain language:**  
Hospitals need to anticipate which medications will be needed in the future to avoid shortages, reduce waste, and improve patient care. Manual forecasting is difficult because disease patterns change over time and vary across populations.

**Technical:**  
ICD codes provide a standardized representation of patient diagnoses. Leveraging their temporal structure enables predictive modeling of downstream medication usage, offering a data-driven alternative to heuristic-based planning.

---

## Data
**Plain language:**  
The project uses real-world healthcare data that records diagnoses and medication usage over time.

**Technical:**  
The dataset includes:
- ICD-9 and ICD-10 diagnosis codes
- Medication usage counts
- Time-indexed hospital or facility-level records

ICD codes are encoded into structured time series aligned with medication usage intervals.

---

## Methodology
**Plain language:**  
We transform diagnosis histories into time-based signals and train models to predict how medication usage will change in the future.

**Technical:**  

1. **Data Encoding**
   - ICD-9/10 codes mapped into numerical representations
   - Aggregation into fixed time windows
   - Construction of multivariate ICD time series

2. **Forecasting Models**
   - Naive baseline models
   - Vector Autoregression (VAR)
   - Deep learning models, including:
     - Temporal Fusion Transformer (TFT)
     - Attention-based Recurrent LSTM (AR-LSTM)

3. **Training and Evaluation**
   - Sliding-window forecasting
   - Multi-step prediction
   - Comparison across forecasting horizons

---

## Key Findings
**Plain language:**  
- Models that account for long-term trends and interactions perform better than simple baselines.
- Deep learning models are more accurate when disease patterns are complex and non-linear.
- The system can detect when predictions become unreliable and trigger retraining.

**Technical:**  
- Transformer-based models achieve the highest forecasting accuracy.
- Attention mechanisms improve interpretability by highlighting influential ICD codes.
- Performance drift detection enables adaptive retraining to maintain reliability over time.

---

## Figures
All figures referenced in the paper are included in this repository.

Example:
```md
![Medication Usage Forecast](figures/medication_forecast.png)

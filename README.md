# Customer Churn Prediction — Reproducible ML Comparison

**Big Data / Machine Learning coursework project (BOZH 2).**

This is an **adapted educational project inspired by the user-provided Kaggle customer-churn notebook**. The original notebook did **not** include the input dataset. To provide ready-to-use, verified results **without requiring a Kaggle download**, this project includes a newly generated **synthetic** telecom dataset. The measurements below apply to this synthetic dataset only, not to the original Kaggle benchmark.

## Experimental results (executed)

| Algorithm | Accuracy | Precision (churn) | Recall (churn) | F1 (churn) | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 80.27% | 55.71% | 26.53% | 35.94% | 79.82% |
| Neural Network (MLP) | 79.91% | 58.21% | 13.27% | 21.61% | 76.75% |
| Random Forest | 79.35% | 51.18% | 22.11% | 30.88% | 76.14% |

The best overall ROC-AUC was achieved by Logistic Regression (79.82%). Accuracy differences are percentage points, not relative improvement percentages. **All models have limited churn recall**, so accuracy alone is not sufficient to assess real-world churn detection.

## Included files

- `data/synthetic_telco_customers.csv` — **7,043 synthetic customer records**
- `run_project.py` — complete reproducible training and evaluation script
- `notebooks/customer_churn_results.ipynb` — ready-to-run notebook
- `results/metrics.csv` — actual computed scores
- `results/summary.txt` — exact text output
- `results/model_comparison.png`, `roc_curves.png`, `confusion_matrix.png` — charts generated from predictions

## How to run

Install Python 3.10+ then run:

```bash
pip install -r requirements.txt
python run_project.py
```

The script regenerates results from the included CSV; fixed random seed = 42. The dataset is created once if absent.

## Data and attribution

Source inspiration: **Customer Churn Prediction** Kaggle notebook, uploaded by the project author for adaptation. Original IBM/Kaggle Telco Customer Churn data is not included; the data provided here is artificial and must not be presented as original Kaggle data. This implementation extends the ML comparison to Logistic Regression, Random Forest and MLP, and supplies reproducible evaluation artifacts.

## Key findings

Logistic Regression achieved **80.27% test accuracy**, exceeding Random Forest by **0.92 percentage points** and MLP by **0.36 percentage points** on this generated data. Its ROC-AUC was **79.82%**. Results may change with another dataset or train/test split.

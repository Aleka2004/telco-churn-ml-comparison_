# Telco Customer Churn — Machine Learning Comparison

**Academic project: Applying Machine Learning Methods in Big Data Analytics (Independent Study 2).**

This repository is an **adapted educational project based on a Kaggle notebook supplied by the student**, `customer-churn-prediction.ipynb`. It is **not presented as an entirely original Kaggle work**. The original notebook's author and reuse license should be verified on its Kaggle page and credited before public redistribution.

## What was adapted

- Preserved the telco customer churn classification task and neural-network approach (MLP).
- Added **Logistic Regression** and **Random Forest** for comparison.
- Reorganized preprocessing into scikit-learn `Pipeline` and `ColumnTransformer` to reduce data leakage.
- Used a stratified train/test split and compared Accuracy, Precision, Recall, F1-score and ROC-AUC.
- Added comparison plots, confusion matrix and CSV export.

## Dataset

Use the [IBM Telco Customer Churn dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` and place it in `data/`. The dataset is **not included in this ZIP**.

## Run locally

```bash
python -m pip install -r requirements.txt
jupyter notebook notebooks/telco_churn_comparison.ipynb
```

Alternatively, upload the notebook to Kaggle or Google Colab and provide the dataset. Run all cells to obtain experimental results. **No performance numbers are claimed without executing the notebook.**

## Repository structure

- `notebooks/telco_churn_comparison.ipynb`: main experiment
- `data/README.md`: dataset instructions
- `results/README.md`: where evaluation outputs are written
- `docs/PROJECT_OVERVIEW.md`: project overview
- `requirements.txt`: dependencies

## Scope

The notebook runs in scikit-learn on a single machine. It demonstrates machine-learning methods applicable to large-data analytics, **but does not implement distributed Big Data processing**. For a genuinely distributed implementation, migrate the pipeline to Apache Spark MLlib and benchmark it on larger datasets.

## Attribution

**Adapted from:** user-provided Kaggle notebook `customer-churn-prediction.ipynb` (original Kaggle author/license to be confirmed). **Data:** IBM Telco Customer Churn dataset, hosted on Kaggle by blastchar. Retain the original source attribution when publishing derivatives.

# Machine Learning Lab

Lab experiments and mini-projects from my **Machine Learning** course at The National Institute of Engineering (NIE), Mysuru. The repo walks through the fundamentals in order: Python data libraries, linear and polynomial regression, cross-validation, regularization, and a logistic regression classifier.

**Author:** Gagangowda H L · **USN:** 4NI24CI043 · BE Computer Engineering (AI & ML), 2024-2028

---

## Contents

| Notebook | Topic | What it covers |
|----------|-------|----------------|
| [`exp1.ipynb`](./exp1.ipynb) | Python for ML | `math`, NumPy, pandas, and Matplotlib basics (line, bar, histogram, scatter, pie charts) |
| [`LinearRegression.ipynb`](./LinearRegression.ipynb) | Linear regression | Predicting marks from hours studied with scikit-learn, 12/3 train-test split, slope and intercept, train and test MSE, regression line plot |
| [`PolynomialRegression.ipynb`](./PolynomialRegression.ipynb) | Polynomial regression | Degree 2 fit using `PolynomialFeatures`, fitted equation, MSE, smooth curve plot |
| [`k_fold_cv.ipynb`](./k_fold_cv.ipynb) | K-fold cross-validation | 5-fold CV on the study-hours data, per-fold equations and MSE, average CV MSE |
| [`lasso_ridge_regularization.ipynb`](./lasso_ridge_regularization.ipynb) | Regularization | OLS vs Ridge vs Lasso on a clean linear dataset with train/validation/test splits |
| [`activity1_case_b.ipynb`](./activity1_case_b.ipynb) | Outlier sensitivity | Same comparison as above with an intentional outlier (record 5), showing how OLS, Ridge, and Lasso respond |
| [`activity2_banknote.ipynb`](./activity2_banknote.ipynb) | Logistic regression | Banknote authentication: EDA, stratified 80:20 split, scaled pipeline, accuracy, precision, recall, F1, confusion matrix |
| [`activity2_report.pdf`](./activity2_report.pdf) | Mini-project report | Written report for the banknote authentication activity |

### Datasets

| File | Used by | Description |
|------|---------|-------------|
| `LinearRegression.csv` | Linear regression | `Hours_Studied`, `Marks` |
| `k_fold_linear_regression.csv` | K-fold CV | `Student`, `Hours_Studied`, `Marks` |
| `Degree 2 polynomial.csv`, `Degree 3 polynomial.csv` | Polynomial regression | Small `X`, `Y` samples |
| `activity1_case_b.csv` | Outlier activity | `X`, `Y` with one outlier |
| UCI Banknote Authentication | Logistic regression | 1,372 records, 4 wavelet features, binary class. Loaded directly from the UCI repository, so an internet connection is needed |

---

## Repository Structure

```
machine-learning-lab/
├── exp1.ipynb
├── LinearRegression.ipynb
├── LinearRegression.csv
├── PolynomialRegression.ipynb
├── Degree 2 polynomial.csv
├── Degree 3 polynomial.csv
├── k_fold_cv.ipynb
├── k_fold_linear_regression.csv
├── lasso_ridge_regularization.ipynb
├── activity1_case_b.ipynb
├── activity1_case_b.csv
├── activity2_banknote.ipynb
├── activity2_report.pdf
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- Jupyter Notebook, JupyterLab, or VS Code with the Jupyter extension (the notebooks were written in Google Colab and also run there)

### Setup

```bash
git clone https://github.com/KeyboardNoMouse/machine-learning-lab.git
cd machine-learning-lab

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install numpy pandas matplotlib scikit-learn jupyter
jupyter lab
```

### Running the notebooks

Open any notebook and run the cells top to bottom.

> **Note on file paths:** some notebooks were written for Colab and load CSVs by the name used there (`LinearRegression.ipynb` reads `linear_regression_lab_15_records.csv`, and `k_fold_cv.ipynb` reads `kfolds_lab.csv`). When running locally, point `pd.read_csv(...)` at `LinearRegression.csv` and `k_fold_linear_regression.csv` respectively.

---

## Tech Stack

- **Language:** Python
- **Libraries:** NumPy, pandas, Matplotlib, scikit-learn
- **Environment:** Jupyter / Google Colab

## Concepts Covered

- Simple linear regression and MSE
- Polynomial feature expansion
- Train / validation / test splitting
- K-fold cross-validation
- Ridge (L2) and Lasso (L1) regularization, and robustness to outliers
- Logistic regression, feature scaling with pipelines (no test-set leakage)
- Classification metrics: accuracy, precision, recall, F1, confusion matrix

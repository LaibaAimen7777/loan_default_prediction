# Loan Default Prediction Using Machine Learning and Statistical Data Fusion

## 1. Project Overview

This research project investigates how different class-imbalance handling techniques affect the performance of machine-learning models for loan default prediction.

The study uses two datasets:
- **Credit Risk Dataset:** Recipient dataset containing loan and borrower information, including the loan default target.
- **Financial Risk for Loan Approval Dataset:** Donor dataset used to enrich the recipient dataset through statistical data fusion.

## 2. Research Question

How do different class-imbalance handling techniques affect the performance of machine-learning models for loan default prediction?

## 3. Project Pipeline

The project follows these stages:

1. **Data Inspection and EDA:** Examine both datasets, their features, distributions, missing values, and data quality.
2. **Data Preprocessing:** Handle missing values, remove duplicates, and prepare variables for analysis.
3. **Train-Test Splitting:** Split the recipient dataset into training and testing sets using stratified sampling.
4. **Statistical Data Fusion:** Apply nearest-neighbour matching using age, log income, and homeownership to transfer selected attributes from the donor dataset.
5. **Dataset Construction:** Prepare three variants:
   - Original dataset
   - Fusion-enhanced dataset
   - Fusion-enhanced dataset without the eligibility feature
6. **Model Development:** Train Logistic Regression, Random Forest, and XGBoost models.
7. **Imbalance Handling:** Compare four strategies:
   - Baseline (no resampling)
   - SMOTENC
   - Random Over-Sampling (ROS)
   - Random Under-Sampling (RUS)
8. **Model Evaluation:** Use repeated cross-validation and evaluate Accuracy, Precision, Recall, F1-score, ROC-AUC, and PR-AUC.
9. **Statistical Analysis:** Apply corrected statistical comparisons and Holm adjustment for multiple comparisons.
10. **Final Testing:** Evaluate model performance on the held-out test set.

## 4. Repository Structure

```text
loan-default-prediction/
│
├── Loan_Default_Research/
│   ├── datasets/
│   │   ├── processed/              # Cleaned and prepared datasets
│   │   └── raw/                    # Original datasets
│   │
│   ├── notebooks/                  # Jupyter notebooks for each project stage
│   │
│   ├── results/                    # Experimental results and statistical outputs
│   │
│   ├── README.md                   # Research project documentation
│   │
│   └── requirements.txt            # Python dependencies
│
├── Notebook_01_Output/             # Outputs from Notebook 01
│
├── Notebook_02_Output/             # Outputs from Notebook 02
│
├── Notebook_03_Output/             # Outputs from Notebook 03
│
├── Notebook_04_Output/             # Outputs from Notebook 04
│
├── .gitignore                      # Files excluded from Git
│
└── README.md                       # Main repository documentation
```

## 5. Models and Techniques

**Machine-learning models**
- Logistic Regression
- Random Forest
- XGBoost

**Class-imbalance techniques**
- Baseline
- SMOTENC
- Random Over-Sampling
- Random Under-Sampling

**Evaluation metrics**
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

## 6. Tools and Technologies

- Python
- Jupyter Notebook
- Pandas and NumPy
- Scikit-learn
- Imbalanced-learn
- XGBoost
- Matplotlib and Seaborn

## 7. Contributors

- Arzu Ali
- Laiba Aimen

## 8. Project Status

Academic research project — Master of Science in Data Science.

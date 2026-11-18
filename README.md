# Loan Default Prediction Using Machine Learning and Statistical Data Fusion

## **Project Overview**

This project investigates how different class-imbalance handling techniques affect the performance of machine-learning models for loan default prediction.

The study uses a Credit Risk dataset as the recipient dataset and a Financial Risk for Loan Approval dataset as the donor dataset. Statistical data fusion is used to enrich the recipient dataset with additional financial attributes.

## **Research Question**

How do different class-imbalance handling techniques affect the performance of machine-learning models for loan default prediction?

## **Project Pipeline**

The project follows these main stages:

1. **Data Collection and Inspection:** Load and examine the Credit Risk and Financial Risk datasets, including their features, data types, missing values, and distributions.

2. **Data Preprocessing:** Clean the datasets, handle missing values, remove duplicates, and prepare the variables for analysis.

3. **Train-Test Splitting:** Split the recipient dataset into training and testing sets using stratified sampling to preserve the target-class distribution.

4. **Statistical Data Fusion:** Use nearest-neighbour matching based on age, log income, and homeownership to transfer selected financial attributes from the donor dataset to the recipient dataset.

5. **Dataset Construction:** Prepare three dataset variants:
   - Original dataset
   - Fusion-enhanced dataset
   - Fusion-enhanced dataset without the eligibility feature

6. **Model Development:** Train and evaluate Logistic Regression, Random Forest, and XGBoost models.

7. **Class-Imbalance Handling:** Compare four strategies:
   - Baseline (no resampling)
   - SMOTENC
   - Random Over-Sampling (ROS)
   - Random Under-Sampling (RUS)

8. **Model Evaluation:** Perform repeated cross-validation and evaluate the models using Accuracy, Precision, Recall, F1-score, ROC-AUC, and PR-AUC.

9. **Statistical Analysis:** Compare experimental results using corrected statistical tests and Holm adjustment for multiple comparisons.

10. **Final Testing:** Evaluate the selected experimental configurations on the held-out test set.

## **Repository Structure**

```text
loan-default-prediction/
│
├── datasets/
│   ├── raw/              # Original datasets
│   └── processed/        # Cleaned and prepared datasets
│
├── notebooks/
│   ├── 01_Data_Inspection_and_EDA.ipynb
│   ├── 02_Financial_Risk_Inspection_and_EDA.ipynb
│   ├── 03_Statistical_Data_Fusion.ipynb
│   └── 04_Model_Experiments.ipynb
│
├── results/              # Model results and statistical analysis outputs
├── figures/              # Visualizations and research paper figures
│
├── README.md             # Project overview, pipeline, and repository guide
├── requirements.txt      # Python dependencies
└── .gitignore             # Files excluded from version control
```

## **Technologies and Libraries**

- Python
- Jupyter Notebook
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- XGBoost
- Imbalanced-learn

## **Evaluation Metrics**

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

## **Contributors**

- Arzu Ali
- Laiba Aimen

## **Project Status**

Academic research project — Master of Science in Data Science.

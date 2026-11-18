Loan Default Prediction Using Machine Learning

Project Overview

This MS Data Science semester project investigates how different class-imbalance handling techniques affect the performance of machine-learning models for loan default prediction.

The project uses two datasets: a Credit Risk dataset as the recipient dataset and a Financial Risk dataset as the donor dataset. Statistical data fusion is applied to transfer additional financial attributes from donor records to recipient records.

Research Question

How do different class-imbalance handling techniques affect the performance of machine-learning models for loan default prediction?

Project Objectives

• Inspect and preprocess the recipient and donor datasets.
• Apply statistical data fusion using shared attributes.
• Compare original and fused datasets.
• Examine the effect of including and excluding the fusion-eligibility indicator.
• Compare different class-imbalance handling techniques.
• Evaluate Logistic Regression, Random Forest, and XGBoost.
• Assess model performance using repeated stratified cross-validation and a held-out test set.
• Conduct statistical comparisons between experimental configurations.
• Validate the fusion process using distribution checks and a random-donor placebo experiment.

Project Structure

Loan_Default_Research/
│
├── README.md
├── requirements.txt
│
├── datasets/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_Data_Inspection_and_EDA.ipynb
│   ├── 02_Financial_Risk_Inspection_and_EDA.ipynb
│   ├── 03_Statistical_Data_Fusion.ipynb
│   └── 04_Model_Experiments.ipynb
│
└── results/
    ├── figures/
    │   ├── Notebook_01_EDA/
    │   ├── Notebook_02_EDA/
    │   ├── Notebook_03_Fusion/
    │   └── Notebook_04_Experiments/
    │
    ├── all_hyperparameter_tuning_results.csv
    ├── repeated_cv_fold_results.csv
    ├── repeated_cv_summary.csv
    ├── confidence_intervals.csv
    ├── final_test_results.csv
    ├── corrected_statistical_tests.csv
    ├── random_donor_placebo_cv_results.csv
    ├── random_donor_placebo_summary.csv
    ├── random_donor_placebo_statistical_tests.csv
    └── placebo_final_test_comparison.csv
    
Datasets

1. Credit Risk Dataset (Recipient)

• Target column: loan_status
• 0 represents non-default.
• 1 represents default.
• Original dataset: 32,581 rows and 12 columns.
• Cleaned dataset: 32,416 rows after removing duplicates.

2. Financial Risk Dataset (Donor)

• Contains additional financial attributes.
• Used as the donor dataset for statistical data fusion.
• LoanApproved represents a lending decision and is not used as the loan-default target.

Raw datasets are stored in datasets/raw/, while cleaned, processed, and fused datasets are stored in datasets/processed/.

Methodology

1. Data Inspection and Preprocessing

The first two notebooks perform exploratory data analysis and preprocessing of the recipient and donor datasets. They examine missing values, duplicates, data types, distributions, outliers, and relationships between variables.

2. Statistical Data Fusion

The third notebook performs statistical data fusion using shared attributes between the recipient and donor datasets.

• Recipient dataset: Credit Risk.
• Donor dataset: Financial Risk.
• Matching attributes include age, annual income, and homeownership.
• k-nearest neighbours (kNN) matching with k = 5.
• Matching-distance eligibility threshold determined using training data.
• Numerical donor features aggregated using median values.
• Binary donor features aggregated using proportions.
• Original recipient target labels are retained.

Three dataset variants are prepared:

1. Original: Original recipient features.
2. Fused: Original features combined with transferred donor features and the fusion-eligibility indicator.
3. Fused_No_Eligibility: Fused features without the eligibility indicator.

Fusion validation includes distribution comparisons and a random-donor placebo experiment.

3. Class-Imbalance Handling

Four strategies are evaluated:

• Baseline (No Resampling)
• SMOTENC
• Random Over-Sampling
• Random Under-Sampling

ADASYN was explored during preliminary diagnostics but excluded from the final experiment because the tested implementation generated fractional values in one-hot encoded categorical features. Random Over-Sampling was used instead.

Resampling is performed only on training folds. Validation and final test data retain their original class distributions.

4. Machine-Learning Models

Three models are evaluated:

• Logistic Regression
• Random Forest
• XGBoost

Hyperparameters are tuned for each model and dataset variant using baseline pipelines. The selected parameters are then applied across the four imbalance-handling strategies for the corresponding model and dataset variant.

5. Model Evaluation

The main experiment includes:

• 3 dataset variants
• 3 machine-learning models
• 4 imbalance-handling strategies
• 36 experimental configurations
• 25 cross-validation folds per configuration
• 900 total cross-validation fits

Repeated stratified five-fold cross-validation is used for model evaluation.

The following metrics are reported:

• Accuracy
• Precision
• Recall
• F1-score
• ROC-AUC
• PR-AUC

Corrected approximate 95% confidence intervals are calculated using the Nadeau-Bengio correction.

Corrected resampled t-tests are used for statistical comparisons, with Holm adjustment applied to the defined comparison groups.

Final model evaluation is performed using the held-out test set, which is not used for hyperparameter tuning or model selection.

6. Random-Donor Placebo Experiment

A random-donor placebo experiment compares:

• Original dataset
• Matched fused dataset
• Random-donor fused dataset

This experiment investigates whether matching donor records provides useful information beyond randomly assigning donor records.

Tools and Technologies

• Python
• Jupyter Notebook
• pandas
• NumPy
• SciPy
• Matplotlib
• Seaborn
• scikit-learn
• imbalanced-learn
• XGBoost
• statsmodels

Installation and Execution

1. Create a virtual environment

python -m venv .venv

Activate on Windows:

.venv\Scripts\activate

Activate on macOS/Linux:

source .venv/bin/activate

2. Install dependencies

pip install -r requirements.txt

3. Launch Jupyter Notebook

jupyter notebook

4. Execute notebooks in order

1. 01_Data_Inspection_and_EDA.ipynb
2. 02_Financial_Risk_Inspection_and_EDA.ipynb
3. 03_Statistical_Data_Fusion.ipynb
4. 04_Model_Experiments.ipynb

Later notebooks depend on the processed datasets and outputs generated by earlier notebooks.

Note: Notebook 04 includes hyperparameter tuning and repeated cross-validation. The main experiment involves 900 model fits and may require considerable computational time. Previously generated results are available in the results/ directory.

Results

The results/ directory contains:

• Hyperparameter-tuning results.
• Repeated cross-validation fold-level results.
• Cross-validation summary.
• Corrected confidence intervals.
• Final test evaluation results.
• Corrected statistical test results.
• Random-donor placebo cross-validation results.
• Random-donor placebo final test comparison.
• Random-donor placebo statistical test results with Holm-adjusted p-values.

Authors

• Arzu Ali
• Laiba Aimen

Project Type

MS Data Science Semester Project
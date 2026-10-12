# Breast Cancer Diagnostic Classification

## Project Overview
This project builds and evaluates a machine learning pipeline to classify breast cancer tumors as either malignant or benign. Using the Breast Cancer Wisconsin (Diagnostic) dataset, the objective was to establish a baseline predictive model and subsequently improve its recall to minimize false negatives—a critical requirement in medical diagnostics.

## 1. Exploratory Data Analysis (EDA)
The dataset contains 569 instances, comprising 357 benign and 212 malignant cases. To understand the data structure:
* **Count Plot:** Visualized the target variable to confirm the slight class imbalance.
* **Box Plots:** Analyzed the relationship between continuous features and the diagnosis. Visualizations revealed that geometric features representing the mean measurements of the cell nuclei, specifically `radius1` and `area1`, showed distinct separation in median values between classes, indicating strong predictive potential.

## 2. Baseline Model Performance
A baseline `LogisticRegression` model was trained using an 80/20 train-test split (455 training instances, 114 testing instances). 
* **Accuracy:** 95.61%
* **Precision:** 97.50%
* **Recall:** 90.70%
* **F1 Score:** 93.98%

**Limitations:** While the overall accuracy was high, a recall of 90.70% indicates that the model missed nearly 10% of actual malignant tumors. In a clinical setting, false negatives carry severe consequences, meaning recall must be prioritized for improvement.

## 3. Model Optimization & Pipeline
To improve performance, I transitioned from a rigid Logistic Regression algorithm to a non-linear `RandomForestClassifier`. I constructed a Scikit-Learn `Pipeline` with the following steps:
1. **StandardScaler:** Scaled continuous features to ensure uniform weight during feature selection.
2. **SelectKBest:** Combated multicollinearity by isolating the top 15 most distinct features, reducing noise from redundant geometric variables.
3. **GridSearchCV:** Performed hyperparameter tuning optimized specifically for `scoring='recall'`.

## 4. Final Results
The grid search identified the optimal hyperparameters as `max_depth: 5`, `n_estimators: 200`, and `class_weight: 'balanced'`. Forcing the model to penalize errors on the minority malignant class yielded a significant improvement:
* **Accuracy:** 95.61%
* **Precision:** 95.24%
* **Recall:** 93.02%
* **F1 Score:** 94.12%

**Conclusion:** The optimized Random Forest pipeline successfully increased the model's recall to 93.02% without sacrificing the strong overall accuracy of 95.61%. By prioritizing feature selection and balanced class weights, the final model is vastly better suited for medical diagnostic screening where minimizing false negatives is paramount.

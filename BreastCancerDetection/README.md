# Breast Cancer Detection using Decision Tree Classifier

## Overview

This project utilizes a **Decision Tree Classifier** to classify breast cancer tumors as either benign or malignant. The model is built using the **WDBC dataset** and evaluates its performance based on metrics like **accuracy, precision, recall**, and **confusion matrix**. Feature selection techniques were applied to enhance the model's performance.

## Dataset

The dataset used is the **WDBC (Wisconsin Diagnostic Breast Cancer)** dataset, which contains 569 samples, each with 30 features. The features represent various characteristics of cell nuclei present in the breast mass, while the target variable labels the tumor as either:

- **Benign (B)** – non-cancerous
- **Malignant (M)** – cancerous

## Workflow

1. **Data Preprocessing**:
   - Importing and loading the dataset.
   - Handling missing values (if necessary).
   - Splitting the dataset into training and testing sets.
   - Scaling features using **StandardScaler** for normalization.

2. **Model Building**:
   - Initial Decision Tree model is trained on all features.
   - Feature selection is applied to retain only important features based on their importance score.

3. **Model Evaluation**:
   - Model evaluation using metrics such as **accuracy, precision, recall**, and **F1-score**.
   - Confusion matrix and classification report are generated to visualize model performance.
   - **Cross-validation** is performed to ensure robustness.
   - **Hyperparameter tuning** using **GridSearchCV** to optimize model performance.

4. **Visualization**:
   - **Feature importance** is visualized using a bar chart to show the most influential features.
   - **Confusion Matrix** and **Classification Report** are plotted for visual interpretation of model results.

## Key Features

- **Feature Selection**: Features with low importance scores are discarded to improve model efficiency.
- **Hyperparameter Optimization**: The model undergoes grid search for hyperparameter tuning to achieve the best configuration.
- **Evaluation Metrics**: Includes accuracy, precision, recall, F1-score, and confusion matrix.
- **Visualization**: Provides feature importance visualization and confusion matrix.

## Evaluation Metrics

- **Accuracy**: Measures the overall correct predictions.
- **Precision**: The proportion of positive predictions that are actually correct.
- **Recall**: The proportion of actual positive cases that are correctly identified.
- **F1-Score**: The balance between precision and recall.




from evaluation import evaluate_model, plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

# Import dataset and preprocessing
dataset = pd.read_csv('./data/wdbc.data', header=None)
X = dataset.iloc[:, 2:-1].values
feature_names = dataset.columns[2:-1].tolist()
y = np.where(dataset.iloc[:, 1].values == 'M', 1, 0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train the Decision Tree model
classifier1 = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier1.fit(X_train, y_train)
y_pred = classifier1.predict(X_test)

# Evaluate the model
plot_confusion_matrix(y_test, y_pred, "Initial_Model_Confusion_Matrix")
evaluate_model(classifier1, X_test, y_test, feature_names=feature_names, model_name="Initial_Model")

# Filter important features on the entire dataset (X)
important_idx = [i for i, importance in enumerate(classifier1.feature_importances_) if importance > 0.005]
X_important = X[:, important_idx]  # Filtered feature matrix
important_feature_names = [feature_names[i] for i in important_idx]

# Split the filtered dataset
X_train_filtered, X_test_filtered, y_train, y_test = train_test_split(X_important, y, test_size=0.2, random_state=42)

# Re-train with important features
classifier2 = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier2.fit(X_train_filtered, y_train)
y_filtered_pred = classifier2.predict(X_test_filtered)

# Evaluate the updated model
plot_confusion_matrix(y_test, y_filtered_pred, "Filtered_Model_Confusion_Matrix")
evaluate_model(classifier2, X_test_filtered, y_test, feature_names=important_feature_names, model_name="Filtered_Model")

# Cross-validation and hyperparameter tuning
scores = cross_val_score(classifier2, X_important, y, cv=5)
print(f"Cross-Validation Accuracy: {scores.mean():.4f}")

param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
}
grid_search = GridSearchCV(classifier2, param_grid, cv=5)
grid_search.fit(X_important, y)
print("Best Parameters:", grid_search.best_params_)
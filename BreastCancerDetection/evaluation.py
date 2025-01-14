import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Confusion Matrix plot function
def plot_confusion_matrix(y_true, y_pred, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["B", "M"], yticklabels=["B", "M"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(filename)  # Save the confusion matrix plot
    plt.close()  # Close the plot to avoid it being shown again

def evaluate_model(model, X_test, y_test, feature_names=None, output_file="evaluation_output.txt", model_name="model"):
    """
    Evaluate the model and display/save results.

    Parameters:
    - model: Trained classifier.
    - X_test: Test features.
    - y_test: Test targets.
    - feature_names: List of feature names (for feature importance visualization).
    - output_file: Path to the file where results will be saved.
    - model_name: A string identifier for the model (used to differentiate outputs).
    """
    # Update filenames with model_name
    output_file = f"{model_name}_{output_file}"
    plot_file = f"{model_name}_feature_importance_plot.png"

    with open(output_file, "w") as f:
        # Predictions
        y_pred = model.predict(X_test)

        # Classification report
        report = classification_report(y_test, y_pred)
        f.write("Classification Report:\n")
        f.write(report + "\n\n")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")

        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        f.write(f"Accuracy: {accuracy:.4f}\n\n")

        # Feature importance
        if feature_names is not None and hasattr(model, "feature_importances_"):
            feature_importances = model.feature_importances_
            f.write("Feature Importances:\n")
            f.write("Index | Feature Name       | Importance\n")
            f.write("---------------------------------------\n")
            sorted_idx = np.argsort(feature_importances)[::-1]
            for idx in sorted_idx:
                f.write(f"{idx:5} | {feature_names[idx]:<18} | {feature_importances[idx]:.4f}\n")

            # Visualization (save to file)
            plt.figure(figsize=(10, 6))
            sorted_features = [feature_names[i] for i in sorted_idx]
            plt.barh(sorted_features, feature_importances[sorted_idx], color='skyblue')
            plt.xlabel("Feature Importance")
            plt.ylabel("Features")
            plt.title(f"Feature Importance in {model_name}")
            plt.tight_layout()
            plt.savefig(plot_file)  # Save plot as an image

    print(f"Evaluation results saved to {output_file} and feature importance plot to {plot_file}.")

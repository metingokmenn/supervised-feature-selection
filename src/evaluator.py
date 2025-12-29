import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

class Evaluator:
    """
    Handles evaluation on the test set, result logging, and visualization.
    """

    def __init__(self, X_test, y_test, output_dir='results'):
        """
        Args:
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): Test labels.
            output_dir (str): Directory to save results.
        """
        self.X_test = X_test
        self.y_test = y_test
        self.output_dir = output_dir
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def evaluate(self, model, scaler, selected_features, method_name, training_stats):
        """
        Evaluates the model on the test set and returns a summary dictionary.

        Args:
            model: Trained Logistic Regression model.
            scaler: Fitted StandardScaler.
            selected_features (list): List of features used.
            method_name (str): Name of the method (e.g., 'Wrapper').
            training_stats (dict): Stats from the training phase (CV scores, time).

        Returns:
            dict: Full performance metrics.
            array: Predicted labels.
        """
        # Prepare test data
        X_test_sub = self.X_test[selected_features]
        X_test_scaled = scaler.transform(X_test_sub)

        # Predict
        y_pred = model.predict(X_test_scaled)

        # Calculate metrics
        test_acc = accuracy_score(self.y_test, y_pred)
        test_f1 = f1_score(self.y_test, y_pred)

        results = {
            "Method": method_name,
            "Feature Count": len(selected_features),
            "CV Accuracy": training_stats['cv_accuracy'],
            "CV F1": training_stats['cv_f1'],
            "Test Accuracy": test_acc,
            "Test F1-Score": test_f1,
            "Training Time (s)": training_stats['train_time']
        }
        
        return results, y_pred

    def save_results(self, results_list, feature_sets):
        """
        Saves the performance table and selected features list to files.
        """
        # Save Table
        df = pd.DataFrame(results_list)
        csv_path = os.path.join(self.output_dir, "performance_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"[Report] Metrics table saved to: {csv_path}")

        # Save Features
        txt_path = os.path.join(self.output_dir, "selected_features.txt")
        with open(txt_path, 'w') as f:
            for method, feats in feature_sets.items():
                f.write(f"=== {method} ({len(feats)} Features) ===\n")
                f.write(", ".join(feats))
                f.write("\n\n")
        print(f"[Report] Feature lists saved to: {txt_path}")
        return df

    def plot_confusion_matrix(self, y_pred, method_name):
        """
        Plots and saves the confusion matrix for a specific method.
        """
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix: {method_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        file_path = os.path.join(self.output_dir, f"confusion_matrix_{method_name}.png")
        plt.savefig(file_path)
        plt.close()
        print(f"[Report] Confusion Matrix saved to: {file_path}")
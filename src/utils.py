import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

class ReportGenerator:
    """
    A class to handle the generation and saving of reports, including metric tables,
    feature lists, and confusion matrix plots.
    """
    def __init__(self, output_dir='results'):
        """
        Initialize the ReportGenerator.

        Args:
            output_dir (str): The directory where results (reports, plots, tables) will be saved.
                              Defaults to 'results'.
        """
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def save_metrics_table(self, results_list):
        """
        Save the performance metrics of different methods to a CSV file.

        Args:
            results_list (list of dict): A list of dictionaries, where each dictionary contains
                                         performance metrics for a specific method/experiment.

        Returns:
            pd.DataFrame: The DataFrame created from the results list.
        """
        df = pd.DataFrame(results_list)
        cols = ["Method", "Feature Count", "Test Accuracy", "Test F1-Score", "Training Time (s)"]
        file_path = os.path.join(self.output_dir, "performance_metrics.csv")
        df[cols].to_csv(file_path, index=False)
        print(f"[Report] Performance metrics saved to: {file_path}")
        return df

    def save_feature_lists(self, feature_dict):
        """
        Save the selected features for each method into a text file.

        Args:
            feature_dict (dict): A dictionary where keys are method names and values are lists
                                 of selected feature names.
        """
        file_path = os.path.join(self.output_dir, "selected_features.txt")
        with open(file_path, 'w') as f:
            for method, features in feature_dict.items():
                f.write(f"=== {method} ({len(features)} Features) ===\n")
                f.write(", ".join(features))
                f.write("\n\n")
        print(f"[Report] Selected features saved to: {file_path}")

    def plot_confusion_matrix(self, y_true, y_pred, method_name):
        """
        Generate and save a confusion matrix heatmap for a specific method.

        Args:
            y_true (array-like): True labels.
            y_pred (array-like): Predicted labels.
            method_name (str): The name of the method to label the plot and filename.
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix: {method_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        file_path = os.path.join(self.output_dir, f"confusion_matrix_{method_name}.png")
        plt.savefig(file_path)
        plt.close()
        print(f"[Report] Confusion matrix saved to: {file_path}")
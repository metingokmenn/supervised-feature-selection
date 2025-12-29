import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

class ReportGenerator:
    def __init__(self, output_dir='results'):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def save_metrics_table(self, results_list):
        df = pd.DataFrame(results_list)
        # Sütun sırasını düzenle
        cols = ["Yöntem", "Özellik Sayısı", "Test Accuracy", "Test F1-Score", "Eğitim Süresi (sn)"]
        # Eğer tabloda CV sonuçları da istenirse eklenebilir
        file_path = os.path.join(self.output_dir, "performance_metrics.csv")
        df[cols].to_csv(file_path, index=False)
        print(f"[Rapor] Performans tablosu kaydedildi: {file_path}")
        return df

    def save_feature_lists(self, feature_dict):
        file_path = os.path.join(self.output_dir, "selected_features.txt")
        with open(file_path, 'w') as f:
            for method, features in feature_dict.items():
                f.write(f"=== {method} ({len(features)} Features) ===\n")
                f.write(", ".join(features))
                f.write("\n\n")
        print(f"[Rapor] Seçilen özellikler kaydedildi: {file_path}")

    def plot_confusion_matrix(self, y_true, y_pred, method_name):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix: {method_name}')
        plt.xlabel('Tahmin')
        plt.ylabel('Gerçek')
        file_path = os.path.join(self.output_dir, f"confusion_matrix_{method_name}.png")
        plt.savefig(file_path)
        plt.close()
        print(f"[Rapor] Karışıklık matrisi kaydedildi: {file_path}")
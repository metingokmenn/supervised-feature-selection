import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataHandler:
    """
    Handles data loading, cleaning, preprocessing, and splitting for the Online News Popularity dataset.
    """

    def __init__(self, file_path):
        """
        Initializes the DataHandler with the file path.

        Args:
            file_path (str): Path to the CSV dataset file.
        """
        self.file_path = file_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_and_preprocess(self):
        """
        Loads the dataset, removes non-predictive features, generates labels,
        and splits the data into training and testing sets.

        Steps:
        1. Drop 'url' and 'timedelta'.
        2. Create binary labels: 1 if shares >= 1400, else 0.
        3. Split into 80% Train and 20% Test.

        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print(f"[Info] Loading data from: {self.file_path}")
        self.df = pd.read_csv(self.file_path)
        
        # Strip whitespace from column names just in case
        self.df.columns = self.df.columns.str.strip()

        # Step 1a: Remove non-predictive features
        drop_cols = ['url', 'timedelta']
        self.df = self.df.drop(columns=[c for c in drop_cols if c in self.df.columns], errors='ignore')

        # Step 1b: Label generation based on median (1400)
        self.df['label'] = (self.df['shares'] >= 1400).astype(int)
        
        y = self.df['label']
        X = self.df.drop(columns=['shares', 'label'])

        # Step 1c: Split data (80% Train, 20% Test)
        print("[Info] Splitting data into 80% Training and 20% Testing sets...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        
        return self.X_train, self.X_test, self.y_train, self.y_test
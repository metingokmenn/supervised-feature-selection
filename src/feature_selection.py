import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class FeatureSelector:
    """
    Implements Filter, Wrapper, and Embedded feature selection methods.
    """

    def __init__(self, X_train, y_train):
        """
        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
        """
        self.X_train = X_train
        self.y_train = y_train
        
        # Scaling is often required for linear models used in selection
        self.scaler = StandardScaler()
        self.X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.X_train.columns,
            index=self.X_train.index
        )

    def filter_method(self, n_features=15, correlation_threshold=0.9):
        """
        Applies the Filter Method using Pearson Correlation.
        
        1. Removes features that are highly correlated with each other.
        2. Selects top 'n_features' based on correlation with the target.

        Args:
            n_features (int): Number of features to select.
            correlation_threshold (float): Threshold for removing collinear features.

        Returns:
            list: List of selected feature names.
        """
        print("[Process] Running Filter Method (Pearson Correlation)...")
        
        # 1. Feature-Feature Correlation
        corr_matrix = self.X_train.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
        X_reduced = self.X_train.drop(columns=to_drop)
        
        # 2. Feature-Target Correlation
        temp_df = X_reduced.copy()
        temp_df['target'] = self.y_train
        
        correlations = temp_df.corr()['target'].abs().sort_values(ascending=False)
        # Drop target itself and take top n
        selected_features = correlations.drop('target').head(n_features).index.tolist()
        
        return selected_features

    def wrapper_method(self, n_features=15):
        """
        Applies the Wrapper Method using Recursive Feature Elimination (RFE)
        with Logistic Regression.

        Args:
            n_features (int): Number of features to select.

        Returns:
            list: List of selected feature names.
        """
        print("[Process] Running Wrapper Method (RFE with Logistic Regression)...")
        model = LogisticRegression(solver='liblinear', max_iter=1000)
        rfe = RFE(estimator=model, n_features_to_select=n_features)
        rfe.fit(self.X_train_scaled, self.y_train)
        
        selected_features = self.X_train.columns[rfe.support_].tolist()
        return selected_features

    def embedded_method(self, n_features=15):
        """
        Applies the Embedded Method using Random Forest Feature Importance.

        Args:
            n_features (int): Number of features to select.

        Returns:
            list: List of selected feature names.
        """
        print("[Process] Running Embedded Method (Random Forest)...")
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(self.X_train, self.y_train)
        
        importances = pd.Series(model.feature_importances_, index=self.X_train.columns)
        selected_features = importances.nlargest(n_features).index.tolist()
        return selected_features
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler

class ModelTrainer:
    """
    Handles model training and Cross-Validation procedures.
    """

    def __init__(self, X_train, y_train):
        """
        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
        """
        self.X_train = X_train
        self.y_train = y_train

    def train_and_validate(self, selected_features):
        """
        Trains a Logistic Regression model using selected features and performs 5-Fold CV.
        
        Args:
            selected_features (list): The list of feature names to use.

        Returns:
            dict: Dictionary containing the model, training time, and CV scores.
        """
        # Subset data
        X_train_sub = self.X_train[selected_features]

        # Scaling (Essential for Logistic Regression)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sub)

        # Model: Logistic Regression (L2 Regularization is default in sklearn)
        model = LogisticRegression(solver='liblinear', max_iter=1000, random_state=42)

        # 5-Fold Cross Validation
        start_time = time.time()
        cv_results = cross_validate(model, X_train_scaled, self.y_train, cv=5, scoring=['accuracy', 'f1'])
        train_time = time.time() - start_time

        # Final fit on whole training data for testing later
        model.fit(X_train_scaled, self.y_train)

        return {
            "model": model,
            "scaler": scaler,
            "cv_accuracy": cv_results['test_accuracy'].mean(),
            "cv_f1": cv_results['test_f1'].mean(),
            "train_time": train_time
        }
# Supervised Feature Selection: Online News Popularity

This project investigates various supervised feature selection techniques to improve the performance of a Logistic Regression model in predicting the popularity of online news articles. Using the **Online News Popularity Dataset**, we compare Filter, Wrapper, and Embedded methods to identify the most predictive features.

## Project Overview

The primary goal is to determine if reducing the dimensionality of the dataset through targeted feature selection can maintain or enhance predictive accuracy while reducing model complexity.

The project follows these steps:
1.  **Data Preprocessing**: Cleaning the dataset and converting the target variable (`shares`) into a binary classification task (Popular vs. Unpopular).
2.  **Feature Selection**: Implementing three different strategies:
    *   **Filter Method**: Pearson Correlation (removing collinear features and selecting top features based on target correlation).
    *   **Wrapper Method**: Recursive Feature Elimination (RFE) using Logistic Regression.
    *   **Embedded Method**: Feature Importance derived from a Random Forest Classifier.
3.  **Optimization**: Identifying the best selection method and searching for the optimal number of features ($k$).
4.  **Evaluation**: Comparing models using Accuracy, F1-Score, and Confusion Matrices on a held-out test set.

## Project Structure

```text
supervised-feature-selection/
├── dataset/
│   └── OnlineNewsPopularity.csv  # The dataset file
├── src/
│   ├── data_handler.py           # Data loading and preprocessing logic
│   ├── feature_selection.py      # Implementation of Filter, Wrapper, and Embedded methods
│   ├── trainer.py                # Model training and cross-validation
│   ├── evaluator.py              # Performance evaluation and visualization
│   └── utils.py                  # Helper functions
├── results/                      # Output directory for metrics and plots
├── main.py                       # Main entry point for the experiment
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```

### Script Descriptions (`src/`)

Each script in the `src/` directory handles a specific part of the machine learning pipeline:

*   **`data_handler.py`**: 
    *   Responsible for loading the raw CSV data.
    *   Cleans column names and removes non-predictive attributes (like `url` and `timedelta`).
    *   Performs label binarization on the `shares` column (threshold of 1400).
    *   Splits the data into stratified training (80%) and testing (20%) sets.

*   **`feature_selection.py`**: 
    *   Contains the `FeatureSelector` class which implements three core strategies:
        *   **Filter Method**: Uses Pearson Correlation to drop redundant (collinear) features and select the most relevant ones.
        *   **Wrapper Method**: Utilizes Recursive Feature Elimination (RFE) with a Logistic Regression estimator.
        *   **Embedded Method**: Leverages Random Forest feature importance to rank and select features.

*   **`trainer.py`**: 
    *   Handles the training lifecycle using the `ModelTrainer` class.
    *   Implements 5-Fold Cross-Validation to assess model stability and performance during the training phase.
    *   Manages feature scaling (Standardization) which is critical for the Logistic Regression model.

*   **`evaluator.py`**: 
    *   Focuses on testing the final models on held-out data.
    *   Calculates key classification metrics: Accuracy and F1-Score.
    *   Provides utilities for saving results and generating performance reports.

*   **`utils.py`**: 
    *   Contains auxiliary functions for reporting and visualization.
    *   `ReportGenerator` class saves the performance metrics to CSV and writes the list of selected features to a text file.
    *   Includes logic for plotting and saving Confusion Matrix heatmaps using `seaborn` and `matplotlib`.

## Getting Started

### Prerequisites

*   Python 3.8+
*   Pip (Python package installer)

### Installation

1.  **Clone the repository** (or download the source code).
2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    ```
3.  **Activate the virtual environment**:
    *   **Windows**: `venv\Scripts\activate`
    *   **macOS/Linux**: `source venv/bin/activate`
4.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Project

To execute the full experiment, run the `main.py` script:

```bash
python main.py
```

The script will:
*   Load and preprocess the data.
*   Run the initial comparison between All Features, Filter, Wrapper, and Embedded methods.
*   Perform an optimization loop to find the best feature count for the top-performing method.
*   Save all metrics and visualizations to the `results/` directory.

## Seeing Results

Once the execution is complete, you can find the output in the `results/` folder:

1.  **`performance_metrics.csv`**: A detailed table comparing Accuracy, F1-Score, and training time across all scenarios.
2.  **`selected_features.txt`**: A text file listing the specific features chosen by each method.
3.  **`confusion_matrix_{Method_Name}.png`**: A visualization of the model's performance on the test set for the optimized model.

The `main.py` console output will also display a summary table of the final results.


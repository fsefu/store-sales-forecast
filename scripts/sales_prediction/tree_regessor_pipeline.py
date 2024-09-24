import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class TreeBasedRegressorPipeline:
    def __init__(self, model_type='random_forest', random_state=42):
        """
        Initialize the TreeBasedRegressorPipeline class.
        
        Args:
        - model_type (str): Type of tree-based model to use ('random_forest').
        - random_state (int): Random state for reproducibility.
        """
        self.model_type = model_type
        self.random_state = random_state
        self.pipeline = None
        self.model = None

    def build_pipeline(self):
        """
        Build the sklearn pipeline for the regression model.
        
        Returns:
        - pipeline (Pipeline): sklearn pipeline with preprocessing and model steps.
        """
        # Step 1: Data Preprocessing
        # SimpleImputer handles missing values, StandardScaler normalizes the data
        preprocessor = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values
            ('scaler', StandardScaler())  # Normalize the features
        ])
        
        # Step 2: Model selection based on the chosen model type
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(random_state=self.random_state)
        else:
            raise ValueError(f"Model type '{self.model_type}' is not supported.")
        
        # Step 3: Create a full pipeline
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', self.model)
        ])
        return self.pipeline

    def train_model(self, X_train, y_train):
        """
        Train the pipeline model using training data.
        
        Args:
        - X_train (array-like): Training feature data.
        - y_train (array-like): Training target data.
        """
        if self.pipeline is None:
            raise ValueError("Pipeline is not built. Call build_pipeline() first.")
        
        self.pipeline.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model on test data and return performance metrics.
        
        Args:
        - X_test (array-like): Test feature data.
        - y_test (array-like): Test target data.
        
        Returns:
        - dict: Evaluation metrics (MAE, RMSE, R2 Score).
        """
        if self.pipeline is None:
            raise ValueError("Pipeline is not trained. Call train_model() first.")
        
        # Make predictions
        y_pred = self.pipeline.predict(X_test)
        
        # Calculate performance metrics
        metrics = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2_Score': r2_score(y_test, y_pred)
        }
        return metrics

    def tune_hyperparameters(self, X_train, y_train, param_grid, cv=5):
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
        - X_train (array-like): Training feature data.
        - y_train (array-like): Training target data.
        - param_grid (dict): Hyperparameter grid for tuning.
        - cv (int): Number of cross-validation folds.
        
        Returns:
        - best_params_: The best hyperparameters found during the search.
        """
        if self.pipeline is None:
            raise ValueError("Pipeline is not built. Call build_pipeline() first.")
        
        # Perform GridSearchCV
        grid_search = GridSearchCV(self.pipeline, param_grid=param_grid, cv=cv, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        
        # Set the pipeline to the best found pipeline
        self.pipeline = grid_search.best_estimator_
        return grid_search.best_params_

    def predict(self, X):
        """
        Predict target values using the trained model.
        
        Args:
        - X (array-like): Feature data to predict on.
        
        Returns:
        - array-like: Predicted target values.
        """
        if self.pipeline is None:
            raise ValueError("Pipeline is not trained. Call train_model() first.")
        
        return self.pipeline.predict(X)

# # Example usage of the class
# if __name__ == "__main__":
#     # Simulated data for demonstration purposes
#     X, y = np.random.rand(100, 5), np.random.rand(100)  # Replace with real data
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Initialize the model pipeline with RandomForestRegressor
#     model_pipeline = TreeBasedRegressorPipeline(model_type='random_forest')
    
#     # Build the pipeline
#     model_pipeline.build_pipeline()
    
#     # Train the model
#     model_pipeline.train_model(X_train, y_train)
    
#     # Evaluate the model
#     metrics = model_pipeline.evaluate_model(X_test, y_test)
#     print(f"Evaluation Metrics: {metrics}")

#     # Hyperparameter tuning
#     param_grid = {
#         'regressor__n_estimators': [50, 100, 200],
#         'regressor__max_depth': [None, 10, 20]
#     }
#     best_params = model_pipeline.tune_hyperparameters(X_train, y_train, param_grid)
#     print(f"Best Hyperparameters: {best_params}")
    
#     # Predictions
#     predictions = model_pipeline.predict(X_test)
#     print(f"Predictions: {predictions[:5]}")

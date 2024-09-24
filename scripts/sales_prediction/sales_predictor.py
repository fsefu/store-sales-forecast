import numpy as np
import tensorflow as tf
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor

class SalesPredictor:
    def __init__(self, model_type='tensorflow', delta=1.0):
        """
        Initialize the SalesPredictor class.
        
        Args:
        - model_type (str): Type of model to use ('tensorflow' or 'sklearn').
        - delta (float): Threshold for the Huber loss function.
        """
        self.model_type = model_type
        self.delta = delta
        self.model = None

    def build_tensorflow_model(self, input_shape):
        """
        Build a TensorFlow/Keras model for regression.
        
        Args:
        - input_shape (tuple): Shape of the input data (number of features,).
        
        Returns:
        - model (tf.keras.Model): Compiled Keras model with Huber loss.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)  # Regression output
        ])
        
        model.compile(optimizer='adam', 
                      loss=tf.keras.losses.Huber(delta=self.delta), 
                      metrics=['mae'])
        
        self.model = model
        return self.model

    def train_tensorflow_model(self, train_data, train_labels, epochs=100, batch_size=32, validation_split=0.2):
        """
        Train the TensorFlow/Keras model.
        
        Args:
        - train_data (array-like): Training data.
        - train_labels (array-like): Target labels for training.
        - epochs (int): Number of epochs to train.
        - batch_size (int): Batch size for training.
        - validation_split (float): Proportion of training data to use for validation.
        
        Returns:
        - history (History): Training history object.
        """
        if self.model is None:
            raise ValueError("TensorFlow model is not built. Call build_tensorflow_model() first.")
        
        history = self.model.fit(train_data, train_labels, 
                                 epochs=epochs, 
                                 batch_size=batch_size, 
                                 validation_split=validation_split)
        return history

    def custom_huber_loss(self, y_true, y_pred):
        """
        Custom Huber loss function for scikit-learn models.
        
        Args:
        - y_true (array-like): True sales values.
        - y_pred (array-like): Predicted sales values.
        
        Returns:
        - float: Huber loss value.
        """
        error = y_true - y_pred
        is_small_error = np.abs(error) <= self.delta
        squared_loss = 0.5 * np.square(error)
        linear_loss = self.delta * (np.abs(error) - 0.5 * self.delta)
        return np.where(is_small_error, squared_loss, linear_loss).mean()

    def build_sklearn_model(self):
        """
        Build a scikit-learn model (e.g., GradientBoostingRegressor).
        
        Returns:
        - model: Trained scikit-learn model.
        """
        self.model = GradientBoostingRegressor()
        return self.model

    def train_sklearn_model(self, train_data, train_labels):
        """
        Train the scikit-learn model.
        
        Args:
        - train_data (array-like): Training data.
        - train_labels (array-like): Target labels for training.
        
        Returns:
        - model: Trained scikit-learn model.
        """
        if self.model is None:
            raise ValueError("Scikit-learn model is not built. Call build_sklearn_model() first.")
        
        self.model.fit(train_data, train_labels)
        return self.model

    def evaluate_sklearn_model(self, data, labels, cv=5):
        """
        Evaluate the scikit-learn model using cross-validation and Huber loss.
        
        Args:
        - data (array-like): Data for evaluation.
        - labels (array-like): True labels.
        - cv (int): Number of cross-validation folds.
        
        Returns:
        - scores: Cross-validation scores.
        """
        if self.model is None:
            raise ValueError("Scikit-learn model is not trained. Call train_sklearn_model() first.")
        
        # Create a custom scorer using the custom Huber loss function
        huber_scorer = make_scorer(self.custom_huber_loss, greater_is_better=False)
        scores = cross_val_score(self.model, data, labels, cv=cv, scoring=huber_scorer)
        return scores


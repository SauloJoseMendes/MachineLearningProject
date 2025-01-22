import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam


class DeepLearning:
    def __init__(self, input_dim, layer_config, activation, optimizer='adam', epochs=100, batch_size=32, random_state=42):
        """
        Initializes a deep learning model with the provided layer configuration and activation function.

        Args:
        - input_dim: The number of features in the input data.
        - layer_config: Tuple or list specifying the number of units in each hidden layer.
        - activation: The activation function for the hidden layers (e.g., 'relu', 'tanh', etc.)
        - optimizer: The optimizer for the training process (default 'adam').
        - epochs: Number of training epochs (default 100).
        - batch_size: Batch size used for training (default 32).
        - random_state: Random seed (default 42).
        """
        self.input_dim = input_dim
        self.layer_config = layer_config
        self.activation = activation
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

        # Build the model
        self.model = self._build_model()

    def _build_model(self):
        """Build the deep learning model."""
        model = Sequential()

        # Add the input layer
        model.add(Input(shape=(self.input_dim,)))

        # Add the hidden layers
        for units in self.layer_config:
            model.add(Dense(units, activation=self.activation))

        # Output layer for binary classification (change activation for multi-class if needed)
        model.add(Dense(1, activation='sigmoid'))

        # Compile the model
        model.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def train(self, X, T):
        """
        Train the model and evaluate it using a test split.

        Args:
        - X: Features (input data).
        - T: Target (labels).

        Returns:
        - SE: Sensitivity (Recall).
        - SP: Specificity.
        """
        # Split the data into training and testing sets
        X_train, X_test, T_train, T_test = train_test_split(X, T, test_size=0.3, random_state=self.random_state)

        # Train the model
        self.model.fit(X_train, T_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0)

        # Predict on the test set
        predictions = (self.model.predict(X_test) > 0.5).astype(int).flatten()  # Convert probabilities to binary labels

        # Confusion matrix
        cm = confusion_matrix(T_test, predictions)
        TN, FP, FN, TP = cm.ravel()

        # Calculate sensitivity and specificity
        SE = TP / (TP + FN)
        SP = TN / (TN + FP)

        # Print results
        print("Results for trained model:")
        print("Sensitivity (SE):", SE)
        print("Specificity (SP):", SP)

        return SE, SP

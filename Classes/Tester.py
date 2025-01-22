import numpy as np
import optuna
import tensorflow as tf
from optuna.visualization import plot_optimization_history, plot_slice
from sklearn.model_selection import cross_val_score, StratifiedKFold

from Classes.models.ConvolutionalNeuralNetwork import ConvolutionalNeuralNetwork
from Classes.models.DecisionTree import DecisionTree
from Classes.models.DeepLearning import DeepLearning
from Classes.models.NeuralNetwork import NeuralNetowrk


class Tester:
    def __init__(self, x, t, model_type, n_trials=50):
        self.X = x
        self.T = t
        self.model_type = model_type
        self.study = optuna.create_study(direction="maximize")
        self.study.optimize(self.objective, n_trials=n_trials)

    def plot(self):
        """Plot the optimization history and slice plot."""
        plot_optimization_history(self.study).show()
        plot_slice(self.study).show()

    def print_info(self):
        """Print the best parameters and best score."""
        print("Best Parameters:", self.study.best_params)
        print("Best Score:", self.study.best_value)

    def create_model(self, params):
        """Create a model based on the model type and parameters."""
        if self.model_type == 'dt':
            return DecisionTree(**params).model
        elif self.model_type == 'nn':
            return NeuralNetowrk(**params).model
        elif self.model_type == 'cnn':
            return ConvolutionalNeuralNetwork(**params).model
        elif self.model_type == 'dl':
            return DeepLearning(**params).model
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def define_params(self, trial):
        """Define hyperparameters for the model based on the model type."""
        if self.model_type == 'dt':
            return {
                "max_depth": trial.suggest_int('max_depth', 2, 10),
                "min_samples_leaf": trial.suggest_int('min_samples_leaf', 1, 10),
                "max_leaf_nodes": trial.suggest_int('max_leaf_nodes', 2, 20)
            }
        elif self.model_type == 'nn':
            n_layers = trial.suggest_int('n_layers', 1, 3)
            layer_config = tuple(
                trial.suggest_int(f'n_units_l{i}', 1, 4, step=1) for i in range(n_layers)
            )
            activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic'])
            return {
                "layer_config": layer_config,
                "activation": activation
            }
        elif self.model_type == 'cnn':
            n_filters_1 = trial.suggest_int('n_filters_1', 32, 128, step=32)
            n_filters_2 = trial.suggest_int('n_filters_2', 16, 64, step=16)
            pool_size_1 = tuple(trial.suggest_int(f'pool_size1{i}', 2, 4, step=2) for i in range(2))
            pool_size_2 = (2, 2)
            return {
                "n_filters_1": n_filters_1,
                "n_filters_2": n_filters_2,
                "pool_size_1": pool_size_1,
                "pool_size_2": pool_size_2
            }
        elif self.model_type == 'dl':
            n_layers = trial.suggest_int('n_layers', 4, 8)
            layer_config = tuple(
                trial.suggest_int(f'n_units_l{i}', 16, 128, step=16) for i in range(n_layers)
            )
            activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'leaky_relu'])
            return {
                "input_dim": self.X.shape[1],
                "layer_config": layer_config,
                "activation": activation
            }
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def eval_model(self, model):
        """Evaluate the model using cross-validation."""
        if self.model_type == 'dt' or self.model_type == 'nn':
            return cross_val_score(model, self.X, self.T, cv=5, scoring='accuracy').mean()
        elif self.model_type == 'cnn' or self.model_type == 'dl':
            with tf.device('/CPU:0'):
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                scores = []

                for train_index, val_index in skf.split(self.X, self.T):
                    if self.model_type == 'cnn':
                        X_train, X_val = self.X[train_index], self.X[val_index]
                        T_train, T_val = self.T[train_index], self.T[val_index]
                    else:
                        X_train, X_val = self.X.iloc[train_index], self.X.iloc[val_index]
                        T_train, T_val = self.T.iloc[train_index], self.T.iloc[val_index]

                    # Train the model
                    model.fit(X_train, T_train, epochs=10, batch_size=32, verbose=0)

                    # Evaluate on the validation set
                    loss, accuracy = model.evaluate(X_val, T_val, verbose=0)
                    scores.append(accuracy)

                # Return the mean accuracy across folds
                return np.mean(scores)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def objective(self, trial):
        """Objective function for Optuna."""
        params = self.define_params(trial)
        model = self.create_model(params)
        return self.eval_model(model)

    @staticmethod
    def read_config(file_path):
        """
        Reads a configuration file and returns its variables as a dictionary.

        :param file_path: Path to the config.init file.
        :return: Dictionary with configuration variables.
        """
        config = {}
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith("#"):  # Ignore empty lines and comments
                    key, value = map(str.strip, line.split('=', 1))
                    config[key] = value
        return config


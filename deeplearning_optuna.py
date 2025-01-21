import optuna
import tensorflow as tf
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Dropout, ReLU
from Data import Data
from sklearn.model_selection import StratifiedKFold, cross_val_score
from optuna.visualization import plot_optimization_history, plot_slice
import numpy as np

def create_model(trial):
    # Suggest hyperparameters
    num_units_1 = trial.suggest_int('num_units_1', 32, 128, step=32)
    num_units_2 = trial.suggest_int('num_units_2', 16, 64, step=16)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    activation_name = trial.suggest_categorical('activation', ['relu', 'leakyrelu'])

    # Define input layer
    numeric_input = Input(shape=(X.shape[1],), name="Numeric_Input")

    # First hidden layer
    x = Dense(num_units_1)(numeric_input)
    if activation_name == 'relu':
        x = ReLU()(x)
    else:
        x = LeakyReLU()(x)
    x = Dropout(dropout_rate)(x)

    # Second hidden layer
    x = Dense(num_units_2)(x)
    if activation_name == 'relu':
        x = ReLU()(x)
    else:
        x = LeakyReLU()(x)
    x = Dropout(dropout_rate)(x)

    # Output layer
    output = Dense(1, activation="sigmoid")(x)

    # Create model
    model = Model(inputs=numeric_input, outputs=output)

    # Compile model with the suggested learning rate
    
    model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])

    return model

def objective(trial, X, T):
    model = create_model(trial)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_index, val_index in skf.split(X, T):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        T_train, T_val = T[train_index], T[val_index]

        # Train the model
        model.fit(X_train, T_train, epochs=10, batch_size=32, verbose=0)

        # Evaluate on the validation set
        loss, accuracy = model.evaluate(X_val, T_val, verbose=0)
        scores.append(accuracy)

    # Return the mean accuracy across folds
    return np.mean(scores)

with tf.device('/CPU:0'):
    dataset = Data(image_feature_path="feature_vectors.csv")
    dataset.drop_feature(["MARITAL STATUS"])
    # Load data
    X, T = dataset.X, dataset.Y  # Use your `dataset` object

    # Optimize using Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, T), n_trials=50)

    # Print the best parameters and their score
    print("Best Parameters:", study.best_params)
    print("Best Score:", study.best_value)
    plot_optimization_history(study).show()
    plot_slice(study).show()
    print(X.shape)
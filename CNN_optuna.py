import optuna
from Data import Data
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from optuna.visualization import plot_optimization_history, plot_slice
import numpy as np

def build_CNN_model(n_filters_1=64, n_filters_2=32, pool_size_1=(2,2), pool_size_2=(2,2)):
        
    # Define the CNN model
    model = tf.keras.models.Sequential()

    # CNN Layer 1 (accepts input shape 21x21, 1 channel)
    model.add(tf.keras.layers.Conv2D(n_filters_1, (3, 3), activation='relu', input_shape=(21, 21, 1))) 
    model.add(tf.keras.layers.MaxPooling2D(pool_size=pool_size_1))

    # CNN Layer 2
    model.add(tf.keras.layers.Conv2D(n_filters_2, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=pool_size_2))

    # Flatten layer to output a feature vector
    model.add(tf.keras.layers.Flatten()) 

    # Fully connected layer (Dense layer)
    model.add(tf.keras.layers.Dense(16, activation='relu'))

    # Fully connected layer (Dense layer)
    model.add(tf.keras.layers.Dense(8, activation='relu'))  
    
    # Output layer with sigmoid activation for binary classification
    model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

    # Compile the model (not for training, but for compatibility)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def objective(trial):  
    n_filters_1 = trial.suggest_int('n_filters_1', 32, 128, step=32)
    n_filters_2 = trial.suggest_int('n_filters_2', 16, 64, step=16)

    # Dynamically suggest pooling layer sizes
    pool_size_1 = tuple(trial.suggest_int(f'pool_size1{i}', 2, 4, step=2) for i in range(2))
    pool_size_2 = (2,2)

    # Build the CNN model with suggested parameters
    model = build_CNN_model(n_filters_1, n_filters_2, pool_size_1, pool_size_2)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_index, val_index in skf.split(X, T):
        X_train, X_val = X[train_index], X[val_index]
        T_train, T_val = T[train_index], T[val_index]

        # Train the model
        model.fit(X_train, T_train, epochs=10, batch_size=32, verbose=0)

        # Evaluate on the validation set
        loss, accuracy = model.evaluate(X_val, T_val, verbose=0)
        scores.append(accuracy)

    # Return the mean accuracy across folds
    return np.mean(scores)


def find_optimal_hyperparameters():
    dataset = Data(image_dataset_path="COVID_IMG.csv")
    # Load data
    X, T = dataset.images, dataset.Y  # Use your `dataset` object

    # Optimize using Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial), n_trials=50)

    # Print the best parameters and their score
    print("Best Parameters:", study.best_params)
    print("Best Score:", study.best_value)
    plot_optimization_history(study).show()
    plot_slice(study).show()



with tf.device('/CPU:0'):
    dataset = Data(image_dataset_path="COVID_IMG.csv")
    X, T = dataset.images, dataset.Y  # Use your `dataset` object
    find_optimal_hyperparameters()
    
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import optuna
from sklearn.metrics import confusion_matrix
from optuna.visualization import plot_optimization_history, plot_slice
from Data import Data

dataset = Data()
X_train = dataset.X
T_train = dataset.Y

def read_data(file_name):
    df = pd.read_csv(file_name, header=None, skiprows=1)
    X = df.iloc[:, :-1].to_numpy()
    T = df.iloc[:, -1].to_numpy()
    return X, T


def plot_decision_tree(dt):
    plt.figure(figsize=(5, 5))
    plot_tree(dt)
    plt.show()


def train_decision_tree(X, T, leaf_nodes, max_depth, min_samples_leaf):
    # Split data into training and test sets
    X_train, X_test, T_train, T_test = train_test_split(X, T, test_size=0.3, random_state=42)

    # Create decision tree classifier with 'entropy' criterion for ID3-like behavior
    dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_leaf_nodes=leaf_nodes)

    # train decision tree using training set
    dt.fit(X_train, T_train)

    # validate the decision tree using test set
    predictions = dt.predict(X_test)
    cm = confusion_matrix(T_test, predictions)
    TN, FP, FN, TP = cm.ravel()
    SE = TP / (TP + FN)
    SP = TN / (TN + FP)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    print("Results for trained model with maximum depth of:", max_depth, "and max leaf nodes:", leaf_nodes)
    print("Se:" + str(SE))
    print("SP:" + str(SP))
    print("Accuracy:" + str(accuracy))

    plot_decision_tree(dt)

    return SE, SP, accuracy


def objective(trial, X, T):
    # Define hyperparameters to tune
    max_depth = trial.suggest_int('max_depth', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 2, 10)

    # Create a model with these parameters
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_leaf_nodes=max_leaf_nodes
    )

    # Evaluate using cross-validation
    score = cross_val_score(model, X, T, cv=5, scoring='accuracy').mean()
    return score


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

# metrics = train_decision_tree(X, T, 11, 8, 10)
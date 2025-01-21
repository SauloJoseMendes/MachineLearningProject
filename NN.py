import optuna
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from Data import Data
from optuna.visualization import plot_optimization_history, plot_slice

#create multilayer neural network, train and test it
def train_neural_network(layer_config=(100,)):
    Xtrain, Xtest, Ttrain, Ttest = train_test_split(X,T,test_size=0.3, random_state=42)

    mlp = MLPClassifier(hidden_layer_sizes=layer_config, activation='relu',solver='adam',max_iter=500)

    mlp.fit(Xtrain,Ttrain)

    #validate the decision tree using test set
    predictions = mlp.predict(Xtest)
    cm = confusion_matrix(Ttest, predictions)
    TN, FP, FN, TP = cm.ravel()
    SE = TP/(TP+FN)
    SP = TN/(TN+FP)

    #print("Results for trained model:")
    #print("Se:" + str(SE))
    #print("SP:" + str(SP))

    return SE, SP
def find_optimal_hyperparameters():
    # Optimize using Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, T), n_trials=50)

    # Print the best parameters and their score
    print("Best Parameters:", study.best_params)
    print("Best Score:", study.best_value)
    plot_optimization_history(study).show()
    plot_slice(study).show()

def objective(trial, X, T):
    # Suggest number of hidden layers (1 to 3 layers)
    n_layers = trial.suggest_int('n_layers', 1, 3)

    # Suggest units per layer separately and create a tuple dynamically
    layer_config = tuple(
        trial.suggest_int(f'n_units_l{i}', 1, 4, step=1) for i in range(n_layers)
    )

    # Suggest the activation function
    activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic'])

    # Define MLP Classifier with suggested parameters
    mlp = MLPClassifier(hidden_layer_sizes=layer_config, 
                        activation=activation, 
                        solver='adam', 
                        max_iter=500, 
                        random_state=42)

    # Evaluate using cross-validation
    score = cross_val_score(mlp, X, T, cv=5, scoring='accuracy').mean()
    return score  # We aim to maximize accuracy

# Load data
dataset = Data(image_feature_path="feature_vectors.csv")
dataset.drop_feature(["MARITAL STATUS"])
X, T = dataset.X, dataset.Y  # Use your `dataset` object
find_optimal_hyperparameters()
print(X.shape)


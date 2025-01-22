from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


class NeuralNetowrk:
    def __init__(self, layer_config, activation, solver='adam', max_iter=500, random_state=42):
        # Define MLP Classifier with suggested parameters
        self.model = MLPClassifier(hidden_layer_sizes=layer_config,
                                   activation=activation,
                                   solver=solver,
                                   max_iter=max_iter,
                                   random_state=random_state)

    def train(self, X, T):
        Xtrain, Xtest, Ttrain, Ttest = train_test_split(X, T, test_size=0.3, random_state=42)

        self.model.fit(Xtrain, Ttrain)

        # validate the decision tree using test set
        predictions = self.model.predict(Xtest)
        cm = confusion_matrix(Ttest, predictions)
        TN, FP, FN, TP = cm.ravel()
        SE = TP / (TP + FN)
        SP = TN / (TN + FP)

        print("Results for trained model:")
        print("SE:" + str(SE))
        print("SP:" + str(SP))
        return SE, SP

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

class DecisionTree:
    def __init__(self, max_depth, min_samples_leaf,max_leaf_nodes):
        # Create a model with these parameters
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_leaf_nodes=max_leaf_nodes
        )
    def train(self, X, T):
        # Split data into training and test sets
        X_train, X_test, T_train, T_test = train_test_split(X, T, test_size=0.3, random_state=42)

        # train decision tree using training set
        self.model.fit(X_train, T_train)

        # validate the decision tree using test set
        predictions = dt.predict(X_test)
        cm = confusion_matrix(T_test, predictions)
        TN, FP, FN, TP = cm.ravel()
        SE = TP / (TP + FN)
        SP = TN / (TN + FP)
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        print("Se:" + str(SE))
        print("SP:" + str(SP))
        print("Accuracy:" + str(accuracy))
        return SE, SP, accuracy
    
    def plot(self):
        plt.figure(figsize=(5, 5))
        plot_tree(self.model)
        plt.show()
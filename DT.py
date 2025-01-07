from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import numpy
from sklearn.metrics import confusion_matrix
import csv

def read_data(file_name):
    df = pd.read_csv(file_name, header=None, skiprows=1)
    X = df.iloc[:, :-1].to_numpy()
    T = df.iloc[:, -1].to_numpy()
    return X, T

def plot_decision_tree(dt):
    plt.figure(figsize=(5,5))
    plot_tree(dt)
    plt.show()


def train_decision_tree(X, T, leaf_nodes):

    # Split data into training and test sets
    X_train, X_test, T_train, T_test = train_test_split(X, T, test_size=0.3, random_state=42)

    # Create decision tree classifier with 'entropy' criterion for ID3-like behavior
    dt = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=5, min_samples_split=4, min_samples_leaf=2, max_features=None, random_state=42, max_leaf_nodes=leaf_nodes)
    
    #train decision tree using training set
    dt.fit(X_train, T_train)

    #validate the decision tree using test set
    predictions = dt.predict(X_test)
    cm = confusion_matrix(T_test, predictions)
    TN, FP, FN, TP = cm.ravel()
    SE = TP/(TP+FN)
    SP = TN/(TN+FP)

    print("Results for trained model:")
    print("Se:" + str(SE))
    print("SP:" + str(SP))

    #plot_decision_tree(dt)

    return SE, SP


if __name__ == "__main__":
    X, T = read_data("COVID_numerics.csv")

    SEs = []
    SPs = []


    #test decision tree for different leaf nodes
    for leaf_nodes in range(6,13):
        
        SE, SP = train_decision_tree(X,T,leaf_nodes)
        SEs.append(SE)
        SPs.append(SP)
        
    rows = [SEs, SPs]

    with open("results/DT_COVID_numerics.csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)

   
    


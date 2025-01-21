from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix
import csv

#read data
def read_data(file_name):
    df = pd.read_csv(file_name, header=None, skiprows=1)
    df = (df - df.min()) / (df.max() - df.min()) #normalize data
    X = df.iloc[:, :-1].to_numpy()
    T = df.iloc[:, -1].to_numpy()
    
    return X, T

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


if __name__ == "__main__":
    X, T = read_data("COVID_numerics.csv")

    configs=[(4,2),(8,4),(16,8),(32,16),(64,32),(128,64)]
    exps = 10

    SEs = []
    SPs = []

    for config in configs:

        SE_sum=0
        SP_sum=0

        for i in range(0,exps):
            SE, SP = train_neural_network(layer_config=config)

            SE_sum = SE_sum + SE
            SP_sum = SP_sum + SP

        SE_average = SE_sum / exps
        SP_average = SP_sum / exps

        SEs.append(SE_average)
        SPs.append(SP_average)

    rows = [SEs, SPs]

    with open("results/NN/NN_COVID_numerics.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)




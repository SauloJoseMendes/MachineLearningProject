import pandas as pd
from sklearn.preprocessing import StandardScaler

class Data:
    def __init__(self, path="COVID_numerics.csv"):
        self.path = path
        self.data = self.get_data()
        self.X_not_normalized = self.data.drop(columns=['TARGET'])
        self.X = StandardScaler().fit_transform(self.X_not_normalized)
        self.Y = self.data['TARGET']

    def get_data(self):
        numeric_data = pd.read_csv("COVID_numerics.csv")
        return numeric_data

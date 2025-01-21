import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class Data:
    def __init__(self, path="COVID_numerics.csv", image_dataset_path=None):
        self.path = path
        self.numeric_data = self.get_numeric_data()
        self.X_not_normalized = self.numeric_data.drop(columns=['TARGET'])
        self.X = StandardScaler().fit_transform(self.X_not_normalized)
        self.Y = self.numeric_data['TARGET']
        if image_dataset_path is not None:
            self.images = self.get_images()


    def get_numeric_data(self):
        numeric_data = pd.read_csv("COVID_numerics.csv")
        return numeric_data
    def drop_feature(self, feature_to_remove):
        self.X_not_normalized = self.X_not_normalized.drop(columns=feature_to_remove)
        self.X = StandardScaler().fit_transform(self.X_not_normalized)
    def get_images(self):
        image_data = np.loadtxt("COVID_IMG.csv", delimiter=',')
        image_matrices = image_data.reshape(-1, 21, 21)
        image_matrices = image_matrices / 255.0  # normalize data
        image_matrices = np.expand_dims(image_matrices, axis=-1)  # add dimensions
        return image_matrices
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataReader:
    def __init__(self, path="../data/COVID_numerics.csv", image_dataset_path=None, image_feature_path=None):
        self.path = path
        self.numeric_data = self.get_numeric_data()
        self.X_not_normalized = self.numeric_data.drop(columns=['TARGET'])
        self.X = pd.DataFrame(StandardScaler().fit_transform(self.X_not_normalized), columns=self.X_not_normalized.columns)
        self.Y = self.numeric_data['TARGET']
        if image_dataset_path is not None:
            self.image_dataset_path = image_dataset_path
            self.images = self.get_images()
        if image_feature_path is not None:
            img_features_not_normalized = self.get_image_features(image_feature_path)
            self.img_features = pd.DataFrame(StandardScaler().fit_transform(img_features_not_normalized), columns=img_features_not_normalized.columns)
            self.X = pd.concat([self.X, self.img_features], axis=1)

    def get_image_features(self, image_feature_path):
        features = pd.read_csv(image_feature_path)
        return features

    def get_numeric_data(self):
        numeric_data = pd.read_csv(self.path)
        return numeric_data

    def drop_feature(self, feature_to_remove):
        self.X = self.X.drop(columns=feature_to_remove)

    def get_images(self):
        image_data = np.loadtxt(self.image_dataset_path, delimiter=',')
        image_matrices = image_data.reshape(-1, 21, 21)
        image_matrices = image_matrices / 255.0  # Normalize data
        image_matrices = np.expand_dims(image_matrices, axis=-1)  # Add dimensions
        return image_matrices

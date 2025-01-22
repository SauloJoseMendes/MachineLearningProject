import pandas as pd
import tensorflow as tf

from Classes.DataReader import DataReader


class ConvolutionalNeuralNetwork:
    def __init__(self, n_filters_1=64, n_filters_2=32, pool_size_1=(2, 2), pool_size_2=(2, 2)):
        """
        Initialize the CNN model with the specified parameters.

        Parameters:
        - input_shape (tuple): Shape of the input data (height, width, channels).
        - n_filters_1 (int): Number of filters for the first convolutional layer.
        - n_filters_2 (int): Number of filters for the second convolutional layer.
        - pool_size_1 (tuple): Pool size for the first max-pooling layer.
        - pool_size_2 (tuple): Pool size for the second max-pooling layer.
        """
        self.model = self.create_model(
            n_filters_1=n_filters_1,
            n_filters_2=n_filters_2,
            pool_size_1=pool_size_1,
            pool_size_2=pool_size_2
        )

    def create_model(self, n_filters_1, n_filters_2, pool_size_1, pool_size_2):
        """
        Create a CNN model.

        Parameters:
        - n_filters_1 (int): Number of filters for the first convolutional layer.
        - n_filters_2 (int): Number of filters for the second convolutional layer.
        - pool_size_1 (tuple): Pool size for the first max-pooling layer.
        - pool_size_2 (tuple): Pool size for the second max-pooling layer.

        Returns:
        - model (tf.keras.models.Sequential): Compiled CNN model.
        """
        # Define the CNN model
        model = tf.keras.models.Sequential()

        # CNN Layer 1 (accepts dynamic input shape)
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
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def extract_feature_vectors(self, data, output_file):
        """"
        Extracts features  from images using CNN
        Parameters:
            - data (np.array): input data with images
            - output_file (str): path to output file with feature vectors
        Returns:
            - features (np.array): feature vectors
        """
        feature_extractor = tf.keras.models.Model(inputs=self.model.inputs, outputs=self.model.layers[
            -4].output)
        feature_vectors = feature_extractor.predict(data)
        feature_vectors_df = pd.DataFrame(feature_vectors)
        feature_vectors_df = feature_vectors_df.loc[:, (feature_vectors_df != 0).any(axis=0)]
        print(feature_vectors_df.shape)
        feature_vectors_df.to_csv(output_file, index=False)
        print(f"Feature vectors saved to {output_file}")
        return feature_vectors


if __name__ == '__main__':
    model = ConvolutionalNeuralNetwork()
    dataset = DataReader(image_dataset_path="../../data/COVID_IMG.csv")
    dataset.drop_feature(["MARITAL STATUS"])
    X, T = dataset.images, dataset.Y
    model.extract_feature_vectors(X, "../../data/feature_vectors.csv")

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

class CNN:
    def __init__(self, n_filters_1=64, n_filters_2=32, pool_size_1=(2, 2), pool_size_2=(2, 2), input_shape=(21, 21, 1)):
        # Initialize model parameters
        self.n_filters_1 = n_filters_1
        self.n_filters_2 = n_filters_2
        self.pool_size_1 = pool_size_1
        self.pool_size_2 = pool_size_2
        self.input_shape = input_shape
        self.model = self.build_model()

    def build_model(self):
        # Define the CNN model
        model = tf.keras.models.Sequential()

        # CNN Layer 1 (accepts input shape 21x21, 1 channel)
        model.add(tf.keras.layers.Conv2D(self.n_filters_1, (3, 3), activation='relu', input_shape=self.input_shape))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=self.pool_size_1))

        # CNN Layer 2
        model.add(tf.keras.layers.Conv2D(self.n_filters_2, (3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=self.pool_size_2))

        # Flatten layer to output a feature vector
        model.add(tf.keras.layers.Flatten())

        # Fully connected layer (Dense layer)
        model.add(tf.keras.layers.Dense(16, activation='relu'))

        # Fully connected layer (Dense layer)
        model.add(tf.keras.layers.Dense(8, activation='relu'))

        # Output layer with sigmoid activation for binary classification
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        # Compile the model (not for training, but for compatibility)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def train(self, processed_data, T, n_epochs=10):
        # Split data into training and validation sets
        X_train, X_test, T_train, T_test = train_test_split(processed_data, T, test_size=0.3, random_state=42)

        # Train the model
        history = self.model.fit(
            X_train, T_train,
            epochs=n_epochs,
            batch_size=50,
        )

        # Evaluate the model's performance on the test set
        predictions = self.model.predict(X_test)
        predictions = (predictions >= 0.5).astype(int)

        # Compute confusion matrix and performance metrics
        cm = confusion_matrix(T_test, predictions)
        TN, FP, FN, TP = cm.ravel()
        SE = TP / (TP + FN)
        SP = TN / (TN + FP)

        print("Results for trained model:")
        print(f"SE: {SE}")
        print(f"SP: {SP}")

        return SE, SP

    def extract_features(self, images):
        """
        This method takes a batch of images as input, passes them through the CNN layers,
        and returns the extracted features from the last convolutional layer.

        Args:
            images: A batch of images to process (should be in the shape [batch_size, height, width, channels]).

        Returns:
            Features: The extracted features from the last convolutional layer.
        """
        # Remove the output layer and create a model that ends at the last convolutional layer
        feature_extractor = tf.keras.models.Model(inputs=self.model.input, outputs=self.model.layers[-4].output)

        # Get the extracted features
        features = feature_extractor.predict(images)
        return features
import optuna
from Data import Data
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from optuna.visualization import plot_optimization_history, plot_slice
import numpy as np
import pandas as pd

def build_CNN_model(n_filters_1=64, n_filters_2=32, pool_size_1=(2,2), pool_size_2=(2,2)):
        
    # Define the CNN model
    model = tf.keras.models.Sequential()

    # CNN Layer 1 (accepts input shape 21x21, 1 channel)
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
    model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

    # Compile the model (not for training, but for compatibility)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# create new model equivalent to our trained model, but without the dense layers
def create_feature_extractor(model):
    # Create a new model up to the flatten layer (we remove the output layer)
    feature_extractor = tf.keras.models.Model(inputs=model.inputs, outputs=model.layers[-4].output)  # assuming the flatten layer is the 3rd-to-last layer
    return feature_extractor

#extract feature vectors 
def extract_feature_vectors(model, data):
    # Use the feature extractor model to get feature vectors for the entire dataset
    feature_extractor = create_feature_extractor(model)
    
    # Get the feature vectors for all instances in the dataset
    feature_vectors = feature_extractor.predict(data)
    
    return feature_vectors

# Save feature vectors into a .csv file
def save_feature_vectors_to_csv(feature_vectors, output_file):
    # Convert feature vectors to a DataFrame
    feature_vectors_df = pd.DataFrame(feature_vectors)
    # Remove columns where all values are 0.0
    feature_vectors_df = feature_vectors_df.loc[:, (feature_vectors_df != 0).any(axis=0)]
    print(feature_vectors_df.shape)
    # Save DataFrame to a .csv file
    feature_vectors_df.to_csv(output_file, index=False)
    print(f"Feature vectors saved to {output_file}")

def create_model(X,T):
    model = build_CNN_model(n_filters_1=112, n_filters_2=48, pool_size_1=(2,2), pool_size_2=(2,2))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_index, val_index in skf.split(X, T):
        X_train, X_val = X[train_index], X[val_index]
        T_train, T_val = T[train_index], T[val_index]

        # Train the model
        model.fit(X_train, T_train, epochs=10, batch_size=32, verbose=0)

        # Evaluate on the validation setcl
        loss, accuracy = model.evaluate(X_val, T_val, verbose=0)
        scores.append(accuracy)

    # Return the mean accuracy across folds
    print(np.mean(scores))
    vectors = extract_feature_vectors(model, X)
    # Specify the output file path
    output_file = "feature_vectors.csv"
    # Save the feature vectors to the CSV
    save_feature_vectors_to_csv(vectors, output_file)

with tf.device('/CPU:0'):
    dataset = Data(image_dataset_path="COVID_IMG.csv")
    X, T = dataset.images, dataset.Y  # Use your `dataset` object
    create_model(X,T)

    
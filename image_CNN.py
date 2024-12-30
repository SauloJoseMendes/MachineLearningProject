import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import csv

def pre_process_data():
    image_data = np.loadtxt("COVID_IMG.csv", delimiter=',')
    image_matrices = image_data.reshape(-1, 21, 21)
    image_matrices = image_matrices / 255.0 #normalize data
    image_matrices = np.expand_dims(image_matrices, axis=-1) #add dimensions

    #get target
    df = pd.read_csv("COVID_numerics.csv")
    T = df.iloc[:, -1].to_numpy()
    #T = np.expand_dims(T, axis=-1)
    #print(T.shape)
    #print(T)

    return image_matrices, T

def build_CNN_model(n_filters_1=64, n_filters_2=32, pool_size_1=(2,2), pool_size_2=(2,2)):
        
    # Define the CNN model
    model = tf.keras.models.Sequential()

    # CNN Layer 1
    model.add(tf.keras.layers.Conv2D(n_filters_1, (3, 3), activation='relu', input_shape=(21, 21, 1)))  # Input shape: 128x128, 1 channel
    model.add(tf.keras.layers.MaxPooling2D(pool_size=pool_size_1))

    # CNN Layer 2
    model.add(tf.keras.layers.Conv2D(n_filters_2, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=pool_size_2))

    # Flatten layer to output a feature vector
    model.add(tf.keras.layers.Flatten())  # This is where you'll get your feature vector

    # Fully connected layer (Dense layer)
    model.add(tf.keras.layers.Dense(16, activation='relu'))

    # Fully connected layer (Dense layer)
    model.add(tf.keras.layers.Dense(8, activation='relu'))  
    
    # Output layer with sigmoid activation for binary classification
    model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

    # Compile the model (not for training, but for compatibility)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model



def train_model(processed_data, T, model, n_epochs):

    # Split data into training and validation sets
    X_train, X_test, T_train, T_test = train_test_split(processed_data, T, test_size=0.3, random_state=42)

    print(X_train.shape)
    print(T_train.shape)

    val_loss, val_accuracy = model.evaluate(X_test, T_test, verbose=0)
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
    
     # Train the model
    history = model.fit(
        X_train, T_train,
        epochs=n_epochs,
        batch_size=50,
    )

    #print evolution of loss (error between model prediction and correct labels)
    #-print(history.history['loss'])

    #print(model.predict(X_test))

    #val_loss, val_accuracy = model.evaluate(X_test, T_test, verbose=0)
    #print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

    predictions = model.predict(X_test)
    predictions = (predictions >= 0.5).astype(int)
    cm = confusion_matrix(T_test, predictions)
    TN, FP, FN, TP = cm.ravel()
    SE = TP/(TP+FN)
    SP = TN/(TN+FP)

    print("TEST:")
    print("Se:" + str(SE))
    print("SP:" + str(SP))

    return SE, SP


# Assuming you have a trained model already
def create_feature_extractor(model):
    # Create a new model up to the flatten layer (we remove the output layer)
    feature_extractor = tf.keras.models.Model(inputs=model.inputs, outputs=model.layers[-4].output)  # assuming the flatten layer is the 3rd-to-last layer
    return feature_extractor

def extract_feature_vectors(model, data):
    # Use the feature extractor model to get feature vectors for the entire dataset
    feature_extractor = create_feature_extractor(model)
    
    # Get the feature vectors for all instances in the dataset
    feature_vectors = feature_extractor.predict(data)
    
    return feature_vectors






if __name__ == "__main__":

    processed_data, T = pre_process_data()

   

    n_filter_1_list = [64,64,64,128]
    n_filter_2_list = [32,16,32,64]
    pool_1_list = [(2,2),(2,2),(4,4),(4,4)]
    file_names = ["64-32-(2x2)-(2x2).csv","64-16-(2x2)-(2x2).csv","64-32-(4x4)-(2x2).csv","128-64-(4x4)-(2x2).csv"]
    
    for param in range(3,4):

        SE_list = []
        SP_list = []
        file_name = file_names[param]
        
        for epochs in range(25,275,25):

            sum_SE = 0
            sum_SP = 0

            for i in range (0,10):

                print("epochs: " + str(epochs))

                model = build_CNN_model(n_filter_1_list[param],n_filter_2_list[param],pool_1_list[param],(2,2))
                SE, SP = train_model(processed_data, T, model, epochs)
                SE = float(SE)
                SP = float(SP)

                sum_SE = sum_SE + SE
                sum_SP = sum_SP + SP

            average_SE = sum_SE / 10
            average_SP = sum_SP / 10

            SE_list.append(average_SE)
            SP_list.append(average_SP)

        rows = [SE_list, SP_list]
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the rows to the CSV file
            writer.writerows(rows)
           
        
    



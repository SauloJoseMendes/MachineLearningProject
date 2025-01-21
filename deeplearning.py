import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import precision_score, recall_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate, Dropout, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import numpy as np
from Data import Data
import matplotlib.pyplot as plt

# Read and preprocess the image dataset
data = Data()
# data.drop_feature(["MARITAL STATUS"])
image_data = np.loadtxt("COVID_IMG.csv", delimiter=',')
X_images = image_data.reshape(-1, 21, 21, 1)  # Reshape to (samples, 21, 21, 1)

print("NaN in X_images:", np.isnan(X_images).sum())
print("Inf in X_images:", np.isinf(X_images).sum())

# Split data into training and testing sets
X_numeric_train, X_numeric_test, X_images_train, X_images_test, y_train, y_test = train_test_split(
    data.X, X_images, data.Y, test_size=0.2, random_state=42
)

print("X_numeric_train shape:", X_numeric_train.shape)
print("X_images_train shape:", X_images_train.shape)
print("y_train shape:", y_train.shape)

print("Class distribution in y_train:", np.unique(y_train, return_counts=True))
print("Class distribution in y_test:", np.unique(y_test, return_counts=True))

print("Mean of scaled X_numeric_train:", np.mean(X_numeric_train, axis=0))
print("Std of scaled X_numeric_train:", np.std(X_numeric_train, axis=0))

# Visualize an image
plt.imshow(X_images_train[0].squeeze(), cmap='gray')
plt.title("Example Image")
plt.show()

# --- Simplified Model ---
# Define the numeric model
numeric_input = Input(shape=(data.X.shape[1],), name="Numeric_Input")


# Define the image model (flatten before a single dense layer)
image_input = Input(shape=(21, 21, 1), name="Image_Input")

x_numeric = Dense(64, activation='relu')(numeric_input)
x_numeric = Dropout(0.3)(x_numeric)
x_image = Conv2D(32, kernel_size=(3, 3), activation='relu')(image_input)
x_image = MaxPooling2D()(x_image)
x_image = Flatten()(x_image)

# Combine the outputs of both models
combined = Concatenate()([x_numeric, x_image])
x = Dense(16, activation="relu")(combined)  # single hidden layer after concatenation
x = Dropout(0.3)(x)  # Added dropout to avoid overfitting
output = Dense(1, activation="sigmoid")(x)  # output layer

# Define the hybrid model
model = Model(inputs=[numeric_input, image_input], outputs=output)

# Define a lower learning rate and apply gradient clipping
optimizer = Adam(learning_rate=0.0001, clipvalue=1.0)

# Compile the model
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

# Summary of the model
# model.summary()

# Train the model
with tf.device('/CPU:0'):
    history = model.fit(
        [X_numeric_train, X_images_train], y_train,
        validation_data=([X_numeric_test, X_images_test], y_test),
        epochs=100,
        batch_size=32,
    )

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate([X_numeric_test, X_images_test], y_test)

    # Get predictions
    y_pred_probs = model.predict([X_numeric_test, X_images_test])  # Predict probabilities
    y_pred = (y_pred_probs > 0.5).astype(int)  # Convert probabilities to binary predictions

    # Calculate precision and recall
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss Curves')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy Curves')
plt.show()

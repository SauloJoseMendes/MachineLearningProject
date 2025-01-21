import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
from Data import Data

print("GPUs available:", tf.config.list_physical_devices('GPU'))

dataset = Data()

# Ensure dataset.X and dataset.Y are NumPy arrays
X = dataset.X.values if hasattr(dataset.X, 'values') else dataset.X
Y = dataset.Y.values if hasattr(dataset.Y, 'values') else dataset.Y

# Split data into training and testing sets
X_numeric_train, X_numeric_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# --- Pure Numeric Model ---
# Define the numeric model
numeric_input = Input(shape=(X.shape[1],), name="Numeric_Input")  # Ensure the correct shape for input
x_numeric = Dense(64)(numeric_input)  # Increased layer size
x_numeric = LeakyReLU()(x_numeric)
x_numeric = Dropout(0.3)(x_numeric)  # Added dropout for regularization
x_numeric = Dense(32)(x_numeric)  # Additional hidden layer
x_numeric = LeakyReLU()(x_numeric)
output = Dense(1, activation="sigmoid")(x_numeric)  # output layer

# Define the model
model = Model(inputs=numeric_input, outputs=output)

# Define a lower learning rate and apply gradient clipping
optimizer = Adam(learning_rate=0.0001, clipvalue=1.0)

# Compile the model
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

# Summary of the model
# model.summary()

# Train the model
with tf.device('/CPU:0'):
    history = model.fit(
        X_numeric_train, y_train,
        validation_data=(X_numeric_test, y_test),
        epochs=100,
        batch_size=32,
    )

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_numeric_test, y_test)

    # Get predictions
    y_pred_probs = model.predict(X_numeric_test)  # Predict probabilities
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

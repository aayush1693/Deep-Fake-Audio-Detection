# Python
import numpy as np
from sklearn.model_selection import train_test_split
from utils.data_loader import load_data
from models.model import create_model
import tensorflow as tf
from datetime import datetime

# Load and preprocess data
data_dir = 'data/ASVspoof2021'
X, y = load_data(data_dir)
X = X[..., np.newaxis]  # Add channel dimension

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
input_shape = X_train.shape[1:]
model = create_model(input_shape)

# Set up TensorBoard callback
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), callbacks=[tensorboard_callback])

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')
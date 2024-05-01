import os
import sys

import numpy as np
from flwr.client import start_numpy_client, NumPyClient
import keras as ks

from utils import load_partition, read_img, get_labels
from flwr.client import ClientApp
from flwr.client.mod import fixedclipping_mod, secaggplus_mod

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

IMG_SIZE = 160

# Unique client identifier
client_id = int(sys.argv[1])  # Assuming client index is provided as an argument

# Load server address and port number from command-line arguments
server_address = "127.0.0.1"  # sys.argv[2]
port_number = "8080"  # int(sys.argv[3])

# Example usage
# python3 client.py 0 SERVER_IP_ADDRESS 8080

model = ks.Sequential([
    ks.layers.Input(shape=(IMG_SIZE, IMG_SIZE)),
    ks.layers.Flatten(),
    ks.layers.Dense(128, activation='relu'),
    ks.layers.Dense(4)
])

model.compile(
    optimizer='adam',
    loss=ks.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

if len(sys.argv) > 1:
    X_train, X_val, y_train, y_val = load_partition(int(sys.argv[1]))
else:
    print("Not enough arguments... expecting python3 client.py PARTITION_NUMBER; where partition number is 0, 1, 2, 3")
    sys.exit()


class FederatedClient(NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        history = model.fit(X_train, y_train, epochs=100, batch_size=32, steps_per_epoch=5, validation_split=0.1)

        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }

        return model.get_weights(), len(X_train), results

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_val, y_val)
        print("****** CLIENT ACCURACY: ", accuracy, " ******")
        # Predict labels for validation data
        y_pred = np.argmax(model.predict(X_val), axis=1)

        # Calculate the number of correct guesses for each label
        correct_guesses = [np.sum((y_pred == i) & (y_val == i)) for i in range(4)]

        print("Correct Guesses for Each Label:", correct_guesses)

        return loss, len(X_val), {"accuracy": accuracy}


def predict_image(image_path, model):
    img = read_img(image_path)  # Use the read_img function from utils.py
    img = np.array(img) / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Adjust shape for the model

    # Predict using the model
    predictions = model.predict(img)
    predicted_label_index = np.argmax(predictions)
    labels = get_labels()  # This should match the labels used during training
    predicted_label = labels[predicted_label_index]
    probability = np.max(predictions)

    print(f"Predicted Label: {predicted_label}")
    print(f"Probability: {probability:.2f}")

    return predicted_label, probability


def main():
    image_path = 'data/Testing/glioma_tumor/image(1).jpg'  # Path to the image file
    predicted_label, probability = predict_image(image_path, model)
    # print(f"Label: {predicted_label}, Probability: {probability}")
    start_numpy_client(server_address=f"{server_address}:{port_number}", client=FederatedClient())


if __name__ == '__main__':
    main()

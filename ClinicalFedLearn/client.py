import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from flwr.client import start_numpy_client, NumPyClient
import keras as ks

from utils import load_partition, load_testing_data, get_labels

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
    ks.layers.Flatten(input_shape=(IMG_SIZE, IMG_SIZE)),
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

# Load testing data
X_test, y_test = load_testing_data()

# Get labels
labels = get_labels()


# Class to handle federated client
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
        return loss, len(X_val), {"accuracy": accuracy}

    def show_test_samples(self):
        # Get test samples
        test_indices = np.random.choice(len(X_test), size=4, replace=False)
        test_images = X_test[test_indices]
        test_labels = y_test[test_indices]

        # Predict labels for test samples
        predicted_labels = model.predict(test_images)
        predicted_labels = np.argmax(predicted_labels, axis=1)

        # Get the predicted probabilities for each class
        predicted_probs = model.predict_proba(test_images)

        # Display test samples with true and predicted labels
        plt.figure(figsize=(10, 10))
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.imshow(test_images[i], cmap='gray')
            plt.title(
                f"True Label: {labels[test_labels[i]]}\nPredicted Label: {labels[predicted_labels[i]]}\nAccuracy: {predicted_probs[i][predicted_labels[i]]:.2f}")
            plt.axis('off')
        plt.show()


# Start the federated client
if __name__ == '__main__':
    client = FederatedClient()
    client.show_test_samples()
    start_numpy_client(server_address=f"{server_address}:{port_number}", client=client)

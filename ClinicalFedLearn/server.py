import flwr as fl
import keras as ks
from flwr.server import ServerConfig

from utils import load_testing_data
import edgeimpulse as ei

# Constants
IMG_SIZE = 160
EI_API_KEY = "ei_14a485f9272075df56cc980aa5288fdf02655badbf746489e05ed7aed4d446cc"
DEPLOYED_MODEL_FILENAME = "modelfile.eim"

# Load testing data
X_test, y_test = load_testing_data()

# Define model architecture
def create_model():
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
    return model

# Define evaluation function
def evaluate(weights):
    model = create_model()
    model.set_weights(weights)
    loss, accuracy = model.evaluate(X_test, y_test)
    print("****** CENTRALIZED ACCURACY: ", accuracy, " ******")
    return loss, accuracy

# Function to profile and deploy model with Edge Impulse
def profile_and_deploy_with_edge_impulse(model):
    try:
        ei.API_KEY = EI_API_KEY

        # Profile the model
        profile = ei.model.profile(model=model, device='linux-x86-64')
        print(profile.summary())

        # Deploy the model to Edge Impulse
        model_output_type = ei.model.output_type.Classification(labels=["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"])
        deploy_bytes = ei.model.deploy(model=model, model_output_type=model_output_type, deploy_target='runner-linux-aarch64')

        # Save the deployed model
        with open(DEPLOYED_MODEL_FILENAME, 'wb') as f:
            f.write(deploy_bytes)

    except Exception as e:
        print(f"Error during Edge Impulse profiling and deployment: {e}")

# Function to upload trained model to Edge Impulse
def upload_trained_model_to_edge_impulse():
    try:
        ei.API_KEY = EI_API_KEY

        # Load the trained model
        with open(DEPLOYED_MODEL_FILENAME, 'rb') as f:
            model_bytes = f.read()

        # Upload the model to Edge Impulse
        ei.model.upload_model(model_bytes)

        print("Trained model uploaded to Edge Impulse for testing.")

    except Exception as e:
        print(f"Error uploading trained model to Edge Impulse: {e}")

# Define federated averaging strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.75,
    min_available_clients=2
)

if __name__ == '__main__':
    # Define server configuration
    config = ServerConfig(num_rounds=10)

    # Create model
    model = create_model()

    # Profile and deploy model with Edge Impulse
    profile_and_deploy_with_edge_impulse(model)

    # Start federated learning server
    fl.server.start_server(server_address="[::]:8080", strategy=strategy, config=config)

    # Wait for federated learning process to complete
    # Once completed, run the EdgeImpulseClient
    upload_trained_model_to_edge_impulse()

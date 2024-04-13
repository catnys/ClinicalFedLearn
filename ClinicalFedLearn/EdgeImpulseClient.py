import edgeimpulse as ei

class EdgeImpulseClient:
    def __init__(self, api_key):
        self.api_key = api_key
        ei.API_KEY = api_key

    def profile_and_deploy_model(self, model, model_output_type, device='linux-x86-64', deploy_target='runner-linux-aarch64'):
        try:
            # Profile the model
            profile = ei.model.profile(model=model, device=device)
            print(profile.summary())

            # Deploy the model to Edge Impulse
            deploy_bytes = ei.model.deploy(model=model, model_output_type=model_output_type, deploy_target=deploy_target)

            # Return the deployed model bytes
            return deploy_bytes

        except Exception as e:
            print(f"Error during Edge Impulse profiling and deployment: {e}")
            return None

    def upload_trained_model(self, model_bytes):
        try:
            # Upload the model to Edge Impulse
            ei.model.upload_model(model_bytes)

            print("Trained model uploaded to Edge Impulse for testing.")

        except Exception as e:
            print(f"Error uploading trained model to Edge Impulse: {e}")

# Example usage:
if __name__ == '__main__':
    # Initialize Edge Impulse client
    edge_impulse_client = EdgeImpulseClient(api_key="YOUR_EDGE_IMPULSE_API_KEY")

    # Load and train your model
    model = create_model()
    # Train your model...

    # Profile and deploy model with Edge Impulse
    model_output_type = ei.model.output_type.Classification(labels=["class1", "class2", "class3"])
    deploy_bytes = edge_impulse_client.profile_and_deploy_model(model=model, model_output_type=model_output_type)

    # Upload trained model to Edge Impulse for testing
    edge_impulse_client.upload_trained_model(deploy_bytes)

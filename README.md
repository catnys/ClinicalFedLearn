
# Federated Learning in the Clinical Environment

## Introduction
This project explores and implements Federated Learning (FL) techniques in the clinical environment, particularly for medical applications where data privacy is crucial. The focus is on using FL to enhance machine learning models that classify MRI images of brains with various tumor types, while keeping patient data decentralized and secure.

## Installation
Our project is developed in Python 3. For required packages, refer to `requirements.txt`.
```bash
pip install -r requirements.txt
```

## Running the Code (Federated Learning)
**Note:** This code requires a Linux environment (either a VM or WSL).

1. Start the central server in a terminal:
   ```bash
   python3 server.py
   ```
2. Open four additional terminals for the clients. In each, start a client using:
   ```bash
   python3 client.py <index>
   ```
   Replace `<index>` with 0, 1, 2, or 3. The federated learning process begins once all clients are connected.

## Machine Learning
The machine learning models are in the `mri_classification.ipynb` Jupyter Notebook, detailing individual model configurations and evaluations.

## Project Structure
- `server.py`: Manages the central server for federated learning, handling connections and aggregations.
- `client.py`: Each client runs this script, processing local data and contributing to model updates.
- `utils.py`: Includes functions for data loading and preprocessing, adapted to handle MRI image formats.
- `mri_classification.ipynb`: Notebook for training and evaluating machine learning models.
- `requirements.txt`: Lists all Python packages needed.

## Usage
- Use `server.py` to launch the federated server.
- Run `client.py` for each federated client.
- `mri_classification.ipynb` contains the code for training and evaluation.

## Enhancements
Recent updates have optimized model architectures and refined data handling, improving both the robustness and accuracy of tumor classifications.

## Notes
Ensure the `./data/` directory contains the necessary MRI images for training and testing.

## References
Coral, G., Hait, J., Isaak, K., & Watkins, A. (n.d.). KyleIsaak/Federated_Medical_Machine_Learning: A project at Simon Fraser University under CMPT 340, focusing on a federated approach to medical machine learning using MRI data.

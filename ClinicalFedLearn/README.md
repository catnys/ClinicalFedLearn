# Federated Learning in the Clinical Environment

## Introduction
This project explores and implements Federated Learning (FL) techniques in the clinical environment. Federated Learning allows for the training of machine learning models while keeping patient data private, making it suitable for medical applications where data privacy is critical. Specifically, this project focuses on using FL to build and improve machine learning models for classifying MRI images of brains with different tumor types.


## Installation
Our project has been coded in python3. See requirements.txt for all the required python packages.
The requirements can be installed with:

```bash
pip install -r requirements.txt
```

## Running the code (Federated Learning)

Note: A Linux environment is necessary to run this code (either a VM or WSL will work)

1) In a terminal, start the central server using:
```bash
python3 server.py
```

2) Next, open up 4 more terminals, one for each of the 4 Clients. Start each client in its own terminal using the commands:
```bash
python3 client.py 0
python3 client.py 1
python3 client.py 2
python3 client.py 3
```
(Replace `0`, `1`, `2`, `3` with the appropriate client indices.)
4. The federated learning process will begin once all four clients are connected.

### Machine Learning
1. All code for individual machine learning models is located in the Jupyter Notebook `mri_classification.ipynb`.

## Project Structure
- `server.py`: Contains the central server code for federated learning.
- `client.py`: Contains the client code for federated learning.
- `utils.py`: Contains utility functions for data loading and preprocessing.
- `mri_classification.ipynb`: Jupyter Notebook containing code for individual machine learning models.
- `requirements.txt`: Text file listing required Python packages.

## Usage
- The `server.py` script starts the central server for federated learning.
- The `client.py` script runs individual clients for federated learning. Run multiple instances of `client.py` with different indices to simulate multiple clients.
- The `mri_classification.ipynb` notebook contains code for training and evaluating individual machine learning models.

## Notes
- Ensure that the data directory (`./data/`) contains the necessary MRI images for training and testing.

To get more information about this projectplease check paper written by Coral et. al.  
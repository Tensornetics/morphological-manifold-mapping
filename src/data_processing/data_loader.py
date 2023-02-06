import numpy as np
import torch


def load_data(data_path):
    """Loads data from the specified path and returns a PyTorch tensor"""
    # Read and parse data from the file
    data = np.genfromtxt(data_path, delimiter=',')

    # Convert the data to a PyTorch tensor
    data = torch.from_numpy(data)

    return data


def get_dataloader(data, batch_size, shuffle=True):
    """Returns a PyTorch DataLoader for the provided data"""
    # Convert the data to a PyTorch dataset
    dataset = torch.utils.data.TensorDataset(data)

    # Create a DataLoader for the dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader

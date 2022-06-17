# Numpy is needed for data generation
import numpy as np
# Pytorch is needed for model build
import torch

# TensorDataset is needed to prepare the training data in form of tensors
from torch.utils.data import TensorDataset

# To run the model on either the CPU or GPU (if available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Since torch deals with tensors, we convert the numpy arrays into torch tensors
x_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(targets).float()

# Combine the feature tensor and target tensor into torch dataset
train_data = TensorDataset(x_tensor , y_tensor)

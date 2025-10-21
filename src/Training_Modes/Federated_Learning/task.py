from typing import List, Union
import numpy as np
import torch.nn as nn  
import torch  # Importing torch to resolve the undefined error

def get_weights(model: nn.Module) -> List[np.ndarray]:
    """
    Extracts the weights from a model as a list of NumPy arrays.
    Supports various model types by dynamically handling the input.
    """
    if isinstance(model, nn.Module):
        # PyTorch model
        return [param.cpu().detach().numpy() for param in model.state_dict().values()]
    else:
        raise TypeError(f"Unsupported model type: {type(model)}")

def set_weights(model: nn.Module, weights: List[np.ndarray]) -> None:
    """
    Sets the weights of a model from a list of NumPy arrays.
    Supports various model types by dynamically handling the input.
    """
    if isinstance(model, nn.Module):
        # PyTorch model
        state_dict = model.state_dict()
        for param, weight in zip(state_dict.values(), weights):
            param.copy_(torch.tensor(weight))
    else:
        raise TypeError(f"Unsupported model type: {type(model)}")





from typing import List, Union
import numpy as np
import torch.nn as nn
import torch

def get_weights(model: Union[nn.Module, object]) -> List[np.ndarray]:
    """
    Extract weights as NumPy arrays.
    Supports:
      - PyTorch models (state_dict)
      - scikit-learn models (coef_ + intercept_)
      - custom wrappers like StackedLogisticRegression (with get_params)
    """
    # Case 1: PyTorch model
    if isinstance(model, nn.Module):
        return [param.cpu().detach().numpy() for param in model.state_dict().values()]
    
    # Case 2: Custom wrapper with get_params()
    elif hasattr(model, "get_params") and callable(model.get_params):
        params = model.get_params()
        # Ensure we always return NumPy arrays
        return [np.array(p) for p in params]

    # Case 3: scikit-learn model
    elif hasattr(model, "coef_") and hasattr(model, "intercept_"):
        return [model.coef_, model.intercept_]

    else:
        raise TypeError(f"Unsupported model type: {type(model)}")

def set_weights(model: Union[nn.Module, object], weights: List[np.ndarray]) -> None:
    """
    Sets model weights from a list of NumPy arrays.
    Supports both PyTorch and scikit-learn models.
    """
    # Case 1: PyTorch model
    if isinstance(model, nn.Module):
        state_dict = model.state_dict()
        for param, weight in zip(state_dict.values(), weights):
            param.copy_(torch.tensor(weight))
    
    # Case 2: Custom wrapper with set_params()
    elif hasattr(model, "set_params") and callable(model.set_params):
        model.set_params(weights)
        
    # Case 3: scikit-learn model: All linear models in scikit-learn expose the coef and intercept attributes after fitting
    elif hasattr(model, "coef_") and hasattr(model, "intercept_"):
        model.coef_ = weights[0]
        model.intercept_ = weights[1]

    else:
        raise TypeError(f"Unsupported model type: {type(model)}")

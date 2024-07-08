import torch
import numpy as np


def run_on_gpu(func):
    def wrapper(*args, **kwargs):
        # Move input tensors to GPU
        print("moving to gpu")
        args = [arg.cuda() if isinstance(arg, torch.Tensor)
                else arg for arg in args]
        kwargs = {k: v.cuda() if isinstance(v, torch.Tensor)
                  else v for k, v in kwargs.items()}

        # Run the function
        result = func(*args, **kwargs)

        # Move output tensors back to CPU
        print("moving to cpu")
        if isinstance(result, torch.Tensor):
            print("moving to cpu torch")
            return result.cpu()
        elif isinstance(result, (list, tuple)):
            print("moving to cpu list tuple")
            return type(result)(item.cpu() if isinstance(item, torch.Tensor) else item for item in result)
        elif isinstance(result, dict):
            print("moving to cpu dict")
            return {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in result.items()}
        elif isinstance(result, np.ndarray):
            print("moving to cpu numpy array")
            return torch.tensor(result).cpu().numpy()
        return result

    return wrapper

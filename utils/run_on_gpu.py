"""
RUN ON GPU
==========
Simple decoratry module for running calculations on machine gpu
"""

import torch
import numpy as np


def run_on_gpu(func):
    """Decorator that allows for functions to be ran on the GPU

    :param func: Function you want to run
    :type func: function
    """

    def wrapper(*args, **kwargs):
        # Move input tensors to GPU
        print("Running On GPU")
        args = [
            (
                arg.cuda()
                if isinstance(arg, torch.Tensor)
                else arg
            )
            for arg in args
        ]
        kwargs = {
            k: (
                v.cuda()
                if isinstance(v, torch.Tensor)
                else v
            )
            for k, v in kwargs.items()
        }

        # Run the function
        result = func(*args, **kwargs)

        # Move output tensors back to CPU
        if isinstance(result, torch.Tensor):
            print("Return to CPU")
            return result.cpu()
        elif isinstance(result, (list, tuple)):
            print("Return to CPU")
            return type(result)(
                (
                    item.cpu()
                    if isinstance(
                        item, torch.Tensor
                    )
                    else item
                )
                for item in result
            )
        elif isinstance(result, dict):
            print("Return to CPU")
            return {
                k: (
                    v.cpu()
                    if isinstance(v, torch.Tensor)
                    else v
                )
                for k, v in result.items()
            }
        elif isinstance(result, np.ndarray):
            print("Return to CPU")
            return (
                torch.tensor(result).cpu().numpy()
            )
        return result

    return wrapper

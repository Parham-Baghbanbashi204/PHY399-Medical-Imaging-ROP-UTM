import numpy as np
import torch
from timeit import default_timer as timer
from run_on_gpu import run_on_gpu

# Function to run on CPU


def func(a):
    for i in range(10000000):
        a[i] += 1

# Function optimized to run on GPU


@run_on_gpu
def func2(a):
    for i in range(10000000):
        a += 1
    return a


if __name__ == "__main__":
    n = 10000000
    a = np.ones(n, dtype=np.float64)

    # Run on CPU
    start = timer()
    func(a)
    print("without GPU:", timer() - start)

    # Convert numpy array to PyTorch tensor and move to GPU
    a_gpu = torch.ones(n, dtype=torch.float64)

    # Run on GPU
    print("running on GPU")
    start = timer()
    result_gpu = func2(a_gpu)
    print("with GPU:", timer() - start)

    # Move result back to CPU if needed
    result = result_gpu.cpu().numpy()
    print(result[:10])  # Print first 10 elements to verify the result

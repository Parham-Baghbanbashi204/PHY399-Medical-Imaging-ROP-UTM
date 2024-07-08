import numpy as np
import torch
from timeit import default_timer as timer

# Function to run on CPU


def func(a):
    for i in range(10000000):
        a[i] += 1

# Function optimized to run on GPU


def func2(a):
    a += 1


if __name__ == "__main__":
    n = 10000000
    a = np.ones(n, dtype=np.float64)

    # Run on CPU
    start = timer()
    func(a)
    print("without GPU:", timer() - start)

    # Convert numpy array to PyTorch tensor and move to GPU
    a_gpu = torch.ones(n, dtype=torch.float64, device='cuda')

    # Run on GPU
    start = timer()
    func2(a_gpu)
    print("with GPU:", timer() - start)

    # Move result back to CPU if needed
    result = a_gpu.cpu().numpy()

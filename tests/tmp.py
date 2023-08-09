import sys
sys.path.insert(0,'..')

import time
import timeit
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import cpab
from tqdm import tqdm 
import torch.utils.benchmark as benchmark
from itertools import product


tess_size = 50
backend = "pytorch" # ["pytorch", "numpy"]
device = "cpu" # ["cpu", "gpu"]
zero_boundary = True
use_slow = False
outsize = 200
batch_size = 1
basis = "svd"
epsilon_arr = [0,1,2,3,4]
N_arr = [0,1,2,3,4,5,6,7,8]
lr_arr = [1e-3, 1e-4, 1e-5, 1e-6]

colnames = ["epsilon", "N", "lr", "elapsed_time", "error_mean", "error_min"]
configurations = list(product(epsilon_arr, N_arr, lr_arr))
results = []

for config in configurations:
    epsilon, N, lr = config
    T = cpab.Cpab(tess_size, backend, device, zero_boundary, basis)
    T.params.use_slow = use_slow

    grid = T.uniform_meshgrid(outsize)

    theta_1 = T.identity(batch_size, epsilon=epsilon)
    torch.manual_seed(0)
    theta_1 = T.sample_transformation(batch_size)*epsilon
    grid_t1 = T.transform_grid(grid, theta_1)

    theta_2 = torch.autograd.Variable(T.identity(batch_size, epsilon=0.0), requires_grad=True)

    optimizer = torch.optim.Adam([theta_2], lr=lr)
    # optimizer = torch.optim.SGD([theta_2], lr=lr, momentum=0)
    # for the same step size => faster convergence with increasing N, due to smaller theta norm

    # torch.set_num_threads(1)
    loss_values = []
    maxiter = 1500
    start_time = time.process_time_ns()
    for i in range(maxiter):
        optimizer.zero_grad()
        grid_t2 = T.transform_grid_ss(grid, theta_2, method="closed_form", time=1.0, N=N)
        loss = torch.norm(grid_t2 - grid_t1, dim=1).sum()
        loss.backward()
        optimizer.step()
        loss_values.append(loss.item())
    stop_time = time.process_time_ns()
    error_min = np.min(loss_values)
    error_mean = np.mean(loss_values)
    elapsed_time = (stop_time-start_time) * 1e-6

    results.append([epsilon, N, lr, elapsed_time, error_mean, error_min])

results = pd.DataFrame(results, columns=colnames)
results.to_csv("scaling_squaring_training.csv", index=False)


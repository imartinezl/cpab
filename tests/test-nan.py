# %%
import sys
sys.path.insert(0,'..')

import time
import timeit
import numpy as np
import torch
import matplotlib.pyplot as plt
import cpab
from tqdm import tqdm  
import pickle

tess_size = 50
backend = "pytorch" # ["pytorch", "numpy"]
device = "cpu" # ["cpu", "gpu"]
zero_boundary = True
use_slow = False
outsize = 1000
batch_size = 1
method = "closed_form"

T = cpab.Cpab(tess_size, backend, device, zero_boundary)
T.params.use_slow = use_slow

grid = T.uniform_meshgrid(outsize)
theta = T.sample_transformation(batch_size)*2
theta = T.identity(batch_size, epsilon=1)*1

with open("theta.pkl", "rb") as f:
  theta = pickle.load(f).detach()

grid_t = T.transform_grid(grid, theta)
grad = T.gradient_grid(grid, theta)
plt.plot(grid_t.T)

# for k in range(theta.shape[1]):
# for k in np.arange(160, 163):
#     plt.plot(grad[0,:,k].T)

# plt.figure()
# for k in np.arange(0, 160):
#     plt.plot(grad[0,:,k].T)

# plt.plot(grad[0,:,15].T)
# plt.plot(grad[0,:,164].T)
T.visualize_velocity(theta)
T.visualize_deformgrid(theta)
T.visualize_deformgrid(theta, "numeric")
T.visualize_gradient(theta, n_points=1000)
T.visualize_gradient(theta, "numeric", n_points=1000)

torch.any(torch.isnan(grid_t))


# a = grad[0,:,164]
# a[328].numpy()
# a[319].numpy()
# a[66].numpy()

# %%
# closed form
output = T.backend.cpab_cpu.integrate_closed_form_trace(grid, theta, T.params.B, T.params.xmin, T.params.xmax, T.params.nc)
grad_theta = T.backend.cpab_cpu.derivative_closed_form(grid, theta, T.params.B, T.params.xmin, T.params.xmax, T.params.nc) # [n_batch, n_points, d]


# plt.figure()
# grid_t = output[:,:,0]
# plt.plot(grid_t.T)

plt.figure()
for k in range(theta.shape[1]):
    plt.plot(grad_theta[0,:,k])

# numeric
grid_t = T.backend.cpab_cpu.integrate_numeric(grid, theta, T.params.B, T.params.xmin, T.params.xmax, T.params.nc, T.params.nSteps1, T.params.nSteps2)
h = 1e-2
grad_theta = T.backend.cpab_cpu.derivative_numeric_trace(grid_t, grid, theta, T.params.B, T.params.xmin, T.params.xmax, T.params.nc, T.params.nSteps1, T.params.nSteps2, h)
        
# plt.figure()
# grid_t = output[:,:,0]
# plt.plot(grid_t.T)

plt.figure()
for k in range(theta.shape[1]):
    plt.plot(grad_theta[0,:,k])


# %%

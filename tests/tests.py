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
import seaborn as sns
import pandas as pd

# %% Study diffeomorphic properties

from itertools import product
for tess_size, basis in product([3,4,5,6,7], ["sparse", "rref", "qr", "svd"]):
    # tess_size = 6
    backend = "pytorch" # ["pytorch", "numpy"]
    device = "cpu" # ["cpu", "gpu"]
    zero_boundary = True
    use_slow = False
    outsize = 501
    batch_size = 1
    method = "closed_form"
    # basis = "sparse"
    # basis = "rref"
    # basis = "qr"
    # basis = "svd"

    T = cpab.Cpab(tess_size, backend, device, zero_boundary, basis)
    T.params.use_slow = use_slow

    grid = T.uniform_meshgrid(outsize)
    theta = T.identity(batch_size, epsilon=1)
    # theta = T.sample_transformation(batch_size)
    grid_t = T.transform_grid(grid, theta, method=method)[0]
    grid_t_inv_theta = T.transform_grid(grid, -theta, method=method, time=1.0)[0]
    grid_t_inv_time = T.transform_grid(grid, theta, method=method, time=-1.0)[0]

    grad_t = T.gradient_grid(grid, theta, method=method)[0,:,0]
    grad_t_inv_theta = T.gradient_grid(grid, -theta, method=method, time=1.0)[0,:,0]
    grad_t_inv_time = T.gradient_grid(grid, theta, method=method, time=-1.0)[0,:,0]

    # x => f => y => f-1 => x
    # i = 200
    # x = grid[i].item()
    # y = grid_t[0,i].item()
    # xi = grid_t_inv[0,int(y*outsize)].item()
    # print(i, x, y, xi)

    def funinv(x, y):
        return np.interp(x, y, x)

    plt.subplots(figsize=(6,6))
    plt.plot(grid, grid_t, label=f"$\\phi(x,t,\\theta)$")
    plt.plot(grid, funinv(grid, grid_t), label=f"$\\phi^{{-1}}(x,t,\\theta)$")
    plt.plot(grid, grid_t_inv_theta, label=f"$\\phi(x,t,-\\theta)$")
    plt.plot(grid, grid_t_inv_time, label=f"$\\phi(x,-t,\\theta)$")
    for x in np.linspace(0,1,tess_size+1):
        plt.axvline(x, c="k", ls="--", lw=0.5)
        plt.axhline(x, c="k", ls="--", lw=0.5)
    plt.legend()
    plt.axis("equal")

    fig, axs = plt.subplots(1,3,figsize=(10.5,3.5), sharey=True)
    which = np.linspace(0,len(grid)-1,tess_size+1).astype(int)
    dotsize=10
    axs[0].plot(grid, grid_t, label=f"$\\phi(x,t,\\theta)$", color="black")
    axs[0].scatter(grid[which], grid_t[which], s=dotsize, color="black")
    axs[0].plot(grid, funinv(grid, grid_t), label=f"$\\phi^{{-1}}(x,t,\\theta)$", color="blue", ls="dashed")
    axs[0].scatter(grid[which], funinv(grid, grid_t)[which], s=dotsize, color="blue")

    axs[1].plot(grid, grid_t, label=f"$\\phi(x,t,\\theta)$", color="black")
    axs[1].scatter(grid[which], grid_t[which], s=dotsize, color="black")
    axs[1].plot(grid, grid_t_inv_theta, label=f"$\\phi(x,t,-\\theta)$", color="red", ls="dashed")
    axs[1].scatter(grid[which], grid_t_inv_theta[which], s=dotsize, color="red")

    axs[2].plot(grid, grid_t, label=f"$\\phi(x,t,\\theta)$", color="black")
    axs[2].scatter(grid[which], grid_t[which], s=dotsize, color="black")
    axs[2].plot(grid, grid_t_inv_time, label=f"$\\phi(x,-t,\\theta)$", color="orange", ls="dashed")
    axs[2].scatter(grid[which], grid_t_inv_time[which], s=dotsize, color="orange")

    for i in range(3):
        # for x in np.linspace(0,1,tess_size+1):
        #     axs[i].axvline(x, c="k", ls="--", lw=0.5, alpha=0.5)
        #     axs[i].axhline(x, c="k", ls="--", lw=0.5, alpha=0.5)
        # axs[i].legend(loc=(0.05,0.75))
        axs[i].legend()
        axs[i].axis("equal")
        axs[i].set_xticks(np.linspace(0,1,3))
        axs[i].set_yticks(np.linspace(0,1,3))
        axs[i].set_xlabel(f"$x$")

    axs[0].set_ylabel(f"$\\phi$")

    plt.tight_layout()
    plt.savefig("diffeomorphic_properties_" + basis + "_" + str(tess_size) + ".pdf")
# %%
plt.figure()
plt.plot(grid, grad_t, label=f"$\\partial\\phi(x,t,\\theta)/\\partial\\theta$")
# plt.plot(grid, funinv(grid, grad_t), label=f"$\\partial\\phi^{{-1}}(x,t,\\theta)/\\partial\\theta$")
plt.plot(grid, grad_t_inv_theta, label=f"$\\partial\\phi(x,t,-\\theta)/\\partial\\theta$")
plt.plot(grid, grad_t_inv_time, label=f"$\\partial\\phi(x,-t,\\theta)/\\partial\\theta$")
plt.legend()

# %%
x = np.linspace(0,1,50)
y = x**2
plt.plot(x, y)
plt.plot(x, funinv(x,y))

# %% Study metric properties

def deuclidean(x, y):
    return np.sqrt(np.mean((x-y)**2))

def dcpab(x, y, tess_size, zero_boundary, outsize, nx, ny):
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    # tess_size = 3
    backend = "pytorch" # ["pytorch", "numpy"]
    device = "cpu" # ["cpu", "gpu"]
    # zero_boundary = True
    use_slow = False
    # outsize = 500
    batch_size = 1
    method = "closed_form"
    basis = "sparse"
    basis = "rref"
    basis = "qr"

    T = cpab.Cpab(tess_size, backend, device, zero_boundary, basis)
    T.params.use_slow = use_slow

    grid = T.uniform_meshgrid(outsize)
    theta = T.identity(batch_size, epsilon=0)
    theta = torch.autograd.Variable(theta, requires_grad=True)

    lr = 1e-3
    optimizer = torch.optim.Adam([theta], lr=lr)

    maxiter = 150
    loss_values = []
    for i in range(maxiter):
        optimizer.zero_grad()
        
        N = 4
        xaligned = T.transform_data_ss(x, theta, outsize, N=N)
        loss = torch.norm(xaligned - y, dim=1).mean()
        loss.backward()
        optimizer.step()
        loss_values.append(loss.item())

    x = x.detach().numpy()
    y = y.detach().numpy()
    xaligned = xaligned.detach().numpy()

    # plt.figure()
    # plt.plot(loss_values)

    plt.figure()
    plt.plot(x[0,:,0], label=nx)
    plt.plot(y[0,:,0], label=ny)
    plt.plot(xaligned[0,:,0], label=nx + " aligned")
    plt.xlabel("time")
    plt.ylabel("value")
    plt.axis("off")
    plt.legend()
    plt.tight_layout()
    plt.savefig("metric_" + nx + "_vs_" + ny + ".pdf")

    return deuclidean(xaligned, y)

def assert_metric(d12, d13, d23, d21, d31, d32):
    print(r"$d(x,z) \leq d(x,y) + d(y,z)$", d13 <= d12 + d23, d13, d12 + d23)
    print(r"$d(y,z) \leq d(y,x) + d(x,z)$", d23 <= d21 + d13, d23, d21 + d13)
    print(r"$d(x,y) \leq d(x,z) + d(z,y)$",d12 <= d13 + d32, d12, d13 + d32)

    print(r"$d(z,x) \leq d(z,y) + d(y,x)$", d31 <= d32 + d21, d31, d32 + d21)
    print(r"$d(z,y) \leq d(z,x) + d(x,y)$", d32 <= d31 + d12, d32, d31 + d12)
    print(r"$d(y,x) \leq d(y,z) + d(z,x)$", d21 <= d23 + d31, d21, d23 + d31)


channels = 1
width = 100
a = np.zeros((batch_size, channels))
b = np.ones((batch_size, channels)) * 2 * np.pi
noise = np.random.normal(0, 0.03, (batch_size, width, channels))
x = np.linspace(a, b, width, axis=1)
y1 = 0.5 + np.sin(x) + noise
y2 = 0.5 + np.sin(x-np.pi/6) + noise
y3 = 0.5 + np.sin(x-np.pi/3) + noise

plt.figure()
plt.plot(y1[0,:,0], label="x")
plt.plot(y2[0,:,0], label="y")
plt.plot(y3[0,:,0], label="z")
plt.xlabel("time")
plt.ylabel("value")
plt.axis("off")
plt.legend()
plt.tight_layout()
plt.savefig("metric_xyz.pdf")

deuclidean_12 = deuclidean(y1,y2)
deuclidean_13 = deuclidean(y1,y3)
deuclidean_23 = deuclidean(y2,y3)

deuclidean_21 = deuclidean(y2,y1)
deuclidean_31 = deuclidean(y3,y1)
deuclidean_32 = deuclidean(y3,y2)

print("EUCLIDEAN")
assert_metric(deuclidean_12, deuclidean_13, deuclidean_23, deuclidean_21, deuclidean_31, deuclidean_32)

tess_size = 15
zero_boundary = False
dcpab_12 = dcpab(y1, y2, tess_size, zero_boundary, width, "x", "y")
dcpab_13 = dcpab(y1, y3, tess_size, zero_boundary, width, "x", "z")
dcpab_23 = dcpab(y2, y3, tess_size, zero_boundary, width, "y", "z")

dcpab_21 = dcpab(y2, y1, tess_size, zero_boundary, width, "y", "x")
dcpab_31 = dcpab(y3, y1, tess_size, zero_boundary, width, "z", "x")
dcpab_32 = dcpab(y3, y2, tess_size, zero_boundary, width, "z", "y")

print("CPAB")
assert_metric(dcpab_12, dcpab_13, dcpab_23, dcpab_21, dcpab_31, dcpab_32)




# %% Study gradient wrt x

tess_size = 3
backend = "pytorch" # ["pytorch", "numpy"]
device = "cpu" # ["cpu", "gpu"]
zero_boundary = True
use_slow = False
outsize = 500
batch_size = 1
method = "closed_form"
basis = "sparse"
basis = "rref"
basis = "qr"

T = cpab.Cpab(tess_size, backend, device, zero_boundary, basis)
T.params.use_slow = use_slow

grid = T.uniform_meshgrid(outsize)
theta = T.identity(batch_size, epsilon=1)
# theta = T.sample_transformation(batch_size)
grid_t = T.transform_grid(grid, theta, method=method)
grid_t2 = T.transform_grid(grid_t, theta, method=method)

v = T.calc_velocity(grid, theta)
plt.plot(grid, v.T)

plt.figure()
plt.plot(grid, grid_t.T)
plt.plot(grid, grid_t2.T)

phi = grid_t
phip1 = np.gradient(phi, grid, axis=1)
plt.figure()
plt.plot(grid, phip1.T)

phip2 = np.gradient(phip1, grid, axis=1)
plt.figure()
plt.plot(grid, phip2.T)

np.where(phip2 > 0)[1] # if second derivative is zero, then we have not jump to other cells



# %% Compare integration methods: numeric vs gradient

tess_size = 20
backend = "pytorch" # ["pytorch", "numpy"]
device = "cpu" # ["cpu", "gpu"]
zero_boundary = True
outsize = 200
batch_size = 1
basis = "svd"

T = cpab.Cpab(tess_size, backend, device, zero_boundary, basis)

grid = T.uniform_meshgrid(outsize)

nSteps1_arr = np.arange(1,21)
nSteps2_arr = np.arange(1,21)
reps = 20

from itertools import product

results = np.zeros((len(nSteps2_arr), len(nSteps1_arr), reps))
for nSteps1, nSteps2 in product(nSteps1_arr, nSteps2_arr):
    for i in range(reps):
        print( nSteps1, nSteps2, i)
        # theta = T.identity(batch_size, epsilon=1)
        theta = T.sample_transformation(batch_size)

        grid_t1 = T.transform_grid(grid, theta, method="closed_form")

        T.params.nSteps1 = nSteps1
        T.params.nSteps2 = nSteps2

        grid_t2 = T.transform_grid(grid, theta, method="numeric")

        # error = (grid_t1 - grid_t2).numpy()
        # rms = np.sqrt(np.mean(error**2))

        # error = 100*(grid_t2-grid_t1)/grid_t1
        # rms = np.mean(np.abs(error.numpy()))

        error = (grid_t1 - grid_t2).numpy()
        rms = np.max(np.abs(error))

        results[nSteps2-1, nSteps1-1, i] = rms
        
    #     results[nSteps2-1, nSteps1-1] += rms
    # results[nSteps2-1, nSteps1-1] /= reps
results = np.median(results, axis=2)
results_integration = results
print("DONE")
# %% 
from matplotlib.colors import LogNorm

sns.set_style("whitegrid")
sns.set_context("paper")
fig, ax = plt.subplots(constrained_layout=True, figsize=(5,4))
sns.heatmap(
    results, norm=LogNorm(results.min(),results.max()),
    square=True, xticklabels=nSteps1_arr, yticklabels=nSteps2_arr,
    # cbar_kws = {"label": "RMS Error"}, ax=ax
    cbar_kws = {"label": "Precision"}, ax=ax, cmap="inferno"
)
ax.set_xlabel("$N_{steps}$")
ax.set_ylabel("$n_{steps}$")
plt.yticks(rotation=0)
plt.savefig("error_integration.pdf", tight_layout=True)

# %% Compare gradient methods: numeric vs gradient

tess_size = 20
backend = "pytorch" # ["pytorch", "numpy"]
device = "cpu" # ["cpu", "gpu"]
zero_boundary = True
outsize = 200
batch_size = 1
basis = "svd"

T = cpab.Cpab(tess_size, backend, device, zero_boundary, basis)

grid = T.uniform_meshgrid(outsize)

nSteps1_arr = np.arange(1,21)
nSteps2_arr = np.arange(1,21)
reps = 20

from itertools import product

results = np.zeros((len(nSteps2_arr), len(nSteps1_arr), reps))
for nSteps1, nSteps2 in product(nSteps1_arr, nSteps2_arr):
    for i in range(reps):
        print(nSteps1, nSteps2, i)
        # theta = T.identity(batch_size, epsilon=1)
        theta = T.sample_transformation(batch_size)

        grid_t1 = T.gradient_grid(grid, theta, method="closed_form")

        T.params.nSteps1 = nSteps1
        T.params.nSteps2 = nSteps2

        grid_t2 = T.gradient_grid(grid, theta, method="numeric")

        # error = (grid_t1 - grid_t2).numpy()
        # rms = np.sqrt(np.mean(error**2))

        # error = 100*(grid_t2-grid_t1)/grid_t1
        # rms = np.mean(np.abs(error.numpy()))

        error = (grid_t1 - grid_t2).numpy()
        rms = np.max(np.abs(error))

        results[nSteps2-1, nSteps1-1, i] = rms

    #     results[nSteps2-1, nSteps1-1] += rms
    # results[nSteps2-1, nSteps1-1] /= reps
results = np.median(results, axis=2)
results_gradient = results
print("DONE")
# %%
from matplotlib.colors import LogNorm

sns.set_style("whitegrid")
sns.set_context("paper")
fig, ax = plt.subplots(constrained_layout=True, figsize=(5,4))
sns.heatmap(
    results, norm=LogNorm(results.min(),results.max()),
    square=True, xticklabels=nSteps1_arr, yticklabels=nSteps2_arr,
    cbar_kws = {"label": "Precision"}, ax=ax, cmap="inferno", #vmin=1e-4, vmax=1e1
)
ax.set_xlabel("$N_{steps}$")
ax.set_ylabel("$n_{steps}$")
plt.yticks(rotation=0)
plt.savefig("error_gradient.pdf", tight_layout=True)

# %%
from matplotlib.colors import LogNorm

sns.set_style("whitegrid")
sns.set_context("paper")
fig, (ax0, ax1) = plt.subplots(1,2, constrained_layout=True, figsize=(5,4))

r_min = min(results_integration.min(), results_gradient.min())
r_max = max(results_integration.max(), results_gradient.max())
sns.heatmap(
    results_integration, norm=LogNorm(r_min, r_max),
    square=True, xticklabels=nSteps1_arr, yticklabels=nSteps2_arr,
    cbar_kws = {"label": "Precision"}, ax=ax0, cmap="inferno"
)
sns.heatmap(
    results_gradient, norm=LogNorm(r_min, r_max),
    square=True, xticklabels=nSteps1_arr, yticklabels=nSteps2_arr,
    cbar_kws = {"label": "Precision"}, ax=ax1, cmap="inferno"
)
# ax.set_xlabel("$N_{steps}$")
# ax.set_ylabel("$n_{steps}$")
# plt.savefig("error_gradient.pdf", tight_layout=True)

# %% Study composition of grids

grid = T.uniform_meshgrid(outsize)
theta = T.identity(batch_size, epsilon=0.1)
theta = T.sample_transformation(batch_size)
grid_t = T.transform_grid(grid, theta)
grid_t2 = T.transform_grid(grid_t, theta)
grid_t3 = T.transform_grid(grid_t2, theta)

plt.plot(grid)
plt.plot(grid_t.T)
plt.plot(grid_t2.T)
plt.plot(grid_t3.T)

# %%
channels = 3
width = 100
a = np.zeros((batch_size, channels))
b = np.ones((batch_size, channels)) * 2 * np.pi
noise = np.random.normal(0, 0.1, (batch_size, width, channels))
x = np.linspace(a, b, width, axis=1)
data = 0.5 + np.sin(x-noise)
data = torch.tensor(data)

N = 0
data_t = T.transform_data_ss(data, theta / 2**N, width, N=N)
print(data.shape, data_t.shape)
# data_t2 = T.transform_data(data_t, theta, width)
# data_t3 = T.transform_data(data_t2, theta, width)
# data_t4 = T.interpolate(data, grid_t3, width)

plt.figure()
plt.plot(data[:,:,1].T)
plt.plot(data_t[:,:,1].T)
# plt.plot(data_t2[:,:,0].T)
# plt.plot(data_t3[:,:,0].T)
# plt.plot(data_t4[:,:,0].T)

# T.visualize_velocity(theta)
# T.visualize_deformgrid(theta)
# T.visualize_gradient(theta)

T.visualize_deformdata(data, 2**N * theta)

# %% Alignment of time series samples

batch_size = 1
channels = 1
outsize = 100

a = np.zeros((batch_size, channels))
b = np.ones((batch_size, channels)) * 2 * np.pi
noise = np.random.normal(0, 0.1, (batch_size, outsize, channels))
x = np.linspace(a, b, outsize, axis=1)
dataA = np.sin(x)
dataB = np.sin(x + 0.3)

def normalize(x):
    return (x - np.min(x)) / np.ptp(x)

def normalize(x):
    return (x - np.mean(x)) / np.std(x)

dataA = torch.tensor(normalize(dataA))
dataB = torch.tensor(normalize(dataB))

tess_size = 10
backend = "pytorch" # ["pytorch", "numpy"]
device = "cpu" # ["cpu", "gpu"]
zero_boundary = False
use_slow = False
method = "closed_form"
basis = "svd"
basis = "sparse"
basis = "rref"

T = cpab.Cpab(tess_size, backend, device, zero_boundary, basis)
T.params.use_slow = use_slow

grid = T.uniform_meshgrid(outsize)
theta = T.identity(batch_size, epsilon=0)
theta = torch.autograd.Variable(theta, requires_grad=True)

lr = 1e-3
optimizer = torch.optim.Adam([theta], lr=lr)
# optimizer = torch.optim.SGD([theta_2], lr=1e1)

# torch.set_num_threads(1)
loss_values = []
maxiter = 50
with tqdm(desc='Alignment of samples', unit='iters', total=maxiter,  position=0, leave=True) as pb:
    for i in range(maxiter):
        optimizer.zero_grad()
        
        N = 4
        dataT = T.transform_data_ss(dataA, theta, outsize, N=N)
        loss = torch.norm(dataT - dataB, dim=1).mean()
        loss.backward()
        optimizer.step()

        if torch.any(torch.isnan(theta)):
            print("AHSDASD")
            break

        loss_values.append(loss.item())
        pb.update()
        pb.set_postfix({'loss': loss.item()})
    pb.close()

plt.figure()
plt.plot(loss_values)
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.figure()
plt.plot(dataA[:,:,0].T, label="A original")
plt.plot(dataB[:,:,0].T, label="B original")
plt.plot(dataT.detach()[:,:,0].T, label="A aligned to B")
plt.legend()

T.visualize_deformgrid(2**N*theta.detach())

# %% time benchmark
%%timeit -r 20 -n 1

N = 3
t = 1.0 / 2**N
method = "numeric"
method = "closed_form"
# grid_t1 = T.transform_grid(grid, theta, method, time=t)
# grid_t1_num = T.transform_grid(grid, theta, "numeric", time=t)
# grid_t2 = T.transform_grid_ss(grid, theta, method, time=t, N=0)
grad_t = T.gradient_grid(grid, theta, method, time=t)
# plt.plot(grad_t[0,:,:])

# %% scaling squaring time benchmark
%%timeit -r 200 -n 1

N = 2
t = 1.0 / 2**N 
method = "closed_form"
grid_t2 = T.transform_grid(grid, theta, method, time=t)
for j in range(N):
    grid_t2 = T.backend.interpolate_grid(grid_t2)
    # grid_t3 = np.interp(grid_t2[0], grid_t2[0], grid_t2[0])[np.newaxis, :]
# grad_t = T.gradient_grid(grid, theta, method, time=t)
# error = np.linalg.norm(grid_t - grid_t2)

# %% Visualize functions
t = 1.0
# T.visualize_tesselation()
T.visualize_velocity(theta)
T.visualize_deformgrid(theta, time=t)
T.visualize_deformgrid(theta, method="numeric", time=t)
T.visualize_gradient(theta, time=t)
T.visualize_gradient(theta, "numeric", time=t)

plt.figure()
# plt.plot(T.transform_grid(grid, theta, method="numeric", time=0.0)[0]-grid)
plt.plot(T.transform_grid(grid, theta, method="closed_form", time=0.0)[0]-grid)

# %% OPTIMIZATION BY GRADIENT

tess_size = 500
backend = "pytorch" # ["pytorch", "numpy"]
device = "cpu" # ["cpu", "gpu"]
zero_boundary = False
use_slow = False
outsize = 100
batch_size = 1
method = "closed_form"
basis = "sparse"
basis = "svd"
basis = "rref"

from scipy.interpolate import interp1d

results= {}
for basis in ["sparse", "svd", "rref", "qr"]:
    start_time = time.time()

    T = cpab.Cpab(tess_size, backend, device, zero_boundary, basis)
    T.params.use_slow = use_slow
    # T.params.B = torch.tensor(B2, dtype=torch.float)

    grid = T.uniform_meshgrid(outsize)

    torch.manual_seed(0)
    theta_1 = T.sample_transformation(batch_size)*10
    theta_1 = T.identity(batch_size, epsilon=1.0)
    grid_t1 = T.transform_grid(grid, theta_1, method)
    x = [0, .2, .4, .6, .8, 1]
    y = [0, .5, .6, .7, .75, 1]
    f = interp1d(x, y, kind="linear", fill_value="extrapolate")
    xv = np.linspace(0,1,outsize)
    yv = f(xv)
    # grid_t1 = torch.tile(torch.linspace(0,1,outsize)**2, (batch_size,1))
    grid_t1 = torch.tile(torch.tensor(yv, dtype=torch.float), (batch_size, 1))

    torch.manual_seed(1)
    theta_2 = torch.autograd.Variable(T.sample_transformation(batch_size), requires_grad=True)
    theta_2 = torch.autograd.Variable(T.identity(batch_size, epsilon=0.0), requires_grad=True)

    lr = 1e-2
    optimizer = torch.optim.Adam([theta_2], lr=lr)
    # optimizer = torch.optim.SGD([theta_2], lr=1e1)

    # torch.set_num_threads(1)
    loss_values = []
    maxiter = 250
    with tqdm(desc='Alignment of samples', unit='iters', total=maxiter,  position=0, leave=True) as pb:
        for i in range(maxiter):
            optimizer.zero_grad()
            
            # output = T.backend.cpab_cpu.integrate_closed_form_trace(grid, theta_2, T.params.B, T.params.xmin, T.params.xmax, T.params.nc)
            # grad_theta = T.backend.cpab_cpu.derivative_closed_form_trace(output, grid, theta_2, T.params.B, T.params.xmin, T.params.xmax, T.params.nc) # [n_batch, n_points, d]
            # theta_backup = theta_2.clone()
            
            grid_t2 = T.transform_grid(grid, theta_2, method=method)
            loss = torch.norm(grid_t2 - grid_t1)
            loss = torch.norm(grid_t2 - grid_t1, dim=1).mean()
            loss.backward()
            optimizer.step()

            if torch.any(torch.isnan(theta_2)):
                print("AHSDASD")
                break

            loss_values.append(loss.item())
            pb.update()
            pb.set_postfix({'loss': loss.item()})
        pb.close()

    end_time = time.time()

    if False:
        plt.figure()
        plt.plot(loss_values)
        plt.axhline(color="black", ls="dashed")
        # plt.yscale('log')

        plt.figure()
        plt.plot(grid, grid_t1.t())
        plt.plot(grid, grid_t2.detach().t())

        plt.figure()
        plt.plot(grid_t1.t() - grid_t2.detach().t())
        plt.axhline(color="black", ls="dashed")

    results[basis] = {}
    results[basis]["loss"] = loss_values
    results[basis]["B"] = T.params.B
    results[basis]["grid_t1"] = grid_t1.detach().numpy()
    results[basis]["grid_t2"] = grid_t2.detach().numpy()
    results[basis]["theta"] = theta_2.detach().numpy()
    results[basis]["time"] = end_time - start_time

# %%

plt.figure()
for k, v in results.items():
    plt.plot(v["loss"])
plt.title("Loss")
plt.legend(results.keys())
plt.axhline(0, ls="dashed", c="black")

plt.figure()
plt.plot(v["grid_t1"].T, c="black")
for k, v in results.items():
    plt.plot(v["grid_t2"].T)
plt.title("Grid Deform")
plt.legend(["Original"] + list(results.keys()))

print("Loss")
for k, v in results.items():
    print(k, ": ", np.mean(v["loss"]))

print("Time")
for k, v in results.items():
    print(k, ": ", v["time"])

print("Theta")
for k, v in results.items():
    print(k, ": ", np.linalg.norm(v["theta"]))

# %% OPTIMIZATION BY MCMC SAMPLING

tess_size = 5
backend = "pytorch" # ["pytorch", "numpy"]
device = "cpu" # ["cpu", "gpu"]
zero_boundary = True
use_slow = False
outsize = 100
batch_size = 1

T = cpab.Cpab(tess_size, backend, device, zero_boundary)
T.params.use_slow = use_slow

grid = T.uniform_meshgrid(outsize)

theta_ref = T.sample_transformation(batch_size)
grid_ref = T.transform_grid(grid, theta_ref)

current_sample = T.sample_transformation(batch_size)
grid_t = T.transform_grid(grid, current_sample)

current_error = np.linalg.norm(grid_t - grid_ref)
accept_ratio = 0

samples = []
maxiter = 1500
pb = tqdm(desc='Alignment of samples', unit='samples', total=maxiter)
for i in range(maxiter):
    # Sample and transform 
    theta = T.sample_transformation(batch_size, mean=current_sample.flatten())
    grid_t = T.transform_grid(grid, theta, method="closed_form")

    # Calculate new error
    new_error = np.linalg.norm(grid_t - grid_ref)

    samples.append( T.backend.tonumpy(theta[0]))
    
    if new_error < current_error:
        current_sample = theta
        current_error = new_error
        accept_ratio += 1
    pb.update()
print('Acceptence ratio: ', accept_ratio / maxiter * 100, '%')
pb.close()

# samples = np.array(samples)
# for i in range(len(theta[0])):
#     plt.figure()
#     plt.hist(samples[:,i])
#     plt.axvline(theta_ref[0][i], c="red", ls="dashed")


theta = np.mean(samples, axis=0)[np.newaxis,:]
grid_t = T.transform_grid(grid, T.backend.to(theta), method="closed_form")

plt.figure()
plt.plot(grid, grid_ref[0])
plt.plot(grid, grid_t[0])




# %% TRANSFORM DATA

tess_size = 5
backend = "pytorch" # ["pytorch", "numpy"]
device = "cpu" # ["cpu", "gpu"]
zero_boundary = False
use_slow = False
outsize = 100
batch_size = 2

T = cpab.Cpab(tess_size, backend, device, zero_boundary)
T.params.use_slow = use_slow

theta = T.sample_transformation(batch_size)
grid = T.uniform_meshgrid(outsize)
grid_t = T.transform_grid(grid, theta)

T.visualize_deformgrid(theta)

width = 50
channels = 2

# data generation
# data = np.random.normal(0, 1, (batch_size, width, channels))
a = np.zeros((batch_size, channels))
b = np.ones((batch_size, channels)) * 2 * np.pi
noise = np.random.normal(0, 0.1, (batch_size, width, channels))
x = np.linspace(a, b, width, axis=1)
data = np.sin(x)

data_t = T.transform_data(torch.tensor(data), theta, outsize)

# plot data
batch_size, width, channels = data.shape

fig, ax = plt.subplots(nrows=channels, ncols=1, sharex=True, squeeze=False)
for i in range(channels):
    ax[i, 0].plot(data[:, :, i].T, color="blue", alpha=0.1)

# plot transformed data per batch
fig, ax = plt.subplots(nrows=channels, ncols=batch_size, sharex=True, squeeze=False)
for i in range(channels):
    for j in range(batch_size):
        ax[i, j].plot(np.linspace(0,1,width), data[j, :, i], color="blue")
        ax[i, j].plot(np.linspace(0,1,outsize), data_t[j, :, i], color="red")

fig, ax = plt.subplots(nrows=channels, ncols=1, sharex=True, squeeze=False)
for i in range(channels):
    ax[i, 0].plot(np.linspace(0,1,width), data[:, :, i].T, color="blue")
    ax[i, 0].plot(np.linspace(0,1,outsize), data_t[:, :, i].T, color="red")


# %% OPTIMIZATION BY GRADIENT (DATA)

tess_size = 50
backend = "pytorch" # ["pytorch", "numpy"]
device = "cpu" # ["cpu", "gpu"]
zero_boundary = True
use_slow = False
outsize = 100
batch_size = 1
method = "closed_form"

basis = "qr"
basis = "svd"
basis = "sparse"
basis = "rref"

width = 100
channels = 1

# %%
import scipy
from scipy.interpolate import UnivariateSpline
def gradient_spline(time, f, smooth=False):
    """
    This function takes the gradient of f using b-spline smoothing
    :param time: vector of size N describing the sample points
    :param f: numpy array of shape (N) with N samples
    :param smooth: smooth data (default = F)
    :rtype: tuple of numpy ndarray
    :return f0: smoothed functions functions
    :return g: first derivative of each function
    :return g2: second derivative of each function
    """
    
    if smooth:
        spar = len(time) * (.025 * np.fabs(f).max()) ** 2
    else:
        spar = 0
    spline = UnivariateSpline(time, f, s=spar)
    f0 = spline(time)
    g = spline(time, 1)
    g2 = spline(time, 2)

    return f0, g, g2

def f_to_srsf(f, time, smooth=False):
    """
    converts f to a square-root slope function (SRSF)
    :param f: vector of size N samples
    :param time: vector of size N describing the sample points
    :rtype: vector
    :return q: srsf of f
    """
    eps = np.finfo(np.double).eps
    f0, g, g2 = gradient_spline(time, f, smooth)
    q = g / np.sqrt(np.fabs(g) + eps)
    return q

from scipy.integrate import cumtrapz
def srsf_to_f(q, time, f0=0.0):
    """
    converts q (srsf) to a function
    :param q: vector of size N samples of srsf
    :param time: vector of size N describing time sample points
    :param f0: initial value
    :rtype: vector
    :return f: function
    """
    integrand = q*np.fabs(q)
    f = f0 + cumtrapz(integrand, time, initial=0)
    return f


def gradient_custom(f, axis=1, edge_order = 1):
    N = f.ndim 
    slice1 = [slice(None)]*N
    slice2 = [slice(None)]*N
    slice3 = [slice(None)]*N
    slice4 = [slice(None)]*N

    slice1[axis] = slice(1, -1)
    slice2[axis] = slice(None, -2)
    slice3[axis] = slice(1, -1)
    slice4[axis] = slice(2, None)
    
    backend = np if isinstance(f, np.ndarray) else torch
    g = backend.empty_like(f, dtype=backend.float64)

    g[tuple(slice1)] = (f[tuple(slice4)] - f[tuple(slice2)]) / 2.

    if edge_order == 1:
        slice1[axis] = 0
        slice2[axis] = 1
        slice3[axis] = 0

        g[tuple(slice1)] = f[tuple(slice2)] - f[tuple(slice3)]

        slice1[axis] = -1
        slice2[axis] = -1
        slice3[axis] = -2
        g[tuple(slice1)] = f[tuple(slice2)] - f[tuple(slice3)]

    else:
        slice1[axis] = 0
        slice2[axis] = 0
        slice3[axis] = 1
        slice4[axis] = 2

        a = -1.5
        b = 2. 
        c = -0.5
        g[tuple(slice1)] = a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]

        slice1[axis] = -1
        slice2[axis] = -3
        slice3[axis] = -2
        slice4[axis] = -1

        a = 0.5
        b = -2.
        c = 1.5
        g[tuple(slice1)] = a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]

    return g

def curve2srvf(f):
    eps = np.finfo(np.double).eps
    g = np.gradient(f, axis=1, edge_order=2)
    # g = gradient_custom(f, axis=1, edge_order=2)
    q = g / np.sqrt(np.fabs(g) + eps)
    return q

    i = 2*f[:,0,:] - f[:,1,:]
    i = np.expand_dims(i, 1)
    fp = np.diff(f, axis=1, prepend=i)
    tmp =  np.sign(fp) * np.sqrt(np.abs(fp))

    q = fp / np.sqrt(np.fabs(fp) + eps)

    return tmp

def tupleset(t, i, value):
    l = list(t)
    l[i] = value
    return tuple(l)

def cumtrapz(y, axis=1, initial=0):
    d = 1
    nd = len(y.shape)
    slice1 = tupleset((slice(None),)*nd, axis, slice(1, None))
    slice2 = tupleset((slice(None),)*nd, axis, slice(None, -1))

    backend = np if isinstance(y, np.ndarray) else torch
    cat = np.concatenate if isinstance(y, np.ndarray) else torch.cat
    res = backend.cumsum(d * (y[slice1] + y[slice2]) / 2.0, axis=axis)

    if initial is not None:
        shape = list(res.shape)
        shape[axis] = 1
        res = cat([backend.ones(shape) * initial, res], axis)

    return res

def srvf2curve(q, f0=0.0):
    integrand = q*np.abs(q)
    return f0 + cumtrapz(integrand, axis=1, initial=0)
    # return f0 + np.cumsum(integrand, axis=1)

# def curve2srvf(f):
#     i = 2*f[:,0,:] - f[:,1,:]
#     i = torch.unsqueeze(i, 1)
#     fp = torch.diff(f, axis=1, prepend=i)
#     return torch.sign(fp) * torch.sqrt(torch.abs(fp))

# def srvf2curve(q, f0=0.0):
#     integrand = q*torch.abs(q)
#     return f0 + torch.cumsum(integrand, dim=1)

# Generation
batch_size = 1
channels = 1
width = 100
a = np.zeros((batch_size, channels))
b = np.ones((batch_size, channels)) * 2 * np.pi
noise = np.random.normal(0, 0.1, (batch_size, width, channels))
x = np.linspace(a, b, width, axis=1)
data = np.sin(x)
data = torch.tensor(data)
srvf = False
if srvf:
    data = curve2srvf(data)

# newdata = np.empty_like(data)
# olddata = np.empty_like(data)
# for i in range(batch_size):
#     for j in range(channels):
#         newdata[i,:,j] = f_to_srsf(data[i,:,j], x[i,:,j], smooth=False)

#         olddata[i,:,j] = srsf_to_f(newdata[i,:,j], x[i,:,j], f0=data[i,0,j])

newdata2 = curve2srvf(data)
olddata2 = srvf2curve(newdata2, f0=data[:,0,:])

plt.figure()
plt.plot(data[0,:,0])
# plt.plot(olddata[0,:,0])
plt.plot(olddata2[0,:,0])

plt.figure()
# plt.plot(newdata[0,:,0])
plt.plot(newdata2[0,:,0])

# %%
T = cpab.Cpab(tess_size, backend, device, zero_boundary, basis)
T.params.use_slow = use_slow

# custom deformation
alpha = 1
n = outsize
np.random.seed(0)
grid_t = np.cumsum(np.random.dirichlet(alpha=[alpha] * n))
grid_t = np.linspace(0,1,n)**3
data_t1 = T.interpolate(data + noise*0, torch.tensor(grid_t), outsize)
# theta_1 = T.identity(batch_size, epsilon=1.0)
# data_t1 = data

# torch.manual_seed(0)
# theta_1 = T.sample_transformation(batch_size)*1
# data_t1 = T.transform_data(data, theta_1, outsize, method)

theta_2 = torch.autograd.Variable(T.sample_transformation(batch_size), requires_grad=True)
# theta_2 = torch.autograd.Variable(T.identity(batch_size), requires_grad=True)
data_t2 = T.transform_data(data, theta_2, outsize, method)

lr = 1e-2
optimizer = torch.optim.Adam([theta_2], lr=lr)
# optimizer = torch.optim.SGD([theta_2], lr=lr, momentum=0.9)

# torch.set_num_threads(1)
loss_values = []
maxiter = 500
with tqdm(desc='Alignment of samples', unit='iters', total=maxiter,  position=0, leave=True) as pb:
    for i in range(maxiter):
        optimizer.zero_grad()
        
        data_t2 = T.transform_data(data, theta_2, outsize, method)
        loss = torch.norm(data_t2 - data_t1)
        # loss = torch.norm(data_t2 - data_t1, dim=1).mean()
        loss.backward()
        optimizer.step()

        if torch.any(torch.isnan(theta_2)):
            print("AHSDASD")
            break

        loss_values.append(loss.item())
        pb.update()
        pb.set_postfix({'loss': loss.item()})
    pb.close()

plt.figure()
plt.plot(loss_values)
# plt.axhline(color="black", ls="dashed")
plt.yscale('log')

plt.figure()
plt.plot(data_t1[:,:,0].t())
plt.plot(data_t2[:,:,0].detach().t())

plt.figure()
plt.plot(data_t1[:,:,0].t() - data_t2[:,:,0].detach().t())
plt.axhline(color="black", ls="dashed")
# theta_1, theta_2

if srvf:
    plt.figure()
    plt.plot(srvf2curve(data_t1)[:,:,0].t())
    plt.plot(srvf2curve(data_t2)[:,:,0].detach().t())



# %%

x = np.linspace(0,1, 100)
y = np.sin(2*np.pi*x) + np.random.normal(0,0.1, (10, 100))
plt.figure()
for i in range(10):
    y[i,:] = y[i,:] * np.random.uniform(0.5,1)
    plt.plot(y[i,:])

c = np.cov(y, rowvar=False)
u, s, vh = np.linalg.svd(c)

plt.figure()
for i in range(10):
    plt.plot(u[:,i]*s[i])
# %%
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
import importlib
importlib.reload(cpab)
from tqdm import tqdm

# COMPARE NUMERIC VS CLOSED_FORM

torch.random.manual_seed(0)
tess_size = 5
n_points = 1000
T = cpab.Cpab(tess_size=tess_size, backend="pytorch", device="cpu", zero_boundary=True, basis="svd")
T.params.use_slow = False
theta = T.identity(n_sample=1, epsilon=1)
grid = T.uniform_meshgrid(n_points)

# PARAMETERS
theta = torch.autograd.Variable(theta, requires_grad=True)
grid = torch.autograd.Variable(grid, requires_grad=True)

method1 = "numeric"
theta1 = theta.clone()
grid1 = grid.clone()
# y1 = T.transform_grid(grid1, theta1, method1)
y1 = T.gradient_space(grid1, theta1, method1)

method2 = "closed_form"
theta2 = theta.clone()
grid2 = grid.clone()
# y2 = T.transform_grid(grid2, theta2, method2)
y2 = T.gradient_space(grid2, theta2, method2)

y1.retain_grad()
theta1.retain_grad()
grid1.retain_grad()
loss1 = torch.norm(y1)
loss1.backward()

y2.retain_grad()
theta2.retain_grad()
grid2.retain_grad()
loss2 = torch.norm(y2)
loss2.backward()

# with torch.no_grad():
#     plt.figure()
#     plt.plot(grid, y1.T, label=method1)
#     plt.plot(grid, y2.T, label=method2)
#     plt.title("Y1, Y2")
#     plt.legend()

#     plt.figure()
#     plt.plot(grid, y1.grad.T, label=method1)
#     plt.plot(grid, y2.grad.T, label=method2)
#     plt.title("DY1, DY2")
#     plt.legend()

#     plt.figure()
#     plt.plot(grid, (y1.grad - y2.grad).T)
#     plt.title("DY2 - DY1")

#     plt.figure()
#     plt.plot(grid, grid1.grad, label=method1)
#     plt.plot(grid, grid2.grad, label=method2)
#     plt.title("DGRID1, DGRID2")
#     plt.legend()

#     plt.figure()
#     plt.plot(grid, grid1.grad - grid2.grad)
#     plt.title("DGRID1 - DGRID2")

theta1.grad, theta2.grad
# %%
loss_values = []
maxiter = 500
with tqdm(total=maxiter) as pbar:
    for i in range(maxiter):
        optimizer.zero_grad()
        
        # y = T.calc_velocity(grid, theta)
        # y = T.gradient_grid(grid, theta, "numeric")
        method = "numeric"
        method = "closed_form"
        y = T.gradient_space(grid, theta, method)
        # loss = torch.norm(grid)*torch.norm(theta)
        loss = torch.norm(y)
        y.retain_grad()
        # theta.retain_grad()
        # loss.retain_grad()

        loss.backward()
        optimizer.step()
        loss_values.append(loss.item())

        pbar.set_description("Loss %.3f" % loss.item())
        pbar.update()

plt.figure()
plt.plot(loss_values)

# plt.figure()
# plt.plot(y.grad.T)
print(y.grad.shape)

# %%
tess_size = 3
n_points = 1000
T = cpab.Cpab(tess_size=tess_size, backend="pytorch", zero_boundary=True, basis="svd")
T.params.use_slow=True
theta = T.identity(n_sample=1, epsilon=1)
theta = torch.autograd.Variable(theta, requires_grad=True)
grid = T.uniform_meshgrid(n_points)
# grid = torch.autograd.Variable(grid, requires_grad=True)
# y = T.transform_grid(grid, theta)
method = "numeric"
phi_d = T.gradient_space(grid, theta, method)
phi_d = torch.norm(grid)*torch.norm(theta)
phi_d.retain_grad()
z = torch.norm(phi_d)
print(phi_d.requires_grad)

h = 1e-2
phi_1 = T.transform_grid(grid, theta)
phi_2 = T.transform_grid(grid+h, theta)
phi_d_emp = (phi_2-phi_1)/h

eps = 1e-3
d = T.params.d
u_d = []
for i in range(d):
    h = torch.zeros(d)
    h[i] = eps
    u1 = T.gradient_space(grid, theta)
    u2 = T.gradient_space(grid, theta+h)
    ud = (u2 - u1)/eps
    u_d.append(ud)

u_d = torch.cat(u_d)

# with torch.no_grad():
#     plt.figure()
#     plt.plot(grid, phi_d.T)
#     plt.plot(grid, phi_d_emp.T)
    
#     plt.figure()
#     plt.plot(grid, u_d.T)

z.backward()
# %%

# z.register_hook(lambda grad: print("dz/dz", grad.shape, grad))
# y.register_hook(lambda grad: print("dz/dy", grad.shape, grad))
# theta.register_hook(lambda grad: print("dy/dtheta", grad.shape)) 


plt.plot(grid, y.grad.T)

# %%
tess_size = 50
backend = "numpy"
backend = "pytorch"
device = "cpu"
zero_boundary = True
basis = "svd"

T = cpab.Cpab(tess_size, backend, device, zero_boundary, basis)
T.params.use_slow = False

n_points = 1001
grid = T.uniform_meshgrid(n_points)#[torch.randperm(n_points)]
theta = T.identity(n_sample=1, epsilon=10)
theta = torch.autograd.Variable(theta, requires_grad=True)
method = "numeric"
method = "closed_form"
phi = T.transform_grid(grid, theta, method=method)
phi_dx = T.gradient_space(grid, theta, method=method)
phi_dx.retain_grad()

loss = torch.norm(phi_dx)
loss.backward()

# grid = grid.cpu()
# phi = phi.cpu()
# phi_dx = phi_dx.cpu()

with torch.no_grad():
    phi_dx_empirical = np.gradient(phi, grid, edge_order=2, axis=1)
    # phi_dx_empirical = np.diff(phi, axis=1, append=phi[0,-1])*n_points

    plt.figure()
    for x in np.linspace(0,1,tess_size+1):
        plt.axvline(x, c="k", ls="--", lw=0.5)
        plt.axhline(x, c="k", ls="--", lw=0.5)
    plt.plot(grid, phi.T)

    plt.figure()
    # plt.plot(grid, phi_dx.T, marker='o', ms=5, label="computed")
    plt.plot(grid, phi_dx.T, label="computed")
    plt.plot(grid, phi_dx_empirical.T, label="empirical")
    plt.legend()

    plt.figure()
    plt.plot(grid, phi_dx.T - phi_dx_empirical.T, label="difference")
    plt.legend()

    plt.figure()
    plt.plot(grid, phi_dx.grad.T, label="difference")
    plt.legend()

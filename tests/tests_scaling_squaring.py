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
from tqdm import tqdm 
import torch.utils.benchmark as benchmark
from itertools import product


# %% INTEGRATION

results = []

tess_size_arr = [50]
backend_arr = ["pytorch"] # ["pytorch", "numpy"]
device_arr = ["cpu"] # ["cpu", "gpu"]
zero_boundary_arr = [True]
use_slow_arr = [False]
outsize_arr = [200]
batch_size_arr = [20]
basis_arr = ["svd"] # ["svd", "rref", "sparse", "qr"]
method_arr = ["closed_form", "numeric"] # ["closed_form", "numeric"]
N_arr = [0,1,2,3,4,5,6,7,8]
epsilon_arr = [0,1,2,3,4]
num_threads_arr = [1]

colnames = [
    "tess_size", "backend", "device", "zero_boundary", "use_slow", 
    "outsize", "batch_size", "basis", "method", "N", "epsilon", "num_threads",
    "error", "elapsed_time"]
configurations = list(product(tess_size_arr, backend_arr, device_arr, zero_boundary_arr, use_slow_arr, outsize_arr, 
    batch_size_arr, basis_arr, method_arr, N_arr, epsilon_arr, num_threads_arr))
n_configurations = len(configurations)
with tqdm(desc='Benchmark', unit='iters', total=n_configurations,  position=0, leave=True) as pb:
    for config in configurations:
        tess_size, backend, device, zero_boundary, use_slow, outsize, batch_size, basis, method, N, epsilon, num_threads = config

        T = cpab.Cpab(tess_size, backend, device, zero_boundary, basis)
        T.params.use_slow = use_slow

        # compute
        grid = T.uniform_meshgrid(outsize)
        theta = T.identity(batch_size, epsilon=epsilon)
        torch.manual_seed(0)
        theta = T.sample_transformation(batch_size)*epsilon
        grid_ref = T.transform_grid(grid, theta, method=method, time=1.0)
        grid_t = T.transform_grid_ss(grid, theta / 2**N, method=method, time=1.0, N=N) 

        # error
        error = np.linalg.norm(grid_ref - grid_t)

        # time
        t0 = benchmark.Timer(
            stmt="""grid_t = T.transform_grid_ss(grid, theta / 2**N, method=method, time=1.0, N=N)""", 
            globals={"T": T, "grid": grid, "theta": theta, "method": method, "N": N},
            num_threads = num_threads
        )
        measure = t0.blocked_autorange(min_run_time = 1)
        # measure = t0.timeit(1000)
        elapsed_time = measure.mean * 1e6 # microseconds us

        # save
        to_save = [tess_size, backend, device, zero_boundary, use_slow, outsize, batch_size, basis, method, N, epsilon, num_threads,
            error, elapsed_time]
        results.append(to_save)

        pb.update()
    pb.close()

results = pd.DataFrame(results, columns=colnames)
results.to_csv("scaling_squaring_integration.csv", index=False)

# %%
results = pd.read_csv("scaling_squaring_integration.csv")

value_vars = ["error", "elapsed_time"]
id_vars = [e for e in results.columns if e not in value_vars]
results_melt = results.melt(id_vars=id_vars, value_vars=value_vars)

sns.set_style("whitegrid", {'axes.grid' : False})
sns.set_context("paper")
palette = "blend:orange,blue"
g = sns.relplot(x="N", y="value", hue="epsilon", style="epsilon",
    row="variable", #col="method", 
    kind="line", 
    data=results_melt[results_melt.method=="closed_form"], 
    legend="full", palette=palette, height=3, aspect=1.2,
    ci=None, markersize=8, markers=True, dashes=False, lw=2, 
    facet_kws = {
        "sharex": True, "sharey":False, "despine": False, "legend_out": False
    })

for ax in g.axes.flatten():
    ax.set_xticks([0,1,2,3,4,5,6,7,8])
# g.set_titles(template="Method: {col_name}")
g.set_titles(template="")
g.set(ylim=(-0.02, None))
g.axes[0,0].set_ylabel('RMS Error')
g.axes[1,0].set_ylabel('Time ($\mu s$)')
g.axes[1,0].set_xlabel('Number of squarings $N$')
# g.axes[1,1].set_xlabel('Scaling parameter $N$')
g.axes[0,0].legend(title="$\\varepsilon$")

plt.tight_layout()
plt.savefig("scaling_squaring_integration.pdf")

# %% GRADIENT

import torch.utils.benchmark as benchmark
from itertools import product

results = []

tess_size_arr = [50]
backend_arr = ["pytorch"] # ["pytorch", "numpy"]
device_arr = ["cpu"] # ["cpu", "gpu"]
zero_boundary_arr = [True]
use_slow_arr = [False]
outsize_arr = [200]
batch_size_arr = [20]
basis_arr = ["svd"] # ["svd", "rref", "sparse", "qr"]
method_arr = ["closed_form", "numeric"] # ["closed_form", "numeric"]
N_arr = [0,1,2,3,4,5,6,7,8]
epsilon_arr = [0,1,2,3,4]
num_threads_arr = [1]

colnames = [
    "tess_size", "backend", "device", "zero_boundary", "use_slow", 
    "outsize", "batch_size", "basis", "method", "N", "epsilon", "num_threads",
    "error", "elapsed_time"]
configurations = list(product(tess_size_arr, backend_arr, device_arr, zero_boundary_arr, use_slow_arr, outsize_arr, 
    batch_size_arr, basis_arr, method_arr, N_arr, epsilon_arr, num_threads_arr))
n_configurations = len(configurations)
with tqdm(desc='Benchmark', unit='iters', total=n_configurations,  position=0, leave=True) as pb:
    for config in configurations:
        tess_size, backend, device, zero_boundary, use_slow, outsize, batch_size, basis, method, N, epsilon, num_threads = config

        T = cpab.Cpab(tess_size, backend, device, zero_boundary, basis)
        T.params.use_slow = use_slow

        # compute
        grid = T.uniform_meshgrid(outsize)
        theta = T.identity(batch_size, epsilon=epsilon)
        torch.manual_seed(0)
        theta = T.sample_transformation(batch_size)*epsilon

        # grad
        theta_ref = torch.autograd.Variable(theta, requires_grad=True)
        grid_ref = T.transform_grid(grid, theta_ref, method=method, time=1.0)
        loss_ref = torch.norm(grid_ref)
        grid_ref.retain_grad()
        loss_ref.retain_grad()
        loss_ref.backward()

        theta_t = torch.autograd.Variable(theta, requires_grad=True)
        grid_t = T.transform_grid_ss(grid, theta_t / 2**N, method=method, time=1.0, N=N)
        loss_t = torch.norm(grid_t)
        grid_t.retain_grad()
        loss_t.retain_grad()
        loss_t.backward()

        # error
        # error = np.linalg.norm(theta_ref.grad - theta_t.grad)
        error = np.linalg.norm(grid_ref.grad - grid_t.grad) 

        # time
        t0 = benchmark.Timer(
            stmt="""
                theta_t = torch.autograd.Variable(theta, requires_grad=True)
                grid_t = T.transform_grid_ss(grid, theta_t / 2**N, method=method, time=1.0, N=N)
                loss_t = torch.norm(grid_t)
                loss_t.backward()
            """, 
            globals={"T": T, "grid": grid, "theta": theta, "method": method, "N": N, "torch": torch},
            num_threads = num_threads
        )
        measure = t0.blocked_autorange(min_run_time = 1)
        # measure = t0.timeit(1000)
        elapsed_time = measure.mean * 1e3 # miliseconds us

        # save
        to_save = [tess_size, backend, device, zero_boundary, use_slow, outsize, batch_size, basis, method, N, epsilon, num_threads,
            error, elapsed_time]
        results.append(to_save)

        pb.update()
        
    pb.close()

results = pd.DataFrame(results, columns=colnames)
results.to_csv("scaling_squaring_gradient.csv", index=False)

# %%
results = pd.read_csv("scaling_squaring_gradient.csv")

# fig, ax = plt.subplots(2,2, constrained_layout=True, figsize=(8,8))
plt.figure(constrained_layout=True, figsize=(8,6))

value_vars = ["error", "elapsed_time"]
id_vars = [e for e in results.columns if e not in value_vars]
results_melt = results.melt(id_vars=id_vars, value_vars=value_vars)

sns.set_style("whitegrid", {'axes.grid' : False})
sns.set_context("paper")
palette = "blend:orange,blue"
g = sns.relplot(x="N", y="value", hue="epsilon", style="epsilon",
    row="variable", #col="method",
    kind="line", height=3, aspect=1.2,
    data=results_melt[results_melt.method=="closed_form"], 
    legend="full", palette=palette, 
    ci=None, markersize=8, markers=True, dashes=False, lw=2,
    facet_kws = {
        "sharex": True, "sharey":False, "despine": False, "legend_out": False
    })
for ax in g.axes.flatten():
    ax.set_xticks([0,1,2,3,4,5,6,7,8])
# g.set_titles(template="Method: {col_name}")
g.set_titles(template="")
g.axes[0,0].set_ylabel('RMS Error')
g.axes[1,0].set_ylabel('Time (ms)')
g.axes[1,0].set_xlabel('Number of squarings $N$')
# g.axes[1,1].set_xlabel('Scaling parameter $N$')
g.axes[0,0].legend(title="$\\varepsilon$")

plt.tight_layout()
plt.savefig("scaling_squaring_gradient.pdf")

# %% TRAINING

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

with tqdm(desc='Benchmark', unit='iters', total=len(configurations),  position=0, leave=True) as pb:
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

# %%

results = pd.read_csv("scaling_squaring_training.csv")

# fig, ax = plt.subplots(2,2, constrained_layout=True, figsize=(8,8))
plt.figure(figsize=(8,6))

value_vars = ["error_mean", "error_min", "elapsed_time"]
value_vars = ["error_mean", "elapsed_time"]
id_vars = [e for e in results.columns if e not in value_vars]
results_melt = results.melt(id_vars=id_vars, value_vars=value_vars)
results_melt = results_melt[results_melt.lr != 1e-3]

sns.set_style("whitegrid", {'axes.grid' : False})
sns.set_context("paper")
palette = "blend:orange,blue"
g = sns.relplot(x="N", y="value", hue="epsilon", style="epsilon",
    row="variable", col="lr", kind="line", height=4,
    data=results_melt, legend="full", palette=palette, 
    ci=None, markersize=8, markers=True, dashes=False, lw=2,
    facet_kws = {
        "sharex": True, "sharey":"row", "despine": False, "legend_out": True
    })
g.set_titles(template="Learning rate: {col_name}", size=12)
g.axes[0,0].set_ylabel('Loss avg', size=12)
# g.axes[1,0].set_ylabel('Loss min')
g.axes[1,0].set_ylabel('Time (ms)', size=12)
# g.axes[0,0].legend(title="$\\varepsilon$")
g._legend.set_title("$\\varepsilon$")
g._legend.set_bbox_to_anchor([0.98, 0.85])

plt.tight_layout()
plt.savefig("scaling_squaring_training.pdf")

# %%

results = pd.read_csv("scaling_squaring_training.csv")

for lr in [1e-4, 1e-5, 1e-6]:
    m = results[(results.lr == lr)]

    with sns.axes_style("ticks"):
        sns.set_context("paper")

        fig, ax = plt.subplots(figsize=(8,6))
        sns.lineplot(
            data=m,
            x="elapsed_time",
            y="error_mean",
            hue="N",
            marker="o",
            sort=False,
            ax=ax,
            palette="blend:black,black",
            legend=None,
            lw=1,
            ls="dashed"
        )

        for key, row in m[m.epsilon==4].iterrows():
            plt.text(x=row.elapsed_time, y=row.error_mean*1.02, 
            s="N: " + str(int(row.N)), ha="left", va="baseline")

        sns.lineplot(
            data=m,
            x="elapsed_time",
            y="error_mean",
            hue="epsilon",
            marker="o",
            sort=False,
            ax=ax,
            palette= "blend:orange,blue",
            legend="full",
            lw=2,
        )

        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Loss avg')
        ax.legend(title="$\\varepsilon$")

        plt.tight_layout()
        plt.savefig("scaling_squaring_training_" + str(lr) + ".pdf")

# %%
results = pd.read_csv("scaling_squaring_training.csv")

fix, axes = plt.subplots(1, 3, sharey=True, figsize=(12.6,4.2), )
sns.set_style("whitegrid", {'axes.grid' : False})
sns.set_context("paper")

k = 0
for lr in [1e-6, 1e-5, 1e-4]:
    m = results[(results.lr == lr)]
    ax = axes[k]

    sns.lineplot(
        data=m,
        x="elapsed_time",
        y="error_mean",
        hue="N",
        marker="o",
        sort=False,
        ax=ax,
        palette="blend:black,black",
        legend=None,
        lw=0.5,
        ls="dashed",
    )

    for key, row in m[m.epsilon==4].iterrows():
        ax.text(x=row.elapsed_time, y=row.error_mean*1.02, 
        s="N: " + str(int(row.N)), ha="left", va="baseline", rotation=50)

    for key, row in m[m.N==0].iterrows():
        ax.text(x=row.elapsed_time-5, y=row.error_mean, 
        s="$\\varepsilon$: " + str(int(row.epsilon)), ha="right", va="center", rotation=-20)

    sns.lineplot(
        data=m,
        x="elapsed_time",
        y="error_mean",
        hue="epsilon",
        marker="o",
        sort=False,
        ax=ax,
        palette= "blend:orange,blue",
        legend="full" if k == 2 else None,
        lw=2,
    )

    ax.set_xlabel('Time (ms)', size=12)
    ax.set_ylabel('Loss avg', size=12)
    ax.set_title("Learning rate: " + str(lr), size=12)
    ax.set_xlim((870, 1280))
    ax.set_ylim((None, 2.1))
    if k == 2:
        ax.legend(title="$\\varepsilon$")
    k += 1

plt.tight_layout()
plt.savefig("scaling_squaring_training_detail.pdf")

########################################################### %%
# %%timeit -r 20 -n 10

tess_size = 5
backend = "pytorch" # ["pytorch", "numpy"]
device = "cpu" # ["cpu", "gpu"]
zero_boundary = True
use_slow = False
outsize = 1000
batch_size = 1
basis = "rref"

T = cpab.Cpab(tess_size, backend, device, zero_boundary, basis)
T.params.use_slow = use_slow

grid = T.uniform_meshgrid(outsize)

torch.manual_seed(0)
theta_1 = T.identity(batch_size, epsilon=0.7)
theta_1 = T.sample_transformation(batch_size)
grid_t1 = T.transform_grid(grid, theta_1)

theta_2 = torch.autograd.Variable(theta_1, requires_grad=True)
# theta_2 = torch.autograd.Variable(T.identity(batch_size, epsilon=0.3), requires_grad=True)

optimizer = torch.optim.SGD([theta_2], lr=1e-2)
optimizer.zero_grad()

N = 0
t = 1.0
# grid_t2 = T.transform_grid_ss(grid, theta_2, method="closed_form", time=1.0, N=N)       
# t = t / 2**N

# grid_t2 = T.transform_grid(grid, theta_2, method="closed_form", time=1.0)
grid_t2 = T.transform_grid(grid, theta_2, method="closed_form", time = t / 2**N )
grid_t3 = grid_t2
for i in range(N):
    # grid_t3 = T.backend.interpolate_grid_slow(grid_t3)
    grid_t3 = T.backend.interpolate_grid(grid_t3, T.params)

loss = torch.norm(grid_t3 - grid_t1, dim=1).mean()
grid_t2.retain_grad()
grid_t3.retain_grad()
loss.retain_grad()
loss.backward()
optimizer.step()

plt.axline((0,0),(1,1), color="black", ls="dashed")
plt.plot(grid, grid_t1.T)
plt.plot(grid, grid_t2.detach().T)
plt.plot(grid, grid_t3.detach().numpy().T)
plt.axhline(0, color='black', ls="dashed")
plt.axhline(1, color='black', ls="dashed")

grid_t2.grad.numpy(), grid_t3.grad, loss.grad, theta_2.detach().numpy()

# %% 

# %%timeit -r 20 -n 100

import torch
torch.manual_seed(0)

tess_size = 50
backend = "pytorch" # ["pytorch", "numpy"]
device = "cpu" # ["cpu", "gpu"]
zero_boundary = True
use_slow = False
outsize = 1000
batch_size = 2
basis = "rref"

T = cpab.Cpab(tess_size, backend, device, zero_boundary, basis)
T.params.use_slow = use_slow

theta_1 = T.sample_transformation(batch_size)
# theta_1 = T.identity(batch_size, epsilon=1.0)
theta_2 = torch.autograd.Variable(theta_1, requires_grad=True)
grid = T.uniform_meshgrid(outsize)
grid_t1 = T.transform_grid(grid, theta_2)

from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU], record_shapes=False) as prof:
    with record_function("record function"):
        for i in range(100):
            grid_t3 = T.backend.interpolate_grid(grid_t1, T.params)
            loss = grid_t3.norm().mean()
            loss.backward()

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

# plt.plot(grid, grid_t1.T)
# plt.plot(grid, grid_t3.T)

# %%
import numpy as np
xp = grid
yp = grid_t1[0,:]
xnew = grid_t1[0,:]
a = np.interp(xnew, xp, yp)
from scipy.interpolate import interp1d
f = interp1d(xp, yp, fill_value="extrapolate")
b = f(xnew)
plt.plot(grid_t1.T)
plt.plot(grid_t2.T)
plt.plot(a)
plt.plot(b)

# %%

import torch
import contextlib

class Interp1d(torch.autograd.Function):
    def __call__(self, x, y, xnew, out=None):
        return self.forward(x, y, xnew, out)

    def forward(ctx, x, y, xnew, out=None):
        """
        Linear 1D interpolation on the GPU for Pytorch.
        This function returns interpolated values of a set of 1-D functions at
        the desired query points `xnew`.
        This function is working similarly to Matlab™ or scipy functions with
        the `linear` interpolation mode on, except that it parallelises over
        any number of desired interpolation problems.
        The code will run on GPU if all the tensors provided are on a cuda
        device.
        Parameters
        ----------
        x : (N, ) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values.
        y : (N,) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values. The length of `y` along its
            last dimension must be the same as that of `x`
        xnew : (P,) or (D, P) Pytorch Tensor
            A 1-D or 2-D tensor of real values. `xnew` can only be 1-D if
            _both_ `x` and `y` are 1-D. Otherwise, its length along the first
            dimension must be the same as that of whichever `x` and `y` is 2-D.
        out : Pytorch Tensor, same shape as `xnew`
            Tensor for the output. If None: allocated automatically.
        """
        # making the vectors at least 2D
        is_flat = {}
        require_grad = {}
        v = {}
        device = []
        eps = torch.finfo(y.dtype).eps
        for name, vec in {'x': x, 'y': y, 'xnew': xnew}.items():
            assert len(vec.shape) <= 2, 'interp1d: all inputs must be '\
                                        'at most 2-D.'
            if len(vec.shape) == 1:
                v[name] = vec[None, :]
            else:
                v[name] = vec
            is_flat[name] = v[name].shape[0] == 1
            require_grad[name] = vec.requires_grad
            device = list(set(device + [str(vec.device)]))
        assert len(device) == 1, 'All parameters must be on the same device.'
        device = device[0]

        # Checking for the dimensions
        assert (v['x'].shape[1] == v['y'].shape[1]
                and (
                     v['x'].shape[0] == v['y'].shape[0]
                     or v['x'].shape[0] == 1
                     or v['y'].shape[0] == 1
                    )
                ), ("x and y must have the same number of columns, and either "
                    "the same number of row or one of them having only one "
                    "row.")

        reshaped_xnew = False
        if ((v['x'].shape[0] == 1) and (v['y'].shape[0] == 1)
           and (v['xnew'].shape[0] > 1)):
            # if there is only one row for both x and y, there is no need to
            # loop over the rows of xnew because they will all have to face the
            # same interpolation problem. We should just stack them together to
            # call interp1d and put them back in place afterwards.
            original_xnew_shape = v['xnew'].shape
            v['xnew'] = v['xnew'].contiguous().view(1, -1)
            reshaped_xnew = True

        # identify the dimensions of output and check if the one provided is ok
        D = max(v['x'].shape[0], v['xnew'].shape[0])
        shape_ynew = (D, v['xnew'].shape[-1])
        if out is not None:
            if out.numel() != shape_ynew[0]*shape_ynew[1]:
                # The output provided is of incorrect shape.
                # Going for a new one
                out = None
            else:
                ynew = out.reshape(shape_ynew)
        if out is None:
            ynew = torch.zeros(*shape_ynew, device=device)

        # moving everything to the desired device in case it was not there
        # already (not handling the case things do not fit entirely, user will
        # do it if required.)
        for name in v:
            v[name] = v[name].to(device)

        # calling searchsorted on the x values.
        ind = ynew.long()

        # expanding xnew to match the number of rows of x in case only one xnew is
        # provided
        if v['xnew'].shape[0] == 1:
            v['xnew'] = v['xnew'].expand(v['x'].shape[0], -1)

        torch.searchsorted(v['x'].contiguous(),
                           v['xnew'].contiguous(), out=ind)

        # the `-1` is because searchsorted looks for the index where the values
        # must be inserted to preserve order. And we want the index of the
        # preceeding value.
        ind -= 1
        # we clamp the index, because the number of intervals is x.shape-1,
        # and the left neighbour should hence be at most number of intervals
        # -1, i.e. number of columns in x -2
        ind = torch.clamp(ind, 0, v['x'].shape[1] - 1 - 1)

        # helper function to select stuff according to the found indices.
        def sel(name):
            if is_flat[name]:
                return v[name].contiguous().view(-1)[ind]
            return torch.gather(v[name], 1, ind)

        # activating gradient storing for everything now
        enable_grad = False
        saved_inputs = []
        for name in ['x', 'y', 'xnew']:
            if require_grad[name]:
                enable_grad = True
                saved_inputs += [v[name]]
            else:
                saved_inputs += [None, ]
        # assuming x are sorted in the dimension 1, computing the slopes for
        # the segments
        is_flat['slopes'] = is_flat['x']
        # now we have found the indices of the neighbors, we start building the
        # output. Hence, we start also activating gradient tracking
        with torch.enable_grad() if enable_grad else contextlib.suppress():
            v['slopes'] = (
                    (v['y'][:, 1:]-v['y'][:, :-1])
                    /
                    (eps + (v['x'][:, 1:]-v['x'][:, :-1]))
                )

            # now build the linear interpolation
            ynew = sel('y') + sel('slopes')*(
                                    v['xnew'] - sel('x'))

            if reshaped_xnew:
                ynew = ynew.view(original_xnew_shape)

        ctx.save_for_backward(ynew, *saved_inputs)
        return ynew

    @staticmethod
    def backward(ctx, grad_out):
        inputs = ctx.saved_tensors[1:]
        gradients = torch.autograd.grad(
                        ctx.saved_tensors[0],
                        [i for i in inputs if i is not None],
                        grad_out, retain_graph=True)
        result = [None, ] * 5
        pos = 0
        for index in range(len(inputs)):
            if inputs[index] is not None:
                result[index] = gradients[pos]
                pos += 1
        return (*result,)

c = Interp1d()(xp, yp, xnew)
plt.plot(a)
plt.plot(b)
plt.plot(c.T)
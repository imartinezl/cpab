# %%
import sys
sys.path.insert(0,'..')

import time
import timeit
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import cpab
from tqdm import tqdm 

import pandas as pd
import torch.utils.benchmark as benchmark

# %%
from numpy.linalg import cond
from scipy.linalg import orth, norm, pinv

def plot_basis_velocity(T, B=None):
    if B is not None:
        T.params.B = B
    
    plt.figure(constrained_layout=True, figsize=(6,4))
    with sns.axes_style("whitegrid"):
        sns.set_context("paper")
        for k in range(T.params.d):
            theta = T.identity()

            theta[0][k] = 1

            grid = T.uniform_meshgrid(outsize)
            v = T.calc_velocity(grid, theta)

            # Plot
            plt.plot(grid, v.T)
            # plt.title(T.params.basis + " CPA velocity basis")
            plt.xlabel("$x$")
            plt.ylabel("$v(x)$")
            plt.grid(0)
    plt.savefig("basis_" + T.params.basis + ".pdf", tight_layout=True)


def plot_basis_velocity(T, B=None):
    if B is not None:
        T.params.B = B
    
    fig, ax = plt.subplots(1, constrained_layout=True, figsize=(3.75,3.5))
    with plt.style.context("ieee"):
        # plt.style.use('nature')
        # sns.set_context("paper")
        colors = ["#D90368","#FF7733","#51D345","#4A7B9D"]
        colors = ["#EF476F","#06D6A0","#118AB2","#FDC500"]
        for k in range(T.params.d):
            theta = T.identity()

            theta[0][k] = 1

            grid = T.uniform_meshgrid(outsize)
            v = T.calc_velocity(grid, theta)

            # Plot
            ax.axhline(0, color="gray", ls="--", lw=1)
            ax.plot(grid, v.T, lw=2, color=colors[k])
            which = np.linspace(0,len(grid)-1,tess_size+1).astype(int)
            # which = [0,20,40,60,80,99]
            ax.scatter(grid[which], v.T[which], lw=2, color=colors[k])
            # plt.title(T.params.basis + " CPA velocity basis")
            ax.set_xlabel("$x$")
            ax.set_ylabel("$v(x)$")
            ax.set_xticks([0,0.5,1])
            ax.set_yticks([-0.2,0,0.2])
            ax.set_ylim(-0.25,0.25)
            # ax.grid(0)
            # ax.grid()
    plt.savefig("basis_" + T.params.basis + "_2.pdf", tight_layout=True)
def sparsity(sparse):
    return (sparse == 0).sum() / sparse.size

def orthonormality(A):
    return norm(A.T.dot(A) - np.eye(A.shape[1]))

def orthonormal(A):
    return np.allclose(A.T.dot(A), np.eye(A.shape[1]))

def orthogonality(A):
    E = A.T.dot(A)
    return np.sum(~np.isclose(E - np.diag(np.diagonal(E)), np.zeros_like(E))) / A.size

def orthogonal(A):
    return orthogonality(A) == 0

# %% 
# The Sparse Null Space Basis Problem

tess_size = 50
backend = "numpy" # ["pytorch", "numpy"]
device = "cpu" # ["cpu", "gpu"]
zero_boundary = True
outsize = 101
batch_size = 1

basis = "svd"
T = cpab.Cpab(tess_size, backend, device, zero_boundary, basis)
B_svd = T.params.B
# plot_basis_velocity(T)

basis = "rref"
T = cpab.Cpab(tess_size, backend, device, zero_boundary, basis)
B_rref = T.params.B
# plot_basis_velocity(T)

basis = "sparse"
T = cpab.Cpab(tess_size, backend, device, zero_boundary, basis)
B_sparse = T.params.B
# plot_basis_velocity(T)

basis = "qr"
T = cpab.Cpab(tess_size, backend, device, zero_boundary, basis)
B_qr = T.params.B
# plot_basis_velocity(T)

L = T.tess.constrain_matrix()
sparsity(L)

# %%
tess_size = 5
backend = "numpy" # ["pytorch", "numpy"]
device = "cpu" # ["cpu", "gpu"]
zero_boundary = True
outsize = 101
batch_size = 1
basis = "sparse"
T = cpab.Cpab(tess_size, backend, device, zero_boundary, basis)
T.tess.B

def bmatrix(a):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv +=  [r'\end{bmatrix}']
    return '\n'.join(rv)

print(bmatrix(np.round(T.tess.constrain_matrix(),1)))
print(bmatrix(np.round(T.tess.B,3)))
# print(bmatrix(T.tess.B))

# %% Benchmark build times

tess_size_arr = [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
basis_arr = ["svd", "rref", "qr", "sparse"]

backend = "numpy" # ["pytorch", "numpy"]
device = "cpu" # ["cpu", "gpu"]
zero_boundary = True

from itertools import product

results = []
for (tess_size, basis) in product(tess_size_arr, basis_arr):
    print(tess_size, basis)
    T = cpab.Cpab(tess_size, backend, device, zero_boundary, basis)

    t0 = benchmark.Timer(
    stmt="T = cpab.Cpab(tess_size, backend, device, zero_boundary, basis)", 
    globals={
        "cpab": cpab, "tess_size": tess_size, "backend": backend, 
        "device": device, "zero_boundary": zero_boundary, "basis": basis
    })
    measure = t0.blocked_autorange(min_run_time=2)

    elapsed_time_median = measure.median
    elapsed_time_iqr = measure.iqr

    results.append([tess_size, basis, elapsed_time_median, elapsed_time_iqr])

results = pd.DataFrame(results, columns=["tess_size", "basis", "elapsed_time_median", "elapsed_time_iqr"])
results.to_csv("basis_benchmark.csv")
# %%

results = pd.read_csv("basis_benchmark.csv")

from scipy.optimize import curve_fit
def func(x, a, b):
    return a*np.exp(b*x)

colors = np.repeat(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], 2)
colors = np.repeat(['#7570b3', '#d95f02', '#1b9e77', '#e7298a'], 2)
colors = np.repeat(["#D90368","#FF7733","#51D345","#4A7B9D"], 2)
k = 0

with sns.axes_style("whitegrid"):
    sns.set_context("paper")
    fig, ax = plt.subplots(figsize=(6,4), constrained_layout=True)
    for basis in ["svd", "qr", "sparse", "rref"]:
        m = results[results.basis==basis]
        xdata = m.tess_size
        xdata_norm = (xdata - np.min(xdata)) / np.ptp(xdata)
        ydata = m.elapsed_time_median
        popt, pcov = curve_fit(func, xdata_norm, ydata)

        ax.plot(xdata, ydata, color=colors[k], marker="o", ms=4, lw=1)
        ax.plot(xdata, func(xdata_norm, *popt), ls="--", lw=1, color=colors[k+1], 
            label=basis.upper().ljust(13-len(basis)) + ' $y = $%.5f $e^{%.2f x}$' % tuple(popt))
        # ax.text(0, popt[0], "%.1e" % popt[0], ha="right", va="top", size=6)

        ax.set_xlabel("Tessellation size")
        ax.set_ylabel("Elapsed time [s]")
        ax.set_yscale("log")
        ax.grid(True, axis="y", which="both", ls="--", c='gray', alpha=0.3)
        plt.legend(fontsize=9)

        k += 2

    plt.savefig("basis_benchmark_fitted_2.pdf")


# %%

with sns.axes_style("whitegrid"):
    sns.set_context("paper")
    plt.figure(figsize=(8,6), constrained_layout=True)
    g = sns.lineplot(
        data=results, 
        x="tess_size", 
        y="elapsed_time_median", 
        hue="basis",
        hue_order= ["svd", "qr", "sparse", "rref"],
        marker="o",
    )
    g.set(yscale="log")
    g.set_ylabel("Elapsed time [s]")
    g.set_xlabel("Tessellation size")
    g.legend(title="Basis")

    plt.savefig("basis_benchmark.pdf")


# %% Properties
def properties_table(results):
    df = []
    header = ["basis", "sparsity", "cond num", "orth", "orthogonality", "orthnorm", "orthonormality", "norm", "norm inv"]
    print(" | ".join(header))
    print("-"*70)
    dec = 3
    tol = 1e-7
    for v in results:
        name = v[0]
        B = v[1]
        B[np.abs(B) < tol] = 0
        Binv = pinv(B)
        values = [
            name, 
            np.round(sparsity(B),dec), 
            np.round(cond(B),dec), 
            orthogonal(B), 
            np.round(orthogonality(B), dec), 
            orthonormal(B), 
            np.round(orthonormality(B), dec), 
            np.round(norm(B),dec), 
            np.round(norm(Binv),dec)
        ]
        print("\t".join(map(str, values)))
        df.append(values)
    df = pd.DataFrame(df, columns=header)
    return df

results = [("svd", B_svd), ("rref", B_rref), ("sparse", B_sparse), ("qr", B_qr)]
properties_table(results).to_csv("basis_properties.csv", index=False)
properties_table(results).to_latex("basis_properties.tex", index=False)

# %%

def properties(name, B):
    return [
        name, sparsity(B), cond(B), 
        orthogonal(B), orthogonality(B), 
        orthonormal(B), orthonormality(B),
        norm(B),
        norm(pinv(B))
    ]

names = ["svd", "qr", "sparse", "rref"]
Bs = [B_svd, B_qr, B_sparse, B_rref]

results = []
for k in range(4):
    results.append(properties(names[k], Bs[k]))

results = pd.DataFrame(results, 
    columns=["name", "sparsity", "condition_number", "orthogonal", "orthogonality", 
    "orthonormal", "orthonormality", "norm", "norm_inv"])

results
# %% MATLAB

from scipy.io import loadmat
B1 = loadmat("/home/imartinez/Documents/MATLAB/B1.mat")["B"]
B2 = loadmat("/home/imartinez/Documents/MATLAB/B2.mat")["B"]

B1 = B1 / norm(B1, axis=0)

plot_basis_velocity(T, B1)
plot_basis_velocity(T, B2)

properties_table([("B1", B1), ("B2", B2)])

# %%
 
def metric1(B):
    data = []
    for i in np.arange(0, B.shape[0], 2):
        for j in np.arange(0, B.shape[0], 2):
            d = np.sum(B[i, ] * B[j, ])
            data.append([i, j, d])

    return data

def metric2(B):
    d = 0.0
    for j in np.arange(B.shape[1]):
        for i in np.arange(0, B.shape[0], 2):
            d += np.abs(B[i,j])

    return d

def metric3(B):
    d = 0.0
    for j in np.arange(B.shape[1]):
        for i in np.arange(0, B.shape[0], 2):
            d += np.abs(B[i,j])

    return d

def metric4(B):
    data = []
    data = np.empty((B.shape[1], B.shape[1]))
    for i in np.arange(0, B.shape[1]):
        for j in np.arange(0, B.shape[1]):
            d = np.dot(B[:, i], B[:, j])
            # data.append([i, j, d])
            data[i,j] = d

    return np.round(data,2)

metric = metric4
metric(B_svd), metric(B_rref), metric(B_sparse)

# %%

plt.figure()
plt.spy(metric(B_svd), precision=1e-7)

plt.figure()
plt.spy(metric(B_rref), precision=1e-7)

plt.figure()
plt.spy(metric(B_sparse), precision=1e-7)

# %%
names = ["svd", "qr", "sparse", "rref"]
k = 0
with sns.axes_style("white"):
    sns.set_context("paper")
    fig, ax = plt.subplots(ncols=4, figsize=(6,4), sharex=True, sharey=True, constrained_layout=True)
    for B in [B_svd, B_qr, B_sparse, B_rref]:
        ax[k].spy(B, alpha=0.5)
        ax[k].set_title("Basis " + names[k])
        k += 1
plt.savefig("basis_spy.pdf")
# %%

tess_size = 50
backend = "numpy" # ["pytorch", "numpy"]
device = "cpu" # ["cpu", "gpu"]
zero_boundary = True
outsize = 101
batch_size = 1

basis = "svd"
T = cpab.Cpab(tess_size, backend, device, zero_boundary, basis)
B_svd = T.params.B

basis = "rref"
T = cpab.Cpab(tess_size, backend, device, zero_boundary, basis)
B_rref = T.params.B

basis = "sparse"
T = cpab.Cpab(tess_size, backend, device, zero_boundary, basis)
B_sparse = T.params.B

basis = "qr"
T = cpab.Cpab(tess_size, backend, device, zero_boundary, basis)
B_qr = T.params.B

L = T.tess.constrain_matrix()
sparsity(L)

names = ["svd", "qr", "sparse", "rref"]
k = 0

def fastspy(A, ax, color):
    m, n = A.shape
    x, y = np.where(B.T)
    ax.set_xlim((-1,n))
    ax.set_ylim((-1,m))
    ax.set_aspect("equal")
    ax.apply_aspect()
    r = 1
    r_ = ax.transData.transform([r,0])[0] - ax.transData.transform([0,0])[0] 
    ax.scatter(x, y, s=1.7*r_**2, marker='s',edgecolors='none', color=color)
    
    # ax.scatter(x, y, s=r_**2, marker='s',edgecolors='none', color=color)
    # ax.set_xticks(range(0,n))
    # ax.set_yticks(range(0,m,2))

    ax.invert_yaxis()


# with sns.axes_style("white"):
    # sns.set_context("paper")
fig, ax = plt.subplots(ncols=4, figsize=(6,3), sharex=True, sharey=True, constrained_layout=True)
colors = ["#D90368","#FF7733","#51D345","#4A7B9D"]
tol = 1e-7
for B in [B_svd, B_qr, B_sparse, B_rref]:
    # ax[k].spy(B, alpha=0.5, origin="upper")
    B[np.abs(B) < tol] = 0
    fastspy(B, ax[k], colors[k])
    ax[k].set_title(names[k].upper())
    k += 1

# plt.subplots_adjust(wspace=0)
plt.tight_layout()    
plt.savefig("basis_spy_3.pdf")

# %%

x = 0
y = 0
k = 0
h = 0
g = 0
with sns.axes_style("whitegrid"):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8,6), 
        sharex=False, sharey=False, constrained_layout=True)
    sns.set_context("paper")
    for B in [B_svd, B_qr, B_sparse, B_rref]:
        B = orth(B)
        for j in range(B.shape[1]):
            dx = B[0, j]
            dy = B[2, j]
            ax[h, g].arrow(x, y, dx, dy, head_width=0.04, color="black")

        ax[h, g].axis("equal")
        ax[h, g].scatter(0,0, color="red")
        ax[h, g].set_title("Basis " + names[k])
        
        k += 1

        g += 1
        if g == 2:
            h += 1
            g = 0
plt.savefig("basis_arrow.pdf")

# %%
from sklearn.decomposition import PCA

names = ["svd", "qr", "sparse", "rref"]
k = 0
h = 0
g = 0
with sns.axes_style("whitegrid"):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8,6), constrained_layout=True)
    sns.set_context("paper")

for B in [B_svd, B_qr, B_sparse, B_rref]:
    pca = PCA(n_components=2)
    pca.fit(B)

    for i in range(4):
        v = pca.components_[:,i]
        ax[h, g].arrow(0,0, v[0], v[1], head_width=0.04, color="black")
        
    ax[h, g].set_title("Basis " + names[k])
    ax[h, g].axis("equal")
    ax[h, g].scatter(0,0, color="red")

    g += 1
    if g == 2:
        h += 1
        g = 0
    k += 1

plt.savefig("basis_arrow_pca.pdf")
# %%

import numpy as np
import scipy
from scipy.optimize import LinearConstraint, minimize
from scipy.linalg import null_space, norm

fun = lambda x: norm(x-c)**2

n = 3
c = np.ones(n)*1
x0 = np.zeros(n)
fun(x0)

A = np.array([
    [1,4,0],
    [4,0,2]
])
m = A.shape[0]
ub = np.zeros(m)
lb = np.zeros(m)
constraint = LinearConstraint(A, lb, ub)
# res = minimize(fun, x0, constraints=constraint, method="SLSQP")
res = minimize(fun, x0, constraints=constraint, method="trust-constr",
    options={
        "factorization_method": None,
        "verbose": 3
        })
res.x

A = np.array([
    [2,3,5],
    [-4,2,3]
])
Ap = null_space(A)
Ap


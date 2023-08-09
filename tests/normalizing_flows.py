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
from scipy.stats import beta, uniform

# %%
def moving_average(x, w):
    x_padded = np.pad(x, (w//2, w-1-w//2), mode='edge')
    return np.convolve(x_padded, np.ones((w,))/w, mode='valid')

# %%

def get_values(n, tess_size, mean=0, length_scale=3.8, output_variance=1.4, theta=None):
    np.random.seed(0)

    a, b = 3, 3
    z = np.linspace(0,1,n)
    # z = np.random.beta(a, b,size=n)
    # z = np.sort(z)
    pz = beta.pdf(z, a, b)

    # tess_size = 25
    backend = "numpy" # ["pytorch", "numpy"]
    device = "cpu" # ["cpu", "gpu"]
    zero_boundary = True
    use_slow = False
    outsize = n
    batch_size = 1
    method = "numeric"
    basis = "sparse"
    basis = "rref"
    basis = "qr"
    basis = "svd"

    T = cpab.Cpab(tess_size, backend, device, zero_boundary, basis)
    T.params.use_slow = use_slow

    grid = T.uniform_meshgrid(outsize)
    if theta is None:
        theta = T.sample_transformation_with_prior(1, mean=np.ones(tess_size-1,dtype=np.float32)*mean, length_scale=length_scale, output_variance=output_variance)
    fz = T.transform_grid(grid, theta, method=method)[0]
    fx = T.transform_grid(grid, -theta, method=method)[0]
    x = T.transform_data(np.atleast_3d(z), theta, outsize, method=method).flatten()
    dfdz = T.gradient_space(grid, theta, method=method)[0]
    px = pz * dfdz
    px /= np.max(px)

    return z, pz, fz, x, px, fx, dfdz, grid, T, theta, outsize, method


# %%
from matplotlib import cm
from matplotlib.collections import LineCollection

fig, axs = plt.subplots(3,1, figsize=(5, 6), sharex=True, gridspec_kw=dict(height_ratios=[1,1.5,1]))
cmap = "plasma"
palette = cm.get_cmap(cmap)

def plot_linecolor(ax, x, y, z):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(z.min(), z.max())
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(z)
    lc.set_linewidth(5)
    ax.plot(x, y, alpha=0.1)
    line = ax.add_collection(lc)


# tess_size = 8
# mean = 0.3
# length_scale=0.04
# output_variance=1.

# tess_size = 6
# mean = 0.2
# length_scale=0.28
# output_variance=0.9

# tess_size = 24
# mean = 0.3
# length_scale = 0.2
# output_variance = 1

n = 1000
tess_size = 20
mean = 0.
length_scale = 0.2
output_variance = 1.
theta = None

z, pz, fz, x, px, fx, dfdz, grid, T, theta, outsize, method = get_values(n, tess_size, mean, length_scale, output_variance, theta)

from scipy.stats import gaussian_kde
plot_linecolor(axs[2], z, pz, pz)
plot_linecolor(axs[0], x, px, px)

n = 60
z, pz, fz, x, px, fx, dfdz, grid, T, theta, outsize, method = get_values(n, tess_size, mean, length_scale, output_variance, theta)

dt = 0.05
for t in np.arange(0, 1, dt):
    x = T.transform_data(np.atleast_3d(z), theta, outsize, method=method, time=dt).flatten()
    dfdz = T.gradient_space(np.atleast_3d(z), theta, method=method, time=dt)[0]
    px = pz * dfdz
    px /= np.max(px)

    for i in range(n):
        axs[1].plot([z[i], x[i]], [t, t+dt], color=palette(px[i]), alpha=0.6)
        
    z = x
    pz = px


for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])

axs[0].set_xlim(0,1)
axs[1].set_ylim(0,1)
axs[0].set_ylabel(r"Target density $p(x)$")
axs[1].set_ylabel(r"Time $t$")
axs[2].set_ylabel(r"Base density $q(z)$")
axs[2].set_xlabel(r"$z$")
axs[0].set_xlabel(r"$x$")

plt.tight_layout()
plt.savefig("nf_simple1.pdf", bbox_inches="tight")
plt.close()



# %% HERE NEW CLEAN ATTEMPT

n = 1000
tess_size = 20
mean = 0.
length_scale = 0.2
output_variance = 1.
theta = None

z, pz, fz, x, px, fx, dfdz, grid, T, theta, outsize, method = get_values(n, tess_size, mean, length_scale, output_variance, theta)

with plt.style.context('nature'):
    fig, axs = plt.subplots(1,3, figsize=(8.5/1.5,3/1.5), gridspec_kw={"width_ratios":[1,1,1]}, constrained_layout=True)

    bw = 0.035
    bins = int(1/bw)+1
    # axs[0].hist(z, bins=bins, density=True, color="gray", alpha=0.25)
    axs[0].plot(z, pz, color="black", lw=2)
    # sns.kdeplot(z, ax=axs[0], color="black", lw=2)
    axs[1].plot(grid, fz, color="#D90368", lw=2)
    axs[2].plot(x, px, color="black", lw=2)
    # axs[2].hist(x.flatten(), bins=bins, density=True, color="gray", alpha=0.25)
    # sns.kdeplot(x.flatten(), ax=axs[2], color="black", lw=2)

    axs[0].set_xlim(0,1)
    axs[2].set_xlim(0,1)
    axs[1].set_xlim(0,1)
    axs[1].set_ylim(0,1)

    for ax in axs:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

    fontsize = 7
    axs[0].set_title(r"Base distribution", fontsize=fontsize)
    axs[0].set_xlabel(r"$z$")
    axs[0].set_ylabel(r"$q(z)$")
    axs[1].set_title(r"Bijective function $f$", fontsize=fontsize)
    axs[1].set_xlabel(r"$z$")
    axs[1].set_ylabel(r"$x=f(z)$")
    axs[2].set_title(r"Target distribution", fontsize=fontsize)
    axs[2].set_xlabel(r"$x$")
    axs[2].set_ylabel(r"$p(x)$")

    plt.tight_layout()
    plt.savefig("nf_simple.pdf", bbox_inches="tight")
    plt.close()

# %%

import matplotlib.gridspec as gridspec
n = 1000
tess_size = 20
mean = 0.
length_scale = 0.2
output_variance = 1.
theta = None

z, pz, fz, x, px, fx, dfdz, grid, T, theta, outsize, method = get_values(n, tess_size, mean, length_scale, output_variance, theta)

with plt.style.context('nature'):

    fig = plt.figure(constrained_layout=True, figsize=(4,3))
    gs = fig.add_gridspec(4,5,width_ratios=[1,0.01,1,0.01,1])

    ax0 = fig.add_subplot(gs[1:3,0])
    ax1 = fig.add_subplot(gs[0:2,2])
    ax2 = fig.add_subplot(gs[2:,2])
    ax3 = fig.add_subplot(gs[1:3,4])
    axs = [ax0, ax1, ax2, ax3]


    # axs[0].hist(z, density=True, bins=20)
    axs[0].plot(z, pz, color="black", lw=2)
    # sns.kdeplot(z, ax=axs[0])
    axs[0].set_ylim(0.1,2)
    # sns.rugplot(z, ax=axs[0], color="black", alpha=0.1)

    axs[1].plot(grid, fz, color="#D90368", lw=2)
    axs[2].plot(grid, dfdz, color="#00CC66", lw=2)

    # axs[3].hist(x, density=True, bins=20)
    axs[3].plot(x, px, color="black", lw=2)
    # sns.kdeplot(x, ax=axs[3], color="black", lw=2)
    # sns.rugplot(x, ax=axs[3], color="black", lw=2, alpha=0.1)

    axs[0].set_xlim(0,1)
    axs[2].set_xlim(0,1)
    axs[3].set_xlim(0,1)
    axs[1].set_xlim(0,1)
    axs[1].set_ylim(0,1)

    fontsize = 7
    axs[0].set_title(r"Base distribution", fontsize=fontsize)
    axs[0].set_xlabel(r"$z$")
    axs[0].set_ylabel(r"$q(z)$")
    axs[1].set_title(r"Bijective function $f$", fontsize=fontsize)
    axs[1].set_xlabel(r"$z$")
    axs[1].set_ylabel(r"$x=f(z)$")
    axs[2].set_title(r"Change of variable $\frac{df^{-1}}{dz}$", fontsize=fontsize)
    axs[2].set_xlabel(r"$z$")
    # axs[2].set_ylabel(r"$p(x)=q(z)  \frac{df^{-1}}{dz}$")
    axs[2].set_ylabel(r"$(df/dz)^{-1}$")
    axs[3].set_title(r"Target distribution", fontsize=fontsize)
    axs[3].set_xlabel(r"$x$")
    axs[3].set_ylabel(r"$p(x)$")

    for ax in axs:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

    # plt.tight_layout()
    plt.savefig("nf_simple2.pdf", bbox_inches="tight")
    plt.close()


# %%

n = 1000
tess_size = 20
mean = 0.
length_scale = 0.2
output_variance = 1.

tess_size = 8
mean = -2
length_scale = 2
output_variance = 2.

theta = None

z, pz, fz, x, px, fx, dfdz, grid, T, theta, outsize, method = get_values(n, tess_size, mean, length_scale, output_variance, theta)

with plt.style.context('nature'):
    # fig, axs = plt.subplots(1,4, figsize=(11/1.4,3/1.4), gridspec_kw={"width_ratios":[1,1,1,1]}, constrained_layout=True)
    # fig, axs = plt.subplots(1,4, figsize=(11,3), gridspec_kw={"width_ratios":[1,1,1,1]}, constrained_layout=True)
    
    # fig, axs = plt.subplots(2,2, figsize=(4,4), gridspec_kw={"width_ratios":[1,1]}, constrained_layout=True)
    # axs = axs.flatten()
    # axs = axs[[0,2,3,1]]
    
    fig, axs = plt.subplots(2,3, figsize=(6,4), gridspec_kw={"width_ratios":[1,1,1]}, constrained_layout=True)
    axs = axs.flatten()
    for k, ax in enumerate(axs):
        if k in [0,1,4,2]:
            continue
        ax.axis("off")
    axs = axs[[0,1,4,2]]


    # axs[0].hist(z, density=True, bins=20)
    axs[0].plot(z, pz, color="black", lw=2)
    # sns.kdeplot(z, ax=axs[0])
    # sns.rugplot(z, ax=axs[0], color="black", alpha=0.1)

    axs[1].plot(grid, fz, color="#D90368", lw=2)
    axs[2].plot(grid, dfdz, color="#00CC66", lw=2)

    # axs[3].hist(x, density=True, bins=20)
    axs[3].plot(x, px, color="black", lw=2)
    # sns.kdeplot(x, ax=axs[3])
    # sns.rugplot(x, ax=axs[3], alpha=0.1)

    axs[0].set_xlim(0,1)
    axs[2].set_xlim(0,1)
    axs[3].set_xlim(0,1)
    axs[1].set_xlim(0,1)
    axs[1].set_ylim(0,1)


    fontsize = 9
    axs[0].set_title(r"Base distribution", fontsize=fontsize)
    # axs[0].set_xlabel(r"$z$")
    # axs[0].set_ylabel(r"$q(z)$")
    axs[0].text(0.5, 0.5, r"$q(z)$", fontsize=10, transform=axs[0].transAxes, ha="center", va="center")
    axs[1].set_title(r"Bijective function $f$", fontsize=fontsize)
    # axs[1].set_xlabel(r"$z$")
    # axs[1].set_ylabel(r"$x=f(z)$")
    axs[1].text(0.5, 0.5, r"$x=f(z)$", fontsize=10, transform=axs[1].transAxes, ha="center", va="center")
    # axs[2].set_title(r"Change of variable $\frac{df^{-1}}{dz}$", fontsize=fontsize)
    # axs[2].set_xlabel(r"$z$")
    # axs[2].set_ylabel(r"$p(x)=q(z) \times \frac{df^{-1}}{dz}$")
    # axs[2].text(0.5, 0.5, r"Change of variable", fontsize=10, transform=axs[2].transAxes, ha="center", va="center")
    axs[2].text(0.5, 0.5, r"$\dfrac{df(z)}{dz}$", fontsize=10, transform=axs[2].transAxes, ha="center", va="center")
    axs[3].set_title(r"Target distribution", fontsize=fontsize)
    # axs[3].set_xlabel(r"$x$")
    # axs[3].set_ylabel(r"$p(x)$")
    axs[3].text(0.5, 0.5, r"$p(x) = q(z) \times \dfrac{df}{dz}$", fontsize=10, transform=axs[3].transAxes, ha="center", va="center")

    for ax in axs:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

    # plt.tight_layout()
    plt.savefig("nf_simple5.pdf", bbox_inches="tight")
    # plt.savefig("nf_simple4.pdf", bbox_inches="tight")
    plt.close()



# %%


import matplotlib.gridspec as gridspec

n = 100
tess_size = 5
mean = -0.1
length_scale = 1.8
output_variance = 1.2

mean = 0.3
length_scale = 0.38
output_variance = 0.6

theta = None
# theta = np.array([[0.2,1,0,1]])
# theta = np.array([[0.5,0.3,-0.4,1]])

z, pz, fz, x, px, fx, dfdz, grid, T, theta, outsize, method = get_values(n, tess_size, mean, length_scale, output_variance, theta)

with plt.style.context('nature'):
    fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(7/1.25, 3/1.25), 
        gridspec_kw=dict(width_ratios=[1,0.1,1]))

    which = np.linspace(0,len(grid)-1,tess_size+1).astype(int)
    axs[0].plot(grid, fz, color="#D90368", lw=2, label="Forward")
    axs[0].plot(grid, fx, color="#FF8F00", lw=2, label="Inverse")
    axs[0].scatter(grid[which], fz[which], color="black", lw=2, label="Knots", zorder=10)
    axs[0].plot([0,0],[1,0], color="gray", lw=1, ls="--")
    axs[0].plot([0,1],[1,1], color="gray", lw=1, ls="--")
    axs[0].plot([0,1],[0,0], color="gray", lw=1, ls="--")
    axs[0].plot([1,1],[0,1], color="gray", lw=1, ls="--")
    
    h = 0.16
    axs[0].plot([0,-h],[0,-h], color="#D90368", lw=2)
    axs[0].plot([1,1+h],[1,1+h], color="#D90368", lw=2)
    axs[0].legend()
    axs[0].set_xlim(-0.2, 1.2)
    axs[0].set_ylim(-0.2, 1.2)
    axs[0].set_xticks([0,0.5,1])
    axs[0].set_yticks([0,0.5,1])
    axs[0].set_xticklabels([r"-$B$",0,r"$B$"])
    axs[0].set_yticklabels([r"-$B$",0,r"$B$"])

    xp = grid
    xp = np.insert(xp, [0,0], [-h,0])
    xp = np.append(xp, [1,1+h])
    yp = 1/dfdz
    yp = np.insert(yp, [0,0], [1,1])
    yp = np.append(yp, [1,1])
    axs[2].plot(xp, yp, color="#00CC66", lw=2)
    # axs[2].scatter(xp[which+2], yp[which+2], color="black", lw=1, label="Knots", zorder=10)
    axs[2].fill_between(xp, yp, color="#00CC66", alpha=0.1)
    axs[2].set_xlim(0, 1)
    axs[2].set_ylim(0, None)
    axs[2].set_xlim(-h, 1+h)
    axs[2].set_xticklabels([r"-$B$",0,r"$B$"])
    axs[2].set_xticks([0,0.5,1])
    axs[2].set_yticks([0,1,2,3])
    # axs[1].scatter(grid[which], 1/dfdz[which], color="#00CC66", lw=2)

    axs[0].set_xlabel(r"$z$", fontsize=8)
    axs[0].set_ylabel(r"$h(z)$", fontsize=8)
    axs[2].set_xlabel(r"$z$", fontsize=8)
    # axs[2].set_ylabel(r"$\frac{\partial f}{\partial z}$", fontsize=8)
    axs[2].set_ylabel(r"$\partial h/\partial z$", fontsize=8)

    # for ax in axs:
    #     ax.set_yticklabels([])
    #     ax.set_xticklabels([])
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    axs[1].axis("off")

    plt.tight_layout()
    plt.savefig("nf_simple3.pdf", bbox_inches="tight")
    plt.close()

# %%

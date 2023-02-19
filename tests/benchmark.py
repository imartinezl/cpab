# %%

import sys

sys.path.insert(0, "..")

import time
import timeit
import numpy as np
import torch
import matplotlib.pyplot as plt
import cpab

import torch.autograd.profiler as profiler
import torch.utils.benchmark as benchmark
import pandas as pd
import pickle
import seaborn as sns
import matplotlib

# %% SETUP
tess_size = 32
backend = "pytorch"  # ["pytorch", "numpy"]
device = "cpu"  # ["cpu", "gpu"]
zero_boundary = False
use_slow = False
outsize = 200
batch_size = 1
method = "closed_form"
# method = "numeric"
basis = "rref"
N = 6

T = cpab.Cpab(tess_size, backend, device, zero_boundary, basis)
T.params.use_slow = use_slow

grid = T.uniform_meshgrid(outsize)
theta = T.sample_transformation(batch_size)
# theta = T.identity(batch_size, epsilon=1.0)

grid_t = T.transform_grid(grid, theta, method)
# grid_t = T.transform_grid_ss(grid, theta / 2**N, method, N=N)

def nonuniform_meshgrid(outsize):
    x = torch.rand(outsize-1)
    total = x.sum()
    x = torch.cumsum(x, axis=0) / total
    return torch.cat([torch.zeros(1), x])

grid2 = nonuniform_meshgrid(outsize)
grid_t2 = T.transform_grid(grid2, theta, method)
# grid_t2 = T.transform_grid_ss(grid2, theta / 2**N, method, N=N)

plt.figure()
plt.plot(grid, grid_t.T)
plt.plot(grid2, grid_t2.T)

plt.figure()
plt.plot(grid, (grid_t2 - grid_t).T)


# %% PYTORCH BENCHMARK
t0 = benchmark.Timer(
    stmt="""
    theta_grad = torch.autograd.Variable(theta, requires_grad=True)
    grid_t = T.transform_grid(grid, theta_grad, method)
    loss = torch.norm(grid_t)
    loss.backward() 
    """, 
    globals={"T": T, "grid": grid, "theta": theta, "method": method}
)
# t0.timeit(1)
t0.blocked_autorange(min_run_time=0.5)
# %% CPROFILE

import cProfile

cProfile.run(
    """
theta_grad = torch.autograd.Variable(theta, requires_grad=True)
for i in range(1000): 
    grid_t = T.transform_grid(grid, theta_grad, method)
    # loss = torch.norm(grid_t)
    # loss.backward()
""",
    sort="cumtime",
)
# %% YEP + PPROF
import yep

# torch.set_num_threads(1)

theta_grad = torch.autograd.Variable(theta, requires_grad=True)
yep.start("profile.prof")
for i in range(100):
    grid_t = T.transform_grid(grid, theta_grad, method)
    # loss = torch.norm(grid_t)
    # loss.backward()

yep.stop()

# %% TIMEIT

repetitions = 1000
n = 10
timing = timeit.Timer(
    lambda: T.transform_grid(grid, theta),
    # setup="gc.enable()"
).repeat(repetitions, n)
print("Time: ", np.mean(timing) / n, "+-", np.std(timing) / np.sqrt(n))

# %% PYTORCH PROFILER

with profiler.profile(with_stack=True, profile_memory=True) as prof:
    T.transform_grid(grid, theta, method)

print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=50))
# prof.export_chrome_trace("trace.json")

# %% snakeviz
# %prun -D program.prof T.transform_grid(grid, theta)

# %%

from itertools import product

results = []

num_threads_arr = [1] # [1, 2, 4]

backend_arr = ["pytorch"] # ["pytorch", "numpy"]
device_arr = ["cpu"] # ["cpu", "gpu"]
method_arr = ["closed_form"] # ["closed_form", "numeric"]
use_slow_arr = [False] # [True, False]
zero_boundary_arr = [True] # [True, False]

tess_size_arr = [50]
outsize_arr = [1000]
batch_size_arr = [200]

for (
    backend,
    device,
    method,
    use_slow,
    zero_boundary,
    tess_size,
    outsize,
    batch_size,
) in product(
    backend_arr,
    device_arr,
    method_arr,
    use_slow_arr,
    zero_boundary_arr,
    tess_size_arr,
    outsize_arr,
    batch_size_arr,
):
    # SETUP
    T = cpab.Cpab(tess_size, backend, device, zero_boundary)
    T.params.use_slow = use_slow

    grid = T.uniform_meshgrid(outsize)
    theta = T.identity(batch_size, epsilon=1)

    label = "CPAB: backend, device, method, use_slow, zero_boundary, tess_size, outsize, batch_size"
    # sub_label = f"[{backend}, {device}, {method}, {'slow' if use_slow else 'fast'}, {'zero_boundary' if zero_boundary else 'no_zero_boundary'}, {tess_size}, {outsize}, {batch_size}]"
    sub_label = f"[{backend}, {device}, {method}, {use_slow}, {zero_boundary}, {tess_size}, {outsize}, {batch_size}]"
    print(sub_label)
    for num_threads in num_threads_arr:
        repetitions = 1

        # FORWARD
        t0 = benchmark.Timer(
            stmt=
            """
            grid_t = T.transform_grid(grid, theta, method)
            """,
            globals={"T": T, "grid": grid, "theta": theta, "method": method},
            num_threads=num_threads,
            label=label,
            sub_label=sub_label,
            description="Forward",
        )
        # results.append(t0.timeit(repetitions))
        results.append(t0.blocked_autorange(min_run_time=0.5))
        # results.append(t0.adaptive_autorange())

        # BACKWARD
        t1 = benchmark.Timer(
            stmt=
            """
            theta_grad = torch.autograd.Variable(theta, requires_grad=True)
            grid_t = T.transform_grid(grid, theta_grad, method)
            loss = torch.norm(grid_t)
            loss.backward()            
            """,
            globals={"T": T, "grid": grid, "theta": theta, "method": method},
            num_threads=num_threads,
            label=label,
            sub_label=sub_label,
            description="Backward",
        )
        # results.append(t1.timeit(repetitions))
        results.append(t1.blocked_autorange(min_run_time=0.5))
        # results.append(t1.adaptive_autorange())


# %%
compare = benchmark.Compare(results)
compare.trim_significant_figures()
compare.colorize()
compare.print()

# %% RESULTS TO LATEX

df = [
    pd.DataFrame({
        'experiment': t.as_row_name.replace('[', '').replace(']', ''), 
        'description': t.task_spec.description,
        'threads': t.task_spec.num_threads,
        # 'time': t.raw_times,
        'time_mean': np.mean(t.raw_times),
        'time_std': np.std(t.raw_times),
        }, index=[0])
    for t in results
]
df = pd.concat(df, ignore_index=True)


header = ['Backend', 'Device', 'Method', 'Speed', 'Boundary', 'Tess Size', 'Grid Size', 'Batch Size']
parameters = pd.DataFrame(df["experiment"].str.split(',', expand=True).values, columns=header)

df = pd.concat([parameters, df], axis=1).drop(columns=['experiment'])
# df.to_latex("test.tex", index=False, multirow=True)
df

# %%


# %% RESULTS TO PLOT

df = [
    pd.DataFrame({
        'experiment': t.as_row_name, 
        'description': t.task_spec.description,
        'threads': t.task_spec.num_threads,
        'time': t.raw_times})
    for t in results
]
df = pd.concat(df, ignore_index=True)
df['experiment_id'] = df.groupby('experiment', sort=False).ngroup().apply(str)

n = pd.unique(df.experiment_id)
exps = pd.unique(df.experiment)
caption = '\n'.join([k + ": " + exps[int(k)] for k in n])

header = ['Backend', 'Device', 'Method', 'Speed', 'Boundary', 'Tess Size', 'Grid Size', 'Batch Size']
cell_text = [e.replace('[','').replace(']','').split(', ') for e in exps]

vlen = np.vectorize(len)
w = np.max(vlen(cell_text + [header]), axis=0)

# %%

with sns.axes_style("whitegrid"):
    g = sns.catplot(
        x="time", y="experiment_id", 
        hue="threads", col="description",
        data=df, kind="box", ci=None, sharex=True,
        fliersize=2, linewidth=1, width=0.75)
    sns.despine(top=False, right=False, left=False, bottom=False)
    plt.xticks(np.logspace(-10,-1, num=10))
    # plt.figtext(0, -0.1, caption, wrap=True, 
    #     verticalalignment='top', horizontalalignment='left', fontsize=10)

    table = plt.table(
        cellText=cell_text, 
        rowLabels=n, 
        colLabels=header, 
        colWidths = w,
        cellLoc='center',
        loc='bottom',
        # fontsize=50
        bbox=[-1.0,-0.5, 1.2, 0.35]
        )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    # table.auto_set_column_width(n)
    # table.scale(1, 1)

    for ax in g.axes[0]:
        ax.set_xscale('log')
        ax.grid(axis="x", which="minor", ls="--", c='gray', alpha=0.2)

plt.savefig('example.png')


# %% IMPORT RESULTS


with open("benchmark_cpab_new.pickle", 'rb') as f:
    results = pickle.load(f)

df = [
    pd.DataFrame({
        'experiment': t.as_row_name.replace('[', '').replace(']', ''), 
        'task': t.task_spec.description,
        'threads': t.task_spec.num_threads,
        'time_mean': [t.mean],
        'time_median': t.median,
        'time_iqr': t.iqr,
        # 'time': t.raw_times,
    })
    for t in results
]
df = pd.concat(df, ignore_index=True)
header = ['Backend', 'Device', 'Method', 'Speed', 'Boundary', 'Tess Size', 'Grid Size', 'Batch Size']
parameters = pd.DataFrame(df["experiment"].str.split(', ', expand=True).values, columns=header)

df["Library"] = "cpab"
df_cpab = pd.concat([parameters, df], axis=1).drop(columns=['experiment'])


with open("benchmark_libcpab_new.pickle", 'rb') as f:
    results = pickle.load(f)

df = [
    pd.DataFrame({
        'experiment': t.as_row_name.replace('[', '').replace(']', ''), 
        'task': t.task_spec.description,
        'threads': t.task_spec.num_threads,
        'time_mean':  [t.mean],
        'time_median': t.median,
        'time_iqr': t.iqr,
        # 'time': t.raw_times,
    })
    for t in results
]
df = pd.concat(df, ignore_index=True)
header = ['Backend', 'Device', 'Speed', 'Boundary', 'Tess Size', 'Grid Size', 'Batch Size']
parameters = pd.DataFrame(df["experiment"].str.split(', ', expand=True).values, columns=header)

df["Library"] = "libcpab"
df["Method"] = "libcpab"
df_libcpab = pd.concat([parameters, df], axis=1).drop(columns=['experiment'])

df = pd.concat([df_cpab, df_libcpab])
df = df[df.Boundary == "True"].reset_index()

# %%

with sns.axes_style("ticks"):
    sns.set_context("paper")
    fig = plt.figure(constrained_layout = True)
    g = sns.catplot(
        data=df, x="task", y="time_median", 
        hue="Method", row="Device", col="Speed", 
        kind="bar", sharey=False, ci=False, 
        errwidth=1, capsize=0.1,
        facet_kws={
            "margin_titles": True,
            "despine": False
        },
        palette=["#16BFE0", "#E0CF2B", "#E0005D"]
    )

    g.set(yscale="log")
    for key, ax in g.axes_dict.items():
        ax.grid(True, axis="y", which="both", ls="--", c='gray', alpha=0.3)
    g.set_xlabels("")
    g.set_ylabels("Time (s)")
    plt.savefig("benchmark_times.pdf")

# %% LOG SCALE

sns.set_style("whitegrid", {'axes.grid' : False})
sns.set_context("paper")
g = sns.catplot(
    data=df[df.Speed == "False"], 
    x="task", y="time_median", 
    hue="Method", col="Device", 
    kind="bar", sharey=True,
    facet_kws={
        "margin_titles": True,
        "despine": False
    },
    legend=False,
    ci=False, 
    palette=["#16BFE0", "#E0CF2B", "#E0005D"],
    height=4, aspect=0.6,
)


new_labels = ['Closed form (ours)', 'Numeric (ours)', 'Libcpab']
leg = plt.legend(loc='upper center')
# leg.set_title("Method")
leg._legend_box.align = "left"
for t, l in zip(leg.texts, new_labels):
    t.set_text(l)


g.set(yscale="log")
plt.ylim((5e-5, 2))
for key, ax in g.axes_dict.items():
    ax.grid(True, axis="y", which="both", ls="--", c='gray', alpha=0.3)
    ax.tick_params(pad=0)
    ax.set_xticklabels(["Solve ODE\n$\phi$","Gradient\n$\partial \\phi/\partial \\theta$"])
g.set_xlabels("")
g.set_ylabels("")
# plt.yticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0], labels=["$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "$1$"])
plt.yticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0], labels=["$0.1$", "$1$", "$10$", "$100$", "$1000$"])

plt.text(-2.7,2.6,"Time (ms)", ha="center")
g.col_names = ["CPU", "GPU"]
g.set_titles(col_template="{col_name}", fontsize=18)

for index, row in df[df.Speed == "False"].iterrows():
    hx = -0.27
    hy = 0.0
    hz = 0.265
    x = (row.task == "Backward")*1 + (row.Method == "closed_form")*hx + (row.Method == "numeric")*hy + (row.Method == "libcpab")*hz
    # g.axes_dict[row.Device].text(x,row.time_median, round(row.time_median, 5), ha="center")
    # g.axes_dict[row.Device].text(x,row.time_median, str(round(row.time_median*1000, 2)) + "$\mu s$", ha="center", fontsize=9, rotation=0)
    units = "\nμs " if index == 0 else ""
    label = row.time_median*1000
    label = round(label, 2) if label < 1 else (int(round(label,0)) if label > 10 else round(label,1))
    g.axes_dict[row.Device].text(x,row.time_median, str(label),# + " " + units,
        va="bottom", ha="center", rotation=0, fontsize=10, fontname="Roboto Medium", zorder=100)


plt.savefig("benchmark_times_compiled.pdf", bbox_inches="tight")
# %% LINEAR SCALE

sns.set_style("whitegrid", {'axes.grid' : False})
sns.set_context("paper")
g = sns.catplot(
    data=df[df.Speed == "False"], 
    x="task", y="time_median", 
    hue="Method", col="Device", 
    kind="bar", sharey=False,
    facet_kws={
        "margin_titles": True,
        "despine": False
    },
    legend=False,
    ci=False, 
    palette=["#16BFE0", "#E0CF2B", "#E0005D"],
    height=4, aspect=0.7,
)   


new_labels = ['Closed form (ours)', 'Numeric (ours)', 'Libcpab']
leg = plt.legend(loc='upper center')
# leg.set_title("Method")
leg._legend_box.align = "left"
for t, l in zip(leg.texts, new_labels):
    t.set_text(l)


# g.set(yscale="log")
# plt.ylim((0, 1.2))
ylim = {"cpu": 1.15, "gpu": 0.21}
yticks = {"cpu": [0, 0.2, 0.4, 0.6, 0.8, 1], "gpu": [0, 0.05, 0.1, 0.15]}
ylabels = {"cpu": [0, 200, 400, 600, 800, 1000], "gpu": [0, 50, 100, 150]}
for key, ax in g.axes_dict.items():
    ax.grid(True, axis="y", which="both", ls="--", c='gray', alpha=0.3)
    ax.set_ylim((0, ylim[key]))
    ax.set_yticks(yticks[key])
    ax.set_yticklabels(ylabels[key])
    ax.tick_params(pad=0)
    ax.set_xticklabels(["Solve ODE\n$\phi$","Gradient\n$\partial \\phi/\partial \\theta$"])
    if key=="gpu":
        ax.yaxis.tick_right()
        ax.yaxis.set_ticks_position('none')

plt.tight_layout()

g.set_xlabels("")
g.set_ylabels("")
# plt.yticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0], labels=["$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "$1$"])
# plt.yticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0], labels=["$0.1$", "$1$", "$10$", "$100$", "$1000$"])


plt.text(-2.9,0.2, "Time\n(ms)", ha="center", fontsize=10, fontname="Roboto Medium")
plt.text(1.75,0.2, "Time\n(ms)", ha="center", fontsize=10, fontname="Roboto Medium")
g.col_names = ["CPU", "GPU"]
g.set_titles(col_template="{col_name}", fontsize=18)

for index, row in df[df.Speed == "False"].iterrows():
    hx = -0.27
    hy = 0.0
    hz = 0.265
    x = (row.task == "Backward")*1 + (row.Method == "closed_form")*hx + (row.Method == "numeric")*hy + (row.Method == "libcpab")*hz
    # g.axes_dict[row.Device].text(x,row.time_median, round(row.time_median, 5), ha="center")
    # g.axes_dict[row.Device].text(x,row.time_median, str(round(row.time_median*1000, 2)) + "$\mu s$", ha="center", fontsize=9, rotation=0)
    units = "\nμs " if index == 0 else ""
    label = row.time_median*1000
    label = round(label, 1) if label < 1 else (int(round(label,0)) if label > 10 else round(label,1))
    y = row.time_median
    # y += 0.02 if (row.Device == "gpu") and (row.task == "Forward") and (row.Method == "numeric") else 0
    g.axes_dict[row.Device].text(x,y, str(label),# + " " + units,
        va="bottom", ha="center", rotation=0, fontsize=10, fontname="Roboto Medium", zorder=100)

plt.savefig("benchmark_times_compiled_linear.pdf", bbox_inches="tight")


# %% LINEAR SCALE

sns.set_style("whitegrid", {'axes.grid' : False})
sns.set_context("paper")
g = sns.catplot(
    data=df[(df.Speed == "False") & (df.Method != "numeric")], 
    x="task", y="time_median", 
    hue="Method", col="Device", 
    kind="bar", sharey=False,
    facet_kws={
        "margin_titles": True,
        "despine": False
    },
    legend=False,
    ci=False, 
    palette=["#16BFE0",  "#E0005D", "#E0CF2B"],
    height=4, aspect=0.7,
)   


new_labels = ['Closed form (ours)',  'Libcpab (Detlefsen, 2018)', 'Numeric (ours)']
leg = plt.legend(loc='upper center')
# leg.set_title("Method")
leg._legend_box.align = "left"
for t, l in zip(leg.texts, new_labels):
    t.set_text(l)


# g.set(yscale="log")
# plt.ylim((0, 1.2))
ylim = {"cpu": 0.3, "gpu": 0.20}
yticks = {"cpu": [0, 0.05, 0.10, 0.15, 0.20, 0.25], "gpu": [0, 0.05, 0.1, 0.15]}
ylabels = {"cpu": [0, 50, 100, 150, 200, 250], "gpu": [0, 50, 100, 150]}
for key, ax in g.axes_dict.items():
    ax.grid(True, axis="y", which="both", ls="--", c='gray', alpha=0.3)
    ax.set_ylim((0, ylim[key]))
    ax.set_yticks(yticks[key])
    ax.set_yticklabels(ylabels[key])
    ax.tick_params(pad=0)
    ax.set_xticklabels(["Solve ODE\n$\phi$","Gradient\n$\partial \\phi/\partial \\theta$"])
    if key=="gpu":
        ax.yaxis.tick_right()
        ax.yaxis.set_ticks_position('none')

plt.tight_layout()

g.set_xlabels("")
g.set_ylabels("")
# plt.yticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0], labels=["$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "$1$"])
# plt.yticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0], labels=["$0.1$", "$1$", "$10$", "$100$", "$1000$"])


plt.text(-2.85,0.182, "Time\n(ms)", ha="center", fontsize=10, fontname="Roboto Medium")
plt.text(1.73,0.182, "Time\n(ms)", ha="center", fontsize=10, fontname="Roboto Medium")
g.col_names = ["CPU", "GPU"]
g.set_titles(col_template="{col_name}", fontsize=18)

for index, row in df[(df.Speed == "False") & (df.Method != "numeric")].iterrows():
    hx = -0.19
    hy = 0.0
    hz = 0.19
    x = (row.task == "Backward")*1 + (row.Method == "closed_form")*hx + (row.Method == "numeric")*hy + (row.Method == "libcpab")*hz
    # g.axes_dict[row.Device].text(x,row.time_median, round(row.time_median, 5), ha="center")
    # g.axes_dict[row.Device].text(x,row.time_median, str(round(row.time_median*1000, 2)) + "$\mu s$", ha="center", fontsize=9, rotation=0)
    units = "\nμs " if index == 0 else ""
    label = row.time_median*1000
    label = round(label, 2) if label < 1 else (int(round(label,0)) if label > 10 else round(label,1))
    y = row.time_median
    # y += 0.02 if (row.Device == "gpu") and (row.task == "Forward") and (row.Method == "numeric") else 0
    g.axes_dict[row.Device].text(x,y, str(label),# + " " + units,
        va="bottom", ha="center", rotation=0, fontsize=10, fontname="Roboto Medium", zorder=100)

plt.savefig("benchmark_times_compiled_linear.pdf", bbox_inches="tight")

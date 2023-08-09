# %%

# Import cpab library
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("talk")
plt.rcParams["figure.figsize"] = (10, 8)
plt.rcParams["figure.constrained_layout.use"] = True
import cpab
path = "./docs/source/_static/figures/"

# Transformation instance 
T = cpab.Cpab(tess_size=5, backend="numpy", device="cpu", zero_boundary=True, basis="qr")

# Generate grid
grid = T.uniform_meshgrid(100)

# Transformation parameters
theta = T.identity(epsilon=1)

# Transform grid
grid_t = T.transform_grid(grid, theta)

# Visualize tesselation
T.visualize_tesselation()
plt.savefig(path + "visualize_tesselation.png")

# Visualize velocity field
T.visualize_velocity(theta)
plt.savefig(path + "visualize_velocity.png")

# Visualize grid deformation
T.visualize_deformgrid(theta)
plt.savefig(path + "visualize_deformgrid.png")

# Visualize transformation gradient w.r.t parameters
T.visualize_gradient(theta)
plt.savefig(path + "visualize_gradient.png")


# Data generation
import numpy as np
batch_size = 1
width = 50
channels = 2
a = np.zeros((batch_size, channels))
b = np.ones((batch_size, channels)) * 2 * np.pi
x = np.linspace(a, b, width, axis=1)
data = np.sin(x)

# Transform data
T.visualize_deformdata(data, theta)
plt.savefig(path + "visualize_deformdata.png")


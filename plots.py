import numpy as np
import matplotlib.pyplot as plt
import logging
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.collections import LineCollection


def plot_mu_var(mu, var, dir):
    if len(mu) == 0 or len(var) == 0:
        print("Warning: mu or var is empty!")
        return

    mu_values = np.linspace(-0.15, 0.15, 100)  # Mean around 0
    sigma2_values = np.linspace(0.2, 1.2, 100)  # Variance around 0.33
    MU, SIGMA2 = np.meshgrid(mu_values, sigma2_values)
    Z = np.exp(-((MU - 0.03)**2 / (2 * 0.025**2) + (SIGMA2 - 0.5)**2 / (2 * 0.015**2)))

    points = np.array([mu, var]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create 2D histogram to get density
    H, xedges, yedges = np.histogram2d(mu, var, bins=10)

    # Function to get bin index
    def get_bin_value(x, y):
        xi = np.searchsorted(xedges, x) - 1
        yi = np.searchsorted(yedges, y) - 1
        if 0 <= xi < H.shape[0] and 0 <= yi < H.shape[1]:
            return H[xi, yi]
        else:
            return 0

    # Get density at each segment midpoint
    midpoints = 0.5 * (points[:-1].squeeze() + points[1:].squeeze())
    densities = np.array([get_bin_value(x, y) for x, y in midpoints])

    # Normalize densities
    norm = Normalize(vmin=densities.min(), vmax=densities.max())

    # Create a colormap (e.g., darker = more frequent)
    cmap = plt.cm.plasma

    # Build colored segments
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(densities)
    lc.set_linewidth(2)

    plt.figure(figsize=(6, 6))
    plt.contour(MU, SIGMA2, Z, levels=15, cmap='rainbow')  # Contour plot
    plt.gca().add_collection(lc)
    plt.colorbar(lc, label="Density")

    plt.xlabel(r'$\mu$')
    plt.ylabel(r'$\sigma^2$')
    plt.grid()
    plt.savefig(dir)

    logging.info(f"Figure saved at {dir}")

def plot_steps(steps, dir):

    t_values = [1.0, 0.75, 0.5, 0.25, 0.0]  # Time labels

    # Create a 1x5 subplot
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))

    for i, ax in enumerate(axes):
        ax.imshow(steps[i].detach().cpu().permute(1, 2, 0).numpy())
        ax.set_title(f"t = {t_values[i]:.2f}")
        ax.axis("off")  # Hide axes

    fig.savefig(dir, dpi=300, bbox_inches="tight")
    logging.info(f'Figure saved at {dir}')

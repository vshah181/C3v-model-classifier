import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from user_options_local import GRADIENT, INTERCEPT


def plot_phase_diagram(fname, delim=None, plotname="phase_diagram.pdf"):
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Arial"
    plt.rcParams["mathtext.it"] = "Arial:italic"
    plt.rcParams["mathtext.bf"] = "Arial:bold"
    plt.rcParams['svg.fonttype'] = 'none'

    # Prepare the data for plotting
    data = np.loadtxt(fname, delimiter=delim, comments="#")

    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    x_vals = np.unique(x)
    y_vals = (np.unique(y) * GRADIENT) + INTERCEPT
    nx = len(x_vals)
    ny = len(y_vals)

    grid = np.empty((nx, ny), dtype=int)

    iz = 0
    for ix, xpoint in enumerate(x_vals):
        for iy, ypoint in enumerate(y_vals):
            grid[iy, ix] = int(z[iz])
            iz += 1

    # Make the figure and the axes
    colours = ["#999999", "#6463FA", "#FF6978", "#2EAA7F", "#FFB363"]
    cmap = ListedColormap(colours)
    bounds = np.arange(-0.5, 5, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    extent = [np.min(x_vals), np.max(x_vals), np.min(y_vals), np.max(y_vals)]
    ax.imshow(grid, origin="lower", extent=extent, cmap=cmap, norm=norm,
              interpolation="nearest", aspect="auto")
    ax.set_xlabel(r"$c \prime$")
    if (INTERCEPT > 0):
        sign = r" + "
    else:
        sign = r" - "
    y_label_str = r"$a_{12} = " + f"{GRADIENT:.3f}" + r"\times a_{5}" + sign
    y_label_str += f"{abs(INTERCEPT):.3f}" + r"$"
    ax.set_ylabel(y_label_str)

    value_labels = {
        0: "unknown",
        1: "band insulator",
        2: "weak TI",
        3: "strong TI",
        4: "Weyl semimetal"
    }

    legend_handles = []
    for val, label in value_labels.items():
        patch = Patch(color=colours[val], label=label)
        legend_handles.append(patch)

    ax.legend(handles=legend_handles, title="Phase", loc='center left',
              bbox_to_anchor=(1, 0.5), frameon=False)

    plt.tight_layout()
    #plt.show()
    plt.savefig(plotname)


def main():
    plot_phase_diagram("phase_diagram.csv", ",", "phase_diagram.pdf")


if __name__ == "__main__":
    main()

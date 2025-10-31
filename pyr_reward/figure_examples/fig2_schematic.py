import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import matplotlib as mpl

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
plt.rc('font', size=16)
plt.rcParams["font.family"] = "Arial"

# --- Example data ---
epochs = ["Epoch 1", "Epoch 2", "Epoch 3", "Epoch 4"]
n_cells = 4
cell_activity = np.array([
    [1, 1, 0, 0],  # Cell 1
    [1, 1, 1, 0],  # Cell 2
    [1, 1, 1, 1],  # Cell 3
    [1, 1, 1, 0],  # Cell 4
])

# Example tuning curve peaks for each cell in active epochs
# Each row = cell, each column = epoch (x-location of peak)
tuning_peaks = np.array([
    [0.3, 0.35, np.nan, np.nan],
    [0.5, 0.55, 0.6, np.nan],
    [0.2, 0.25, 0.3, 0.35],
    [0.7, 0.65, 0.6, np.nan],
])

# --- Plot setup ---
fig, ax = plt.subplots(figsize=(7,4))
ax.set_xlim(-0.5, len(epochs)-0.5)
ax.set_ylim(-0.5, n_cells-0.5)
ax.set_xticks(range(len(epochs)))
ax.set_xticklabels(epochs)
ax.set_yticks(range(n_cells))
ax.set_yticklabels([f"Cell {i+1}" for i in range(n_cells)])
ax.invert_yaxis()
ax.set_title("Cartoon tuning curves across epochs")

colors = {1: "indigo", 0: "lightgray"}

# --- Function to draw tuning curve ---
def draw_tuning_curve(ax, x_center, y_center, peak_pos, active=1):
    if np.isnan(peak_pos) or active==0:
        # gray silent tuning
        ax.plot([x_center-0.15, x_center, x_center+0.15],
                [y_center, y_center, y_center],
                color='lightgray', lw=2)
    else:
        # cartoon bell-shaped tuning curve
        x = np.linspace(-0.15, 0.15, 50) + x_center
        y = np.exp(-((x - (x_center+peak_pos*0.3))**2)/(2*0.005)) + y_center
        ax.plot(x, y, color=colors[active], lw=2)

# --- Draw all cells ---
for i in range(n_cells):
    for j in range(len(epochs)):
        draw_tuning_curve(ax, j, i, tuning_peaks[i,j], cell_activity[i,j])

# --- Draw arrows for consistency ---
for i in range(n_cells):
    for j in range(len(epochs)-1):
        if cell_activity[i, j] == 1 and cell_activity[i, j+1] == 1:
            ax.arrow(j, i, 0.8, 0, head_width=0.1, head_length=0.1,
                     fc='black', ec='black', zorder=3)
        elif cell_activity[i, j] == 1 and cell_activity[i, j+1] == 0:
            ax.arrow(j, i, 0.8, 0, head_width=0.1, head_length=0.1,
                     fc='gray', ec='gray', linestyle='dashed', zorder=3)

plt.show()
n
"""
zahra

lick selectivity across trials
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random
import matplotlib.patches as patches
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10s
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"

def wilcoxon_r(x, y):
    # x, y are paired arrays (same subjects)
    W, p = scipy.stats.wilcoxon(x, y)
    diffs = x - y
    n = np.count_nonzero(diffs)  # exclude zero diffs
    if n == 0:
        return np.nan, p
    # Normal approximation for Wilcoxon (no ties/zeros correction here)
    mean_W = n * (n + 1) / 4
    sd_W = np.sqrt(n * (n + 1) * (2*n + 1) / 24)
    Z = (W - mean_W) / sd_W
    # Enforce direction from the actual mean difference
    Z = np.sign(np.nanmean(diffs)) * abs(Z)
    r = Z / np.sqrt(n)
    return r, p
df=pd.read_csv(r'C:\Users\Han\Documents\MATLAB\vip_reward_cell_copy_w_raw_data\raw_data\cell\fig1f.csv')
# Melt for plotting later (per epoch)
df_long_all = df.melt(
    id_vars=['animal', 'day', 'epoch'],
    value_vars=['first', 'last8'],
    var_name='condition',
    value_name='value'
)

# A unique identifier per individual
df_long_all['animal_day'] = (
    df_long_all['animal'].astype(str) + "_" +
    df_long_all['day'].astype(str)
)

# ---------------------------------------------------
# Run Wilcoxon test PER EPOCH
# ---------------------------------------------------
epoch_stats = {}
for ep in sorted(df['epoch'].unique()):
    sub = df[df['epoch'] == ep][['first', 'last8']].dropna()
    if len(sub) >= 5:  # need enough pairs
        W, p = wilcoxon_r(sub['first'], sub['last8'])
    else:
        W, p = np.nan, np.nan
    epoch_stats[ep] = (W, p)
    print(f"Epoch {ep}:  Wilcoxon W={W:.3f}, p={p:.4g}")
import matplotlib.colors as mcolors

# -----------------------------------------------------
# Epoch base colors
# -----------------------------------------------------
cond_colors = {
    'first':  '#006b6b',   # dark teal (filled)
    'last8':  '#66b2b2'    # lighter teal (edge only)
}


fig,ax = plt.subplots(figsize=(5,3.5))
# ---------------------------------------------------
# Bar plot (average)
# ---------------------------------------------------
sns.barplot(
    data=df_long_all,
    x='epoch',
    y='value',
    hue='condition',
    fill=False,
    estimator=np.mean,
    errorbar='se'
)

# ---------------------------------------------------
# Draw connecting lines for each animal × day × epoch
# ---------------------------------------------------
for (animal, day, epoch), subdf in df_long_all.groupby(['animal', 'day', 'epoch']):
    
    # Expect 2 rows: condition = first, last8
    subdf_sorted = subdf.sort_values('condition')
    
    if len(subdf_sorted) == 2:
        conds = subdf_sorted['condition'].values
        vals = subdf_sorted['value'].values

        # Map condition → bar x-position
        x_positions = {
            'first': epoch - 1 - 0.15,   # left offset inside epoch
            'last8': epoch - 1 + 0.15    # right offset inside epoch
        }
        xs = [x_positions[c] for c in conds]

        plt.plot(xs, vals, color='gray', alpha=0.5, linewidth=1)
ax.spines[['top','right']].set_visible(False) 
# ---------------------------------------------------
# Styling
# ---------------------------------------------------
ax.set_xlabel('Epoch')
ax.set_ylabel('Lick selectivity')
plt.title(f"Lick selectivity, first vs. last 8 trials")
plt.tight_layout()
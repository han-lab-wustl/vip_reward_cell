"""get lick rate in old reward zone 
make lick tuning curve
get licks in old vs. new reward zone pos
trial by trial tuning curve
current vs previous licks per trial
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random
import matplotlib.patches as patches
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
plt.rc('font', size=14)          # controls default text sizes

from scipy import stats
from statannotations.Annotator import Annotator


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
 
# Define group order and palette
order = ['Control','VIP Inhibition','VIP Excitation']
pl = {'Control': "slategray", 'VIP Inhibition': "red", 'VIP Excitation': 'darkgoldenrod'}
# first 8 
df_plot=pd.read_csv(r'C:\Users\Han\Documents\MATLAB\vip_reward_cell_copy_w_raw_data\raw_data\cell\fig6m_first.csv')
# Get hue offsets (where seaborn puts the stripplot points)
n_hue = df_plot["zone"].nunique()
dodge_amount = 0.4  # adjust if needed

from statsmodels.stats.multitest import multipletests
# Store stats
stats_list = []
# Comparisons: Previous vs Current per condition
for cond in order:
    sub = df_plot[df_plot["condition"] == cond]
    # Pivot to have Previous and Current in columns per session
    pivot = sub.pivot(index="sess", columns="zone", values="val").dropna()
    if pivot.shape[0] == 0:
        continue
    # Paired Wilcoxon test
    stat, p = wilcoxon_r(pivot["Current"],pivot["Previous"])
    stats_list.append({
        "Condition": cond,
        "Test": "Wilcoxon (paired)",
        "Statistic": stat,
        "p-value": p,
        'n': len(pivot["Current"])
    })

# Correct for multiple comparisons
pvals = [d['p-value'] for d in stats_list]
rej, pvals_corrected, _, _ = multipletests(pvals, method='fdr_bh')
for i, d in enumerate(stats_list):
    d['p-corrected'] = pvals_corrected[i]
    d['Significant'] = rej[i]

# Convert to DataFrame
df_stats = pd.DataFrame(stats_list)
print(df_stats)

# ----------------- Plot with bars -----------------
fig, ax = plt.subplots(figsize=(3.5,3))
sns.barplot(
    data=df_plot, x="condition", y="val", hue="zone",
    order=order, errorbar='se', palette=['k','royalblue'],
    ax=ax, fill=False
)

# Add connecting lines per session
dodge_amount = 0.4
for cond in order:
    sub = df_plot[df_plot["condition"] == cond]
    for sess in sub["sess"].unique():
        sdat = sub[sub["sess"] == sess]
        if {"Previous", "Current"} <= set(sdat["zone"]):
            prev_y = sdat[sdat["zone"]=="Previous"]["val"].values[0]
            curr_y = sdat[sdat["zone"]=="Current"]["val"].values[0]

            xpos = order.index(cond)
            prev_x = xpos - dodge_amount/2
            curr_x = xpos + dodge_amount/2
            ax.plot([prev_x, curr_x], [prev_y, curr_y],
                    color="gray", alpha=0.5, linewidth=.8, zorder=0)

# Add significance asterisks
for i, row in df_stats.iterrows():
    xpos = order.index(row['Condition'])
    y_max = df_plot[df_plot["condition"]==row['Condition']]['val'].max()-1
    if row['Significant']:
        if row['p-corrected']<0.05:
            star='*'
        if row['p-corrected']<0.01:
            star='**'
        if row['p-corrected']<0.001:
            star='***'
        if row['p-corrected']>0.05: star=''
        ax.text(xpos, y_max, star, ha='center', va='bottom', fontsize=30)
    else:
        ax.text(xpos, y_max, "ns", ha='center', va='bottom', fontsize=14)

# # Clean up
ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'], rotation=20)
ax.spines[['top','right']].set_visible(False)
ax.set_ylabel("Lick rate (Hz)")
ax.set_xlabel("")
ax.legend(title="Condition")
fig.suptitle('First 8 trials')

# last 8
df_plot=pd.read_csv(r'C:\Users\Han\Documents\MATLAB\vip_reward_cell_copy_w_raw_data\raw_data\cell\fig6m.csv')
# Get hue offsets (where seaborn puts the stripplot points)
n_hue = df_plot["zone"].nunique()
dodge_amount = 0.4  # adjust if needed

from statsmodels.stats.multitest import multipletests
# Store stats
stats_list = []
# Comparisons: Previous vs Current per condition
for cond in order:
    sub = df_plot[df_plot["condition"] == cond]
    # Pivot to have Previous and Current in columns per session
    pivot = sub.pivot(index="sess", columns="zone", values="val").dropna()
    if pivot.shape[0] == 0:
        continue
    # Paired Wilcoxon test
    stat, p = wilcoxon_r(pivot["Current"],pivot["Previous"])
    stats_list.append({
        "Condition": cond,
        "Test": "Wilcoxon (paired)",
        "Statistic": stat,
        "p-value": p,
        'n': len(pivot["Current"])
    })

# Correct for multiple comparisons
pvals = [d['p-value'] for d in stats_list]
rej, pvals_corrected, _, _ = multipletests(pvals, method='fdr_bh')
for i, d in enumerate(stats_list):
    d['p-corrected'] = pvals_corrected[i]
    d['Significant'] = rej[i]

# Convert to DataFrame
df_stats = pd.DataFrame(stats_list)
print(df_stats)

# ----------------- Plot with bars -----------------
fig, ax = plt.subplots(figsize=(3.5,3))
sns.barplot(
    data=df_plot, x="condition", y="val", hue="zone",
    order=order, errorbar='se', palette=['k','royalblue'],
    ax=ax, fill=False
)

# Add connecting lines per session
dodge_amount = 0.4
for cond in order:
    sub = df_plot[df_plot["condition"] == cond]
    for sess in sub["sess"].unique():
        sdat = sub[sub["sess"] == sess]
        if {"Previous", "Current"} <= set(sdat["zone"]):
            prev_y = sdat[sdat["zone"]=="Previous"]["val"].values[0]
            curr_y = sdat[sdat["zone"]=="Current"]["val"].values[0]

            xpos = order.index(cond)
            prev_x = xpos - dodge_amount/2
            curr_x = xpos + dodge_amount/2
            ax.plot([prev_x, curr_x], [prev_y, curr_y],
                    color="gray", alpha=0.5, linewidth=.8, zorder=0)

# Add significance asterisks
for i, row in df_stats.iterrows():
    xpos = order.index(row['Condition'])
    y_max = df_plot[df_plot["condition"]==row['Condition']]['val'].max()-1
    if row['Significant']:
        if row['p-corrected']<0.05:
            star='*'
        if row['p-corrected']<0.01:
            star='**'
        if row['p-corrected']<0.001:
            star='***'
        if row['p-corrected']>0.05: star=''
        ax.text(xpos, y_max, star, ha='center', va='bottom', fontsize=30)
    else:
        ax.text(xpos, y_max, "ns", ha='center', va='bottom', fontsize=14)


# # Clean up
ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'], rotation=20)
ax.spines[['top','right']].set_visible(False)
ax.set_ylabel("Lick rate (Hz)")
ax.set_xlabel("")
ax.legend(title="Condition")
fig.suptitle('Last 8 trials')
#%%

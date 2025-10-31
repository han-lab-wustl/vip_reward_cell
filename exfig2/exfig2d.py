
"""
zahra
pure time cells (not rew or place)
"""
#%%
import numpy as np, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
# plt.rc('font', size=20)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
#%%

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
 
# number of epochs vs. reward cell prop incl combinations    
# combine the dataframes 
df_new=pd.read_csv(r'C:\Users\Han\Documents\MATLAB\vip_reward_cell\raw_data\exfig2d.csv')
df_new['prop_diff'] = df_new['goal_cell_prop'] - df_new['goal_cell_prop_shuffle']
df_av = df_new.groupby(['animals', 'type']).median(numeric_only=True)
df_av = df_av.reset_index()
distance_diff = df_av[df_av['type'] == 'Distance']['prop_diff'].reset_index(drop=True)
time_diff = df_av[df_av['type'] == 'Time']['prop_diff'].reset_index(drop=True)
# Make sure they're aligned properly — this assumes same number and order
t_stat, p_val = wilcoxon_r(distance_diff, time_diff)
df_new = df_new.reset_index()

fig,axes = plt.subplots(figsize=(11,3.5),ncols=3,width_ratios=[3,3,1])
ax=axes[0]
custom_palette = ['cornflowerblue', 'k']
sns.set_palette(custom_palette)
hue_order = ['Distance', 'Time']
# av across mice
# sns.stripplot(x='num_epochs', y='goal_cell_prop',hue='type',
        # data=df_new,s=10,alpha=0.7,ax=ax,dodge=True,hue_order=hue_order)
sns.barplot(x='num_epochs', y='goal_cell_prop',hue='type',
        data=df_new,hue_order=hue_order,
        fill=False,ax=ax, errorbar='se')
# bar plot of shuffle instead
ax = sns.barplot(data=df_new, # correct shift
        x='num_epochs', y='goal_cell_prop_shuffle',hue='type',color='grey', hue_order=hue_order,
        alpha=0.5, err_kws={'color': 'grey'},errorbar=None,ax=ax)
ax.spines[['top','right']].set_visible(False)
ax.set_ylabel('Cell %')
ax.set_xlabel('')
ans = df_new.animals.unique()
# lines
alpha=0.5
# One value per animal per epoch per type
df_lines = df_new.groupby(['animals', 'num_epochs', 'type'])['goal_cell_prop'].median().reset_index()
for epoch in df_lines['num_epochs'].unique():
    df_ep = df_lines[df_lines['num_epochs'] == epoch]
    for animal in df_ep['animals'].unique():
        sub = df_ep[df_ep['animals'] == animal]
        if len(sub) == 2:  # Ensure both Distance and Time are present
            x = ['Distance', 'Time']
            y = sub.sort_values('type')['goal_cell_prop'].values
            ax.plot([epoch-2 - 0.2, epoch-2 + 0.2], y, color='grey', alpha=0.5, linewidth=1.5)
ax.legend().set_visible(False)

ax=axes[1]
# sns.stripplot(x='num_epochs', y='prop_diff',hue='type',
#         data=df_new,s=10,alpha=0.7,ax=ax,dodge=True,hue_order=hue_order)
sns.barplot(x='num_epochs', y='prop_diff',hue='type',
        data=df_new,hue_order=hue_order,
        fill=False,ax=ax, errorbar='se')
ax.spines[['top','right']].set_visible(False)
ax.set_ylabel('Cell %-shuffle')
ax.set_xlabel('# of epochs')
# ax.text(.5, 25, f'p={p_val:.3g}', fontsize=12)
# One value per animal per epoch per type
df_lines = df_new.groupby(['animals', 'num_epochs', 'type'])['prop_diff'].median().reset_index()
for epoch in df_lines['num_epochs'].unique():
    df_ep = df_lines[df_lines['num_epochs'] == epoch]
    for animal in df_ep['animals'].unique():
        sub = df_ep[df_ep['animals'] == animal]
        if len(sub) == 2:  # Ensure both Distance and Time are present
            x = ['Distance', 'Time']
            y = sub.sort_values('type')['prop_diff'].values
            ax.plot([epoch-2 - 0.2, epoch-2 + 0.2], y, color='grey', alpha=0.5, linewidth=1.5)
ax.legend().set_visible(False)

ax=axes[2]
sns.barplot(x='type', y='prop_diff',hue='type',
        data=df_av, fill=False,ax=ax, errorbar='se')
ax.spines[['top','right']].set_visible(False)
ax.set_ylabel('Cell %-shuffle')
ax.set_xlabel('')
ax.text(.5, 20, f'r={t_stat:.3g}\np={p_val:.3g}\nn={len(df_lines.animals.unique())}', fontsize=12)
# One value per animal per epoch per type
df_lines = df_av.groupby(['animals', 'type'])['prop_diff'].median().reset_index()
df_ep = df_lines
for animal in df_ep['animals'].unique():
        sub = df_ep[df_ep['animals'] == animal]
        if len(sub) == 2:  # Ensure both Distance and Time are present
                x = ['Distance', 'Time']
                y = sub.sort_values('type')['prop_diff'].values
                ax.plot([0,1], y, color='grey', alpha=0.5, linewidth=1.5)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# Add comparison bar with significance asterisk
# Adjust y-bar height as needed
y_bar = df_av['prop_diff'].max() + 5
y_bracket_top = y_bar + 1

x1, x2 = 0, 1  # positions of "Distance" and "Time" on x-axis
# Draw bracket: vertical up from bars, horizontal top, vertical down
ax.plot([x1, x1], [y_bar, y_bracket_top], color='black', linewidth=1.5)  # left |
ax.plot([x2, x2], [y_bar, y_bracket_top], color='black', linewidth=1.5)  # right |
ax.plot([x1, x2], [y_bracket_top, y_bracket_top], color='black', linewidth=1.5)  # top —
ax.set_title('Mean of epochs')
# Add significance text
ax.text((x1 + x2) / 2, y_bracket_top + 1, '*' if p_val < 0.05 else 'n.s.',
        ha='center', va='bottom', fontsize=14)

fig.suptitle('Reward-time v. Reward-distance cells during movement')

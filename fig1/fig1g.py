
#%%
"""
get average licks correct v incorrect
"""
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd, os
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"]=10
mpl.rcParams["ytick.major.size"]=10
# plt.rc('font', size=16)          # controls default text sizes
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
#%%
# lick rate correct vs. incorrect
# per epoch
plt.rc('font', size=20) 
df=pd.read_csv(r'C:\Users\Han\Documents\MATLAB\vip_reward_cell\raw_data\fig1g.csv')
df_long = df.melt(
    id_vars=['animal', 'day'],
    value_vars=['lick_rate_corr', 'lick_rate_incorr_pre','lick_rate_incorr'],
    var_name='trial_type',
    value_name='lick_rate'
)
# Clean up trial_type labels
df_long['trial_type'] = df_long['trial_type'].map({
    'lick_rate_corr': 'Correct Pre-Reward',
    'lick_rate_incorr_pre': 'Incorrect Pre-Reward',
    'lick_rate_incorr': 'Incorrect All'
})
df=df_long.groupby(['animal','trial_type']).mean(numeric_only=True)
pl=['seagreen','firebrick','firebrick']
order=['Correct Pre-Reward','Incorrect Pre-Reward','Incorrect All']
fig,ax=plt.subplots(figsize=(3,4))
sns.barplot(x='trial_type',y='lick_rate',data=df,fill=False,palette=pl,order=order,errorbar='se')
# sns.stripplot(x='trial_type',y='lick_rate',data=df,palette=pl)
# --- Draw connecting lines per animal ---
df_wide = df.reset_index().pivot(index='animal', columns='trial_type', values='lick_rate')
x_pos = {'Correct Pre-Reward': 0, 'Incorrect Pre-Reward': 1,
         'Incorrect All': 2}  # x-axis positions

for animal, row in df_wide.iterrows():
    x_vals = [x_pos['Correct Pre-Reward'], x_pos['Incorrect Pre-Reward'],x_pos['Incorrect All']]
    y_vals = [row['Correct Pre-Reward'], row['Incorrect Pre-Reward'],row['Incorrect All']]
    ax.plot(x_vals, y_vals, color='gray', alpha=0.5, linewidth=1.5)
ax.spines[['top','right']].set_visible(False)
ax.set_xlabel('Trial type')
ax.set_ylabel('Lick rate (licks/s)')
import scipy.stats as stats
# --- Perform pairwise Wilcoxon signed-rank tests ---
comparisons = [
    ('Correct Pre-Reward', 'Incorrect Pre-Reward'),
    ('Correct Pre-Reward', 'Incorrect All')
]
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')

results = []
for a, b in comparisons:
    valid_df = df_wide.dropna(subset=[a, b])
    stat, p = wilcoxon_r(valid_df[a].values, valid_df[b].values)
    results.append({'comp': f'{a} vs {b}', 'stat': stat, 'p': p, 'n': len(valid_df)})
from statsmodels.stats.multitest import multipletests
# FDR correction
pvals = [r['p'] for r in results]
reject, pvals_fdr, _, _ = multipletests(pvals, method='fdr_bh')

for i, r in enumerate(results):
    r['p_fdr'] = pvals_fdr[i]
    r['signif'] = (
        '***' if pvals_fdr[i] < 0.001 else
        '**' if pvals_fdr[i] < 0.01 else
        '*' if pvals_fdr[i] < 0.05 else 'ns'
    )
# --- Annotate significance on plot ---
y_max = df['lick_rate'].max()
height = y_max * 0.03
for i, (comp, label) in enumerate(zip(comparisons, results)):
    x1 = x_pos[comp[0]]
    x2 = x_pos[comp[1]]
    y = y_max + (i + 1) * height * 4  # stack bars
    ax.plot([x1, x1, x2, x2], [y, y+height, y+height, y], lw=1.5, c='k')
    ax.text((x1 + x2) / 2, y + height * 1.2, label['signif'], ha='center', va='bottom', fontsize=18)
ax.text((x1 + x2) / 2, y + height * 2.0, f"n = {label['n']}", ha='center', va='bottom', fontsize=10)

"""
get reward distance cells between opto and non opto conditions
oct 2024
mods in june 2025
control vs. opto epoch only

"""

#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
from statsmodels.stats.multitest import multipletests
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
plt.rcParams["font.family"] = "Arial"
# plot dff
import itertools
conds = ['ctrl', 'vip', 'vip_ex']
# dfnew
fig, ax = plt.subplots(figsize=(4.7,5.5 ))
# expand opto vs. prev
pl=['k','slategray']
dfnew=pd.read_csv(r'C:\Users\Han\Documents\MATLAB\vip_reward_cell_copy_w_raw_data\raw_data\cell\fig7b.csv')
# ---- aggregate for other panel (as you had) ----
dfagg = dfnew.groupby(['animals','condition','epoch']).mean(numeric_only=True).reset_index()
dfagg['dff']=dfagg['dff']*100
sns.barplot(x="condition", y="dff", hue='epoch', data=dfagg,
            palette=pl, errorbar='se', fill=False, ax=ax,legend=False)
# sns.stripplot(x="condition", y="dff", hue='epoch', data=dfagg,
#               palette=pl, alpha=a, s=s, ax=ax,dodge=True)
ax.spines[['top', 'right']].set_visible(False)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'], rotation=20)
ax.set_ylabel('')
ax.set_xlabel('')

# ---- connect pre vs opto per animal ----
for cond in conds:
    sub = dfagg[dfagg['condition'] == cond]
    for an in sub['animals'].unique():
        an_sub = sub[sub['animals'] == an]
        if set(an_sub['epoch']) >= {'prev', 'opto'}:
            prev_val = an_sub.loc[an_sub['epoch'] == 'prev', 'dff'].values[0]
            opto_val = an_sub.loc[an_sub['epoch'] == 'opto', 'dff'].values[0]
            x = conds.index(cond)
            # line connecting pre (x - offset) and opto (x + offset)
            ax.plot([x - 0.2, x + 0.2], [prev_val, opto_val],
                    color='gray', alpha=0.5, lw=1)


# Add significance annotations
def add_sig(ax, i, y_pos, stat,pval, h=.5,f=.5):
    x1=i
    x2=i+f
    x_center=i+(f/2)
    plt.plot([x1, x1, x2, x2], [y_pos, y_pos + h, y_pos + h, y_pos], lw=1.5, color='black')
    if pval < 0.001:
        sig = '***'
    elif pval < 0.01:
        sig = '**'
    elif pval < 0.05:
        sig = '*'
    else:
        sig = 'ns'
    plt.text(x_center, y_pos+h, sig, ha='center', va='bottom', fontsize=14)
    plt.text(x_center, y_pos-h, f'r={stat:.3g},p={pval:.3g}', ha='center', fontsize=8)
# Pairwise Mann-Whitney U testsn (Wilcoxon rank-sum)
p_vals = []
stats_=[]
for cond in dfagg.condition.unique():
    x1 = dfagg[(dfagg['condition'] == cond) & (dfagg['epoch']=='prev')]['dff'].dropna()
    x2 = dfagg[(dfagg['condition'] == cond) & (dfagg['epoch']=='opto')]['dff'].dropna()
    stat, p = scipy.stats.ttest_rel(x1.values, x2.values)
    p_vals.append(p)
    stats_.append(stat)
# Correct for multiple comparisons
reject, p_vals_corrected, _, _ = multipletests(p_vals, method='fdr_bh')
# Plot all pairwise comparisons
y_start = dfagg['dff'].max()
gap = .2
for i, c in enumerate(dfagg.condition.unique()):
    add_sig(ax,i-.25, y_start, stats_[i],p_vals_corrected[i])
    y_start += gap
# ax.set_ylim([-35,10])
ax.set_title('Per mouse')
ax.set_ylabel(rf'Mean % $\Delta F/F$')
fig.suptitle('All pyramidal cells')
plt.tight_layout()

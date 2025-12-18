
"""
zahra
lick corr trial by trial
split into corr vs incorr
shuffle: circ shuffle and calculate tuning curve
repeat x 100
"""
#%%
import numpy as np, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
from collections import Counter 
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
from scipy.ndimage import gaussian_filter1d
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
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
 

pltdf=pd.read_csv(r'C:\Users\Han\Documents\MATLAB\vip_reward_cell_copy_w_raw_data\raw_data\cell\fig4h.csv')
# summary fig
pl=['indigo','lightpink']
fig,ax=plt.subplots(figsize=(4,3))
sns.lineplot(x='trial_num',y='mean_dff_far',hue='condition',data=pltdf,palette=pl)
ax.spines[['top','right']].set_visible(False)
ax.set_ylabel(r'Opp. loc. mean % $\Delta F/F$')
ax.set_xlabel(r'Trial #')
fig.suptitle(r'Place (COM $<\pi/4$ and $>0$)')
plt.tight_layout()
def add_pair_lines(ax, data, x='condition', y='', order=None, id_col='animal', color='gray', alpha=0.5):
   """Draw paired lines connecting conditions for each animal."""
   for a, sub in data.groupby(id_col):
      if order is not None:
         sub = sub.set_index(x).loc[order].reset_index()
      if sub.shape[0] == 2:  # only draw if both conditions exist
         ax.plot([0,1], sub[y], color=color, alpha=alpha, zorder=0,linewidth=1.5)
         
def add_sig(ax, data, y, order, id_col='animal', ypos=None,h=.1):
   """Run Wilcoxon signed-rank test and add significance annotation to ax."""
   # wide pivot for paired comparison
   wide = data.pivot(index=id_col, columns='condition', values=y)
   wide = wide[order].dropna()  # ensure both conditions present
   stat, p = wilcoxon_r(wide[order[0]], wide[order[1]])
   # y position for the annotatio
   if ypos is None:
      ypos = data[y].max() * 1.1
   ax.text(0.5, ypos, f"cohen's r={stat:.3g}\np={p:.3g}", ha='center', va='bottom',fontsize=8)
   return stat, p

bigdf=pd.read_csv(r'C:\Users\Han\Documents\MATLAB\vip_reward_cell_copy_w_raw_data\raw_data\cell\fig4i.csv')

plt.rc('font', size=14)  
order=['nofarlick','farlick']
lbls=['No', 'Yes']

fig,axes=plt.subplots(ncols=2,figsize=(4,4),sharey=True)
ax=axes[0]
sns.barplot(x='condition',y='mean_dff_near',data=bigdf,fill=False,palette=pl,order=order,ax=ax)
ax.spines[['top','right']].set_visible(False)
add_sig(ax, bigdf, y='mean_dff_near', order=order)
add_pair_lines(ax, bigdf, x='condition', y='mean_dff_near', order=order)
ax.set_xticklabels(lbls)
ax.set_xlabel('Licks after passing reward zone?')
ax.set_ylabel(r'Mean % $\Delta F/F$')
ax.set_title('Current reward zone\n(Area 1)')
ax=axes[1]
sns.barplot(x='condition',y='mean_dff_far',data=bigdf,fill=False,palette=pl,order=order,ax=ax)
ax.set_xticklabels(lbls)
ax.set_xlabel('')
add_sig(ax, bigdf, y='mean_dff_far', order=order)
add_pair_lines(ax, bigdf, x='condition', y='mean_dff_far', order=order)
ax.spines[['top','right']].set_visible(False)
ax.set_ylabel(r'Mean $\Delta F/F$')
ax.set_title('After reward zone')
fig.suptitle(r'Place (COM $<\pi/4$ and $>0$)')
plt.tight_layout()

# %%

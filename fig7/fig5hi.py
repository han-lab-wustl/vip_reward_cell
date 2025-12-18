#%%
"""
8/31/25
- train in 70% of opto trials, all trials in rest of epochs
- test on 30% of opto trials only
- no min window for changepoint
"""
import matplotlib.pyplot as plt
import ruptures as rpt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import poisson
from scipy.special import logsumexp
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import torch
from torch.utils.data import TensorDataset, DataLoader
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error, mean_absolute_error

import numpy as np, sys
import scipy.io, scipy.interpolate, scipy.stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from hmmlearn import hmm
import random
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.opto.behavior.behavior import smooth_lick_rate
from projects.pyr_reward.placecell import make_tuning_curves_by_trialtype_w_darktime, intersect_arrays,make_tuning_curves
from projects.opto.behavior.behavior import smooth_lick_rate
import matplotlib.backends.backend_pdf, matplotlib as mpl
from sklearn.preprocessing import StandardScaler
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
# plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"


df=pd.read_csv(r'C:\Users\Han\Documents\MATLAB\vip_reward_cell_copy_w_raw_data\raw_data\fig5_bayesian_goal_decoding_70_30_split.csv')
df_shuffle = pd.read_csv(r'C:\Users\Han\Documents\MATLAB\vip_reward_cell_copy_w_raw_data\raw_data\fig5_decoding_shuffle.csv')

pl=['slategray','red','darkgoldenrod']
order = ['ctrl','vip','vip_ex']

fig,axes=plt.subplots(ncols=2)
ax=axes[0]
var='opto_s_rate'
df=df.groupby(['animals','days','type']).mean(numeric_only=True)
dfan=df.groupby(['animals','type']).mean(numeric_only=True)
sns.barplot(x='type',y=var,data=df,fill=False,ax=ax,palette=pl)
sns.stripplot(x='type',y=var,data=df,ax=ax,alpha=0.3,palette=pl)
sns.stripplot(x='type',y=var,data=dfan,ax=ax,s=10,alpha=.7,palette=pl)
sns.barplot(data=df_shuffle, # correct shift
        x='type', y=var,color='grey', 
        label='shuffle', alpha=0.4, err_kws={'color': 'grey'},errorbar=None,ax=ax)

ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'], rotation=20)
df = df.reset_index()
group_counts = df.groupby("type")[var].count()
# Add text annotations above each bar
for i, t in enumerate(order):
    n = group_counts.get(t, 0)
    y = df[df['type'] == t][var].mean()  # bar height (mean)
    ax.text(i, y + 0.05*y, f"n={n}", ha='center', va='bottom', fontsize=10)
ax.set_xlabel('')
# sns.barplot(x='condition',y='f_rate',data=df)
# Group by animal, type, and condition to get per-animal means
def add_sig_bar(ax, x1, x2, y, h, p,t):
    """
    Draws a significance bar with asterisks between x1 and x2 at height y+h.
    """
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='k')
    if p < 0.001:
        stars = '***'
    elif p < 0.01:
        stars = '**'
    elif p < 0.05:
        stars = '*'
    else:
        stars = 'n.s.'
    ax.text((x1+x2)/2, y+h, f'p={p:.3g}', ha='center', va='bottom',fontsize=10)
    ax.text((x1+x2)/2, y-h*4, f't={t:.3g}', ha='center', va='bottom',fontsize=10)
# Example: compare ctrl vs vip and ctrl vs vipex
df = df.reset_index()
ymax = df[var].max()
h = ymax * 0.1  # bracket height step

pairs = [('ctrl','vip'), ('ctrl','vip_ex')]
for i,(a,b) in enumerate(pairs):
   x1 = df.loc[df.type==a, var].dropna().values
   x2 = df.loc[df.type==b, var].dropna().values
   t, p = scipy.stats.ranksums(x1, x2)
   print(a,b,p)
   add_sig_bar(ax,
               order.index(a),
               order.index(b),
               ymax*1.05 + i*h,
               h*0.2,  # short vertical tick
               p,t)
ax.set_ylabel('Accuracy (%)')

var='opto_time_before_predict_s'
ax=axes[1]# df=df.groupby(['animals','days','type']).mean(numeric_only=True)
sns.barplot(x='type',y=var,data=df,fill=False,ax=ax,palette=pl)
sns.stripplot(x='type',y=var,data=df,ax=ax,alpha=0.3,hue='type',palette=pl)
sns.stripplot(x='type',y=var,data=dfan,ax=ax,s=10,alpha=.7,hue='type',palette=pl)
df_grouped = df.reset_index()
dfan=dfan.reset_index()
# Add lines per animal
ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'], rotation=20)
ax.set_xlabel('')
group_counts = df.groupby("type")[var].count()
# Add text annotations above each bar
for i, t in enumerate(order):
    n = group_counts.get(t, 0)
    y = df[df['type'] == t][var].mean()  # bar height (mean)
    ax.text(i, y + 0.05*y, f"n={n}", ha='center', va='bottom', fontsize=10)

# Final tweaks
ax.set_ylabel('% Time in correct prediction')
# Example: compare ctrl vs vip and ctrl vs vipex
df = df.reset_index()
order = ['ctrl','vip','vip_ex']
ymax = df[var].max()
h = ymax * 0.1  # bracket height step

pairs = [('ctrl','vip'), ('ctrl','vip_ex')]
for i,(a,b) in enumerate(pairs):
   x1 = df.loc[df.type==a, var].dropna().values
   x2 = df.loc[df.type==b, var].dropna().values
   t, p = scipy.stats.ranksums(x1, x2)
   print(a,b,p)
   add_sig_bar(ax,
               order.index(a),
               order.index(b),
               ymax*1.05 + i*h,
               h*0.2,  # short vertical tick
               p,t)

sns.despine()
fig.suptitle('Goal decoding performance (LED on)')
plt.tight_layout()

# %%
# df=df[~((df.animals=='e217')&((df.days.isin([9,11,29,30]))))]
# df=df[~((df.animals=='e201')&((df.days>77)))]

fig,ax=plt.subplots(figsize=(3.2,4))
var='pos_mae'
df=df.groupby(['animals','days','type']).mean(numeric_only=True)
dfan=df.groupby(['animals','type']).mean(numeric_only=True).reset_index()
sns.barplot(x='type',y=var,data=df,fill=False,ax=ax,palette=pl)
sns.stripplot(x='type',y=var,data=df,ax=ax,alpha=0.3,palette=pl)
sns.stripplot(x='type',y=var,data=dfan,ax=ax,s=10,alpha=.7,palette=pl)

ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'], rotation=20)
df = df.reset_index()
group_counts = df.groupby("type")[var].count()
# Add text annotations above each bar
for i, t in enumerate(order):
    n = group_counts.get(t, 0)
    y = df[df['type'] == t][var].mean()  # bar height (mean)
    ax.text(i, y + 0.05*y, f"n={n}", ha='center', va='bottom', fontsize=10)
ax.set_xlabel('')
# sns.barplot(x='condition',y='f_rate',data=df)
# Group by animal, type, and condition to get per-animal means
def add_sig_bar(ax, x1, x2, y, h, p,t):
    """
    Draws a significance bar with asterisks between x1 and x2 at height y+h.
    """
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='k')
    if p < 0.001:
        stars = '***'
    elif p < 0.01:
        stars = '**'
    elif p < 0.05:
        stars = '*'
    else:
        stars = 'n.s.'
    ax.text((x1+x2)/2, y+h, f'p={p:.3g}', ha='center', va='bottom',fontsize=10)
    ax.text((x1+x2)/2, y-h*4, f't={t:.3g}', ha='center', va='bottom',fontsize=10)
# Example: compare ctrl vs vip and ctrl vs vipex
ymax = df[var].max()-10
h = ymax * 0.1  # bracketheight step

pairs = [('ctrl','vip'), ('ctrl','vip_ex')]
for i,(a,b) in enumerate(pairs):
   x1 = df.loc[df.type==a, var].dropna().values
   x2 = df.loc[df.type==b, var].dropna().values
   t, p = scipy.stats.ranksums(x1, x2)
   print(a,b,p)
   add_sig_bar(ax,
               order.index(a),
               order.index(b),
               ymax*1.05 + i*h,
               h*0.2,  # short vertical tick
               p,t)
ax.set_ylabel('Mean absolute error (cm)')
sns.despine()
# fig.suptitle('Position decoding performance (LED on)')
plt.tight_layout()
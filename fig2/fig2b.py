
"""
zahra
2025
get # of spatially tuned cells
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
# plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"

# df=df[(df.animal!='e189')]
df=pd.read_csv(r"C:\Users\Han\Documents\MATLAB\vip_reward_cell_copy_w_raw_data\raw_data\cell\fig2b.csv")
colors = ['k', 'slategray', 'darkcyan', 'darkgoldenrod', 'orchid']
s=10
a=.7
fig,ax=plt.subplots(figsize=(3.3,4))
sns.barplot(x='epoch', y='spatial_tuned_per_ep_all', data=df, hue='epoch', palette=colors,fill=False,errorbar='se',legend=False)
# sns.stripplot(x='epoch', y='spatial_tuned_per_ep_all', data=df, hue='epoch', palette=colors,legend=False,s=s,alpha=a)
# sns.stripplot(x='epoch', y='spatial_tuned_per_ep_all', data=df)
# df=
grouped = df.groupby('animal')

for name, group in grouped:
    group_sorted = group.sort_values('epoch')
    ax.plot(
        group_sorted['epoch'] - 1,  # bar/strip x-positions start at 0
        group_sorted['spatial_tuned_per_ep_all'],
        color='gray', alpha=0.5, linewidth=1.5
    )

ax.spines[['top','right']].set_visible(False)
ax.set_ylabel('Spatially tuned cells %')
ax.set_xlabel('Epoch #')
ax.set_ylim([0,80])

# Group values by epoch
grouped = [group['spatial_tuned_per_ep_all'].values for _, group in df.groupby('epoch')]

# Run Kruskal-Wallis test
stat, pval = scipy.stats.kruskal(*grouped)

print(f"Kruskal-Wallis H = {stat:.3f}, p = {pval:.4g}")
import scikit_posthocs as sp

# Dunn's post hoc test with Holm correction
posthoc = sp.posthoc_dunn(df, val_col='spatial_tuned_per_ep_all', group_col='epoch', p_adjust='fdr_bh')

print("Dunn's test with Holm correction:\n", posthoc)
counts = df.groupby('epoch')['spatial_tuned_per_ep_all'].count()
print("Number of observations per epoch:\n", counts)


"""
zahra
behavior metrics
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

df_plt = pd.read_csv(r"C:\Users\Han\Documents\MATLAB\vip_reward_cell_copy_w_raw_data\raw_data\cell\fig1d.csv")
# plot goal cells across epochs
plt.rc('font', size=20)
colors=['k','slategray','darkcyan','darkgoldenrod']

# number of epochs vs. rates    
fig,ax = plt.subplots(figsize=(4,4))
# av across mice
sns.barplot(x='epoch', y='rates',hue='epoch',palette=colors,
        data=df_plt,
        fill=False,ax=ax, color='k', errorbar='se',legend=False)
ax.spines[['top','right']].set_visible(False)
ax.set_ylabel('% Correct trials')
ax.set_xlabel('Epoch')
# Group rates by number of epochs
grouped = df_plt.groupby('epoch')['rates'].apply(list)
# draw connecting lines per animal
for animal, dfa in df_plt.groupby('animals'):
        dfa=dfa.reset_index()
        ax.plot(
                dfa['epoch']-1, dfa['rates'],
                linewidth=1.5, alpha=0.5, color='gray'
        )
# Filter out groups with <2 data points (optional for robustness)
valid_groups = [g for g in grouped if len(g) > 1]

# Run test if at least 2 valid groups
if len(valid_groups) > 1:
    stat, p = scipy.stats.kruskal(*valid_groups)
    print("Kruskal-Wallis Test: Effect of number of epochs on % correct trials")
    print(f"H-statistic = {stat:.3f}, p = {p:.4f}")
else:
    print("Not enough valid groups to run Kruskal-Wallis test.")
summary = df_plt.groupby('epoch')['rates'].agg(['mean', scipy.stats.sem]).reset_index()
summary.columns = ['epoch', 'mean_rate', 'sem_rate']

print("Mean ± SEM of % correct trials per number of epochs:")
print(summary)
ax.set_title('Task performance')

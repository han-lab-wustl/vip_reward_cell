

"""
zahra
nov 2024
quantify reward-relative cells post reward
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd, os
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
# plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"

df=pd.read_csv(r'C:\Users\Han\Documents\MATLAB\vip_reward_cell\raw_data\fig3k.csv')
#%%
# histogram of latencies
# fig 3
plt.rc('font', size=20) 
# Prepare data
subset = df[df.behavior == 'Reward']
animals = subset['animal'].unique()
palette = sns.color_palette('tab10', len(animals))

fig, axes = plt.subplots(ncols=2,figsize=(10, 5),sharey=True)
ax=axes[0]
x_vals = np.linspace(subset['latency (s)'].min(), subset['latency (s)'].max(), 500)
kde_vals_all = []
# Plot KDE for each animal and store for averaging
for i, animal in enumerate(animals):
    data_animal = subset[subset['animal'] == animal]['latency (s)'].dropna()
    if len(data_animal)>1:
        kde = scipy.stats.gaussian_kde(data_animal)
        y_vals = kde(x_vals)
    else:
        kde = scipy.stats.norm.pdf(x_vals, loc=data_animal.values[0], scale=.5)
        y_vals=kde
    kde_vals_all.append(y_vals)
    sns.lineplot(x=x_vals, y=y_vals, ax=ax, label=animal, color=palette[i], linewidth=1.5)
# Compute mean KDE across animals
mean_kde_vals = np.mean(kde_vals_all, axis=0)
ax.plot(x_vals, mean_kde_vals, color='black', linewidth=3, label='Mean KDE')
mean_val = np.nanmean(df[df.behavior=='Reward']['latency (s)'].values)
sem= scipy.stats.sem(df[df.behavior=='Reward']['latency (s)'].values,nan_policy='omit')
ax.axvline(mean_val, color='k', linestyle='--', linewidth=2)
ax.text(mean_val, ax.get_ylim()[1], f'Mean={mean_val:.2g} +/- {sem:.2g}', color='k', ha='center', va='bottom', fontsize=14)
ax.spines[['top','right']].set_visible(False)
ax.set_ylabel('# Post-reward cells')
ax.set_xlabel('Latency from reward (s)')

ax=axes[1]
subset = df[df.behavior == 'Movement Start']
kde_vals_all = []
x_vals = np.linspace(subset['latency (s)'].min(), subset['latency (s)'].max(), 500)
# Plot KDE for each animal and store for averaging
for i, animal in enumerate(animals):
    data_animal = subset[subset['animal'] == animal]['latency (s)'].dropna()
    if len(data_animal)>1:
        kde = scipy.stats.gaussian_kde(data_animal)
        y_vals = kde(x_vals)
        kde_vals_all.append(y_vals)
    
        sns.lineplot(x=x_vals, y=y_vals, ax=ax, label=animal, color=palette[i], linewidth=1.5)
# Compute mean KDE across animals
mean_kde_vals = np.mean(kde_vals_all, axis=0)
ax.plot(x_vals, mean_kde_vals, color='black', linewidth=3, label='Mean KDE')
mean_val = np.nanmean(df[df.behavior=='Movement Start']['latency (s)'].values)
sem= scipy.stats.sem(df[df.behavior=='Movement Start']['latency (s)'].values,nan_policy='omit')
ax.axvline(mean_val, color='k', linestyle='--', linewidth=2)
ax.text(mean_val, ax.get_ylim()[1], f'Mean={mean_val:.2g} +/- {sem:.2g}', color='k', ha='center', va='bottom', fontsize=14)
ax.spines[['top','right']].set_visible(False)
ax.set_ylabel('# Post-reward cells')
ax.set_xlabel('Latency from movement start (s)')

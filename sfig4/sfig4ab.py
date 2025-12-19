
"""
zahra
get tuning curves with dark time
reward cell p vs. behavior
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
plt.rc('font', size=20)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"

df=pd.read_csv(r'C:\Users\Han\Documents\MATLAB\vip_reward_cell_copy_w_raw_data\raw_data\cell\sfig4ab.csv')
# --- Scatterplot with hue per animal ---
fig,axes = plt.subplots(ncols=3,figsize=(11,4),sharey=True)
metrics = ['rates', 'lick_selectivity', 'lick_rate']
lbl=['$\Delta$ % Correct trials', '$\Delta$ Lick selectivity', '$\Delta$ Lick rate']
for ii,m in enumerate(metrics):
    ax=axes[ii]
    r, p = scipy.stats.pearsonr(df[m], df['reward_cell'])
    n = len(df)
    if ii==2: legend=True
    else: legend=False
    sns.scatterplot(
        data=df, x=m, y='reward_cell',
        hue='animals', palette='tab10', s=60, ax=ax,legend=legend,alpha=0.5
    )
    # Optional: regression line on top of all points
    sns.regplot(
        data=df, x=m, y='reward_cell',
        scatter=False, color='black', ci=None,ax=ax
    )
    # --- Annotate stats ---
    ax.text(
        0.05, 0.95,
        f"r = {r:.3g}\np = {p:.3g}\nn = {n}",
        transform=ax.transAxes,
        verticalalignment='top',fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )
    ax.spines[['top','right']].set_visible(False)
    # plt.xlabel("% Correct trials")
    if ii==2:
        ax.legend(title="Animal", bbox_to_anchor=(1.05, 1), loc='upper left',fontsize=12)
    else: ax.set_ylabel("")
    ax.set_xlabel(lbl[ii])
    ax.set_ylabel("Place cell %")

fig.suptitle('Place cell % vs. performance metrics')
plt.tight_layout()
# plt.savefig(os.path.join(savedst, "performance_v_placecell.svg"))

#%%
# --- Correlation stats ---
df=df.groupby(['animals']).mean(numeric_only=True).reset_index()
# --- Scatterplot with hue per animal ---
fig,axes = plt.subplots(ncols=3,figsize=(11,4),sharey=True)
metrics = ['rates', 'lick_selectivity', 'lick_rate']
lbl=['$\Delta$ % Correct trials', '$\Delta$ Lick selectivity', '$\Delta$ Lick rate']
for ii,m in enumerate(metrics):
    ax=axes[ii]
    r, p = scipy.stats.pearsonr(df[m], df['reward_cell'])
    n = len(df)
    if ii==2: legend=True
    else: legend=False
    sns.scatterplot(
        data=df, x=m, y='reward_cell',
        hue='animals', palette='tab10', s=60, ax=ax,legend=legend,
    )
    # Optional: regression line on top of all points
    sns.regplot(
        data=df, x=m, y='reward_cell',
        scatter=False, color='black', ci=None,ax=ax
    )
    # --- Annotate stats ---
    ax.text(
        0.05, 0.95,
        f"r = {r:.3g}\np = {p:.3g}\nn = {n}",
        transform=ax.transAxes,
        verticalalignment='top',fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )
    ax.spines[['top','right']].set_visible(False)
    # plt.xlabel("% Correct trials")
    if ii==2:
        ax.legend(title="Animal", bbox_to_anchor=(1.05, 1), loc='upper left',fontsize=12)
    else: ax.set_ylabel("")
    ax.set_xlabel(lbl[ii])
    ax.set_ylabel("Place cell %")

fig.suptitle('Place cell % vs. performance metrics, per mouse')
plt.tight_layout()

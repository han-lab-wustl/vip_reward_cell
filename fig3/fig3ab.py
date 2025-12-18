
"""
zahra
april 2025
get place cells 
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
plt.rc('font', size=16)          # controls default text sizes
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

# for figure; place cells in any epoch
# consecutive epochs
df_permsav=pd.read_csv(r"C:\Users\Han\Documents\MATLAB\vip_reward_cell_copy_w_raw_data\raw_data\cell\fig3a.csv")
fig,ax = plt.subplots(figsize=(3,4))
sns.barplot(x='epoch_comparison', y='place_cell_prop',
        data=df_permsav,
        fill=False,ax=ax, color='indigo', errorbar='se')
sns.barplot(data=df_permsav, # correct shift
        x='epoch_comparison', y='place_cell_prop_shuffle',color='grey', 
        label='shuffle', alpha=0.5, err_kws={'color': 'grey'},errorbar=None,ax=ax)

ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.01, 1.05))
ax.set_xticklabels(ax.get_xticklabels(),rotation=20)
eps = df_permsav.epoch_comparison.unique()
pvalues=[]
for ep in eps:
        # rewprop = df_plt.loc[(df_plt.num_epochs==ep), 'goal_cell_prop']
        rewprop = df_permsav.loc[(df_permsav.epoch_comparison==ep), 
        'place_cell_prop'].values
        shufprop = df_permsav.loc[(df_permsav.epoch_comparison==ep), 
                'place_cell_prop_shuffle'].values
        t,pval = wilcoxon_r(rewprop, shufprop)
        print(f'{ep} epochs, pval: {pval},w = {t},n={len(rewprop)}')
        pvalues.append(pval)

from statsmodels.stats.multitest import multipletests
reject, pvals_corrected, _, _ = multipletests(pvalues, method='fdr_bh')

y=32
fs=38
for ii,ep in enumerate(eps):
        pval=pvals_corrected[ii]
        # statistical annotation        
        if pval < 0.001:
                ax.text(ii, y, "***", ha='center', fontsize=fs)
        elif pval < 0.01:
                ax.text(ii, y, "**", ha='center', fontsize=fs)
        elif pval < 0.05:
                ax.text(ii, y, "*", ha='center', fontsize=fs)

ax.set_xlabel('')
ax.set_ylabel('Place cell %')
ans = df_permsav.animals.unique()
for i in range(len(ans)):
    ax = sns.lineplot(x=np.arange(len(df_permsav[df_permsav.animals==ans[i]])), y='place_cell_prop', 
    data=df_permsav[df_permsav.animals==ans[i]],
    errorbar=None, color='dimgray', linewidth=1.5, alpha=0.5,ax=ax)

groups = [df_permsav.loc[df_permsav.epoch_comparison==ep, 'place_cell_prop'].values
          for ep in df_permsav.epoch_comparison.unique()]

H, p_kw = scipy.stats.kruskal(*groups)
print(f"Kruskal–Wallis: H={H:.3f}, p={p_kw:.3g}")

# =========================
# Post-hoc pairwise Dunn test if KW significant
# =========================
import scikit_posthocs as sp_post
if p_kw < 0.05:
        dunn = sp_post.posthoc_dunn(df_permsav, val_col='place_cell_prop', 
                                group_col='epoch_comparison', p_adjust='fdr_bh')
        print(dunn)
        
import itertools

# Set label
ax.set_ylabel('Place cell %')

# Add subject-level lines
ans = df_permsav.animals.unique()
for i in range(len(ans)):
    ax = sns.lineplot(x=np.arange(len(df_permsav[df_permsav.animals==ans[i]])),
                      y='place_cell_prop',
                      data=df_permsav[df_permsav.animals==ans[i]],
                      errorbar=None, color='dimgray',
                      linewidth=1.5, alpha=0.2, ax=ax)

# Kruskal–Wallis
groups = [df_permsav.loc[df_permsav.epoch_comparison==ep, 'place_cell_prop'].values
          for ep in df_permsav.epoch_comparison.unique()]
H, p_kw = scipy.stats.kruskal(*groups)
print(f"Kruskal–Wallis: H={H:.3f}, p={p_kw:.3g}")
h=1.5
# Post-hoc Dunn test
import scikit_posthocs as sp_post
if p_kw < 0.05:
    dunn = sp_post.posthoc_dunn(df_permsav,
                                val_col='place_cell_prop',
                                group_col='epoch_comparison',
                                p_adjust='fdr_bh')
    print(dunn)

    # =============================
    # Draw comparison bars manually
    # =============================
    xticks = df_permsav.epoch_comparison.unique()
    y_max = df_permsav['place_cell_prop'].max()
    bar_height = y_max * 0.1   # height increment for bars
    current_y = y_max + bar_height

    for (i, j) in itertools.combinations(range(len(xticks)), 2):
        pval = dunn.iloc[i, j]
        if pval < 0.05:
            # draw line
            x1, x2 = i, j
            ax.plot([x1, x1, x2, x2],
                    [current_y, current_y+h, current_y+h, current_y],
                    lw=1.5, c='k')
            # significance stars
            if pval < 0.001:
                stars = '***'
            elif pval < 0.01:
                stars = '**'
            else:
                stars = '*'
            ax.text((x1+x2)/2, current_y+0.015, stars,
                    ha='center', va='bottom', fontsize=fs)
            current_y += bar_height  # increment for next bar
fig.suptitle('Place cells between two epochs')
#%%
df_plt2=pd.read_csv(r"C:\Users\Han\Documents\MATLAB\vip_reward_cell\raw_data\fig2b.csv")
fig,axes = plt.subplots(ncols=2,figsize=(6.5,4))
ax=axes[0]
sns.barplot(x='num_epochs', y='place_cell_prop',
        data=df_plt2,
        fill=False,ax=ax, color='indigo', errorbar='se')
# bar plot of shuffle instead
sns.barplot(data=df_plt2, # correct shift
        x='num_epochs', y='place_cell_prop_shuffle',color='grey', 
        label='shuffle', alpha=0.5, err_kws={'color': 'grey'},errorbar=None,ax=ax)

ax.spines[['top','right']].set_visible(False)
ax.legend()#.set_visible(False)
ax.set_ylabel('Place cell %')
eps = [2,3,4]
y = 28
pshift = 1
fs=46
pvalues=[]
ts=[]
for ii,ep in enumerate(eps):
        rewprop = df_plt2.loc[(df_plt2.num_epochs==ep), 'place_cell_prop']
        shufprop = df_plt2.loc[(df_plt2.num_epochs==ep), 'place_cell_prop_shuffle']
        tstat,pval = wilcoxon_r(rewprop, shufprop)
        pvalues.append(pval)
        ts.append(tstat)
        print(f'{ep} epochs, pval: {pval}, r ={tstat}, n={len(rewprop)}')
# correct pvalues
from statsmodels.stats.multitest import multipletests
reject, pvals_corrected, _, _ = multipletests(pvalues, method='fdr_bh')

for ii,ep in enumerate(eps):
        pval=pvals_corrected[ii]
        # statistical annotation        
        if pval < 0.001:
                ax.text(ii, y, "***", ha='center', fontsize=fs)
        elif pval < 0.01:
                ax.text(ii, y, "**", ha='center', fontsize=fs)
        elif pval < 0.05:
                ax.text(ii, y, "*", ha='center', fontsize=fs)
        # ax.text(ii-0.5, y-pshift*2, f't={ts[ii]:.3g}\np={pval:.3g}',fontsize=10,rotation=45)
# make lines
ans = df_plt2.animals.unique()
for i in range(len(ans)):
    ax = sns.lineplot(x=df_plt2.num_epochs-2, y='place_cell_prop', 
    data=df_plt2[df_plt2.animals==ans[i]],
    errorbar=None, color='dimgray', linewidth=1.5, alpha=0.5,ax=ax)
ax.set_title('Place cells',pad=30)
ax.set_xlabel('')
ax.set_ylim([0,35])
ax=axes[1]
# subtract from shuffle
# df_plt2=df_plt2.reset_index()
df_plt2['place_cell_prop_sub_shuffle'] = df_plt2['place_cell_prop']-df_plt2['place_cell_prop_shuffle']
# av across mice
# sns.stripplot(x='num_epochs', y='place_cell_prop_sub_shuffle',color='cornflowerblue',data=df_plt2,s=10,alpha=0.7,ax=ax)
sns.barplot(x='num_epochs', y='place_cell_prop_sub_shuffle',
        data=df_plt2,
        fill=False,ax=ax, color='indigo', errorbar='se')
# make lines
df_plt2=df_plt2.reset_index()
ans = df_plt2.animals.unique()
for i in range(len(ans)):
    ax = sns.lineplot(x=df_plt2.num_epochs-2, y='place_cell_prop_sub_shuffle', 
    data=df_plt2[df_plt2.animals==ans[i]],
    errorbar=None, color='dimgray', linewidth=1.5, alpha=0.5,ax=ax)
y=15
for ii,ep in enumerate(eps):
        pval=pvals_corrected[ii]
        # statistical annotation        
        if pval < 0.001:
                ax.text(ii, y, "***", ha='center', fontsize=fs)
        elif pval < 0.01:
                ax.text(ii, y, "**", ha='center', fontsize=fs)
        elif pval < 0.05:
                ax.text(ii, y, "*", ha='center', fontsize=fs)
        # ax.text(ii-0.5, y-pshift*2, f't={ts[ii]:.3g}\np={pval:.3g}',fontsize=10,rotation=45)
        
ax.spines[['top','right']].set_visible(False)
ax.set_xlabel('# of epochs')
ax.set_ylabel('')
ax.set_title('Place cell %-shuffle',pad=30)
ax.set_ylim([-1,20
             ])
fig.suptitle('Dedicated place cells')

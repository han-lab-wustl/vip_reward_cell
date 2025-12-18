
"""
zahra
get tuning curves with dark time
get cells in 2, 3, or 4 epochs
july 2025
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
df_permsav=pd.read_csv(r"C:\Users\Han\Documents\MATLAB\vip_reward_cell_copy_w_raw_data\raw_data\cell\fig3c.csv")

fig,ax = plt.subplots(figsize=(3,4))
sns.barplot(x='epoch_comparison', y='goal_cell_prop',
        data=df_permsav,
        fill=False,ax=ax, color='cornflowerblue', errorbar='se')
sns.barplot(data=df_permsav, # correct shift
        x='epoch_comparison', y='goal_cell_prop_shuffle',color='grey', 
        label='shuffle', alpha=0.5, err_kws={'color': 'grey'},errorbar=None,ax=ax)

ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.01, 1.05))
ax.set_xticklabels(ax.get_xticklabels(),rotation=20)
eps = df_permsav.epoch_comparison.unique()
pvalues=[]
for ep in eps:
        rewprop = df_permsav.loc[(df_permsav.epoch_comparison==ep), 
        'goal_cell_prop'].values
        shufprop = df_permsav.loc[(df_permsav.epoch_comparison==ep), 
                'goal_cell_prop_shuffle'].values
        t,pval = wilcoxon_r(rewprop, shufprop)
        print(f'{ep} epochs, pval: {pval},w = {t},n={len(rewprop)}')
        pvalues.append(pval)

from statsmodels.stats.multitest import multipletests
reject, pvals_corrected, _, _ = multipletests(pvalues, method='fdr_bh')

y=45
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
ans = df_permsav.animals.unique()
for i in range(len(ans)):
    ax = sns.lineplot(x=np.arange(len(df_permsav[df_permsav.animals==ans[i]])), y='goal_cell_prop', 
    data=df_permsav[df_permsav.animals==ans[i]],
    errorbar=None, color='dimgray', linewidth=1.5, alpha=0.5,ax=ax)

groups = [df_permsav.loc[df_permsav.epoch_comparison==ep, 'goal_cell_prop'].values
          for ep in df_permsav.epoch_comparison.unique()]

H, p_kw = scipy.stats.kruskal(*groups)
print(f"Kruskal–Wallis: H={H:.3f}, p={p_kw:.3g}")

# =========================
# Post-hoc pairwise Dunn test if KW significant
# =========================
import scikit_posthocs as sp_post
if p_kw < 0.05:
        dunn = sp_post.posthoc_dunn(df_permsav, val_col='goal_cell_prop', 
                                group_col='epoch_comparison', p_adjust='fdr_bh')
        print(dunn)
        
import itertools

# Set label
ax.set_ylabel('Reward cell %')

# Add subject-level lines
ans = df_permsav.animals.unique()
for i in range(len(ans)):
    ax = sns.lineplot(x=np.arange(len(df_permsav[df_permsav.animals==ans[i]])),
                      y='goal_cell_prop',
                      data=df_permsav[df_permsav.animals==ans[i]],
                      errorbar=None, color='dimgray',
                      linewidth=1.5, alpha=0.2, ax=ax)

# Kruskal–Wallis
groups = [df_permsav.loc[df_permsav.epoch_comparison==ep, 'goal_cell_prop'].values
          for ep in df_permsav.epoch_comparison.unique()]
H, p_kw = scipy.stats.kruskal(*groups)
print(f"Kruskal–Wallis: H={H:.3f}, p={p_kw:.3g}")
h=1.5
# Post-hoc Dunn test
import scikit_posthocs as sp_post
if p_kw < 0.05:
    dunn = sp_post.posthoc_dunn(df_permsav,
                                val_col='goal_cell_prop',
                                group_col='epoch_comparison',
                                p_adjust='fdr_bh')
    print(dunn)

    # =============================
    # Draw comparison bars manually
    # =============================
    xticks = df_permsav.epoch_comparison.unique()
    y_max = df_permsav['goal_cell_prop'].max()
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
fig.suptitle('Reward cells between two epochs')

# compare to shuffle
plt.rc('font', size=20)          # controls default text sizes
df_plt2=pd.read_csv(r'C:\Users\Han\Documents\MATLAB\vip_reward_cell_copy_w_raw_data\raw_data\cell\fig3d.csv')
# number of epochs vs. reward cell prop incl combinations    
fig,axes = plt.subplots(figsize=(6.5,4),ncols=2,sharex=True)
ax=axes[0]
# av across mice
# sns.stripplot(x='num_epochs', y='goal_cell_prop',color='k',
        # data=df_plt2,s=10,alpha=0.7,ax=ax)
sns.barplot(x='num_epochs', y='goal_cell_prop',
        data=df_plt2,
        fill=False,ax=ax, color='cornflowerblue', errorbar='se')
# bar plot of shuffle instead
ax = sns.barplot(data=df_plt2, # correct shift
        x='num_epochs', y='goal_cell_prop_shuffle',color='grey', 
        label='shuffle', alpha=0.5, err_kws={'color': 'grey'},errorbar=None,ax=ax)

ax.spines[['top','right']].set_visible(False)
ax.legend()#.set_visible(False)
# make lines
alpha=0.5
ans = df_plt2.animals.unique()
for i in range(len(ans)):
    sns.lineplot(x=df_plt2.num_epochs-2, y='goal_cell_prop', 
    data=df_plt2[df_plt2.animals==ans[i]],
    errorbar=None, color='gray', alpha=0.5, linewidth=1.5,ax=ax)
ax.set_xlabel('')
ax.set_ylabel('Reward cell % ')

eps = [2,3,4]
y = 37
pshift = 4
fs=38
pvalues = []
ts=[]
# Step 1: Compute p-values
for ep in eps:
    rewprop = df_plt2.loc[(df_plt2.num_epochs == ep), 'goal_cell_prop']
    shufprop = df_plt2.loc[(df_plt2.num_epochs == ep), 'goal_cell_prop_shuffle']
    t, pval = wilcoxon_r(rewprop, shufprop)
    print(f'{ep} epochs, pval: {pval}, r ={t}, n={len(rewprop)}')    
    pvalues.append(pval)
    ts.append(t)
    
from statsmodels.stats.multitest import fdrcorrection
# Step 2: FDR correction
reject, pvals_fdr = fdrcorrection(pvalues, alpha=0.05)

# Step 3: Annotate plot
for ii, (ep, pval_corr, sig) in enumerate(zip(eps, pvals_fdr, reject)):
    if pval_corr < 0.001:
        stars = "***"
    elif pval_corr < 0.01:
        stars = "**"
    elif pval_corr < 0.05:
        stars = "*"
    else:
        stars = "n.s."
    if sig:
        ax.text(ii, y, stars, ha='center', fontsize=fs)
    else:
        ax.text(ii, y, stars, ha='center', fontsize=fs, color='gray')  # Optional: fade non-sig
    ax.text(ii, y-10, f'p={pval_corr:.3g}\nt={ts[ii]:.3g}', ha='center', fontsize=10, color='k')  # Optional: fade non-sig

ax.set_title('Reward cells')

# subtract from shuffle
# df_plt2=df_plt2.reset_index()
df_plt3 = df_plt2.groupby(['num_epochs']).mean(numeric_only=True)
df_plt3=df_plt3.reset_index()
# subtract by average across animals?
sub = []
for ii,xx in df_plt2.iterrows():
        ep = xx.num_epochs
        sub.append(xx.goal_cell_prop-df_plt3.loc[df_plt3.num_epochs==ep, 'goal_cell_prop_shuffle'].values[0])
        
df_plt2['goal_cell_prop_sub_shuffle']=sub
# vs. average within animal
df_plt2['goal_cell_prop_sub_shuffle']=df_plt2['goal_cell_prop']-df_plt2['goal_cell_prop_shuffle']
ax=axes[1]# av across mice
# sns.stripplot(x='num_epochs', y='goal_cell_prop_sub_shuffle',color='cornflowerblue',
#         data=df_plt2,s=10,alpha=0.7,ax=ax)
sns.barplot(x='num_epochs', y='goal_cell_prop_sub_shuffle',
        data=df_plt2,
        fill=False,ax=ax, color='cornflowerblue', errorbar='se')
ax.spines[['top','right']].set_visible(False)
ax.set_title('Reward cell %-shuffle')
ax.set_ylim([0, 22])
# make lines
ans = df_plt2.animals.unique()
for i in range(len(ans)):
    ax = sns.lineplot(x=df_plt2.num_epochs-2, y='goal_cell_prop_sub_shuffle', 
    data=df_plt2[df_plt2.animals==ans[i]],
    errorbar=None, color='dimgray', alpha=0.5, linewidth=1.5)
    
ax.set_xlabel('# of epochs')
ax.set_ylabel('')
y=18
# Step 3: Annotate plot
for ii, (ep, pval_corr, sig) in enumerate(zip(eps, pvals_fdr, reject)):
    if pval_corr < 0.001:
        stars = "***"
    elif pval_corr < 0.01:
        stars = "**"
    elif pval_corr < 0.05:
        stars = "*"
    else:
        stars = "n.s."
    if sig:
        ax.text(ii, y, stars, ha='center', fontsize=fs)
    else:
        ax.text(ii, y, stars, ha='center', fontsize=fs, color='gray')  # Optional: fade non-sig

fig.suptitle('Dedicated reward cells')
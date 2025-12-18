"""get place cells between opto and non opto conditions
april 2025
"""

#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
from statsmodels.formula.api import ols
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
import itertools
from statsmodels.stats.anova import anova_lm  # <-- Correct import
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
plt.rcParams["font.family"] = "Arial"

df=pd.read_csv(r'C:\Users\Han\Documents\MATLAB\vip_reward_cell_copy_w_raw_data\raw_data\cell\fig7h.csv')
pl =['k','slategray']
df_plt = df
df_plt = df_plt.groupby(['animals','condition','opto']).mean(numeric_only=True).reset_index()
fig,axes = plt.subplots(ncols=2,figsize=(8,3),sharex=True,sharey=True)
ax=axes[0]

sns.barplot(x='condition', y='place_cell_prop_early',hue='opto',
        data=df_plt,
        palette=pl,
        fill=False,ax=ax, color='k', errorbar='se',legend=False)
# ----- connecting lines per animal -----
for cond in df_plt['condition'].unique():
    sub = df_plt[df_plt['condition']==cond]
    wide = sub.pivot(index='animals', columns='opto', values='place_cell_prop_early').dropna()
    xpos = list(df_plt['condition'].unique()).index(cond)
    for _, row in wide.iterrows():
        ax.plot([xpos-0.2, xpos+0.2], [row[False], row[True]], color='gray', alpha=0.5, lw=1.5)

# ----- paired stats -----
stats_results = {}
for cond in df_plt['condition'].unique():
    sub = df_plt[df_plt['condition']==cond]
    wide = sub.pivot(index='animals', columns='opto', values='place_cell_prop_early').dropna()
    if wide.shape[1]==2:
        t,p = stats.ttest_rel(wide[False], wide[True])
        stats_results[cond] = {'t':t, 'p':p, 'n':len(wide)}
    else:
        stats_results[cond] = None


# ----- annotate p-values -----
for i, cond in enumerate(df_plt['condition'].unique()):
    res = stats_results[cond]
    if res is not None:
        p = res['p']
        ymax = df_plt[df_plt['condition']==cond]['place_cell_prop_early'].max()
        # choose stars
        if p < 0.001:
            stars = '***'
        elif p < 0.01:
            stars = '**'
        elif p < 0.05:
            stars = '*'
        else:
            stars = f"ns"
        ax.text(i, ymax*1.15, stars, ha='center', va='bottom', fontsize=14)
        ax.text(i, ymax*1.01, f't={res["t"]:.3g}\np={p:.3g}', ha='center', va='bottom', fontsize=9)

ax.spines[['top','right']].set_visible(False)
new_labels = {'ctrl': 'Control', 'vip': 'VIP\nInhibition', 'vip_ex': 'VIP\nExcitation'}
ax.set_xlabel('')
ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'], rotation=20)
ax.set_ylabel('Place cell %')
ax.set_title('First 8 trials')
ax=axes[1]
### late
sns.barplot(x='condition', y='place_cell_prop',hue='opto',
        data=df_plt,
        palette=pl,
        fill=False,ax=ax, color='k', errorbar='se',legend=False)
# ----- connecting lines per animal -----
for cond in df_plt['condition'].unique():
    sub = df_plt[df_plt['condition']==cond]
    wide = sub.pivot(index='animals', columns='opto', values='place_cell_prop').dropna()
    xpos = list(df_plt['condition'].unique()).index(cond)
    for _, row in wide.iterrows():
        ax.plot([xpos-0.2, xpos+0.2], [row[False], row[True]], color='gray', alpha=0.5, lw=1.5)

# ----- paired stats -----
stats_results = {}
for cond in df_plt['condition'].unique():
    sub = df_plt[df_plt['condition']==cond]
    wide = sub.pivot(index='animals', columns='opto', values='place_cell_prop').dropna()
    if wide.shape[1]==2:
        t,p = stats.ttest_rel(wide[False], wide[True])
        stats_results[cond] = {'t':t, 'p':p, 'n':len(wide)}
    else:
        stats_results[cond] = None


# ----- annotate p-values -----
for i, cond in enumerate(df_plt['condition'].unique()):
    res = stats_results[cond]
    if res is not None:
        p = res['p']
        ymax = df_plt[df_plt['condition']==cond]['place_cell_prop'].max()
        # choose stars
        if p < 0.001:
            stars = '***'
        elif p < 0.01:
            stars = '**'
        elif p < 0.05:
            stars = '*'
        else:
            stars = f"ns"
        ax.text(i, ymax*1.15, stars, ha='center', va='bottom', fontsize=14)
        ax.text(i, ymax*1.01, f't={res["t"]:.3g}\np={p:.3g}', ha='center', va='bottom', fontsize=9)

ax.spines[['top','right']].set_visible(False)
new_labels = {'ctrl': 'Control', 'vip': 'VIP\nInhibition', 'vip_ex': 'VIP\nExcitation'}
ax.set_xlabel('')
ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'], rotation=20)
ax.set_title('Last 8 trials')
fig.suptitle('Place cells')


# --- Summary stats (mean ± SEM) ---
summary = (
    df_plt.groupby(["condition", "opto"])
      .agg(mean_goal=('place_cell_prop', 'mean'),
           sem_goal=('place_cell_prop', 'sem'),
           n=('place_cell_prop', 'count'))
      .reset_index()
)
print("Summary by condition, epoch, and opto:")
print(summary)

#%%
# subtract by led off sessions
# ----------------------------------------
# Plotting Stim - No Stim per Animal

# subtract by led off sessions for both
# ----------------------------------------

# Plotting Stim - No Stim per Animal
# ----------------------------------------

df_an = df_plt.copy()
df_an = df_an.sort_values(['animals', 'condition'])
df_an['opto'] = [True if xx==True else False for xx in df_an.opto]

# compute delta for each condition per animal
delta_vals = []
for (animal, condition), group in df_an.groupby(['animals', 'condition']):

    stim = group.loc[group.opto == True].set_index(['animals', 'condition'])[['place_cell_prop', 'place_cell_prop_early', 'other_sp_prop']]
    no_stim = group.loc[group.opto == False].set_index(['animals', 'condition'])[['place_cell_prop', 'place_cell_prop_early','other_sp_prop']]

    if not stim.empty and not no_stim.empty:
        delta_vals.append([animal, condition, 
                            stim.loc[(animal, condition),'place_cell_prop'] - no_stim.loc[(animal, condition), 'place_cell_prop'], 
                            stim.loc[(animal, condition), 'place_cell_prop_early'] - no_stim.loc[(animal, condition), 'place_cell_prop_early'], stim.loc[(animal, condition), 'other_sp_prop'] - no_stim.loc[(animal, condition), 'other_sp_prop']])

df_delta = pd.DataFrame(delta_vals, columns=['animals', 'condition', 'delta_late', 'delta_early', 'delta_other_sp'])
# Combine early and late deltas (mean)
df_delta['delta_combined'] = df_delta[['delta_late', 'delta_early']].mean(axis=1)

# Now we can plot side by side
fig, axs = plt.subplots(1, 3, figsize=(11,6),sharey=True)

a = 0.7
s = 12

# Plotting late
ax = axs[1]
sns.stripplot(data=df_delta, x='condition', y='delta_late', hue='condition',ax=ax, palette=pl, size=s, alpha=a)
sns.barplot(data=df_delta, x='condition', y='delta_late',hue='condition', ax=ax, palette=pl, fill=False,errorbar='se')
ax.set_xlabel('')
ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'], rotation=20)
ax.spines[['top', 'right']].set_visible(False)
ax.set_title('Late Place')
# --- Stats + annotation ---
data=df_delta
conds = data['condition'].unique()
pairs = list(combinations(conds, 2))[:2]
y_max = data['delta_late'].quantile(.99)
y_step = 0.4 * abs(y_max)

for i, (cond1, cond2) in enumerate(pairs):
    vals1 = data[data['condition'] == cond1]['delta_late']
    vals2 = data[data['condition'] == cond2]['delta_late']
    stat, pval = scipy.stats.ranksums(vals1, vals2)
    # Annotation text
    if pval < 0.001:
        text = '***'
    elif pval < 0.01:
        text = '**'
    elif pval < 0.05:
        text = '*'
    else:
        text = f""

    # Get x-locations
    x1, x2 = conds.tolist().index(cond1), conds.tolist().index(cond2)
    y = y_max + y_step * (i + 1)
    ax.plot([x1, x1, x2, x2], [y, y + y_step/3, y + y_step/3, y], lw=1.5, c='k')
    ax.text((x1 + x2)/2, y-y_step*.2, text, ha='center', va='bottom', fontsize=40)
    ax.text((x1 + x2)/2, y-y_step*.3, f'{pval:.3g}', ha='center', va='bottom', fontsize=12)

# Plotting early
ax = axs[0]
sns.stripplot(data=df_delta, x='condition', y='delta_early', ax=ax, 
              palette=pl, size=s, alpha=a)
sns.barplot(data=df_delta, x='condition', y='delta_early', ax=ax, 
            palette=pl, fill=False,errorbar='se')
ax.set_ylabel('$\Delta$ Place cell % \n(LEDon-LEDoff)')
ax.set_xlabel('')
ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'], rotation=20)
ax.spines[['top', 'right']].set_visible(False)
ax.set_title('Early Place')
# --- Stats + annotation ---
data=df_delta
conds = data['condition'].unique()
pairs = list(combinations(conds, 2))[:2]
y_max = data['delta_early'].quantile(.85)
y_step = 0.4 * abs(y_max)

for i, (cond1, cond2) in enumerate(pairs):
    vals1 = data[data['condition'] == cond1]['delta_early']
    vals2 = data[data['condition'] == cond2]['delta_early']
    stat, pval = scipy.stats.ranksums(vals1, vals2)
    # Annotation text
    if pval < 0.001:
        text = '***'
    elif pval < 0.01:
        text = '**'
    elif pval < 0.05:
        text = '*'
    else:
        text = f""

    # Get x-locations
    x1, x2 = conds.tolist().index(cond1), conds.tolist().index(cond2)
    y = y_max + y_step * (i + 1)
    ax.plot([x1, x1, x2, x2], [y, y + y_step/3, y + y_step/3, y], lw=1.5, c='k')
    ax.text((x1 + x2)/2, y-y_step*.2, text, ha='center', va='bottom', fontsize=40)
    ax.text((x1 + x2)/2, y-y_step*.3, f'{pval:.3g}', ha='center', va='bottom', fontsize=12)

# other spatially tuned

# Plotting early
ax = axs[2]
sns.stripplot(data=df_delta, x='condition', y='delta_other_sp', ax=ax, 
              palette=pl, size=s, alpha=a)
sns.barplot(data=df_delta, x='condition', y='delta_other_sp', ax=ax, 
            palette=pl, fill=False,errorbar='se')
ax.set_xlabel('')
ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'], rotation=20)
ax.spines[['top', 'right']].set_visible(False)
ax.set_title('Other spatially tuned')
# --- Stats + annotation ---
data=df_delta
conds = data['condition'].unique()
pairs = list(combinations(conds, 2))[:2]
y_max = data['delta_other_sp'].quantile(.99)
y_step = 0.4 * abs(y_max)

for i, (cond1, cond2) in enumerate(pairs):
    vals1 = data[data['condition'] == cond1]['delta_other_sp']
    vals2 = data[data['condition'] == cond2]['delta_other_sp']
    stat, pval = scipy.stats.ranksums(vals1, vals2)
    # Annotation text
    if pval < 0.001:
        text = '***'
    elif pval < 0.01:
        text = '**'
    elif pval < 0.05:
        text = '*'
    else:
        text = f""

    # Get x-locations
    x1, x2 = conds.tolist().index(cond1), conds.tolist().index(cond2)
    y = y_max + y_step * (i + 1)
    ax.plot([x1, x1, x2, x2], [y, y + y_step/3, y + y_step/3, y], lw=1.5, c='k')
    ax.text((x1 + x2)/2, y-y_step*.2, text, ha='center', va='bottom', fontsize=40)
    ax.text((x1 + x2)/2, y-y_step*.3, f'{pval:.3g}', ha='center', va='bottom', fontsize=12)
# Save the plot
#%%
# just place and other 
pl ={'ctrl': "slategray", 'vip': 'red', 'vip_ex': 'darkgoldenrod'}

fig,axes=plt.subplots(ncols=2,figsize=(7,4),sharey=True)
ax=axes[0]
sns.stripplot(data=df_delta, x='condition', y='delta_combined', ax=ax, 
              palette=pl, size=s, alpha=a)
sns.barplot(data=df_delta, x='condition', y='delta_combined', ax=ax, 
            palette=pl, fill=False, errorbar='se')
ax.set_ylabel('$\Delta$ Cell % (LEDon-off)')
ax.set_xlabel('')
ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'], rotation=20)
ax.spines[['top', 'right']].set_visible(False)
ax.set_title('All trials\nPlace')

# --- Stats + annotation ---
data = df_delta
conds = data['condition'].unique()
pairs = list(combinations(conds, 2))[:2]
y_max = data['delta_combined'].quantile(.85)
y_step = 0.4 * abs(y_max)

for i, (cond1, cond2) in enumerate(pairs):
    vals1 = data[data['condition'] == cond1]['delta_combined']
    vals2 = data[data['condition'] == cond2]['delta_combined']
    stat, pval = scipy.stats.ranksums(vals1, vals2)

    if pval < 0.001:
        text = '***'
    elif pval < 0.01:
        text = '**'
    elif pval < 0.05:
        text = '*'
    else:
        text = 'ns'

    x1, x2 = conds.tolist().index(cond1), conds.tolist().index(cond2)
    y = y_max + y_step * (i + 1)
    ax.plot([x1, x1, x2, x2], [y, y + y_step/3, y + y_step/3, y], lw=1.5, c='k')
    ax.text((x1 + x2)/2, y + y_step * .5, text, ha='center', va='bottom', fontsize=20)
    ax.text((x1 + x2)/2, y - y_step * .3, f'{pval:.3g}', ha='center', va='bottom', fontsize=12)
ax=axes[1]
sns.stripplot(data=df_delta, x='condition', y='delta_other_sp', ax=ax, 
              palette=pl, size=s, alpha=a)
sns.barplot(data=df_delta, x='condition', y='delta_other_sp', ax=ax, 
            palette=pl, fill=False,errorbar='se')
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'], rotation=20)
ax.spines[['top', 'right']].set_visible(False)
ax.set_title('Other spatially tuned')
# --- Stats + annotation ---
data=df_delta
conds = data['condition'].unique()
pairs = list(combinations(conds, 2))[:2]
y_max = data['delta_other_sp'].quantile(.99)
y_step = 0.4 * abs(y_max)

for i, (cond1, cond2) in enumerate(pairs):
    vals1 = data[data['condition'] == cond1]['delta_other_sp']
    vals2 = data[data['condition'] == cond2]['delta_other_sp']
    stat, pval = scipy.stats.ranksums(vals1, vals2)
    # Annotation text
    if pval < 0.001:
        text = '***'
    elif pval < 0.01:
        text = '**'
    elif pval < 0.05:
        text = '*'
    else:
        text = f"ns"

    # Get x-locations
    x1, x2 = conds.tolist().index(cond1), conds.tolist().index(cond2)
    y = y_max + y_step * (i + 1)
    ax.plot([x1, x1, x2, x2], [y, y + y_step/3, y + y_step/3, y], lw=1.5, c='k')
    ax.text((x1 + x2)/2, y+y_step*.5, text, ha='center', va='bottom', fontsize=20)
    ax.text((x1 + x2)/2, y-y_step*.3, f'{pval:.2g}', ha='center', va='bottom', fontsize=12)
#%% 
pl ={'ctrl': "slategray", 'vip': 'red', 'vip_ex': 'darkgoldenrod'}

# correlate with rates diff
beh = pd.read_csv(r'C:\Users\Han\Documents\MATLAB\vip_reward_cell_copy_w_raw_data\raw_data\cell\fig7f.csv')
beh=beh[(beh.animals.isin(df.animals.values))&(beh.days.isin(df.days.values))]
beh = beh.groupby(['animals', 'opto']).mean(numeric_only=True).reset_index()
beh=beh[beh.opto==True]
# take all trial cells
# y = np.nanmean([df_an.loc[(df_an.opto==True), 'place_cell_prop_early'].values,df_an.loc[(df_an.opto==True), 'place_cell_prop'].values],axis=0)
# average sp and place
df_an['sp_av_prop'] = df_an[['place_cell_prop','other_sp_prop']].max(axis=1)
# Compute the difference: on - off
# spatially tuned (not place)
diff = (
    df_an.pivot(index="animals", columns="opto", values="sp_av_prop")
    .assign(difference=lambda x: x[True] - x[False])
    .reset_index()[["animals", "difference"]]
)
y=diff.difference.values
# y=df_delta.delta_early
# Perform regression
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(beh.rates_diff.values, y)
print(f"Correlation (r) = {r_value:.4f}, p-value = {p_value:.3g}")

# Plot scatter plot with regression line
fig,ax=plt.subplots(figsize=(4,4))
sns.scatterplot(x=beh.rates_diff.values, y=y,hue=df_an.loc[(df_an.opto==True),'condition'].values,s=300,alpha=.7,palette=pl,ax=ax)
ax.plot(beh.rates_diff.values, intercept + slope * beh.rates_diff.values, color='steelblue', label='Regression Line',linewidth=3)
ax.legend()
ax.set_xlabel("$\Delta$% Correct trials (LEDon-LEDoff)")
ax.set_ylabel("$\Delta$ Place cell % (LEDon)")
ax.set_title(f"r={r_value:.3g}, p={p_value:.3g}",fontsize=16)
ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('Place cell vs. task performance')

"""
vip behavior
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
plt.rc('font', size=20)          # controls default text sizes
from statsmodels.formula.api import ols
import scipy.stats as stats, itertools
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.anova import anova_lm  # <-- Correct import

df=pd.read_csv(r'C:\Users\Han\Documents\MATLAB\vip_reward_cell\raw_data\fig4i_exfig4.csv')
# plot rates vip vs. ctl led off and on
bigdf_plot = df.groupby(['animals', 'condition', 'opto']).mean(numeric_only=True).reset_index()
# 
# Pairwise Mann-Whitney U testsn (Wilcoxon rank-sum)
conds = ['ctrl', 'vip', 'vip_ex']
comparisons = list(itertools.combinations(conds, 2))[:-1]
#%%
plt.rc('font', size=18)
# correct trials
# Plot
a=.7
s=10
fig, axes = plt.subplots(ncols = 2, figsize=(7.5,4.5),width_ratios=[1.8,1])
# expand opto vs. prev
ax=axes[0]
dfex = pd.DataFrame()
dfex['rates'] = np.concatenate([bigdf_plot.rates_opto, bigdf_plot.rates_prev])
dfex['rates']=dfex['rates']*100
dfex['condition'] = np.concatenate([bigdf_plot.condition]*2)
dfex['animal'] = np.concatenate([bigdf_plot.animals]*2)
dfex['ep'] = np.concatenate([['opto']*len(bigdf_plot), ['prev']*len(bigdf_plot)])
pl=['k','slategray']
sns.barplot(x="condition", y="rates", hue='ep', data=dfex,hue_order=['prev','opto'],
             errorbar='se', fill=False, ax=ax,palette=pl,legend=False)
ax.spines[['top', 'right']].set_visible(False)
ax.set_ylabel('% Correct trials')
ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'], rotation=20)
ax.set_xlabel('')
# --- Add connecting lines ---
# Get the x positions of the dodge stripplot points
pos = {('prev', 0): -0.2, ('opto', 0): 0.2}  # shift per hue
for cond in dfex['condition'].unique():
    cond_mask = dfex['condition'] == cond
    for animal, subdf in dfex[cond_mask].groupby('animal'):
        if {'prev', 'opto'} <= set(subdf['ep']):  # only draw if both present
            x = [list(dfex['condition'].unique()).index(cond) + pos[('prev', 0)],list(dfex['condition'].unique()).index(cond) + pos[('opto', 0)]]
            y = subdf.set_index('ep').loc[['prev', 'opto'], 'rates']
            ax.plot(x, y, color="gray", linewidth=1.5,alpha=0.5, zorder=0)


pl = {'ctrl': "slategray", 'vip': 'red', 'vip_ex':'darkgoldenrod'}
ax.set_ylim([30,100])
ax=axes[1]
sns.barplot(x="condition", y="rates_diff", hue='condition', data=bigdf_plot,
            palette=pl, errorbar='se', fill=False, ax=ax)
sns.stripplot(x="condition", y="rates_diff", hue='condition', data=bigdf_plot,
        palette=pl, alpha=a, s=s, ax=ax)
ax.spines[['top', 'right']].set_visible(False)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'], rotation=20)
ax.set_ylabel('% Correct trials\n(LEDoff-LEDon)')
ax.set_xlabel('')
p_vals = []
for c1, c2 in comparisons:
    x1 = bigdf_plot[bigdf_plot['condition'] == c1]['rates_diff'].dropna()
    x2 = bigdf_plot[bigdf_plot['condition'] == c2]['rates_diff'].dropna()
    stat, p = stats.ranksums(x1, x2, alternative='two-sided')
    p_vals.append(p)
# Correct for multiple comparisons
reject, p_vals_corrected, _, _ = multipletests(p_vals, method='fdr_bh')

# Add significance annotations
def add_sig(ax, group1, group2, y_pos, pval, xoffset=0.05):
    x1 = conds.index(group1)
    x2 = conds.index(group2)
    x_center = (x1 + x2) / 2
    plt.plot([x1, x1, x2, x2], [y_pos, y_pos + 1, y_pos + 1, y_pos], lw=1.5, color='black')
    if pval < 0.001:
        sig = '***'
    elif pval < 0.01:
        sig = '**'
    elif pval < 0.05:
        sig = '*'
    else:
        sig = ''
    plt.text(x_center, y_pos-4, sig, ha='center', va='bottom', fontsize=38)
    # plt.text(x_center, y_pos-2, f'p={pval:.3g}', ha='center', fontsize=8)

# Plot all pairwise comparisons
y_start = bigdf_plot['rates_diff'].max() + 1
gap = 5
for i, (c1, c2) in enumerate(comparisons):
    add_sig(ax, c1, c2, y_start, p_vals_corrected[i])
    y_start += gap
# ax.set_ylim([-35,10])
plt.tight_layout()
fig.suptitle('Entire epoch')
#%%
# Step 1: Calculate the means and standard deviations
group1=x1;group2=x2
mean1 = np.mean(group1)
mean2 = np.mean(group2)
std1 = np.std(group1, ddof=1)
std2 = np.std(group2, ddof=1)
# Step 2: Calculate pooled standard deviation
n1, n2 = len(group1), len(group2)
pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
# Step 3: Calculate Cohen's d
cohens_d = (mean1 - mean2) / pooled_std
# Step 4: Perform Power Analysis using the calculated Cohen's d
alpha = 0.05  # Significance level
power = 0.8   # Desired power
from statsmodels.stats import power as smp
analysis = smp.TTestIndPower()
sample_size = analysis.solve_power(effect_size=cohens_d, alpha=alpha, power=power, alternative='two-sided')
print(f"Cohen's d: {cohens_d:.4f}")
print(f"Required sample size per group: {sample_size:.2f}")
#%%
# velocity

fig, axes = plt.subplots(ncols = 2, figsize=(7.5,4.5),width_ratios=[1.8,1])
# expand opto vs. prev
ax=axes[0]
dfex = pd.DataFrame()
dfex['velocity'] = np.concatenate([bigdf_plot.velocity_opto, bigdf_plot.velocity_prev])
dfex['condition'] = np.concatenate([bigdf_plot.condition]*2)
dfex['animal'] = np.concatenate([bigdf_plot.animals]*2)
dfex['ep'] = np.concatenate([['opto']*len(bigdf_plot), ['prev']*len(bigdf_plot)])
pl=['k','slategray']
sns.barplot(x="condition", y="velocity", hue='ep', data=dfex,hue_order=['prev','opto'],
             errorbar='se', fill=False, ax=ax,palette=pl,legend=False)
ax.spines[['top', 'right']].set_visible(False)
ax.set_ylabel('Velocity (cm/s)')
ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'], rotation=20)
ax.set_xlabel('')
# --- Add connecting lines ---
# Get the x positions of the dodge stripplot points
pos = {('prev', 0): -0.2, ('opto', 0): 0.2}  # shift per hue
for cond in dfex['condition'].unique():
    cond_mask = dfex['condition'] == cond
    for animal, subdf in dfex[cond_mask].groupby('animal'):
        if {'prev', 'opto'} <= set(subdf['ep']):  # only draw if both present
            x = [list(dfex['condition'].unique()).index(cond) + pos[('prev', 0)],
                 list(dfex['condition'].unique()).index(cond) + pos[('opto', 0)]]
            y = subdf.set_index('ep').loc[['prev', 'opto'], 'velocity']
            ax.plot(x, y, color="gray", linewidth=1.5,alpha=0.5, zorder=0)

pl = {'ctrl': "slategray", 'vip': 'red', 'vip_ex':'darkgoldenrod'}
ax=axes[1]

sns.barplot(x="condition", y="velocity_diff", hue='condition', data=bigdf_plot,
            palette=pl, errorbar='se', fill=False, ax=ax)
sns.stripplot(x="condition", y="velocity_diff", hue='condition', data=bigdf_plot,
              palette=pl, alpha=a, s=s, ax=ax)
ax.spines[['top', 'right']].set_visible(False)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'], rotation=20)
ax.set_ylabel('Velocity (cm/s)\n(LEDoff-LEDon)')
ax.set_xlabel('')
# Add significance annotations
def add_sig(ax, group1, group2, y_pos, t,pval, xoffset=0.05):
    x1 = conds.index(group1)
    x2 = conds.index(group2)
    x_center = (x1 + x2) / 2
    plt.plot([x1, x1, x2, x2], [y_pos, y_pos + .5, y_pos + .5, y_pos], lw=1.5, color='black')
    if pval < 0.001:
        sig = '***'
    elif pval < 0.01:
        sig = '**'
    elif pval < 0.05:
        sig = '*'
    else:
        sig = ''
    plt.text(x_center, y_pos, sig, ha='center', va='bottom', fontsize=38)
    plt.text(x_center, y_pos, f't={t:.3g}\np={pval:.3g}', ha='center', fontsize=8)
# Pairwise Mann-Whitney U testsn (Wilcoxon rank-sum)
p_vals = []
tstat=[]
for c1, c2 in comparisons:
    x1 = bigdf_plot[bigdf_plot['condition'] == c1]['velocity_diff'].dropna()
    x2 = bigdf_plot[bigdf_plot['condition'] == c2]['velocity_diff'].dropna()
    stat, p = stats.ranksums(x1, x2, alternative='two-sided')
    p_vals.append(p)
    tstat.append(stat)
# Correct for multiple comparisons
reject, p_vals_corrected, _, _ = multipletests(p_vals, method='fdr_bh')

# Plot all pairwise comparisons
y_start = bigdf_plot['velocity_diff'].max()-1
gap = 2
for i, (c1, c2) in enumerate(comparisons):
    add_sig(ax, c1, c2, y_start, tstat[i],p_vals_corrected[i])
    y_start += gap
# ax.set_ylim([-35,10])
plt.tight_layout()
fig.suptitle('Pre-reward velocity')
#%%
# lick rate early
plt.rc('font', size=18)
# Plot
fig, axes = plt.subplots(ncols = 2, figsize=(7.5,4.5),width_ratios=[1.8,1])
# expand opto vs. prev
ax=axes[0]
dfex = pd.DataFrame()
dfex['lick_rate'] = np.concatenate([bigdf_plot.lick_rate_opto, bigdf_plot.lick_rate_prev])
dfex['condition'] = np.concatenate([bigdf_plot.condition]*2)
dfex['animal'] = np.concatenate([bigdf_plot.animals]*2)
dfex['ep'] = np.concatenate([['opto']*len(bigdf_plot), ['prev']*len(bigdf_plot)])
pl=['k','slategray']
sns.barplot(x="condition", y="lick_rate", hue='ep', data=dfex,hue_order=['prev','opto'],errorbar='se', fill=False, ax=ax,palette=pl,legend=False)
ax.spines[['top', 'right']].set_visible(False)
ax.set_ylabel('Lick rate (Hz)')
ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'], rotation=20)
ax.set_xlabel('')
# --- Add connecting lines ---
# Get the x positions of the dodge stripplot points
pos = {('prev', 0): -0.2, ('opto', 0): 0.2}  # shift per hue
for cond in dfex['condition'].unique():
    cond_mask = dfex['condition'] == cond
    for animal, subdf in dfex[cond_mask].groupby('animal'):
        if {'prev', 'opto'} <= set(subdf['ep']):  # only draw if both present
            x = [list(dfex['condition'].unique()).index(cond) + pos[('prev', 0)],
                    list(dfex['condition'].unique()).index(cond) + pos[('opto', 0)]]
            y = subdf.set_index('ep').loc[['prev', 'opto'], 'lick_rate']
            ax.plot(x, y, color="gray", linewidth=1.5,alpha=0.5, zorder=0)


pl = {'ctrl': "slategray", 'vip': 'red', 'vip_ex':'darkgoldenrod'}
ax=axes[1]
sns.barplot(x="condition", y="lick_rate_diff", hue='condition', data=bigdf_plot,
            palette=pl, errorbar='se', fill=False, ax=ax)
sns.stripplot(x="condition", y="lick_rate_diff", hue='condition', data=bigdf_plot,
        palette=pl, alpha=a, s=s, ax=ax)
ax.spines[['top', 'right']].set_visible(False)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'], rotation=20)
ax.set_ylabel('Lick rate (Hz)')
ax.set_xlabel('')
p_vals = []
stats=[]
for c1, c2 in comparisons:
    x1 = bigdf_plot[bigdf_plot['condition'] == c1]['lick_rate_diff'].dropna()
    x2 = bigdf_plot[bigdf_plot['condition'] == c2]['lick_rate_diff'].dropna()
    stat, p = scipy.stats.ranksums(x1, x2, alternative='two-sided')
    p_vals.append(p)
    stats.append(stat)
# Correct for multiple comparisons
reject, p_vals_corrected, _, _ = multipletests(p_vals, method='fdr_bh')

# Add significance annotations
def add_sig(ax, group1, group2, y_pos, t,pval, xoffset=0.05,h=0.1):
    x1 = conds.index(group1)
    x2 = conds.index(group2)
    x_center = (x1 + x2) / 2
    plt.plot([x1, x1, x2, x2], [y_pos, y_pos + h, y_pos + h, y_pos], lw=1.5, color='black')
    if pval < 0.001:
        sig = '***'
    elif pval < 0.01:
        sig = '**'
    elif pval < 0.05:
        sig = '*'
    else:
        sig = ''
    plt.text(x_center, y_pos, sig, ha='center', va='bottom', fontsize=38)
    plt.text(x_center, y_pos, f't={t:.3g}\np={pval:.3g}', ha='center', fontsize=8)
# Plot all pairwise comparisons
y_start = bigdf_plot['lick_rate_diff'].max()
gap = .5
for i, (c1, c2) in enumerate(comparisons):
    add_sig(ax, c1, c2, y_start, stats[i],p_vals_corrected[i])
    y_start += gap
# ax.set_ylim([-35,10])
plt.tight_layout()
fig.suptitle('Pre-reward lick rate, first 8 trials')
#%%

# lick rate late
plt.rc('font', size=18)
# Plot
fig, axes = plt.subplots(ncols = 2, figsize=(7.5,4.5),width_ratios=[1.8,1])
# expand opto vs. prev
ax=axes[0]
dfex = pd.DataFrame()
dfex['lick_rate'] = np.concatenate([bigdf_plot.lick_rate_late_opto, bigdf_plot.lick_rate_late_prev])
dfex['condition'] = np.concatenate([bigdf_plot.condition]*2)
dfex['animal'] = np.concatenate([bigdf_plot.animals]*2)
dfex['ep'] = np.concatenate([['opto']*len(bigdf_plot), ['prev']*len(bigdf_plot)])
pl=['k','slategray']
sns.barplot(x="condition", y="lick_rate", hue='ep', data=dfex,hue_order=['prev','opto'],errorbar='se', fill=False, ax=ax,palette=pl,legend=False)
ax.spines[['top', 'right']].set_visible(False)
ax.set_ylabel('Lick rate (Hz)')
ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'], rotation=20)
ax.set_xlabel('')
# --- Add connecting lines ---
# Get the x positions of the dodge stripplot points
pos = {('prev', 0): -0.2, ('opto', 0): 0.2}  # shift per hue
for cond in dfex['condition'].unique():
    cond_mask = dfex['condition'] == cond
    for animal, subdf in dfex[cond_mask].groupby('animal'):
        if {'prev', 'opto'} <= set(subdf['ep']):  # only draw if both present
            x = [list(dfex['condition'].unique()).index(cond) + pos[('prev', 0)],
                    list(dfex['condition'].unique()).index(cond) + pos[('opto', 0)]]
            y = subdf.set_index('ep').loc[['prev', 'opto'], 'lick_rate']
            ax.plot(x, y, color="gray", linewidth=1.5,alpha=0.5, zorder=0)


pl = {'ctrl': "slategray", 'vip': 'red', 'vip_ex':'darkgoldenrod'}
ax=axes[1]
sns.barplot(x="condition", y="lick_rate_late_diff", hue='condition', data=bigdf_plot,
            palette=pl, errorbar='se', fill=False, ax=ax)
sns.stripplot(x="condition", y="lick_rate_late_diff", hue='condition', data=bigdf_plot,
        palette=pl, alpha=a, s=s, ax=ax)
ax.spines[['top', 'right']].set_visible(False)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'], rotation=20)
ax.set_ylabel('Lick rate (Hz)')
ax.set_xlabel('')
p_vals = []
stats=[]
for c1, c2 in comparisons:
    x1 = bigdf_plot[bigdf_plot['condition'] == c1]['lick_rate_late_diff'].dropna()
    x2 = bigdf_plot[bigdf_plot['condition'] == c2]['lick_rate_late_diff'].dropna()
    stat, p = scipy.stats.ranksums(x1, x2, alternative='two-sided')
    p_vals.append(p)
    stats.append(stat)
# Correct for multiple comparisons
reject, p_vals_corrected, _, _ = multipletests(p_vals, method='fdr_bh')

# Add significance annotations
def add_sig(ax, group1, group2, y_pos, t,pval, xoffset=0.05,h=0.1):
    x1 = conds.index(group1)
    x2 = conds.index(group2)
    x_center = (x1 + x2) / 2
    plt.plot([x1, x1, x2, x2], [y_pos, y_pos + h, y_pos + h, y_pos], lw=1.5, color='black')
    if pval < 0.001:
        sig = '***'
    elif pval < 0.01:
        sig = '**'
    elif pval < 0.05:
        sig = '*'
    else:
        sig = ''
    plt.text(x_center, y_pos, sig, ha='center', va='bottom', fontsize=38)
    plt.text(x_center, y_pos, f't={t:.3g}\np={pval:.3g}', ha='center', fontsize=8)
# Plot all pairwise comparisons
y_start = bigdf_plot['lick_rate_late_diff'].max()
gap = .5
for i, (c1, c2) in enumerate(comparisons):
    add_sig(ax, c1, c2, y_start, stats[i],p_vals_corrected[i])
    y_start += gap
# ax.set_ylim([-35,10])
plt.tight_layout()
fig.suptitle('Pre-reward lick rate, last 8 trials')
plt.savefig(os.path.join(savedst, 'lick_rate_opto_last8.svg'), bbox_inches='tight')
#%% 
# lick selectivity early
fig, axes = plt.subplots(ncols = 2, figsize=(7.5,4.5),width_ratios=[1.8,1])
# expand opto vs. prev
ax=axes[0]
dfex = pd.DataFrame()
dfex['lick_selectivity_early'] = np.concatenate([bigdf_plot.lick_selectivity_early_opto, bigdf_plot.lick_selectivity_early_prev])
dfex['condition'] = np.concatenate([bigdf_plot.condition]*2)
dfex['animal'] = np.concatenate([bigdf_plot.animals]*2)
dfex['ep'] = np.concatenate([['opto']*len(bigdf_plot), ['prev']*len(bigdf_plot)])
pl=['k','slategray']
sns.barplot(x="condition", y="lick_selectivity_early", hue='ep', data=dfex,hue_order=['prev','opto'],errorbar='se', fill=False, ax=ax,palette=pl,legend=False)
ax.spines[['top', 'right']].set_visible(False)
ax.set_ylabel('Lick selectivity')
ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'], rotation=20)
ax.set_xlabel('')
# --- Add connecting lines ---
# Get the x positions of the dodge stripplot points
pos = {('prev', 0): -0.2, ('opto', 0): 0.2}  # shift per hue
for cond in dfex['condition'].unique():
    cond_mask = dfex['condition'] == cond
    for animal, subdf in dfex[cond_mask].groupby('animal'):
        if {'prev', 'opto'} <= set(subdf['ep']):  # only draw if both present
            x = [list(dfex['condition'].unique()).index(cond) + pos[('prev', 0)],
                    list(dfex['condition'].unique()).index(cond) + pos[('opto', 0)]]
            y = subdf.set_index('ep').loc[['prev', 'opto'], 'lick_selectivity_early']
            ax.plot(x, y, color="gray", linewidth=1.5,alpha=0.5, zorder=0)


pl = {'ctrl': "slategray", 'vip': 'red', 'vip_ex':'darkgoldenrod'}
ax=axes[1]
sns.barplot(x="condition", y="lick_selectivity_early_diff", hue='condition', data=bigdf_plot,
            palette=pl, errorbar='se', fill=False, ax=ax)
sns.stripplot(x="condition", y="lick_selectivity_early_diff", hue='condition', data=bigdf_plot,
        palette=pl, alpha=a, s=s, ax=ax)
ax.spines[['top', 'right']].set_visible(False)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'], rotation=20)
ax.set_ylabel('Lick selectivity')
ax.set_xlabel('')
p_vals = []
stats=[]
for c1, c2 in comparisons:
    x1 = bigdf_plot[bigdf_plot['condition'] == c1]['lick_selectivity_early_diff'].dropna()
    x2 = bigdf_plot[bigdf_plot['condition'] == c2]['lick_selectivity_early_diff'].dropna()
    stat, p = scipy.stats.ranksums(x1, x2, alternative='two-sided')
    p_vals.append(p)
    stats.append(stat)
# Correct for multiple comparisons
reject, p_vals_corrected, _, _ = multipletests(p_vals, method='fdr_bh')

# Add significance annotations
def add_sig(ax, group1, group2, y_pos, t,pval, xoffset=0.05,h=0.01):
    x1 = conds.index(group1)
    x2 = conds.index(group2)
    x_center = (x1 + x2) / 2
    plt.plot([x1, x1, x2, x2], [y_pos, y_pos + h, y_pos + h, y_pos], lw=1.5, color='black')
    if pval < 0.001:
        sig = '***'
    elif pval < 0.01:
        sig = '**'
    elif pval < 0.05:
        sig = '*'
    else:
        sig = ''
    plt.text(x_center, y_pos, sig, ha='center', va='bottom', fontsize=38)
    plt.text(x_center, y_pos, f't={t:.3g}\np={pval:.3g}', ha='center', fontsize=8)
# Plot all pairwise comparisons
y_start = bigdf_plot['lick_selectivity_early_diff'].max()
gap = .05
for i, (c1, c2) in enumerate(comparisons):
    add_sig(ax, c1, c2, y_start, stats[i],p_vals_corrected[i])
    y_start += gap
# ax.set_ylim([-35,10])
plt.tight_layout()
fig.suptitle('Lick selectivity, first 8 trials')
#%%
#lick selectivity late
# # Plot
fig, axes = plt.subplots(ncols = 2, figsize=(7.5,4.5),width_ratios=[1.8,1])
# expand opto vs. prev
ax=axes[0]
dfex = pd.DataFrame()
dfex['lick_selectivity'] = np.concatenate([bigdf_plot.lick_selectivity_opto, bigdf_plot.lick_selectivity_prev])
dfex['condition'] = np.concatenate([bigdf_plot.condition]*2)
dfex['animal'] = np.concatenate([bigdf_plot.animals]*2)
dfex['ep'] = np.concatenate([['opto']*len(bigdf_plot), ['prev']*len(bigdf_plot)])
pl=['k','slategray']
sns.barplot(x="condition", y="lick_selectivity", hue='ep', data=dfex,hue_order=['prev','opto'],errorbar='se', fill=False, ax=ax,palette=pl,legend=False)
ax.spines[['top', 'right']].set_visible(False)
ax.set_ylabel('Lick selectivity')
ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'], rotation=20)
ax.set_xlabel('')
# --- Add connecting lines ---
# Get the x positions of the dodge stripplot points
pos = {('prev', 0): -0.2, ('opto', 0): 0.2}  # shift per hue
for cond in dfex['condition'].unique():
    cond_mask = dfex['condition'] == cond
    for animal, subdf in dfex[cond_mask].groupby('animal'):
        if {'prev', 'opto'} <= set(subdf['ep']):  # only draw if both present
            x = [list(dfex['condition'].unique()).index(cond) + pos[('prev', 0)],
                    list(dfex['condition'].unique()).index(cond) + pos[('opto', 0)]]
            y = subdf.set_index('ep').loc[['prev', 'opto'], 'lick_selectivity']
            ax.plot(x, y, color="gray", linewidth=1.5,alpha=0.5, zorder=0)


pl = {'ctrl': "slategray", 'vip': 'red', 'vip_ex':'darkgoldenrod'}
ax=axes[1]
sns.barplot(x="condition", y="lick_selectivity_diff", hue='condition', data=bigdf_plot,
            palette=pl, errorbar='se', fill=False, ax=ax)
sns.stripplot(x="condition", y="lick_selectivity_diff", hue='condition', data=bigdf_plot,
        palette=pl, alpha=a, s=s, ax=ax)
ax.spines[['top', 'right']].set_visible(False)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'], rotation=20)
ax.set_ylabel('Lick selectivity')
ax.set_xlabel('')
p_vals = []
stats=[]
for c1, c2 in comparisons:
    x1 = bigdf_plot[bigdf_plot['condition'] == c1]['lick_selectivity_diff'].dropna()
    x2 = bigdf_plot[bigdf_plot['condition'] == c2]['lick_selectivity_diff'].dropna()
    stat, p = scipy.stats.ranksums(x1, x2, alternative='two-sided')
    p_vals.append(p)
    stats.append(stat)
# Correct for multiple comparisons
reject, p_vals_corrected, _, _ = multipletests(p_vals, method='fdr_bh')

# Add significance annotations
def add_sig(ax, group1, group2, y_pos, t,pval, xoffset=0.05,h=0.01):
    x1 = conds.index(group1)
    x2 = conds.index(group2)
    x_center = (x1 + x2) / 2
    plt.plot([x1, x1, x2, x2], [y_pos, y_pos + h, y_pos + h, y_pos], lw=1.5, color='black')
    if pval < 0.001:
        sig = '***'
    elif pval < 0.01:
        sig = '**'
    elif pval < 0.05:
        sig = '*'
    else:
        sig = ''
    plt.text(x_center, y_pos, sig, ha='center', va='bottom', fontsize=38)
    plt.text(x_center, y_pos, f't={t:.3g}\np={pval:.3g}', ha='center', fontsize=8)
# Plot all pairwise comparisons
y_start = bigdf_plot['lick_selectivity_diff'].max()
gap = .05
for i, (c1, c2) in enumerate(comparisons):
    add_sig(ax, c1, c2, y_start, stats[i],p_vals_corrected[i])
    y_start += gap
# ax.set_ylim([-35,10])
plt.tight_layout()
fig.suptitle('Lick selectivity, last 8 trials')

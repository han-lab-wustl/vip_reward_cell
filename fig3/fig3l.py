"""
zahra
quantify reward-relative cells post reward
take all the cells and quantify their transient in rewarded vs. unrewarded stops
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
#%%
with open(r'C:\Users\Han\Documents\MATLAB\vip_reward_cell\raw_data\fig3l.p', "rb") as fp: #unpickle
   dct = pickle.load(fp)
nrewstops_wo_licks_trials_per_an=dct['nrewstops_wo_licks_trials_per_an']
nrewstops_w_licks_trials_per_an=dct['nrewstops_w_licks_trials_per_an']
rewstops_trials_per_an=dct['rewstops_trials_per_an']
# average activity of all cells
# get post rew / stop activity
secs_post_rew=10# window after stop
mov_start=4
range_val,binsize=10, .1
# average of all cells
# subtract from prewidnow
pre=-5
nrewstops_wo_licks_trials_per_an_av = [[np.nanmean(np.hstack(yy)[int((range_val/binsize)+(mov_start/binsize)):int((range_val/binsize)+(secs_post_rew/binsize))],axis=(1,0))-np.nanmean(np.hstack(yy)[int((range_val/binsize)+(pre/binsize)):int((range_val/binsize))],axis=(1,0)) for yy in xx] for xx in nrewstops_wo_licks_trials_per_an]
nrewstops_w_licks_trials_per_an_av = [[np.nanmean(np.hstack(yy)[int((range_val/binsize)+(mov_start/binsize)):int((range_val/binsize)+(secs_post_rew/binsize))],axis=(1,0))-np.nanmean(np.hstack(yy)[int((range_val/binsize)+(pre/binsize)):int((range_val/binsize))],axis=(1,0)) for yy in xx] for xx in nrewstops_w_licks_trials_per_an]
rewstops_trials_per_an_av = [[np.nanmean(np.hstack(yy)[int((range_val/binsize)+(mov_start/binsize)):int((range_val/binsize)+(secs_post_rew/binsize))],axis=(1,0))-np.nanmean(np.hstack(yy)[int((range_val/binsize)+(pre/binsize)):int((range_val/binsize))],axis=(1,0)) for yy in xx] for xx in rewstops_trials_per_an]


#%%
def wilcoxon_r(x, y):
    # x, y are paired arrays (same subjects)
    W, p = scipy.stats.wilcoxon(x, y, zero_method="wilcox")
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
 
animals=['e218', 'e216', 'e217', 'e200', 'e201', 'e186', 'e190', 'e189',
       'z8', 'z9', 'e145', 'e139', 'z16']
# plot
plt.rc('font', size=18) 
import itertools
df=pd.DataFrame()
nrewstops_wo_licks_trials_per_an_av_concat = np.concatenate(nrewstops_wo_licks_trials_per_an_av)
an_nrewstops_wo_licks_trials_per_an_av_concat = np.concatenate([[animals[ii]]*len(xx) for ii,xx in enumerate(nrewstops_wo_licks_trials_per_an_av)])
nrewstops_w_licks_trials_per_an_av_concat = np.concatenate(nrewstops_w_licks_trials_per_an_av)
an_nrewstops_w_licks_trials_per_an_av_concat = np.concatenate([[animals[ii]]*len(xx) for ii,xx in enumerate(nrewstops_w_licks_trials_per_an_av)])
rewstops_trials_per_an_av_concat = np.concatenate(rewstops_trials_per_an_av)
an_rewstops_trials_per_an_av_concat = np.concatenate([[animals[ii]]*len(xx) for ii,xx in enumerate(rewstops_trials_per_an_av)])

df['activity'] = np.concatenate([nrewstops_wo_licks_trials_per_an_av_concat,
            nrewstops_w_licks_trials_per_an_av_concat,rewstops_trials_per_an_av_concat])
df['trial_type'] = np.concatenate([['non_rewarded_stops_wo_licks']*len(nrewstops_wo_licks_trials_per_an_av_concat),
                ['non_reward_stops_w_licks']*len(nrewstops_w_licks_trials_per_an_av_concat),
                ['rewarded_stops']*len(rewstops_trials_per_an_av_concat)])
df['animal'] = np.concatenate([an_nrewstops_wo_licks_trials_per_an_av_concat,
            an_nrewstops_w_licks_trials_per_an_av_concat,an_rewstops_trials_per_an_av_concat])
df=df[(df.animal!='e189')&(df.animal!='e145')&(df.animal!='e139')] # gcamp 6 or missing trial types
# combined nonrew 
df=df.reset_index()
palette = {'rewarded_stops':'seagreen', 'non_rewarded_stops_wo_licks': 'firebrick', 'non_reward_stops_w_licks': 'sienna'}
# plot all cells
fig,ax=plt.subplots(figsize=(3.5,5))
    
df = df.groupby(['animal', 'trial_type']).mean().reset_index()
s=12
order=['non_rewarded_stops_wo_licks','non_reward_stops_w_licks','rewarded_stops']
# sns.stripplot(x='trial_type', y='activity',hue='trial_type', data=df, s=s, alpha=0.7,ax=ax,palette=palette,order=order)
sns.barplot(x='trial_type', y='activity',hue='trial_type', data=df, fill=False,ax=ax,palette=palette,order=order)
ax.spines[['top','right']].set_visible(False)

ax.set_xticklabels(ax.get_xticklabels(), rotation=20)
ans = df.animal.unique()
for i in range(len(ans)):
    ax = sns.lineplot(x='trial_type', y='activity', 
    data=df[df.animal==ans[i]],
    errorbar=None, color='dimgray', linewidth=1.5,alpha=0.5)
import statsmodels.api as sm
import statsmodels.formula.api as smf
import itertools
from statsmodels.stats.multitest import multipletests

# Perform ANOVA
# Group data by trial type
grouped = [group['activity'].values for name, group in df.groupby('trial_type')]
# Perform Kruskal-Wallis test
stat, pval = scipy.stats.kruskal(*grouped)
print(f'Kruskal-Wallis H={stat:.3f}, p={pval:.3g}')
# Get unique trial types
trial_types = df['trial_type'].unique()
# Create all pairwise comparisons
comparisons = list(itertools.combinations(trial_types, 2))
# Store p-values
p_values = []
test_results = []
for group1, group2 in comparisons:
    # Extract paired data (same animals)
    paired_data = df.pivot(index="animal", columns="trial_type", values="activity").dropna()
    x,y=paired_data[group2],paired_data[group1] 
    # Wilcoxon r (signed)
    r_w, p_w = wilcoxon_r(x, y)
    p_values.append(p_w)
    test_results.append([p_w, r_w])
    p_values.append(p_w)

# Apply Bonferroni correction
reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
# Print results
for (group1, group2), corrected_p in zip(comparisons, pvals_corrected):
    print(f"Paired t-test: {group1} vs {group2}, Corrected p-value: {corrected_p:.4f}")
    
# Get unique trial types and positions
group_names = df["trial_type"].unique()
group_positions = {name: i for i, name in enumerate(group_names)}

# Get max y-value for annotation positioning
y_max = df["activity"].max()-.1
y_offset = (y_max - df["activity"].min()) * 0.1  # Adjust spacing

# Define height adjustment for each comparison
bar_heights = [y_max + (i) * y_offset for i in range(len(comparisons))]

# Iterate through pairwise comparisons
for i, ((group1, group2), corrected_p) in enumerate(zip(comparisons, pvals_corrected)):
    x1, x2 = group_positions[group1], group_positions[group2]
    y = bar_heights[i]  # Assign height for this comparison

    # Draw significance bar
    plt.plot([x1, x1, x2, x2], [y, y + y_offset * 0.2, y + y_offset * 0.2, y], 'k', lw=1.5)
    
    # Annotate with corrected p-value
    p_text = f"p = {corrected_p:.3g}"    
    ax.text((x1 + x2) / 2, y + y_offset * 0.3, (f'{test_results[i][1]:.3g},{corrected_p:.3g}'), 
             ha='center', va='bottom', fontsize=10)
    ax.text((x1 + x2) / 2, y, '*', 
            ha='center', va='bottom', fontsize=42)

ax.set_xticklabels(['No licks','Licks','Rewarded'])
ax.set_ylabel('Mean $\Delta F/F$ (after-before mov.)')
ax.set_xlabel('Trial type')
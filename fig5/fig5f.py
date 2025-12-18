
#%%
"""
zahra
2025
dff by trial type
added all cell subtype function
also gets cosine sim
"""
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd, os
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
# plt.rc('font', size=16)          # controls default text sizes
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
 
bigdf=pd.read_csv(r'C:\Users\Han\Documents\MATLAB\vip_reward_cell_copy_w_raw_data\raw_data\cell\fig5f.csv')
cell_order = ['pre', 'post', 'far_pre', 'far_post', 'place']
fig,ax = plt.subplots(figsize=(7.5,4))
# sns.stripplot(x='cell_type', y='mean_dff', data=bigdf,hue='trial_type',
#         dodge=True,palette={'Correct':'seagreen', 'Incorrect': 'firebrick'},
#         s=s,alpha=0.7,order=cell_order)
sns.barplot(x='cell_type', y='mean_dff', data=bigdf,hue='trial_type',
        fill=False,palette={'Correct':'seagreen', 'Incorrect': 'firebrick'},
            order=cell_order,errorbar='se')

ax.spines[['top','right']].set_visible(False)
ax.set_ylabel('Mean $\Delta F/F$ (pre or post-reward)')
ax.set_xlabel('Cell type')

ax.set_xticklabels(['Pre', 'Post', 'Far pre', 'Far post', 'Place'])
# Use the last axis to get handles/labels
handles, labels = ax.get_legend_handles_labels()
# Create a single shared legend with title "Trial type" 
ax.legend(
    handles, labels,
    loc='upper left',
    bbox_to_anchor=(.7, 1),
    borderaxespad=0.,
    title='Trial type'
)
xpos = {ct: i for i, ct in enumerate(cell_order)}

# Draw dim gray connecting lines between paired trial types
for animal in bigdf['animal'].unique():
    for ct in cell_order:
        sub = bigdf[(bigdf['animal'] == animal) & (bigdf['cell_type'] == ct)]
        if len(sub) == 2:  # both trial types present
            # Get x locations for dodge-separated points
            x_base = xpos[ct]
            offsets = [-0.2, 0.2]  # match sns stripplot dodge
            y_vals = sub.sort_values('trial_type')['mean_dff'].values
            x_vals = [x_base + offset for offset in offsets]
            ax.plot(x_vals, y_vals, color='dimgray', alpha=0.5, linewidth=1)

# ans = bigdf.animal.unique()
# for i in range(len(ans)):
#     for j,tr in enumerate(np.unique(bigdf.cell_type.values)):
#         testdf= bigdf[(bigdf.animal==ans[i]) & (bigdf.cell_type==tr)]
#         ax = sns.lineplot(x='trial_type', y='mean_dff', 
#         data=testdf,
#         errorbar=None, color='dimgray', linewidth=2, alpha=0.7,ax=ax)
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM

# 1) Two-way repeated measures ANOVA
aov = AnovaRM(
    data=bigdf,
    depvar='mean_dff',
    subject='animal',
    within=['trial_type','cell_type']
).fit()
print(aov)    # F-stats and p-values for main effects and interaction
aov_table = aov.anova_table

# Print full p-values
with pd.option_context('display.precision', 10):
    print(aov_table)
# 2) Post-hoc paired comparisons: correct vs incorrect within each cell_type
posthoc = []
for ct in cell_order:
    sub = bigdf[bigdf['cell_type']==ct]
    cor = sub[sub['trial_type']=='Correct']['mean_dff'].values
    inc = sub[sub['trial_type']=='Incorrect']['mean_dff'].values
    t, p_unc = wilcoxon_r(cor, inc)
    posthoc.append({
        'cell_type': ct,
        't_stat':    t,
        'p_uncorrected': p_unc
    })

posthoc = pd.DataFrame(posthoc)
# Bonferroni
import statsmodels
reject, pvals_corrected, _, _ = statsmodels.stats.multitest.multipletests(posthoc['p_uncorrected'], method='fdr_bh')
posthoc['p_fdr_bh'] = pvals_corrected
posthoc['significant'] = reject
# map cell_type → x-position
xpos = {ct: i for i, ct in enumerate(cell_order)}
for _, row in posthoc.iterrows():
    x = xpos[row['cell_type']]
    y = bigdf[
        (bigdf['cell_type']==row['cell_type'])
    ]['mean_dff'].quantile(.7) + 0.1  # just above the tallest bar
    p = row['p_fdr_bh']
    stars = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
    ax.text(x, y, stars, ha='center', va='bottom', fontsize=42)    
    r=row['t_stat']
    ax.text(x, y, f'r={r:.3g}\np={p:.3g}', ha='center', va='bottom', fontsize=12)
# Assuming `axes` is a list of subplots and `ax` is the one with the legend (e.g., the last one)

# Example interpretation (fill in with your numbers)
# trial_type (Num DF = 1, Den DF = 9, F = 12.3, p = 0.006)
# -- There is a significant main effect of trial type: across all cell types, mean ΔF/F is different on correct vs. incorrect trials.

# cell_type (Num DF = 3, Den DF = 27, F = 8.7, p < 0.001)
# -- There is a significant main effect of cell type: some cell types have higher overall ΔF/F than others, regardless of trial outcome.

# trial_type × cell_type (Num DF = 3, Den DF = 27, F = 4.2, p = 0.014)
# -- The interaction is significant: the difference between correct vs. incorrect ΔF/F depends on which cell type you look at.

# Because the interaction is significant, you should then examine post-hoc tests (e.g., the paired comparisons you ran) to see for each cell type whether correct vs. incorrect is significant.
# plt.savefig(os.path.join(savedst, 'allcelltype_trialtype.svg'),bbox_inches='tight')
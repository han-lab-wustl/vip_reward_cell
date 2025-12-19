
"""
zahra
find proportion of cells that are considered reward cells for 
multiple epochs and days
1) get day 1 reward cells
2) get the next 2 days of reward cells
3) get proportion of cells that are reward cells across all the epochs
"""
#%%

import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd, matplotlib.backends.backend_pdf
import pickle, seaborn as sns, random, math, os, matplotlib as mpl
from collections import Counter
from itertools import combinations, chain
import scikit_posthocs as sp
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
# plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
df=pd.read_csv(r'C:\Users\Han\Documents\MATLAB\vip_reward_cell_copy_w_raw_data\raw_data\cell\sfig3f.csv')

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

color='indigo'
fig, ax = plt.subplots(figsize=(5,5))
sns.stripplot(x='epoch_number',y='reward_cell_p',data=df, dodge=True,color=color,
    alpha=0.7)
sns.barplot(x='epoch_number',y='reward_cell_p',data=df,fill=False,color='indigo',errorbar='se')
sns.barplot(data=df, # correct shift
        x='epoch_number', y='reward_cell_p_shuf',color='grey',
        alpha=0.5, err_kws={'color': 'grey'},errorbar=None,ax=ax,legend=False)

# make lines
ans = df.animal.unique()
for i in range(len(ans)):
    ax = sns.lineplot(x=df.epoch_number-2, y='reward_cell_p', 
    data=df[df.animal==ans[i]],
    errorbar=None, color='gray', linewidth=1.5,alpha=0.5)
ax.spines[['top','right']].set_visible(False)
ax.set_ylabel('Tracked place cell %')
ax.set_xlabel('# of epochs')

pvals = []; stats_=[]
epochs = [2,3,4]
for ep in epochs:
    dsub = df[df.epoch_number == ep]
    if 'reward_cell_p_shuf' in dsub.columns:
        stat, pval = wilcoxon_r(
            dsub['reward_cell_p'].values, 
            dsub['reward_cell_p_shuf'].values)
        stats_.append(stat)
        stat, pval = scipy.stats.wilcoxon(
            dsub['reward_cell_p'], 
            dsub['reward_cell_p_shuf'],
            alternative='greater'  # adjust if needed
        )
        pvals.append(pval)
    else:
        pvals.append(np.nan)
# 2. FDR correction
from statsmodels.stats.multitest import multipletests
rej, pvals_corr, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
# 3. Annotate plot
for i, ep in enumerate(epochs):
    dsub = df[df.epoch_number == ep]
    y_max = 25
    xpos = ep - 2  # match x-axis shift in lineplot
    # Choose significance level
    if pvals_corr[i] < 0.001:
        star = '***'
    if pvals_corr[i] < 0.01:
        star = '**'
    if pvals_corr[i] < 0.05:
        star = '*'
    if pvals_corr[i] > 0.05: star=''
    # Plot text if significant
    ax.text(xpos, y_max + 0.5, star, ha='center', va='bottom', fontsize=40)
ax.axvline(1.5, linestyle='--',color='k',linewidth=3)
ax.text(0.5, y_max+5, 'Day 1', ha='center', va='bottom', fontsize=20)
ax.text(2.5, y_max+5, 'Day 2', ha='center', va='bottom', fontsize=20)
ax.set_title('Next day cell tracking\n')
plt.tight_layout()

#%%
# plot com
dfplt=df
# dfplt = df[df.animal != 'e218'].dropna(subset=['average_com', 'epoch_number'])
fig, ax = plt.subplots(figsize=(4,5))
sns.barplot(x='epoch_number',y='average_com',data=dfplt,fill=False,color=color,errorbar='se')
# make lines
ans = dfplt.animal.unique()
for i in range(len(ans)):
    ax = sns.lineplot(x=dfplt.epoch_number-2, y='average_com', 
    data=dfplt[dfplt.animal==ans[i]],
    errorbar=None, color='gray', linewidth=1.5,alpha=0.5)
ax.spines[['top','right']].set_visible(False)
ax.set_ylabel('Mean center-of-mass - reward loc. (cm)')
ax.set_xlabel('# of epochs')
ax.axvline(1.5, color='k', linestyle='--', linewidth=3)
# test significance; epoch 2 vs. 3, 4 vs. 5 etc
# Define groups
group_early = dfplt[dfplt.epoch_number.isin([2])]
group_late = dfplt[dfplt.epoch_number.isin([4])]

# Average within animal across those epochs
early_avg = group_early.groupby('animal')['average_com'].mean()
late_avg = group_late.groupby('animal')['average_com'].mean()

# Align both groups to animals that have data in both
common_animals = early_avg.index.intersection(late_avg.index)
early_vals = early_avg.loc[common_animals].values
late_vals = late_avg.loc[common_animals].values
# --- Mann-Whitney U test ---
u_stat, p_val = scipy.stats.ranksums(early_vals[~np.isnan(early_vals)], late_vals[~np.isnan(late_vals)])
# --- Add comparison bar across Day 1 and Day 2 ---
y = 5  # adjust based on your data range
x1, x2 = 0, 2  # bar from first to last bar (epoch 2 to 5 are at x=0 to 3 after subtracting 2)
ax.plot([x1, x1, x2, x2], [y, y+5, y+5, y], lw=1.5, c='k')

# Statistical annotation
if p_val < 0.001:
    star = '***'
elif p_val < 0.01:
    star = '**'
elif p_val < 0.05:
    star = '*'
else:
    star = f"p={p_val:.2g}"

ax.text((x1 + x2)/2, y+0.02, star, ha='center', va='bottom', fontsize=40)

ax.text(0.5, 15, 'Day 1', ha='center', va='bottom', fontsize=20)
ax.text(2.3, 15, 'Day 2', ha='center', va='bottom', fontsize=20)
# ax.set_title('Next day cell tracking')
ax.axhline(0,linewidth=2,color='slategrey')
ax.text(2.2, -.05, 'Reward loc.', ha='center', va='bottom', fontsize=14,color='slategrey')
# plt.tight_layout()

#%%
# example
#print
# from projects.pyr_reward.rewardcell import create_mask_from_coordinates
# import cv2
# bigcom[(bigcom.animal=='e201')]
# cellnm=23

# bigcom[(bigcom.animal=='e201') & (bigcom.tracked_cell_id==cellnm)]
# tracked_lut, days= get_tracked_lut(celltrackpth,'e201',pln)
# animal='e201'
# days = [52,53]
# e201_eg = tracked_lut[days].iloc[cellnm]
# cmap = ['k','yellow']  # Choose your preferred colormap
# cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", cmap)
# # Convert to RGBA and set alpha = 0 for the lowest value
# cmap_with_alpha = cmap(np.linspace(0, 1, 256))  # Get 256 colors from colormap
# cmap_with_alpha[0, -1] = 0  # Set alpha=0 for the lowest value (index 0)
# # Create a new colormap with the adjusted alpha
# transparent_cmap = matplotlib.colors.ListedColormap(cmap_with_alpha)
# fig,axes = plt.subplots(ncols=len(days),nrows=2, figsize=(10,6),
#         gridspec_kw={'height_ratios': [3,1]})
# for ii,day in enumerate(days):
#     params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
#     fall = scipy.io.loadmat(params_pth, variable_names=['stat', 'ops', 'dFF'])
#     stat = fall['stat'][0]
#     # dtype=[('ypix', 'O'), ('xpix', 'O'), ('lam', 'O'), ('med', 'O'), 
#     # ('footprint', 'O'), ('mrs', 'O'), ('mrs0', 'O'), ('compact', 'O'), 
#     # ('solidity', 'O'), ('npix', 'O'), ('npix_soma', 'O'), 
#     # ('soma_crop', 'O'), ('overlap', 'O'), ('radius', 'O'), 
#     # ('aspect_ratio', 'O'), ('npix_norm_no_crop', 'O'), ('npix_norm', 'O'), 
#     # ('skew', 'O'), ('std', 'O'), ('neuropil_mask', 'O')])
#     celln = e201_eg[day]
#     stat_cll = stat[celln]
#     img = fall['ops']['meanImg'][0][0]
#     ypix = stat_cll['ypix'][0][0][0]
#     xpix = stat_cll['xpix'][0][0][0]
#     dFF = fall['dFF']
#     pad = 80  # how much to zoom out from the edges
#     ymin, ymax = max(0, ypix.min() - pad), min(img.shape[0], ypix.max() + pad)
#     xmin, xmax = max(0, xpix.min() - pad), min(img.shape[1], xpix.max() + pad)
#     coords = np.column_stack((xpix, ypix))  
#     mask,cmask,center=create_mask_from_coordinates(coords, 
#             img.shape)                
#     img_crop = img[ymin:ymax, xmin:xmax]
#     mask_crop =cmask[ymin:ymax, xmin:xmax]
#     axes[0,ii].imshow(img_crop, cmap='gray')
#     axes[0,ii].imshow(mask_crop,cmap=transparent_cmap,vmin=1)
#     axes[0,ii].axis('off')
#     axes[1,ii].plot(dFF[:,celln],color=color)
#     axes[1,ii].set_ylabel('$\Delta$ F/F')
#     axes[1,ii].set_xlabel('Time (min)')
#     ax=axes[1,ii]
#     # Get x-axis tick locations from the axes
#     xticks = ax.get_xticks()
#     # Only use the first and last ticks, scaled by 31.25
#     xtick_locs = [xticks[1], xticks[-2]]
#     xtick_labels = [f"{xticks[1]/31.25:.2f}", f"{(xticks[-2]/31.25/60):.2f}"]
#     # Set the ticks and labels
#     ax.set_xticks(xtick_locs)
#     ax.set_xticklabels(xtick_labels)
#     ax.spines[['top','right']].set_visible(False)
#     axes[0,ii].set_title(f'Session {ii+1}')
# fig.suptitle(f'Tracked place cell, animal {animal}')

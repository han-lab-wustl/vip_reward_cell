"""
fig 2 panel
"""
#%%
import os, pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt,scipy
import matplotlib.backends.backend_pdf, matplotlib as mpl, sys
from itertools import combinations
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
plt.rc('font', size=20)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.pyr_reward.rewardcell import wilcoxon_r

df_plt = pd.read_csv(r'C:\Users\Han\Documents\MATLAB\vip_reward_cell\raw_data\fig2fg.csv')
#%% combine pre and post

from statsmodels.stats.multitest import multipletests

plt.rc('font', size=20) 
# number of epochs vs. reward cell prop    
fig, ax = plt.subplots(figsize=(4,4))

order = ['Near', 'Far']
colors = [sns.color_palette("Dark2")[3], (0.6, 0.3, 0.4)]

# Average across animals

# Stripplot + barplot
sns.barplot(x='num_epochs', y='goal_cell_prop',
            data=df_plt, hue='cell_group', hue_order=order,
            palette=colors, fill=False, ax=ax, errorbar='se')
sns.barplot(data=df_plt, x='num_epochs', y='goal_cell_prop_shuffle',
            hue='cell_group', hue_order=order,
            color='grey', alpha=0.3, err_kws={'color': 'grey'}, 
            errorbar=None, ax=ax, legend=False)
# --- Stats ---
results = []
for ep in sorted(df_plt.num_epochs.unique()):
    for i, ct in enumerate(order):
        dsub = df_plt[(df_plt.cell_group == ct) & (df_plt.num_epochs == ep)]
        if len(dsub) >= 2:
            stat, pval = wilcoxon_r(dsub['goal_cell_prop'], dsub['goal_cell_prop_shuffle'])
            results.append({
                'epoch': ep, 'cell_type': ct, 'pval': pval,'stat':stat,
                'xidx': i, 'ymax': dsub[['goal_cell_prop','goal_cell_prop_shuffle']].values.max()
            })

raw_pvals = [r['pval'] for r in results]
rej, pvals_corr, _, _ = multipletests(raw_pvals, method='fdr_bh')

for r, pcorr in zip(results, pvals_corr):
    ep = r['epoch']
    ct = r['cell_type']
    xpos = ep - 2.3 + (.4 * r['xidx'])
    ymax = r['ymax'] - 5
    
    # Stars
    if pcorr < 0.001:
        star = '***'
    elif pcorr < 0.01:
        star = '**'
    elif pcorr < 0.05:
        star = '*'
    else:
        star = ''
    
    if star:
        ax.text(xpos+.1, ymax + 2.2, star, ha='center', fontsize=30)
    ax.text(xpos, ymax + 5, f'p={pcorr:.3g},r={r["stat"]:.3g}', ha='center', fontsize=8)


# --- Final cleanup ---
ax.spines[['top','right']].set_visible(False)
ax.set_ylabel('Reward cell %')
ax.set_xlabel('# of epochs')
ax.legend(title='Cell type', fontsize=14, title_fontsize=14)

#%%

tau_near_post = [3.0489636175501236,
 10,
 3.8341630353568457,
 3.633185663248898,
 1.887423357676723,
 2.024701367032217,
 4.101109387024986,
 1.394899512202025,
 8.285634427250967]

tau_far_post = [0.8124304077806286,
 1.3869893611756274,
 1.1258345007793356,
 1.121918482030937,
 1.2423106981283685,
 0.6417469607531997,
 2.8264561666017674,
 1.1917876466224495,
 0.9725972085268909]
tau_near_pre=[2.229871790705441,
 8.52567178670323,
 2.476094070017733,
 2.3564721703814375,
 1.2707059843024346,
 1.6940739823949498,
 5.334664718351216,
 1.668246974638461,
 3.832971336860993]
tau_far_pre=[0.6836690821310514,
 1.0352307170375459,
 0.7997558933924453,
 0.6322625138957906,
 1.0936686907074666,
 0.7142383247936395,
 1.4717390874448504,
 0.7429834561831965,
 0.9829182265060615]
animals=['e145', 'e186', 'e190', 'e201', 'e216', 'e218', 'z16', 'z8', 'z9']
df = pd.DataFrame({
    'animal': animals * 4,
    'tau': np.concatenate([tau_near_pre, tau_near_post, tau_far_pre, tau_far_post]),
    'location': (['Near']*len(animals) +
                  ['Near']*len(animals) +
                  ['Far']*len(animals) +
                  ['Far']*len(animals)),
   'cell_type': (['Pre-reward']*len(animals) +
                  ['Post-reward']*len(animals) +
                  ['Pre-reward']*len(animals) +
                  ['Post-reward']*len(animals)),
   'hue_cell_type': (['Near pre-reward']*len(animals) +
                  ['Near post-reward']*len(animals) +
                  ['Far pre-reward']*len(animals) +
                  ['Far post-reward']*len(animals))
})
# number of epochs vs. reward cell prop incl combinations    
# make sure outlier numbers aren't there?
# df=df[(df.tau<10) & (df.tau>0)]
# Plot lines for each animal
fig,ax=plt.subplots(figsize=(4,5))
hue_order=['Far','Near']
# Overlay barplot (mean ± SEM)
sns.barplot(x='cell_type', y='tau', hue='location',data=df, hue_order=hue_order,
            errorbar='se', fill=False, ax=ax,palette='Dark2')
ax.spines[['top', 'right']].set_visible(False)
ax.set_ylabel(r'Decay over epochs ($\tau$)')
ax.set_xlabel('')
ax.tick_params(axis='x', rotation=20)

# ----------------------------
# Optional: Paired test + annotation
# ----------------------------
grouped = df.groupby(['cell_type', 'location'])
pvals = {}
for ct in df['cell_type'].unique():
    near = df[(df.cell_type == ct) & (df.location == 'Near')].set_index('animal')['tau']
    far = df[(df.cell_type == ct) & (df.location == 'Far')].set_index('animal')['tau']
    common_animals = near.index.intersection(far.index)
    stat, pval = scipy.stats.wilcoxon(near[common_animals], far[common_animals])
    pvals[ct] = pval

# near pre v. post
pre = df[(df.cell_type == 'Pre-reward') & (df.location == 'Near')].set_index('animal')['tau']
post = df[(df.cell_type == 'Post-reward') & (df.location == 'Near')].set_index('animal')['tau']
common_animals = pre.index.intersection(post.index)
stat, pvalprevpost = scipy.stats.wilcoxon(post[common_animals], pre[common_animals])

# ----------------------------
# Add color bars or significance annotations
# ----------------------------
bar_width = 0.3
xticks = [0, 1]  # for 2 bars: Pre-reward, Post-reward
barh = 0.2
height_offset = 6

for i, ct in enumerate(df['cell_type'].unique()):
   if pvals[ct]>0.05:
      star=''
   if pvals[ct]<0.05:
      star='*'
   if pvals[ct]<.01:
      star='**'
   # Add p-value text
   # ax.text(i, height_offset + 0.2, f'p={pvals[ct]:.3f}', ha='center', va='bottom', fontsize=10)
   ax.text(i, height_offset + 0.2, star, ha='center', va='bottom', fontsize=40)
   ax.plot([i-.2,i-.2,i+.2,i+.2],[height_offset-barh,height_offset,height_offset,height_offset-barh],color='k')
# pre v post
# Add p-value text
height_offset=6.5
i=.4
ax.text(.5, height_offset, f'p={pvalprevpost:.3f}', ha='center', va='bottom', fontsize=10)
ax.plot([i-.2,i-.2,i+.8,i+.8],[height_offset-barh,height_offset,height_offset,height_offset-barh],color='k')
# X positions for bar centers
x_center = {'Pre-reward': 0, 'Post-reward': 1}
bar_offset = 0.2  # spacing between Near and Far within each cell type

for ct in df['cell_type'].unique():
    x = x_center[ct]
    for animal in df['animal'].unique():
         y_near = df[(df.cell_type == ct) & (df.location == 'Far') & (df.animal == animal)].tau.values[0]
         y_far  = df[(df.cell_type == ct) & (df.location == 'Near') & (df.animal == animal)].tau.values[0]
         ax.plot([x - bar_offset, x + bar_offset], [y_near, y_far],
                 
                  color='gray', alpha=0.5, linewidth=1.5)
plt.tight_layout()
#%%
#%% Combine Pre- and Post-reward
df_combined = df.groupby(['animal','location'])['tau'].mean().reset_index()

# Plot
fig, ax = plt.subplots(figsize=(3,5))
order=['Near','Far']
# Overlay barplot (mean ± SEM)
sns.barplot(x='location', y='tau', data=df_combined, palette=colors, errorbar='se', ax=ax,order=order,fill=False)

# Add connecting lines for each animal
for animal in df_combined['animal'].unique():
    y_near = df_combined[(df_combined.animal==animal) & (df_combined.location=='Near')]['tau'].values[0]
    y_far  = df_combined[(df_combined.animal==animal) & (df_combined.location=='Far')]['tau'].values[0]
    ax.plot([0,1], [y_near, y_far], color='gray', alpha=0.5, linewidth=1.5)

near = df[(df.location == 'Near')].set_index('animal')['tau']
far = df[(df.location == 'Far')].set_index('animal')['tau']
common_animals = near.index.intersection(far.index)
stat, pval = wilcoxon_r(near[common_animals], far[common_animals])

# Add significance starn
height = max(df_combined['tau'])
if pval < 0.001:
    star = '***'
elif pval < 0.01:
    star = '**'
elif pval < 0.05:
    star = '*'
else:
    star = ''
ax.text(0.5, height, star, ha='center', va='bottom', fontsize=30)
ax.text(0.5, height, f'{stat:.3g}, {pval:.3g}', ha='center', va='bottom', fontsize=8)

ax.plot([0,0,1,1], [height-0.2,height,height,height-0.2], color='k')


ax.set_ylabel(r'Decay over epochs ($\tau$)')
ax.set_xlabel('')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
# Get summary statistics including SEM
summary = df.groupby(['location'])['tau'].agg([
    'count', 
    'mean', 
    'std', 
    ('sem', lambda x: x.sem()),  # Standard Error of Mean
    'median', 
    'min', 
    'max'
])

print(summary)

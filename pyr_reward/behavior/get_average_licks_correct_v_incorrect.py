
#%%
"""
get average licks correct v incorrect and see if they
correspond to pre-reward activity?
"""
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd, os
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"]=10
mpl.rcParams["ytick.major.size"]=10
# plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.opto.behavior.behavior import get_success_failure_trials
from projects.pyr_reward.rewardcell import get_radian_position,licks_by_trialtype, wilcoxon_r
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
#%%
goal_cm_window=20 # to search for rew cells
lasttr=8 #  last trials
bins=90
# iterate through all animals
dfs = []
tcs_correct_all=[]
tcs_fail_all=[]
lrates_all=[]
vel_all=[]
for ii in range(len(conddf)):
    day = conddf.days.values[ii]
    animal = conddf.animals.values[ii]
    if (animal!='e217') & (conddf.optoep.values[ii]<2):
        if animal=='e145' or animal=='e139': pln=2 
        else: pln=0
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
        tcs_correct,tcs_fail,tcs_probes,tcs_correct_vel, tcs_fail_vel,tcs_probes_vel, lick_rate,vel=licks_by_trialtype(params_pth, animal,bins=90)
        tcs_correct_all.append([tcs_correct,tcs_correct_vel])
        tcs_fail_all.append([tcs_fail,tcs_fail_vel,tcs_probes,tcs_probes_vel])
        lrates_all.append(lick_rate)
        vel_all.append(vel)
#%%
# lick rate correct vs. incorrect
# per epoch
plt.rc('font', size=20) 
lick_rate_corr = [[np.nanmean(yy[0]) for yy in xx] for xx in lrates_all]
lick_rate_incorr_pre = [[np.nanmean(yy[1]) for yy in xx] for xx in lrates_all]
lick_rate_incorr = [[np.nanmean(yy[2]) for yy in xx] for xx in lrates_all]
conddf=conddf.copy()
conddf=conddf[(conddf.animals!='e217') & (conddf.optoep<2)]
df=pd.DataFrame()
df['lick_rate_corr']=np.concatenate(lick_rate_corr)
df['animal']=np.concatenate([[conddf.animals.values[jj]]*len(xx) for jj, xx in enumerate(lick_rate_corr)])
df['day']=np.concatenate([[conddf.days.values[jj]]*len(xx) for jj, xx in enumerate(lick_rate_corr)])
df['lick_rate_incorr_pre']=np.concatenate(lick_rate_incorr_pre)
df['lick_rate_incorr']=np.concatenate(lick_rate_incorr)
df_long = df.melt(
    id_vars=['animal', 'day'],
    value_vars=['lick_rate_corr', 'lick_rate_incorr_pre','lick_rate_incorr'],
    var_name='trial_type',
    value_name='lick_rate'
)

# Clean up trial_type labels
df_long['trial_type'] = df_long['trial_type'].map({
    'lick_rate_corr': 'Correct Pre-Reward',
    'lick_rate_incorr_pre': 'Incorrect Pre-Reward',
    'lick_rate_incorr': 'Incorrect All'
})
df=df_long.groupby(['animal','trial_type']).mean(numeric_only=True)
pl=['seagreen','firebrick','firebrick']
order=['Correct Pre-Reward','Incorrect Pre-Reward','Incorrect All']
fig,ax=plt.subplots(figsize=(3,4))
sns.barplot(x='trial_type',y='lick_rate',data=df,fill=False,palette=pl,order=order,errorbar='se')
# sns.stripplot(x='trial_type',y='lick_rate',data=df,palette=pl)
# --- Draw connecting lines per animal ---
df_wide = df.reset_index().pivot(index='animal', columns='trial_type', values='lick_rate')
x_pos = {'Correct Pre-Reward': 0, 'Incorrect Pre-Reward': 1,
         'Incorrect All': 2}  # x-axis positions

for animal, row in df_wide.iterrows():
    x_vals = [x_pos['Correct Pre-Reward'], x_pos['Incorrect Pre-Reward'],x_pos['Incorrect All']]
    y_vals = [row['Correct Pre-Reward'], row['Incorrect Pre-Reward'],row['Incorrect All']]
    ax.plot(x_vals, y_vals, color='gray', alpha=0.5, linewidth=1.5)
ax.spines[['top','right']].set_visible(False)
ax.set_xlabel('Trial type')
ax.set_ylabel('Lick rate (licks/s)')
import scipy.stats as stats
# --- Perform pairwise Wilcoxon signed-rank tests ---
comparisons = [
    ('Correct Pre-Reward', 'Incorrect Pre-Reward'),
    ('Correct Pre-Reward', 'Incorrect All')
]
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')

results = []
for a, b in comparisons:
    valid_df = df_wide.dropna(subset=[a, b])
    stat, p = wilcoxon_r(valid_df[a].values, valid_df[b].values)
    results.append({'comp': f'{a} vs {b}', 'stat': stat, 'p': p, 'n': len(valid_df)})
from statsmodels.stats.multitest import multipletests
# FDR correction
pvals = [r['p'] for r in results]
reject, pvals_fdr, _, _ = multipletests(pvals, method='fdr_bh')

for i, r in enumerate(results):
    r['p_fdr'] = pvals_fdr[i]
    r['signif'] = (
        '***' if pvals_fdr[i] < 0.001 else
        '**' if pvals_fdr[i] < 0.01 else
        '*' if pvals_fdr[i] < 0.05 else 'ns'
    )
# --- Annotate significance on plot ---
y_max = df['lick_rate'].max()
height = y_max * 0.03
for i, (comp, label) in enumerate(zip(comparisons, results)):
    x1 = x_pos[comp[0]]
    x2 = x_pos[comp[1]]
    y = y_max + (i + 1) * height * 4  # stack bars
    ax.plot([x1, x1, x2, x2], [y, y+height, y+height, y], lw=1.5, c='k')
    ax.text((x1 + x2) / 2, y + height * 1.2, label['signif'], ha='center', va='bottom', fontsize=18)
ax.text((x1 + x2) / 2, y + height * 2.0, f"n = {label['n']}", ha='center', va='bottom', fontsize=10)

plt.savefig(os.path.join(savedst, 'lickrate_trialtype.svg'), 
        bbox_inches='tight')
#%%
# velocity correct vs. incorrect
# per epoch
plt.rc('font', size=20) 
lick_rate_corr = [[np.nanmean(yy[0]) for yy in xx] for xx in vel_all]
lick_rate_incorr_pre = [[np.nanmean(yy[1]) for yy in xx] for xx in vel_all]
lick_rate_incorr = [[np.nanmean(yy[2]) for yy in xx] for xx in vel_all]
conddf=conddf.copy()
conddf=conddf[(conddf.animals!='e217') & (conddf.optoep<2)]
df=pd.DataFrame()
df['vel_corr']=np.concatenate(lick_rate_corr)
df['animal']=np.concatenate([[conddf.animals.values[jj]]*len(xx) for jj, xx in enumerate(lick_rate_corr)])
df['day']=np.concatenate([[conddf.days.values[jj]]*len(xx) for jj, xx in enumerate(lick_rate_corr)])
df['vel_incorr_pre']=np.concatenate(lick_rate_incorr_pre)
df['vel_incorr']=np.concatenate(lick_rate_incorr)
df_long = df.melt(
    id_vars=['animal', 'day'],
    value_vars=['vel_corr', 'vel_incorr_pre','vel_incorr'],
    var_name='trial_type',
    value_name='velocity'
)

# Clean up trial_type labels
df_long['trial_type'] = df_long['trial_type'].map({
    'vel_corr': 'Correct Pre-Reward',
    'vel_incorr_pre': 'Incorrect Pre-Reward',
    'vel_incorr': 'Incorrect All'
})
df=df_long.groupby(['animal','trial_type']).mean(numeric_only=True)
pl=['seagreen','firebrick','firebrick']
order=['Correct Pre-Reward','Incorrect Pre-Reward','Incorrect All']
fig,ax=plt.subplots(figsize=(3,4))
sns.barplot(x='trial_type',y='velocity',data=df,fill=False,palette=pl,order=order,errorbar='se')
# sns.stripplot(x='trial_type',y='lick_rate',data=df,palette=pl)
# --- Draw connecting lines per animal ---
df_wide = df.reset_index().pivot(index='animal', columns='trial_type', values='velocity')
x_pos = {'Correct Pre-Reward': 0, 'Incorrect Pre-Reward': 1,
         'Incorrect All': 2}  # x-axis positions

for animal, row in df_wide.iterrows():
    x_vals = [x_pos['Correct Pre-Reward'], x_pos['Incorrect Pre-Reward'],x_pos['Incorrect All']]
    y_vals = [row['Correct Pre-Reward'], row['Incorrect Pre-Reward'],row['Incorrect All']]
    ax.plot(x_vals, y_vals, color='gray', alpha=0.5, linewidth=1.5)
ax.spines[['top','right']].set_visible(False)
ax.set_xlabel('Trial type')
ax.set_ylabel('Velocity (cm/s)')
import scipy.stats as stats
# --- Perform pairwise Wilcoxon signed-rank tests ---
comparisons = [
    ('Correct Pre-Reward', 'Incorrect Pre-Reward'),
    ('Correct Pre-Reward', 'Incorrect All')
]
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')

results = []
for a, b in comparisons:
    valid_df = df_wide.dropna(subset=[a, b])
    stat, p = stats.wilcoxon(valid_df[a], valid_df[b])
    results.append({'comp': f'{a} vs {b}', 'stat': stat, 'p': p, 'n': len(valid_df)})
from statsmodels.stats.multitest import multipletests
# FDR correction
pvals = [r['p'] for r in results]
reject, pvals_fdr, _, _ = multipletests(pvals, method='fdr_bh')

for i, r in enumerate(results):
    r['p_fdr'] = pvals_fdr[i]
    r['signif'] = (
        '***' if pvals_fdr[i] < 0.001 else
        '**' if pvals_fdr[i] < 0.01 else
        '*' if pvals_fdr[i] < 0.05 else 'ns'
    )
# --- Annotate significance on plot ---
y_max = df['velocity'].max()
height = y_max * 0.03
for i, (comp, label) in enumerate(zip(comparisons, results)):
    x1 = x_pos[comp[0]]
    x2 = x_pos[comp[1]]
    y = y_max + (i + 1) * height * 4  # stack bars
    ax.plot([x1, x1, x2, x2], [y, y+height, y+height, y], lw=1.5, c='k')
    ax.text((x1 + x2) / 2, y + height * 1.2, label['signif'], ha='center', va='bottom', fontsize=18)
ax.text((x1 + x2) / 2, y + height * 2.0, f"n = {label['n']}", ha='center', va='bottom', fontsize=10)

plt.savefig(os.path.join(savedst, 'vel_trialtype.svg'), 
        bbox_inches='tight')

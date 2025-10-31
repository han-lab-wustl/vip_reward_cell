
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
from projects.pyr_reward.rewardcell import get_radian_position,licks_by_trialtype
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
#%%
goal_cm_window=20 # to search for rew cells
# radian_alignment_saved = {} # overwrite
goal_cell_iind = []
goal_cell_prop = []
goal_cell_null = [] 
num_epochs = [] 
pvals = []
rates_all = []
total_cells = []
epoch_perm = []
radian_alignment = {}
lasttr=8 #  last trials
bins=90
# iterate through all animals
dfs = []
tcs_correct_all=[]
tcs_fail_all=[]
tcs_probes_all = []
for ii in range(len(conddf)):
    day = conddf.days.values[ii]
    animal = conddf.animals.values[ii]
    if (animal!='e217') & (conddf.optoep.values[ii]<2):
        if animal=='e145' or animal=='e139': pln=2 
        else: pln=0
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
        tcs_correct,tcs_fail,tcs_probes,tcs_correct_vel, tcs_fail_vel,tcs_probes_vel=licks_by_trialtype(params_pth, animal,bins=90)
        tcs_correct_all.append([tcs_correct,tcs_correct_vel])
        tcs_fail_all.append([tcs_fail,tcs_fail_vel])
        tcs_probes_all.append([tcs_probes,tcs_probes_vel])


#%%
# all animals
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
# Assume conddf, tcs_correct_all, tcs_fail_all, and bins are defined

# Filter valid animals
animals = [xx for ii, xx in enumerate(conddf.animals.values)
           if (conddf.optoep.values[ii] < 2)]

# Collect across all animals
all_tcs_correct = []
all_tcs_fail = []
# no average over epochs!!
for ii, (tcs_corr, tcs_f) in enumerate(zip(tcs_correct_all, tcs_fail_all)):
    if animals[ii] in animals:  # already filtered list
        if tcs_corr[0].shape[1] > 0:
            tc_corr = np.vstack(tcs_corr[0][:, :])
            all_tcs_correct.append(tc_corr)
        if tcs_f[0].shape[1] > 0:
            tc_fail = np.vstack(tcs_f[0][:, :])
            if np.sum(np.isnan(tc_fail)) == 0:
               all_tcs_fail.append(tc_fail)
probe_trials = {0: [], 1: [], 2: []}

# licks
for ii, probes in enumerate(tcs_probes_all):
    if animals[ii] in animals:  # only use filtered animals
        for probe in range(3):
         tc_probe=np.vstack(probes[0][:,probe, :])
         if np.sum(np.isnan(tc_probe))<(2*tc_probe.shape[1]):
            probe_trials[probe].append(tc_probe)  # shape: bin

# Create figure
plt.rc('font', size=16)
fig, axes = plt.subplots(ncols=5, nrows=2, sharex=True, figsize=(15, 8), height_ratios=[2.5, 1])
axes = axes.flatten()
bins = 90  # assumed defined

# --- Heatmap: correct trials ---
ax = axes[0]
if all_tcs_correct:
   tccorr = np.vstack(all_tcs_correct)
   valid_rows = ~np.all(np.isnan(tccorr), axis=1)
   tccorr = tccorr[valid_rows]
   max_per_row = np.nanmax(tccorr, axis=1)
   tccorr = tccorr.T / max_per_row
   im = ax.imshow(tccorr.T, vmin=0, vmax=1,aspect='auto',cmap='Blues')
   ax.axvline(45, color='k', linestyle='--')
   ax.set_xticks(np.arange(0, bins, 30))
   ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi + .6, np.pi), 2), rotation=45)
   ax.set_ylabel('Epochs')
   ax.set_xticks([0,45,90])
   ax.set_xticklabels(['-$\\pi$',0,'$\\pi$'])
   ax.set_xlabel('Reward-relative distance ($\Theta$)')
   ax.set_title('All animals\nCorrect')

# --- Heatmap: incorrect trials ---
ax = axes[1]
if all_tcs_fail:
   tc = np.vstack(all_tcs_fail)
   max_per_row = np.nanmax(tc, axis=1)
   #   max_per_row[max_per_row == 0] = np.nan
   tc=tc.T / max_per_row
   im = ax.imshow(tc.T, vmin=0, vmax=1,aspect='auto',cmap='Blues')
   ax.axvline(45, color='k', linestyle='--')
   ax.set_xticks(np.arange(0, bins, 30))
   ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi + .6, np.pi), 2), rotation=45)
   ax.set_title('Incorrect')

# --- Average trace: correct ---
ax = axes[5]
if all_tcs_correct:
   m = np.nanmean(tccorr.T, axis=0)
   ax.plot(m, color='seagreen')
   ax.fill_between(
      range(0, tccorr.shape[0]),
      m - scipy.stats.sem(tccorr.T, axis=0, nan_policy='omit'),
      m + scipy.stats.sem(tccorr.T, axis=0, nan_policy='omit'),
      alpha=0.5, color='seagreen'
   )
   ax.axvline(45, color='k', linestyle='--')
   ax.spines[['top', 'right']].set_visible(False)
   ax.set_title('Correct')
   ax.set_ylabel('Norm. lick')
   ax.set_ylim([0, 1])

# --- Average trace: incorrect ---
ax = axes[6]
if all_tcs_fail:
   m = np.nanmean(tc.T, axis=0)
   ax.plot(m, color='firebrick')
   ax.fill_between(
      range(0, tc.shape[0]),
      m - scipy.stats.sem(tc.T, axis=0, nan_policy='omit'),
      m + scipy.stats.sem(tc.T, axis=0, nan_policy='omit'),
      alpha=0.5, color='firebrick'
   )
   ax.axvline(45, color='k', linestyle='--')
   ax.set_xticks([0,45,90])
   ax.set_xticklabels(['-$\\pi$',0,'$\\pi$'])
   ax.spines[['top', 'right']].set_visible(False)
   ax.set_title('Incorrect')
   ax.set_ylim([0, .8])
    
colors = ['royalblue', 'orange', 'purple']
for i, probe in enumerate([0, 1, 2]):
   ax = axes[2 + i]  # top row: heatmaps
   tc = np.vstack(probe_trials[probe])
   valid_rows = ~np.all(np.isnan(tc), axis=1)
   tc = tc[valid_rows]
   max_per_row = np.nanmax(tc, axis=1)
   tc = tc.T / max_per_row
   tc=tc.T
   im = ax.imshow(tc, vmin=0, vmax=1, aspect='auto',cmap='Blues')
   ax.axvline(45, color='k', linestyle='--')
   ax.set_xticks(np.arange(0, bins, 30))
   ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi + .6, np.pi), 2), rotation=45)
   ax.set_title(f'Probe {probe+1}')

   # --- Bottom row: mean traces ---
   ax = axes[7 + i]
   m = np.nanmean(tc, axis=0)
   sem = scipy.stats.sem(tc, axis=0, nan_policy='omit')
   ax.plot(m, color=colors[i])
   ax.fill_between(range(0, bins), m - sem, m + sem, alpha=0.5, color=colors[i])
   ax.axvline(45, color='k', linestyle='--')
   ax.set_xticks([0, 45, 90])
   ax.set_xticklabels(['-$\\pi$', 0, '$\\pi$'])
   ax.spines[['top', 'right']].set_visible(False)
   ax.set_title(f'Probe {probe+1}')
   ax.set_ylim([0, .4])
   if i==2:
      ax.set_xlabel('Reward-relative distance ($\Theta$)')

cbar = fig.colorbar(im, ax=axes[4], fraction=0.046)
cbar.ax.set_ylabel('Norm. lick', rotation=270, labelpad=15)

plt.tight_layout()
plt.savefig(os.path.join(savedst, 'lick_correctvfail_mean.svg'),bbox_inches='tight')

#%%
# v velocity
# all animals

# Filter valid animals
animals = [xx for ii, xx in enumerate(conddf.animals.values)
           if (conddf.optoep.values[ii] < 2)]

# Collect across all animals
all_tcs_correct = []
all_tcs_fail = []
# no average over epochs!!
for ii, (tcs_corr, tcs_f) in enumerate(zip(tcs_correct_all, tcs_fail_all)):
    if animals[ii] in animals:  # already filtered list
        if tcs_corr[0].shape[1] > 0:
            tc_corr = np.vstack(tcs_corr[1][:, :])
            all_tcs_correct.append(tc_corr)
        if tcs_f[0].shape[1] > 0:
            tc_fail = np.vstack(tcs_f[1][:, :])
            if np.sum(np.isnan(tc_fail)) == 0:
               all_tcs_fail.append(tc_fail)
probe_trials = {0: [], 1: [], 2: []}

# licks
for ii, probes in enumerate(tcs_probes_all):
    if animals[ii] in animals:  # only use filtered animals
        for probe in range(3):
         tc_probe=np.vstack(probes[1][:,probe, :])
         if np.sum(np.isnan(tc_probe))<(2*tc_probe.shape[1]):
            probe_trials[probe].append(tc_probe)  # shape: bin

# Create figure
plt.rc('font', size=16)
fig, axes = plt.subplots(ncols=5, nrows=2, sharex=True, figsize=(15, 8), height_ratios=[2.5, 1])
axes = axes.flatten()
bins = 90  # assumed defined

# --- Heatmap: correct trials ---
ax = axes[0]
if all_tcs_correct:
   tccorr = np.vstack(all_tcs_correct)
   valid_rows = ~np.all(np.isnan(tccorr), axis=1)
   tccorr = tccorr[valid_rows]
   max_per_row = np.nanmax(tccorr, axis=1)
   tccorr = tccorr.T / max_per_row
   im = ax.imshow(tccorr.T, vmin=0, vmax=1,aspect='auto',cmap='Greys')
   ax.axvline(45, color='k', linestyle='--')
   ax.set_xticks(np.arange(0, bins, 30))
   ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi + .6, np.pi), 2), rotation=45)
   ax.set_ylabel('Epochs')
   ax.set_xticks([0,45,90])
   ax.set_xticklabels(['-$\\pi$',0,'$\\pi$'])
   ax.set_xlabel('Reward-centric distance ($\Theta$)')
   ax.set_title('All animals\nCorrect')

# --- Heatmap: incorrect trials ---
ax = axes[1]
if all_tcs_fail:
   tc = np.vstack(all_tcs_fail)
   max_per_row = np.nanmax(tc, axis=1)
   #   max_per_row[max_per_row == 0] = np.nan
   tc=tc.T / max_per_row
   im = ax.imshow(tc.T, vmin=0, vmax=1,aspect='auto',cmap='Greys')
   ax.axvline(45, color='k', linestyle='--')
   ax.set_xticks(np.arange(0, bins, 30))
   ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi + .6, np.pi), 2), rotation=45)
   ax.set_title('Incorrect')

# --- Average trace: correct ---
ax = axes[5]
if all_tcs_correct:
   m = np.nanmean(tccorr.T, axis=0)
   ax.plot(m, color='seagreen')
   ax.fill_between(
      range(0, tccorr.shape[0]),
      m - scipy.stats.sem(tccorr.T, axis=0, nan_policy='omit'),
      m + scipy.stats.sem(tccorr.T, axis=0, nan_policy='omit'),
      alpha=0.5, color='seagreen'
   )
   ax.axvline(45, color='k', linestyle='--')
   ax.spines[['top', 'right']].set_visible(False)
   ax.set_title('Correct')
   ax.set_ylabel('Norm. velocity')
   ax.set_ylim([0, 1])

# --- Average trace: incorrect ---
ax = axes[6]
if all_tcs_fail:
   m = np.nanmean(tc.T, axis=0)
   ax.plot(m, color='firebrick')
   ax.fill_between(
      range(0, tc.shape[0]),
      m - scipy.stats.sem(tc.T, axis=0, nan_policy='omit'),
      m + scipy.stats.sem(tc.T, axis=0, nan_policy='omit'),
      alpha=0.5, color='firebrick'
   )
   ax.axvline(45, color='k', linestyle='--')
   ax.set_xticks([0,45,90])
   ax.set_xticklabels(['-$\\pi$',0,'$\\pi$'])
   ax.spines[['top', 'right']].set_visible(False)
   ax.set_title('Incorrect')
   # ax.set_ylim([0, .8])
    
colors = ['royalblue', 'orange', 'purple']
for i, probe in enumerate([0, 1, 2]):
   ax = axes[2 + i]  # top row: heatmaps
   tc = np.vstack(probe_trials[probe])
   valid_rows = ~np.all(np.isnan(tc), axis=1)
   tc = tc[valid_rows]
   max_per_row = np.nanmax(tc, axis=1)
   tc = tc.T / max_per_row
   tc=tc.T
   im = ax.imshow(tc, vmin=0, vmax=1, aspect='auto',cmap='Greys')
   ax.axvline(45, color='k', linestyle='--')
   ax.set_xticks(np.arange(0, bins, 30))
   ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi + .6, np.pi), 2), rotation=45)
   ax.set_title(f'Probe {probe+1}')

   # --- Bottom row: mean traces ---
   ax = axes[7 + i]
   m = np.nanmean(tc, axis=0)
   sem = scipy.stats.sem(tc, axis=0, nan_policy='omit')
   ax.plot(m, color=colors[i])
   ax.fill_between(range(0, bins), m - sem, m + sem, alpha=0.5, color=colors[i])
   ax.axvline(45, color='k', linestyle='--')
   ax.set_xticks([0, 45, 90])
   ax.set_xticklabels(['-$\\pi$', 0, '$\\pi$'])
   ax.spines[['top', 'right']].set_visible(False)
   ax.set_title(f'Probe {probe+1}')
   # ax.set_ylim([0, .4])
   if i==2:
      ax.set_xlabel('Reward-centric distance ($\Theta$)')

cbar = fig.colorbar(im, ax=axes[4], fraction=0.046)
cbar.ax.set_ylabel('Norm. velocity', rotation=270, labelpad=15)

plt.tight_layout()
plt.savefig(os.path.join(savedst, 'vel_correctvfail_mean.svg'),bbox_inches='tight')

#%%
# per day per animal
animals =[xx for ii,xx in enumerate(conddf.animals.values) if (xx!='e217') & (conddf.optoep.values[ii]<2)]

# licks =0, vel=1
valid = 0
plt.rc('font', size=16) 
dff_correct_per_an = []; dff_fail_per_an = [] # per cell, av epoch
for animal in np.unique(animals):
    dff_correct=[]; dff_fail=[]
    tcs_correct = []
    for ii,tcs_corr in enumerate(tcs_correct_all):
        if animals[ii]==animal:
            if tcs_corr[valid].shape[1]>0:
                # all cells
                # take average of epochs
                tc= np.vstack(np.nanmean(tcs_corr[valid][:,:],axis=0))
                # tc = tcs_corr[0,0,:]
                tcs_correct.append(tc)
                # pre vs. post reward
                #pre
                # dff_correct.append(np.nanmean(tc[:int(bins/2)],axis=1))
                #posts
                dff_correct.append(np.nanmean(tc[int(bins/2):],axis=1))


    tcs_fail = []
    for ii,tcs_f in enumerate(tcs_fail_all):
        if animals[ii]==animal:
            if tcs_f[valid].shape[1]>0:
                # tc = tcs_f[0,0,:]
                # all cells
                tc= np.vstack(np.nanmean(tcs_f[valid][:,:],axis=0))
                if np.sum(np.isnan(tc))==0:
                    tcs_fail.append(tc)
                    # pre
                    # dff_fail.append(np.nanmean(tc[:int(bins/2)],axis=1))
                    # post
                    dff_fail.append(np.nanmean(tc[int(bins/2):],axis=1))
    dff_correct_per_an.append(dff_correct)
    dff_fail_per_an.append(dff_fail)
    fig, axes=plt.subplots(ncols=2, nrows=2,sharex=True,figsize=(6,8), 
                            height_ratios=[4,1])
    axes=axes.flatten()
    ax=axes[0]
    ax.imshow(np.hstack(tcs_correct).T**.6,vmin=0,vmax=1.5)
    ax.axvline(45,color='w', linestyle='--')
    bins=90
    ax.set_xticks(np.arange(0,bins,30))
    ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi+.6, np.pi),2),rotation=45)
    ax.set_ylabel('Epochs')
    ax.set_xlabel('Reward-relative distance ($\Theta$)')
    ax.set_title(f'{animal}\nPre-reward cells\nCorrect')
    try: # if no fails
        ax=axes[1]
        im=ax.imshow(np.hstack(tcs_fail).T**.6,vmin=0,vmax=1.5)
        ax.axvline(45,color='w', linestyle='--')
        ax.set_xticks(np.arange(0,bins,30))
        ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi+.6, np.pi),2),rotation=45)
        ax.set_title('Incorrect')
    except Exception as e:
        print(e)
    cbar=fig.colorbar(im, ax=ax,fraction=0.046)
    cbar.ax.set_ylabel('$\Delta$ F/F', rotation=270, labelpad=15)

    # mean 
    ax=axes[2]
    m = np.nanmean(np.hstack(tcs_correct).T,axis=0)
    ax.plot(m, color='seagreen')
    ax.fill_between(
    range(0, np.hstack(tcs_correct).shape[0]),
    m - scipy.stats.sem(np.hstack(tcs_correct).T, axis=0, nan_policy='omit'),
    m + scipy.stats.sem(np.hstack(tcs_correct).T, axis=0, nan_policy='omit'),
    alpha=0.5, color='seagreen'
    )             
    ax.axvline(45,color='k', linestyle='--')
    ax.spines[['top','right']].set_visible(False)
    ax.set_title('Correct')
    ax.set_ylabel('$\Delta$ F/F')
    # same y axis
    ax.set_ylim([0,0.7])
    bins=90
    try:
        ax=axes[3]
        m = np.nanmean(np.hstack(tcs_fail).T,axis=0)
        ax.plot(m, color='firebrick')
        ax.fill_between(
        range(0, np.hstack(tcs_fail).shape[0]),
        m - scipy.stats.sem(np.hstack(tcs_fail).T, axis=0, nan_policy='omit'),
        m + scipy.stats.sem(np.hstack(tcs_fail).T, axis=0, nan_policy='omit'),
        alpha=0.5, color='firebrick'
        )             
        ax.axvline(45,color='k', linestyle='--')
        ax.set_xticks(np.arange(0,bins,30))
        ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi+.6, np.pi),2),rotation=45)
        ax.spines[['top','right']].set_visible(False)
        ax.set_xlabel('Reward-relative distance ($\Theta$)')
        ax.set_title('Incorrect')
        # same y axis
        ax.set_ylim([0,0.7])

    except Exception as e:
        print(e)
        # fig.tight_layout()
# plt.savefig(os.path.join(savedst, 'pre_rew_correctvfail_mean.svg'),bbox_inches='tight')
#%%
# recalculate tc
animals_unique = np.unique(animals)
df=pd.DataFrame()
correct = np.concatenate([np.concatenate(xx) for xx in dff_correct_per_an])
incorrect = np.concatenate([np.concatenate(xx) if len(xx)>0 else [np.nan] for xx in dff_fail_per_an])
df['mean_dff'] = np.concatenate([correct,incorrect])
df['trial_type']=np.concatenate([['correct']*len(correct),['incorrect']*len(incorrect)])
ancorr = np.concatenate([[animals_unique[ii]]*len(np.concatenate(xx)) for ii,xx in enumerate(dff_correct_per_an)])
anincorr = np.concatenate([[animals_unique[ii]]*len(np.concatenate(xx)) if len(xx)>0 else [animals_unique[ii]] for ii,xx in enumerate(dff_fail_per_an)])
df['animal'] = np.concatenate([ancorr, anincorr])
bigdf=df
# average
bigdf=bigdf.groupby(['animal', 'trial_type']).mean(numeric_only=True)
bigdf=bigdf.reset_index()
bigdf=bigdf[bigdf.animal!='z16']
s=12
fig,ax = plt.subplots(figsize=(2.5,5))
sns.stripplot(x='trial_type', y='mean_dff', data=bigdf,hue='trial_type',
        palette={'correct':'seagreen', 'incorrect': 'firebrick'},
        s=s,alpha=0.7)
sns.barplot(x='trial_type', y='mean_dff', data=bigdf,hue='trial_type',
        fill=False,palette={'correct':'seagreen', 'incorrect': 'firebrick'})

ax.spines[['top','right']].set_visible(False)
# ax.set_ylabel('Post-reward mean tuning curve (av. licks)')
# ax.set_ylabel('Post-reward mean velocity, cm/s')
# ax.set_ylabel('Pre-reward mean velocity, cm/s')
# ax.set_ylabel('Pre-reward mean licks (binned)')
ax.set_ylabel('Post-reward mean licks (binned)')

ax.set_xlabel('Trial type')
cor = bigdf.loc[(bigdf.trial_type=='correct'), 'mean_dff']
incor = bigdf.loc[(bigdf.trial_type=='incorrect'), 'mean_dff']
t,pval = scipy.stats.wilcoxon(cor,incor)
ans = bigdf.animal.unique()
for i in range(len(ans)):
    ax = sns.lineplot(x='trial_type', y='mean_dff', 
    data=bigdf[bigdf.animal==ans[i]],
    errorbar=None, color='dimgray', linewidth=1.5, alpha=0.5,ax=ax)

# statistical annotation       
ii=0.5
y=.2
pshift=y/7
fs=30
if pval < 0.001:
        plt.text(ii, y, "***", ha='center', fontsize=fs)
elif pval < 0.01:
        plt.text(ii, y, "**", ha='center', fontsize=fs)
elif pval < 0.05:
        plt.text(ii, y, "*", ha='center', fontsize=fs)
ax.text(ii, y+pshift, f'p={pval:.2g}',rotation=45,fontsize=12)

# plt.savefig(os.path.join(savedst, 'lick_correctvfail_pre_rew.svg'),bbox_inches='tight')
plt.savefig(os.path.join(savedst, 'lick_correctvfail_post_rew.svg'),bbox_inches='tight')
# plt.savefig(os.path.join(savedst, 'vel_correctvfail_pre_rew.svg'),bbox_inches='tight')
# plt.savefig(os.path.join(savedst, 'vel_correctvfail_post_rew.svg'),bbox_inches='tight')
"""get lick rate in old reward zone 
make lick tuning curve
get licks in old vs. new reward zone pos
trial by trial tuning curve
current vs previous licks per trial
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random
import matplotlib.patches as patches
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"

df_trials=pd.read_csv(r'C:\Users\Han\Documents\MATLAB\vip_reward_cell\raw_data\fig4l.csv')
#%%
plt.rc('font', size=16)
fig, axes = plt.subplots(ncols=3, figsize=(7,3.5), sharey=True)
colors = {'ctrl':'slategray', 'vip':'red', 'vip_ex':'darkgoldenrod'}
linestyles = {'old':'--', 'new':'-'}
lbls = ['Control','VIP Inhibition', 'VIP Excitation']
# Count animals and sessions per group
group_counts = {}
# df_trials=df_trials[df_trials.value<9]
for ii, group in enumerate(['ctrl','vip','vip_ex']):
    sub = df_trials[df_trials['group']==group]
    # filter high lick rate
    # sub=sub[sub.value<9]
    ans=sub.animal.unique()
    ns = []
    for an in ans:
        ansub=sub[sub.animal==an]
        dys = ansub.day.unique().astype(int)
        ns.append(len(dys))
    for zone in ['old', 'new']:
        grp = sub[sub['zone']==zone].groupby('trial')['value']
        grp_mean = grp.mean()
        grp_sem = grp.std() / np.sqrt(grp.count())

        # Convert to arrays and remove NaNs
        grp_sem_all = grp_sem.values.astype(float)
        valid_mask = ~np.isnan(grp_sem_all)
        grp_mean = grp_mean.values[valid_mask]
        grp_sem = grp_sem_all[valid_mask]
        grp_mean_index = grp_mean_index = grp.mean().index[valid_mask]
        triallim=10
        # Select only the last 20 trials
        if len(grp_mean_index) > triallim:
            grp_mean = grp_mean[-triallim:]
            grp_sem = grp_sem[-triallim:]
            grp_mean_index = grp_mean_index[-triallim:]

        if len(grp_mean) == 0:
            continue

        axes[ii].plot(
            range(triallim), grp_mean,
            color=colors[group], linewidth=1.5,
            linestyle=linestyles[zone],
            label=f"{zone}"
        )
        axes[ii].fill_between(
            range(triallim),
            grp_mean - grp_sem,
            grp_mean + grp_sem,
            color=colors[group], alpha=0.2
        )

    n_an = len(ns)
    n_sess = sum(ns)
    axes[ii].set_title(f"{lbls[ii]}\n(n={n_an} mice, {n_sess} sessions)",fontsize=12)
    axes[ii].spines[['top','right']].set_visible(False)
    axes[ii].set_xticks([0,triallim])
    axes[ii].set_xticklabels([grp_mean_index[-1]-triallim,grp_mean_index[-1]])
    # axes[ii].axvline(8, color='k')
    # axes[ii].set_xlim([0,23])
    axes[ii].legend()
axes[2].set_xlabel("Trials in epoch")
axes[0].set_ylabel("Lick rate (Hz)")
axes[0].set_ylim([-1,9])
fig.suptitle("Trial-by-trial pre- reward area lick rates\nLast 8 trials")
plt.tight_layout()

# plt.close()
# %%
# first 8 trials
# Create figure
fig, axes = plt.subplots(ncols=3, figsize=(7, 3.5), sharey=True)

# Define colors and line styles
colors = {'ctrl': 'slategray', 'vip': 'red', 'vip_ex': 'darkgoldenrod'}
linestyles = {'old': '--', 'new': '-'}
labels = ['Control', 'VIP Inhibition', 'VIP Excitation']
groups = ['ctrl', 'vip', 'vip_ex']

# Set trial limit
trial_limit = 8
# Loop through each experimental group
for ii, group in enumerate(groups):
    # Filter data for current group
    sub = df_trials[df_trials['group'] == group]
    # filter lick rate for spurious control lick threshold
    sub=sub[sub.value<9]
    # Count animals and sessions
    animals = sub.animal.unique()
    session_counts = []
    for animal in animals:
        animal_sub = sub[sub.animal == animal]
        days = animal_sub.day.unique().astype(int)
        session_counts.append(len(days))
    
    n_animals = len(session_counts)
    n_sessions = sum(session_counts)
    
    # Plot old vs new reward zones
    for zone in ['old', 'new']:
        # Group by trial and calculate mean/sem
        zone_data = sub[sub['zone'] == zone].groupby('trial')['value']
        mean_values = zone_data.mean()
        sem_values = zone_data.std() / np.sqrt(zone_data.count())
        
        # Remove NaNs
        sem_array = sem_values.values.astype(float)
        valid_mask = ~np.isnan(sem_array)
        
        mean_values = mean_values.values[valid_mask]
        sem_values = sem_array[valid_mask]
        trial_indices = zone_data.mean().index[valid_mask]
        
        # Select only FIRST N trials (changed from last to first)
        if len(trial_indices) > trial_limit:
            mean_values = mean_values[:trial_limit]  # Changed from [-trial_limit:]
            sem_values = sem_values[:trial_limit]    # Changed from [-trial_limit:]
            trial_indices = trial_indices[:trial_limit]  # Changed from [-trial_limit:]
        
        # Skip if no data
        if len(mean_values) == 0:
            continue
        
        # Plot mean ± SEM
        axes[ii].plot(
            range(trial_limit), mean_values,
            color=colors[group], 
            linewidth=1.5,
            linestyle=linestyles[zone],
            label=zone.capitalize()
        )
        
        axes[ii].fill_between(
            range(trial_limit),
            mean_values - sem_values,
            mean_values + sem_values,
            color=colors[group], 
            alpha=0.2
        )
    
    # Format subplot
    axes[ii].set_title(
        f"{labels[ii]}\n(n={n_animals} mice, {n_sessions} sessions)",
        fontsize=12
    )
    axes[ii].spines[['top', 'right']].set_visible(False)
    axes[ii].set_xticks([0, trial_limit-1])
    axes[ii].set_xticklabels([trial_indices[0], trial_indices[-1]])
    axes[ii].legend(frameon=False)

# Set labels
axes[2].set_xlabel("Trials in epoch")
axes[0].set_ylabel("Lick rate (Hz)")
axes[0].set_ylim([.5,6])
# Set overall title
fig.suptitle(
    f"Trial-by-trial pre-reward area lick rates\nFirst {trial_limit} trials",  # Changed "Last" to "First"
)

plt.tight_layout()

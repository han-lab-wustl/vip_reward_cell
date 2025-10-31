

"""
zahra
nov 2024
quantify reward-relative cells post reward
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
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.pyr_reward.placecell import make_tuning_curves_radians_by_trialtype, intersect_arrays, consecutive_stretch, \
    make_velocity_tuning_curves
from projects.opto.behavior.behavior import get_success_failure_trials
from projects.pyr_reward.rewardcell import get_radian_position, extract_data_nearrew, perireward_binned_activity
from projects.dopamine_receptor.drd import get_moving_time_v3, get_stops_licks
from projects.pyr_reward.rewardcell import perireward_binned_activity_early_late, perireward_binned_activity
from projects.memory.behavior import get_behavior_tuning_curve
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'

goal_window_cm=20 # to search for rew cells
saveddataset = rf'Z:\saved_datasets\radian_tuning_curves_nearreward_cell_bytrialtype_nopto_{goal_window_cm}cm_window.p'
with open(saveddataset, "rb") as fp: #unpickle
    radian_alignment_saved = pickle.load(fp)
#%%
plt.close('all')
dfs=[]
#%%
# radian_alignment_saved=dict(list(radian_alignment_saved.items())[117:])
for k,v in radian_alignment_saved.items():
    if 'e145' not in k and 'e139' not in k:
        radian_alignment=radian_alignment_saved
        tcs_correct, coms_correct, tcs_fail, coms_fail,\
                com_goal, goal_cell_shuf_ps_per_comp_av,\
                        goal_cell_shuf_ps_av=radian_alignment[k]
        track_length=270
        goal_window = goal_window_cm*(2*np.pi/track_length) 
        # change to relative value 
        coms_rewrel = np.array([com-np.pi for com in coms_correct])
        # only get cells near reward        
        perm = list(combinations(range(len(coms_correct)), 2))     
        com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])                
        # tuning curves that are close to each other across epochs
        com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
        # in addition, com near but after goal
        upperbound=np.pi/4
        com_goal = [[xx for xx in com if ((np.nanmedian(coms_rewrel[:,
                xx], axis=0)<=upperbound) & (np.nanmedian(coms_rewrel[:,
                xx], axis=0)>0))] for com in com_goal if len(com)>0]
        # get goal cells across all epochs        
        goal_cells = intersect_arrays(*com_goal)

        goal_cell_iind = goal_cells
        tc = tcs_correct
        animal,day,trash = k.split('_'); day=int(day)
        if animal=='e145': pln=2
        else: pln=0
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"        
        print(params_pth)
        fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 'licks',
        'pyr_tc_s2p_cellind', 'timedFF','ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
                'stat'])
        fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
        Fc3 = fall_fc3['Fc3']
        dFF = fall_fc3['dFF']
        VR = fall['VR'][0][0][()]
        scalingf = VR['scalingFACTOR'][0][0]
        try:
            rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf        
        except:
            rewsize = 10

        Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
        dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]
        skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
        # skew_filter = skew[((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
        # skew_mask = skew_filter>2
        Fc3 = Fc3[:, skew>2] # only keep cells with skew greateer than 2
        dFF = dFF[:, skew>2]
        scalingf=2/3
        ybinned = fall['ybinned'][0]/scalingf
        track_length=180/scalingf    
        forwardvel = fall['forwardvel'][0]    
        changeRewLoc = np.hstack(fall['changeRewLoc'])
        trialnum=fall['trialnum'][0]
        rewards = fall['rewards'][0]
        timedFF=fall['timedFF'][0]
        if animal=='e145':
            ybinned=ybinned[:-1]
            forwardvel=forwardvel[:-1]
            changeRewLoc=changeRewLoc[:-1]
            trialnum=trialnum[:-1]
            rewards=rewards[:-1]        # set vars
            timedFF=timedFF[:-1]
        eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
        velocity = fall['forwardvel'][0]
        veldf = pd.DataFrame({'velocity': velocity})
        velocity = np.hstack(veldf.rolling(5).mean().values)
        # velocity - ndarray: velocity of the animal
        # thres - float: Threshold speed in cm/s
        # Fs - int: Number of frames minimum to be considered stopped
        # ftol - int: Frame tolerance for merging stop periods
        moving_middle,stop = get_moving_time_v3(velocity,2,40,20)
        pre_win_framesALL, post_win_framesALL=31.25*5,31.25*5
        nonrew_stop_without_lick, nonrew_stop_with_lick, rew_stop_without_lick, \
        rew_stop_with_lick,mov_success_tmpts=get_stops_licks(moving_middle, stop, pre_win_framesALL, post_win_framesALL,\
                velocity, (fall['rewards'][0]==.5).astype(int), fall['licks'][0], 
                max_reward_stop=31.25*5)    
        # nonrew,rew = get_stops(moving_middle, stop, pre_win_framesALL, 
        #         post_win_framesALL,velocity, fall['rewards'][0])
        nonrew_stop_without_lick_per_plane = np.zeros_like(fall['changeRewLoc'][0])
        nonrew_stop_without_lick_per_plane[nonrew_stop_without_lick.astype(int)] = 1
        nonrew_stop_with_lick_per_plane = np.zeros_like(fall['changeRewLoc'][0])
        nonrew_stop_with_lick_per_plane[nonrew_stop_with_lick.astype(int)] = 1
        movement_starts=mov_success_tmpts.astype(int)
        rew_per_plane = np.zeros_like(fall['changeRewLoc'][0])
        rew_per_plane[rew_stop_with_lick.astype(int)] = 1
        move_start = np.zeros_like(fall['changeRewLoc'][0])
        move_start[movement_starts.astype(int)] = 1
        range_val=7;binsize=0.1
        gc_latencies_mov=[];gc_latencies_rew=[];cellid=[]
        # get latencies based on average of trials
        for gc in goal_cell_iind:
            # _, meanvelrew, __, velrew = perireward_binned_activity(velocity, move_start, 
            #         fall['timedFF'][0], fall['trialnum'][0], range_val,binsize)
            # _, meanlickrew, __, lickrew = perireward_binned_activity(fall['licks'][0], move_start, 
            #     fall['timedFF'][0], fall['trialnum'][0], range_val,binsize)
            _, meanrew, __, rewall = perireward_binned_activity(dFF[:,gc], rewards==1, 
                fall['timedFF'][0], fall['trialnum'][0], range_val,binsize)
            if np.nanmax(meanrew)>1: # only get highly active cells?
                _, meanrstops, __, rewrstops = perireward_binned_activity(dFF[:,gc], move_start, fall['timedFF'][0], fall['trialnum'][0], range_val,binsize)
                
                iind = np.where(meanrew>(np.nanmean(meanrew[int(range_val/binsize):])+1*np.nanstd(meanrew[int(range_val/binsize):])))[0]
                
                transient_after_rew=iind[iind>int(range_val/binsize)]
                if len(transient_after_rew)>0:
                    transient_after_rew=transient_after_rew[0]
                else:
                    transient_after_rew=np.nan
                gc_latencies_rew.append((transient_after_rew-int(range_val/binsize))*binsize)
                iind = np.where(meanrstops>(np.nanmean(meanrstops[int(range_val/binsize-3/binsize):])+1*np.nanstd(meanrstops[int(range_val/binsize-3/binsize):])))[0]
                transient_before_move=iind[(iind>int(range_val/binsize-5/binsize)) & (iind<int(range_val/binsize+3/binsize))]
                if len(transient_before_move)>0:
                    transient_before_move=transient_before_move[0]
                else:
                    transient_before_move=np.nan
                gc_latencies_mov.append((transient_before_move-int(range_val/binsize))*binsize)
                cellid.append(gc)

            # fig,ax=plt.subplots()
            # ax.scatter(range(len(latencies_to_movement)),latencies_to_movement,color='k')
            # ax2 = ax.twinx()
            # ax2.scatter(range(len(latencies_to_rewards)),latencies_to_rewards,color='orchid')
        # concat by cell
        df=pd.DataFrame()
        df['latency (s)']=np.concatenate([gc_latencies_rew,gc_latencies_mov])
        df['behavior']=np.concatenate([['Reward']*len(gc_latencies_rew),
                        ['Movement Start']*len(gc_latencies_mov)])
        df['animal']=[animal]*len(df)
        df['day']=[day]*len(df)
        df['cellid']=np.concatenate([cellid]*2)

        dfs.append(df)

#%%
#plot all cells
df=pd.concat(dfs)
df = df.reset_index()
# df=df[df.animal=='e201']
# df=dfs[0]
fig,ax=plt.subplots(figsize=(8,5))
sns.stripplot(x='behavior',y='latency (s)',data=df,hue='animal',s=8,alpha=0.3,dodge=True)
sns.boxplot(x='behavior',y='latency (s)',data=df,hue='animal',fill=False,showfliers=False,whis=0)
ax.axhline(0,color='k',linestyle='--')
# for an in df.animal.unique():
#     dfan = df[df.animal==an]
#     for dy in dfan.day.unique():
#         dfdy = dfan[dfan.day==dy]
#         for celliid in dfdy.cellid.unique():
#             sns.lineplot(x='behavior',y='latency (s)',data=dfdy[dfdy.cellid==celliid],
#                 alpha=0.1,color='gray')

# sns.barplot(x='behavior',y='latency (s)',data=df,fill=False)
ax.spines[['top','right']].set_visible(False)
#%%

fig,ax=plt.subplots(figsize=(2.2,5))
sns.stripplot(x='behavior',y='latency (s)',data=df,s=8,alpha=0.1,
        dodge=True, color='k')
sns.boxplot(x='behavior',y='latency (s)',data=df,fill=False,showfliers= False,whis=0, color='k')
ax.axhline(0,color='k',linestyle='--')
for an in df.animal.unique():
    dfan = df[df.animal==an]
    for dy in dfan.day.unique():
        dfdy = dfan[dfan.day==dy]
        for celliid in dfdy.cellid.unique():
            sns.lineplot(x='behavior',y='latency (s)',data=dfdy[dfdy.cellid==celliid],
                alpha=0.1,color='gray')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
# sns.barplot(x='behavior',y='latency (s)',data=df,fill=False)
ax.spines[['top','right']].set_visible(False)
ax.set_ylabel('Latency (s) to transient')
ax.set_xlabel('')
#%%
# histogram of latencies
# fig 4
plt.rc('font', size=20) 
# Prepare data
df=df[df.animal!='e200']
subset = df[df.behavior == 'Reward']
animals = subset['animal'].unique()
palette = sns.color_palette('tab10', len(animals))

fig, axes = plt.subplots(ncols=2,figsize=(10, 5),sharey=True)
ax=axes[0]
x_vals = np.linspace(subset['latency (s)'].min(), subset['latency (s)'].max(), 500)
kde_vals_all = []
# Plot KDE for each animal and store for averaging
for i, animal in enumerate(animals):
    data_animal = subset[subset['animal'] == animal]['latency (s)'].dropna()
    if len(data_animal)>1:
        kde = scipy.stats.gaussian_kde(data_animal)
        y_vals = kde(x_vals)
    else:
        kde = scipy.stats.norm.pdf(x_vals, loc=data_animal.values[0], scale=.5)
        y_vals=kde
    kde_vals_all.append(y_vals)
    sns.lineplot(x=x_vals, y=y_vals, ax=ax, label=animal, color=palette[i], linewidth=1.5)
# Compute mean KDE across animals
mean_kde_vals = np.mean(kde_vals_all, axis=0)
ax.plot(x_vals, mean_kde_vals, color='black', linewidth=3, label='Mean KDE')
mean_val = np.nanmean(df[df.behavior=='Reward']['latency (s)'].values)
sem= scipy.stats.sem(df[df.behavior=='Reward']['latency (s)'].values,nan_policy='omit')
ax.axvline(mean_val, color='k', linestyle='--', linewidth=2)
ax.text(mean_val, ax.get_ylim()[1], f'Mean={mean_val:.2g} +/- {sem:.2g}', color='k', ha='center', va='bottom', fontsize=14)
ax.spines[['top','right']].set_visible(False)
ax.set_ylabel('# Post-reward cells')
ax.set_xlabel('Latency from reward (s)')

ax=axes[1]
subset = df[df.behavior == 'Movement Start']
kde_vals_all = []
x_vals = np.linspace(subset['latency (s)'].min(), subset['latency (s)'].max(), 500)
# Plot KDE for each animal and store for averaging
for i, animal in enumerate(animals):
    data_animal = subset[subset['animal'] == animal]['latency (s)'].dropna()
    if len(data_animal)>1:
        kde = scipy.stats.gaussian_kde(data_animal)
        y_vals = kde(x_vals)
        kde_vals_all.append(y_vals)
    
        sns.lineplot(x=x_vals, y=y_vals, ax=ax, label=animal, color=palette[i], linewidth=1.5)
# Compute mean KDE across animals
mean_kde_vals = np.mean(kde_vals_all, axis=0)
ax.plot(x_vals, mean_kde_vals, color='black', linewidth=3, label='Mean KDE')
mean_val = np.nanmean(df[df.behavior=='Movement Start']['latency (s)'].values)
sem= scipy.stats.sem(df[df.behavior=='Movement Start']['latency (s)'].values,nan_policy='omit')
ax.axvline(mean_val, color='k', linestyle='--', linewidth=2)
ax.text(mean_val, ax.get_ylim()[1], f'Mean={mean_val:.2g} +/- {sem:.2g}', color='k', ha='center', va='bottom', fontsize=14)
ax.spines[['top','right']].set_visible(False)
ax.set_ylabel('# Post-reward cells')
ax.set_xlabel('Latency from movement start (s)')

plt.savefig(os.path.join(savedst, 'latency_postrew_hist.svg'))
#%%
# boxplot per animal
dfagg = df.groupby(['animal','behavior']).mean(numeric_only=True)
dfagg=dfagg.reset_index()
sns.stripplot(x='behavior',y='latency (s)',data=dfagg,s=13,alpha=0.7,
        dodge=True, color='k')
sns.boxplot(x='behavior',y='latency (s)',data=dfagg,fill=False,color='k')
ax.axhline(0,color='k',linestyle='--')
for an in dfagg.animal.unique():
    dfan = dfagg[dfagg.animal==an]
    for dy in dfan.day.unique():
        dfdy = dfan[dfan.day==dy]
        for celliid in dfdy.cellid.unique():
            sns.lineplot(x='behavior',y='latency (s)',data=dfdy[dfdy.cellid==celliid],
                alpha=0.1,color='gray')

#%%
plt.close('all')
# per animal pair
ansq = int(np.sqrt(len(df.animal.unique())))
fig,axes=plt.subplots(nrows=ansq,ncols=ansq,figsize=(8,12),sharex=True, sharey=True)
axes = axes.flatten()
for ii,an in enumerate(df.animal.unique()):
    dfan = df[df.animal==an]
    ax=axes[ii]
    sns.stripplot(x='behavior',y='latency (s)',data=dfan,ax=ax,s=8,alpha=0.3,dodge=True)
    sns.boxplot(x='behavior',y='latency (s)',data=dfan,ax=ax,fill=False,showfliers= False,whis=0)
    for dy in dfan.day.unique():
        dfdy = dfan[dfan.day==dy]
        for celliid in dfdy.cellid.unique():
            sns.lineplot(x='behavior',y='latency (s)',data=dfdy[dfdy.cellid==celliid],
                alpha=0.1,color='gray',ax=ax)
    ax.set_title(an)
fig.tight_layout()

#%%

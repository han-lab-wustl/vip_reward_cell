"""
zahra
nov 2024
quantify reward-relative cells post reward
# TODO: take all the cells and quantify their transient in rewarded vs. unrewarded stops
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
plt.rc('font', size=20)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.pyr_reward.placecell import make_tuning_curves_radians_by_trialtype, intersect_arrays
from projects.opto.behavior.behavior import get_success_failure_trials, smooth_lick_rate
from projects.pyr_reward.rewardcell import get_radian_position,extract_data_nearrew,perireward_binned_activity
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
#%%
goal_window_cm=20 # to search for rew cells
saveddataset = rf'Z:\saved_datasets\radian_tuning_curves_nearreward_cell_bytrialtype_nopto_20cm_window.p'
with open(saveddataset, "rb") as fp: #unpickle
    radian_alignment_saved = pickle.load(fp)

#%%
# test
iinds = radian_alignment_saved.keys()
iinds=[xx for xx in iinds if 'e190' in xx]
iinds = ['e186_002_index155','e186_006_index159','e186_010_index163','e186_014_index167','e186_017_index170','e186_021_index174','e186_026_index179','e186_030_index183','e186_035_index188','e201_051_index131', 'e201_053_index133', 'e201_054_index134','e190_027_index193',
 'e190_029_index195',
 'e190_034_index197']
# from projects.pyr_reward.rewardcell import perireward_binned_activity
datarasters=[]
for iind in iinds:
   radian_alignment=radian_alignment_saved
   tcs_correct, coms_correct, tcs_fail, coms_fail,\
         com_goal, goal_cell_shuf_ps_per_comp_av,\
                  goal_cell_shuf_ps_av=radian_alignment[iind]
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
   com_goal = [com for com in com_goal if len(com)>0]
   # get goal cells across all epochs        
   goal_cells = intersect_arrays(*com_goal)
   goal_cells=np.unique(np.concatenate(com_goal))
   post_rew_goal_cells = [xx for xx in goal_cells if (((np.nanmedian(coms_correct[:,xx],axis=0)-np.pi)>0) and ((np.nanmedian(coms_correct[:,xx],axis=0)-np.pi)<upperbound))]
   goal_cell_iind = post_rew_goal_cells
   tc = tcs_correct
   # for gc in goal_cell_iind:
   #     plt.figure()
   #     for ep in range(len(tc)):
   #         plt.plot(tcs_correct[ep,gc,:])
   #         plt.title(gc)
   animal,day,pln = iind.split('_')[0], int(iind.split('_')[1]),0
   params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"        
   fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 'licks',
   'pyr_tc_s2p_cellind', 'timedFF','ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
         'stat'])
   fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
   Fc3 = fall_fc3['Fc3']
   dFF = fall_fc3['dFF']
   Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
   dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
   skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
   # skew_filter = skew[((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
   # skew_mask = skew_filter>2
   Fc3 = Fc3[:, skew>2] # only keep cells with skew greateer than 2
   dFF = dFF[:, skew>2]
   dt = np.nanmedian(np.diff(fall['timedFF'][0]))
   lick_rate = smooth_lick_rate(fall['licks'][0],dt)

   scalingf=2/3
   ybinned = fall['ybinned'][0]/scalingf
   track_length=180/scalingf    
   forwardvel = fall['forwardvel'][0]    
   changeRewLoc = np.hstack(fall['changeRewLoc'])
   trialnum=fall['trialnum'][0]
   rewards = fall['rewards'][0]
   if animal=='e145':
         ybinned=ybinned[:-1]
         forwardvel=forwardvel[:-1]
         changeRewLoc=changeRewLoc[:-1]
         trialnum=trialnum[:-1]
         rewards=rewards[:-1]        # set vars
   eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))

   velocity = fall['forwardvel'][0]
   veldf = pd.DataFrame({'velocity': velocity})

   from projects.dopamine_receptor.drd import get_moving_time_v3, get_stops_licks
   from projects.pyr_reward.rewardcell import perireward_binned_activity_early_late, perireward_binned_activity
   velocity = np.hstack(veldf.rolling(3).mean().values)
   # velocity - ndarray: velocity of the animal
   # thres - float: Threshold speed in cm/s
   # Fs - int: Number of frames minimum to be considered stopped
   # ftol - int: Frame tolerance for merging stop periods
   moving_middle,stop = get_moving_time_v3(velocity,5,20,20)
   pre_win_framesALL, post_win_framesALL=31.25*5,31.25*5
   nonrew_stop_without_lick, nonrew_stop_with_lick, rew_stop_without_lick, rew_stop_with_lick,\
         mov_success_tmpts=get_stops_licks(moving_middle, stop, pre_win_framesALL, post_win_framesALL,\
         velocity, (fall['rewards'][0]==.5).astype(int), fall['licks'][0], 
         max_reward_stop=31.25*5)
   # nonrew,rew = get_stops(moving_middle, stop, pre_win_framesALL, 
   #         post_win_framesALL,velocity, fall['rewards'][0])
   nonrew_stop_without_lick_per_plane = np.zeros_like(fall['changeRewLoc'][0])
   nonrew_stop_without_lick_per_plane[nonrew_stop_without_lick.astype(int)] = 1
   nonrew_stop_with_lick_per_plane = np.zeros_like(fall['changeRewLoc'][0])
   nonrew_stop_with_lick_per_plane[nonrew_stop_with_lick.astype(int)] = 1
   rew_per_plane = np.zeros_like(fall['changeRewLoc'][0])
   rew_per_plane[rew_stop_with_lick.astype(int)] = 1
   movement_starts=mov_success_tmpts.astype(int)
   rew_per_plane = np.zeros_like(fall['changeRewLoc'][0])
   rew_per_plane[rew_stop_with_lick.astype(int)] = 1
   frame_lim=100
   move_start_unrew_idx=[[xx for xx in movement_starts if yy>xx-frame_lim and yy<xx+frame_lim] for yy in nonrew_stop_with_lick]
   move_start_unrew_idx=[xx for xx in move_start_unrew_idx if len(xx)>0]
   move_start_unrew_idx2=[[xx for xx in movement_starts if yy>xx-frame_lim and yy<xx+frame_lim] for yy in nonrew_stop_without_lick]
   move_start_unrew_idx2=[xx for xx in move_start_unrew_idx2 if len(xx)>0]
   if len(move_start_unrew_idx)>0:
      move_start_unrew_idx=np.concatenate([xx if len(xx)==1 else [xx[1]] for xx in move_start_unrew_idx])
   if len(move_start_unrew_idx2)>0:
      move_start_unrew_idx2=np.concatenate([xx if len(xx)==1 else [xx[1]] for xx in move_start_unrew_idx2])
   else:
      move_start_unrew_idx2=[]
   # unrewarded movement starts
   # do not combione without lick for now
   move_start_unrew_idx=np.append(move_start_unrew_idx,move_start_unrew_idx2)
   move_start_unrew = np.zeros_like(fall['changeRewLoc'][0])
   move_start_unrew[move_start_unrew_idx.astype(int)] = 1
   move_start = np.zeros_like(fall['changeRewLoc'][0])
   movement_starts=[xx for xx in movement_starts if xx not in move_start_unrew_idx]
   move_start[movement_starts] = 1

   # per trial
   gc=goal_cell_iind[0]
   range_val,binsize=3, .3
   # TODO: make condensed
   # REWARDED ONLY
   _, meanrstops, __, rewrstops = perireward_binned_activity(Fc3[:,gc], move_start, 
   fall['timedFF'][0], fall['trialnum'][0], range_val,binsize)
   _, meanvelrew, __, velrew = perireward_binned_activity(velocity, move_start, 
         fall['timedFF'][0], fall['trialnum'][0], range_val,binsize)
   _, meanlickrew, __, lickrew = perireward_binned_activity(lick_rate, move_start, 
      fall['timedFF'][0], fall['trialnum'][0], range_val,binsize)
   _, meanrewrewstops, __, rewrewstops = perireward_binned_activity(fall['rewards'][0], move_start, fall['timedFF'][0], fall['trialnum'][0], range_val,binsize)
   # reward/CS
   range_val,binsize=10, .3
   rew=(fall['rewards'][0]==.5)
   rew=rew.astype(int)
   rew[:5000]=0
   _, meanrew, __, rewall = perireward_binned_activity(Fc3[:,gc], rew, 
   fall['timedFF'][0], fall['trialnum'][0], range_val,binsize)
   _, meanvel, __, velall = perireward_binned_activity(velocity, rew, fall['timedFF'][0], fall['trialnum'][0], range_val,binsize)
   _, meanlick, __, lickall = perireward_binned_activity(lick_rate, rew, fall['timedFF'][0], fall['trialnum'][0], range_val,binsize)
   # unrew
   range_val,binsize=3, .3
   _, meanunrstops, __, unrewrstops = perireward_binned_activity(Fc3[:,gc], move_start_unrew, 
   fall['timedFF'][0], fall['trialnum'][0], range_val,binsize)
   _, meanvelunrew, __, velunrew = perireward_binned_activity(velocity, move_start_unrew, 
         fall['timedFF'][0], fall['trialnum'][0], range_val,binsize)
   _, meanlickunrew, __, lickunrew = perireward_binned_activity(lick_rate, move_start_unrew, 
      fall['timedFF'][0], fall['trialnum'][0], range_val,binsize)
   _, meanrewunrewstops, __, rewunrewstops = perireward_binned_activity(fall['rewards'][0], move_start_unrew, fall['timedFF'][0], fall['trialnum'][0], range_val,binsize)
   plt.figure()
   plt.plot(meanrstops) 
   plt.title(iind); plt.show()
   # collect data per day
   datarasters.append([rewall,velall,lickall,rewrstops,velrew,lickrew,unrewrstops,velunrew,lickunrew,rewunrewstops])
#%%
plt.rc('font', size=16) 
fig, axes = plt.subplots(ncols=3,nrows=3,figsize=(7,7),height_ratios=[3,1,1])
# axes=axes.flatten()
# rewall,velall,lickall,rewrstops,velrew,lickrew,unrewrstops,velunrew,lickunrew
rewall=np.hstack([xx[0]/np.nanmax(xx[0]) for xx in datarasters])
velall=np.hstack([xx[1] for xx in datarasters])
lickall=np.hstack([xx[2] for xx in datarasters])
rewrstops=np.hstack([xx[3]/np.nanmax(xx[3]) for xx in datarasters])
velrew=np.hstack([xx[4] for xx in datarasters])
lickrew=np.hstack([xx[5] for xx in datarasters])
lickunrew=np.hstack([xx[8] for xx in datarasters])
velunrew=np.hstack([xx[7] for xx in datarasters])

time_bins = np.arange(velunrew.shape[0])  # 0..19 time bins
# COM for each trial
total_licks = np.sum(lickunrew, axis=0)  # sum over time
com_lick = np.divide(
   np.sum(velunrew * time_bins[:, None], axis=0),  # weighted sum over time
   total_licks,
   out=np.full_like(total_licks, np.nan, dtype=float),
   where=total_licks != 0
)
sort_idx = np.argsort(com_lick)
velunrew = velunrew[:, sort_idx]
lickunrew=np.hstack([xx[8] for xx in datarasters])[:, sort_idx]
unrewrstops=np.hstack([xx[6] for xx in datarasters])[:, sort_idx]
velunrew=np.hstack([xx[7] for xx in datarasters])[:, sort_idx]

rewunrewstops=np.hstack([xx[9] for xx in datarasters])[:, sort_idx]
mask=np.sum(rewunrewstops,axis=0)>0
mask = np.ones_like(mask).astype(bool)
mask2 = np.nanmax(velunrew[10:],axis=0)>40
mask=mask&mask2
velunrew=velunrew[:,mask]
lickunrew=lickunrew[:,mask]
unrewrstops=unrewrstops[:,mask]
unrewrstops[np.isnan(unrewrstops)]=0
unrewrstops=np.array([xx/np.nanmax(xx) if np.nanmax(xx)>0 else np.zeros_like(xx) for xx in unrewrstops.T]).T
vmax=1
vmax_vel=70
vmin_vel=0
vmax_lick=25
vmin_lick=0

range_val,binsize=10, .3
rewall[np.isnan(rewall)]=0
ax1 = axes[0,0]  # base axis for first cell
im1=ax1.imshow(rewall.T, aspect='auto',vmax=vmax)
ax1.set_ylabel('Trials')
ax1.set_xticks([0,(range_val/binsize),(range_val/binsize)*2])
ax1.set_xticklabels([])
ax1.set_yticks([0,len(rewall.T)-1])
ax1.set_yticklabels([1,len(rewall.T)])

ax1.axvline(int(range_val / binsize), color='w', linestyle='--')

ax=axes[1,0]
velall[np.isnan(velall)]=0
im2=ax.imshow(velall.T, aspect='auto',cmap='Greys',vmax=vmax_vel,vmin=vmin_vel)
ax.axvline(int(range_val / binsize), color='k', linestyle='--')
ax.set_xticks([0,(range_val/binsize),(range_val/binsize)*2])
ax.set_xticklabels([])
ax.set_ylabel('Trials')
ax.set_yticks([0,len(rewall.T)-1])
ax.set_yticklabels([1,len(rewall.T)])
ax.legend().set_visible(False)

ax=axes[2,0]
lickall[np.isnan(lickall)]=0
im2=ax.imshow(lickall.T, aspect='auto',cmap='Blues',vmax=vmax_lick,vmin=vmin_lick)
ax.axvline(int(range_val / binsize), color='k', linestyle='--')
# ax.set_ylabel('Velocity (cm/s)')
ax.set_ylabel('Trials')
ax.set_xticks([0,(range_val/binsize),(range_val/binsize)*2])
ax.set_xticklabels([-(range_val),0,(range_val)])
ax.set_yticks([0,len(rewall.T)-1])
ax.set_yticklabels([1,len(rewall.T)])
ax.legend().set_visible(False)
ax.set_xlabel('Time from CS (s)')

range_val,binsize=3, .3
rewrstops[np.isnan(rewrstops)]=0
ax1 = axes[0,1]  # base axis for first cell
im1=ax1.imshow(rewrstops.T, aspect='auto',vmax=vmax)
ax1.axvline(int(range_val / binsize), color='w', linestyle='--')
velrew[np.isnan(velrew)]=0
ax1.set_xticks([0,(range_val/binsize),((range_val/binsize)*2)-1])
ax1.set_xticklabels([])
ax1.set_yticks([0,len(rewrstops.T)-1])
ax1.set_yticklabels([1,len(rewrstops.T)])


ax=axes[1,1]
im2=ax.imshow(velrew.T, aspect='auto',cmap='Greys',vmax=vmax_vel,vmin=vmin_vel)
ax.axvline(int(range_val / binsize), color='k', linestyle='--')
# ax.set_ylabel('Velocity (cm/s)')
ax.legend().set_visible(False)
ax.set_xticks([0,(range_val/binsize),((range_val/binsize)*2)-1])
ax.set_xticklabels([])
ax.set_yticks([0,len(rewrstops.T)-1])
ax.set_yticklabels([])
# ax1.set_yticklabels([1,len(rewrstops.T)])

lickrew[np.isnan(lickrew)]=np.nanmean(lickrew)
ax=axes[2,1]
im2=ax.imshow(lickrew.T, aspect='auto',cmap='Blues',vmax=vmax_lick,vmin=vmin_lick)
ax.axvline(int(range_val / binsize), color='k', linestyle='--')
# ax.set_ylabel('Velocity (cm/s)')
ax.set_xticks([0,(range_val/binsize),(range_val/binsize)*2])
ax.set_xticklabels([-(range_val),0,(range_val)])
ax.set_yticks([0,len(rewrstops.T)-1])
ax.set_yticklabels([])
ax.legend().set_visible(False)
ax.set_xlabel('Time from movement start (s)\nRewarded')

ax1 = axes[0,2]  # base axis for first cell
im1=ax1.imshow(unrewrstops.T, aspect='auto',vmax=vmax)
ax1.axvline(int(range_val / binsize), color='w', linestyle='--')
velrew[np.isnan(velrew)]=0
ax1.set_xticks([0,(range_val/binsize),((range_val/binsize)*2)-1])
ax1.set_xticklabels([])
ax1.set_yticks([0,len(unrewrstops.T)-1])
ax1.set_yticklabels([1,len(unrewrstops.T)])
cax1 = fig.add_axes([0.92, 0.6, 0.02, 0.2])  # [left, bottom, width, height] in figure coords
cbar1 = fig.colorbar(im1, cax=cax1, orientation='vertical')
cbar1.set_label(r'Norm. $\Delta F/F$')

ax=axes[1,2]
im2=ax.imshow(velunrew.T, aspect='auto',cmap='Greys',vmax=vmax_vel,vmin=vmin_vel)
ax.axvline(int(range_val / binsize), color='k', linestyle='--')
# ax.set_ylabel('Velocity (cm/s)')
ax.legend().set_visible(False)
ax.set_yticks([0,len(velunrew.T)-1])
ax.set_yticklabels([])
ax.set_xticks([0,(range_val/binsize),((range_val/binsize)*2)-1])
ax.set_xticklabels([])
cbar1 = fig.colorbar(im2, ax=ax, orientation='vertical', fraction=0.05, pad=0.04)
cbar1.set_label('Velocity (cm/s)')

ax=axes[2,2]
im2=ax.imshow(lickunrew.T, aspect='auto',cmap='Blues',vmax=vmax_lick,vmin=vmin_lick)
ax.axvline(int(range_val / binsize), color='k', linestyle='--')
# ax.set_ylabel('Velocity (cm/s)')
ax.legend().set_visible(False)
ax.set_xticks([0,(range_val/binsize),((range_val/binsize)*2)-1])
ax.set_xticklabels([-(range_val),0,(range_val)])
ax.set_yticks([0,len(velunrew.T)-1])
ax.set_yticklabels([])
ax.set_xlabel('Time from movement start (s)\nUnrewarded')
cbar1 = fig.colorbar(im2, ax=ax, orientation='vertical', fraction=0.05, pad=0.04)


cbar1.set_label('Velocity (cm/s)')
# ax(im2, ax=ax, orientation='vertical', fraction=0.05, pad=0.04)
cbar1.set_label('Lick rate')
# plt.tight_layout()
plt.savefig(os.path.join(savedst, f'trail_postrew_traces_cell{gc}.svg'))
# %%

#%% 
# get lick bouts 
import numpy as np

def find_lick_bout_ends(lick_rate, rate_thresh=1.0, min_bout_duration=3):
    """
    Find timepoints where lick rate transitions from high (above threshold) to 0.
    
    Parameters:
        lick_rate (1D np.array): Time series of lick rate.
        rate_thresh (float): Minimum rate to consider as part of a bout.
        min_bout_duration (int): Minimum number of timepoints above threshold to count as a bout.
        
    Returns:
        bout_end_indices (list): Indices where a lick bout ends (just after rate drops to 0).
    """
    # Create a binary mask for lick bout
    in_bout = lick_rate > rate_thresh

    # Find bout start and end indices
    bout_ends = []
    i = 0
    while i < len(in_bout):
        if in_bout[i]:
            # Start of a bout
            start = i
            while i < len(in_bout) and in_bout[i]:
                i += 1
            end = i  # First zero after high lick rate
            if end - start >= min_bout_duration:
                bout_ends.append(end)  # Index where it drops to 0
        else:
            i += 1
    return bout_ends
lick=fall['licks'][0]
lick_rate = smooth_lick_rate(lick,1/31.25)
bout_ends = find_lick_bout_ends(lick_rate, rate_thresh=7.5, min_bout_duration=3)
bout_ind = np.zeros_like(lick_rate)
bout_ind[bout_ends]=1
bout_ind=bout_ind.astype(bool)
range_val,binsize=3, .3
# TODO: make condensed
_, meanrlick, __, rewrlick = perireward_binned_activity(Fc3[:,gc], bout_ind, 
fall['timedFF'][0], fall['trialnum'][0], range_val,binsize)
_, meanvellick, __, vellick = perireward_binned_activity(velocity, bout_ind, 
        fall['timedFF'][0], fall['trialnum'][0], range_val,binsize)
_, meanlicklick, __, licklick = perireward_binned_activity(fall['licks'][0], bout_ind, 
    fall['timedFF'][0], fall['trialnum'][0], range_val,binsize)
#%%
# incl lick stop
fig, axes = plt.subplots(ncols=3,nrows=3,figsize=(8,7),height_ratios=[3,1,1])
axes=axes.flatten()

range_val,binsize=10, .3
rewall[np.isnan(rewall)]=0
ax1 = axes[0]  # base axis for first cell
im1=ax1.imshow(rewall.T, aspect='auto')
ax1.set_ylabel('Trials')
ax1.set_xticks([0,(range_val/binsize),(range_val/binsize)*2])
ax1.set_xticklabels([])
ax1.axvline(int(range_val / binsize), color='w', linestyle='--')

ax=axes[3]
velall[np.isnan(velall)]=0
im2=ax.imshow(velall.T, aspect='auto',cmap='Greys')
ax.axvline(int(range_val / binsize), color='k', linestyle='--')
ax.set_xticks([0,(range_val/binsize),(range_val/binsize)*2])
ax.set_xticklabels([])
ax.set_ylabel('Trials')
ax.legend().set_visible(False)

ax=axes[6]
lickall[np.isnan(lickall)]=0
im2=ax.imshow(lickall.T, aspect='auto',cmap='Blues')
ax.axvline(int(range_val / binsize), color='k', linestyle='--')
# ax.set_ylabel('Velocity (cm/s)')
ax.set_ylabel('Trials')
ax.set_xticks([0,(range_val/binsize),(range_val/binsize)*2])
ax.set_xticklabels([-(range_val),0,(range_val)])
ax.legend().set_visible(False)
ax.set_xlabel('Time from reward (s)')

range_val,binsize=3, .3
rewrstops[np.isnan(rewrstops)]=0

ax1 = axes[1]  # base axis for first cell
im1=ax1.imshow(rewrstops.T, aspect='auto')
ax1.axvline(int(range_val / binsize), color='w', linestyle='--')
velrew[np.isnan(velrew)]=0
ax1.set_xticks([0,(range_val/binsize),(range_val/binsize)*2])
ax1.set_xticklabels([])
# ax1.set_yticklabels([])

ax=axes[4]
im2=ax.imshow(velrew.T, aspect='auto',cmap='Greys')
ax.axvline(int(range_val / binsize), color='k', linestyle='--')
# ax.set_ylabel('Velocity (cm/s)')
ax.legend().set_visible(False)
ax.set_xticks([0,(range_val/binsize),(range_val/binsize)*2])
ax.set_xticklabels([])
# ax.set_yticklabels([])

lickrew[np.isnan(lickrew)]=np.nanmean(lickrew)
ax=axes[7]
im2=ax.imshow(lickrew.T, aspect='auto',cmap='Blues')
ax.axvline(int(range_val / binsize), color='k', linestyle='--')
# ax.set_ylabel('Velocity (cm/s)')
ax.set_xticks([0,(range_val/binsize),(range_val/binsize)*2])
ax.set_xticklabels([-(range_val),0,(range_val)])
# ax.set_yticklabels([])

ax.legend().set_visible(False)
ax.set_xlabel('Time from movement start (s)')

# lick bout end

range_val,binsize=3, .3
rewrlick[np.isnan(rewrlick)]=0

ax1 = axes[2]  # base axis for first cell
im1=ax1.imshow(rewrlick.T, aspect='auto')
ax1.axvline(int(range_val / binsize), color='w', linestyle='--')
ax1.set_xticks([0,(range_val/binsize),(range_val/binsize)*2])
ax1.set_xticklabels([])
# ax1.set_yticklabels([])
cbar1 = fig.colorbar(im1, ax=ax1, orientation='vertical', fraction=0.05, pad=0.04)
cbar1.set_label('Norm. $\Delta F/F$')
vellick[np.isnan(vellick)]=0

ax=axes[5]
im2=ax.imshow(vellick.T, aspect='auto',cmap='Greys')
ax.axvline(int(range_val / binsize), color='k', linestyle='--')
# ax.set_ylabel('Velocity (cm/s)')
ax.legend().set_visible(False)
ax.set_xticks([0,(range_val/binsize),(range_val/binsize)*2])
ax.set_xticklabels([])
# ax.set_yticklabels([])
cbar1 = fig.colorbar(im2, ax=ax, orientation='vertical', fraction=0.05, pad=0.04)
cbar1.set_label('Velocity (cm/s)')

licklick[np.isnan(licklick)]=np.nanmean(licklick)
ax=axes[8]
im2=ax.imshow(licklick.T, aspect='auto',cmap='Blues')
ax.axvline(int(range_val / binsize), color='k', linestyle='--')
# ax.set_ylabel('Velocity (cm/s)')
ax.set_xticks([0,(range_val/binsize),(range_val/binsize)*2])
ax.set_xticklabels([-(range_val),0,(range_val)])
# ax.set_yticklabels([])

ax.legend().set_visible(False)
ax.set_xlabel('Time from lick stop (s)')
cbar1 = fig.colorbar(im2, ax=ax, orientation='vertical', fraction=0.05, pad=0.04)
cbar1.set_label('Norm. licks')
plt.tight_layout()
plt.savefig(os.path.join(savedst, f'trail_postrew_traces_cell{gc}.svg'))
#%%
# only for unrewarded stops
unrewarded_stops = nonrew_stop_with_lick_per_plane+nonrew_stop_without_lick_per_plane

unrewarded_stops=unrewarded_stops.astype(bool)
range_val,binsize=12, .3
# TODO: make condensed
_, meanunrew, __, unrewall = perireward_binned_activity(Fc3[:,gc], unrewarded_stops, 
fall['timedFF'][0], fall['trialnum'][0], range_val,binsize)
_, meanvelunrew, __, velunrew = perireward_binned_activity(velocity, unrewarded_stops, 
        fall['timedFF'][0], fall['trialnum'][0], range_val,binsize)
_, meanlickunrew, __, lickunrew = perireward_binned_activity(fall['licks'][0], unrewarded_stops, 
    fall['timedFF'][0], fall['trialnum'][0], range_val,binsize)

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
# plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.pyr_reward.placecell import make_tuning_curves_radians_by_trialtype, intersect_arrays
from projects.opto.behavior.behavior import get_success_failure_trials
from projects.pyr_reward.rewardcell import get_radian_position,extract_data_nearrew,perireward_binned_activity
from projects.dopamine_receptor.drd import get_moving_time_v3, get_stops_licks
from projects.pyr_reward.rewardcell import perireward_binned_activity_early_late, perireward_binned_activity

# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
#%%
goal_window_cm=20 # to search for rew cells
saveddataset = rf'Z:\saved_datasets\radian_tuning_curves_nearreward_cell_bytrialtype_nopto_{goal_window_cm}cm_window.p'
with open(saveddataset, "rb") as fp: #unpickle
    radian_alignment_saved = pickle.load(fp)
# test
# just one mouse as eg.?
radian_alignment=radian_alignment_saved
dff_per_an_day_per_trial_type = {}
for k,v in radian_alignment.items():
    iind=k    
    animal,day = k.split('_')[0],k.split('_')[1]  # Extracts '012'
    # if animal==an:
    day=int(day)
    if animal=='e145' or animal=='e139': pln=2 
    else: pln=0
    params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"        
    fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 'licks',
    'timedFF','ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
            'stat'])
    fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
    Fc3 = fall_fc3['Fc3']
    dFF = fall_fc3['dFF']
    # do not remove bordercells:  & (~fall['bordercells'][0].astype(bool))
    Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
    dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]
    skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
    Fc3 = Fc3[:, skew>2] # only keep cells with skew greateer than 2
    dFF = dFF[:, skew>2]
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
    tcs_correct, coms_correct, tcs_fail, coms_fail,\
        com_goal, goal_cell_shuf_ps_per_comp_av,\
                goal_cell_shuf_ps_av=radian_alignment[iind]
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

    velocity = np.hstack(veldf.rolling(5).mean().values)
    # velocity - ndarray: velocity of the animal
    # thres - float: Threshold speed in cm/s
    # Fs - int: Number of frames minimum to be considered stopped
    # ftol - int: Frame tolerance for merging stop periods
    moving_middle,stop = get_moving_time_v3(velocity,2,40,20)
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
    
    range_val,binsize=10, .1
    trials_dff_rstops=[]
    trials_dff_nrstops_wo_licks=[]
    trials_dff_nrstops_w_licks=[]
    for gc in goal_cell_iind:
        # TODO: make condensed
        _, meannstops, __, rewnstops = perireward_binned_activity(Fc3[:,gc], nonrew_stop_without_lick_per_plane, 
                fall['timedFF'][0], fall['trialnum'][0], range_val,binsize)
        _, meanvelnonrew, __, velnonrew = perireward_binned_activity(velocity, nonrew_stop_without_lick_per_plane, 
                fall['timedFF'][0], fall['trialnum'][0], range_val,binsize)
        _, meanlicknonrew, __, licknonrew = perireward_binned_activity(fall['licks'][0], nonrew_stop_without_lick_per_plane, 
            fall['timedFF'][0], fall['trialnum'][0], range_val,binsize)
        _, meanrewnonrew, __, rewnonrew = perireward_binned_activity(fall['rewards'][0], nonrew_stop_without_lick_per_plane, 
        fall['timedFF'][0], fall['trialnum'][0], range_val,binsize)

        _, meannlstops, __, rewnlstops = perireward_binned_activity(Fc3[:,gc], nonrew_stop_with_lick_per_plane, 
                fall['timedFF'][0], fall['trialnum'][0], range_val,binsize)
        _, meanvelnonrewl, __, velnonrewl = perireward_binned_activity(velocity, nonrew_stop_with_lick_per_plane, 
                fall['timedFF'][0], fall['trialnum'][0], range_val,binsize)
        _, meanlicknonrewwl, __, licknonrewwl = perireward_binned_activity(fall['licks'][0], nonrew_stop_with_lick_per_plane, 
            fall['timedFF'][0], fall['trialnum'][0], range_val,binsize)
        _, meanrewnonrewl, __, rewnonrewl = perireward_binned_activity(fall['rewards'][0], nonrew_stop_with_lick_per_plane, 
            fall['timedFF'][0], fall['trialnum'][0], range_val,binsize)

        _, meanrstops, __, rewrstops = perireward_binned_activity(Fc3[:,gc], rew_per_plane, 
        fall['timedFF'][0], fall['trialnum'][0], range_val,binsize)
        _, meanvelrew, __, velrew = perireward_binned_activity(velocity, rew_per_plane, 
                fall['timedFF'][0], fall['trialnum'][0], range_val,binsize)
        _, meanlickrew, __, lickrew = perireward_binned_activity(fall['licks'][0], rew_per_plane, 
            fall['timedFF'][0], fall['trialnum'][0], range_val,binsize)
        _, meanrewrewstops, __, rewrewstops = perireward_binned_activity(fall['rewards'][0], rew_per_plane, 
            fall['timedFF'][0], fall['trialnum'][0], range_val,binsize)
        # save per cell
        trials_dff_rstops.append([rewrstops,velrew,lickrew,rewrewstops])
        trials_dff_nrstops_wo_licks.append([rewnstops,velnonrew,licknonrew,rewnonrew])
        trials_dff_nrstops_w_licks.append([rewnlstops,velnonrewl,licknonrewwl,rewnonrewl])
        
    dff_per_an_day_per_trial_type[k]=[trials_dff_rstops,trials_dff_nrstops_wo_licks,
                            trials_dff_nrstops_w_licks]

#%% 
# get nonrew stops with or without licks
animals = conddf.animals.unique()

# animal x day x cells x trials
# get 0 ind for dff
# 1 - vel; 2 - lick; 3 - reward
nrewstops_wo_licks_trials_per_an =[[[yy[0] for yy in v[1]] for k,v in dff_per_an_day_per_trial_type.items() \
                                    if an in k and len(v[1])>0] for an in animals]
nrewstops_w_licks_trials_per_an =[[[yy[0] for yy in v[2]] for k,v in dff_per_an_day_per_trial_type.items() \
                                    if an in k and len(v[1])>0] for an in animals]
rewstops_trials_per_an =[[[yy[0] for yy in v[0]] for k,v in dff_per_an_day_per_trial_type.items() \
                                    if an in k and len(v[1])>0] for an in animals]

vel_nrewstops_wo_licks_trials_per_an =[[[yy[1] for yy in v[1]] for k,v in dff_per_an_day_per_trial_type.items() \
                                    if an in k and len(v[1])>0] for an in animals]
vel_nrewstops_w_licks_trials_per_an =[[[yy[1] for yy in v[2]] for k,v in dff_per_an_day_per_trial_type.items() \
                                    if an in k and len(v[1])>0] for an in animals]
vel_rewstops_trials_per_an =[[[yy[1] for yy in v[0]] for k,v in dff_per_an_day_per_trial_type.items() \
                                    if an in k and len(v[1])>0] for an in animals]

lick_nrewstops_wo_licks_trials_per_an =[[[yy[2] for yy in v[1]] for k,v in dff_per_an_day_per_trial_type.items() \
                                    if an in k and len(v[1])>0] for an in animals]
lick_nrewstops_w_licks_trials_per_an =[[[yy[2] for yy in v[2]] for k,v in dff_per_an_day_per_trial_type.items() \
                                    if an in k and len(v[1])>0] for an in animals]
lick_rewstops_trials_per_an =[[[yy[2] for yy in v[0]] for k,v in dff_per_an_day_per_trial_type.items() \
                                    if an in k and len(v[1])>0] for an in animals]


#%%
# plot
# get both rewarded and unrewarded stops with licks/without licks
dst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper\panels_main_figures'
def moving_average(x, window_size):
    return np.convolve(x, np.ones(window_size)/window_size, mode='same')
plt.rc('font', size=20) 
a=.3
colors = ['firebrick','seagreen']
for an_eg in range(len(animals)):
    try:
        all_cells = []
        for i in nrewstops_wo_licks_trials_per_an[an_eg]: # animal 1 
            for j in i: # sessions?
                all_cells.append(j) # average across cells
        nrewstops_wo_licks_trials_cells = np.hstack(all_cells)

        all_cells = []
        # combine types of stops
        for i in nrewstops_w_licks_trials_per_an[an_eg]: # animal 1 
            for j in i: # sessions?
                all_cells.append(j) # average across cells
        nrewstops_w_licks_trials_cells = np.hstack(all_cells)

        all_cells = []
        for i in rewstops_trials_per_an[an_eg]: # animal 1 
            for j in i: # sessions?
                all_cells.append(j) # average across cells
        rewstops_trials_cells = np.hstack(all_cells)

        # per animal average
        fig, axes = plt.subplots(nrows = 3, sharex=True, figsize=(3,5))
        ax=axes[0]
        m=np.nanmean(nrewstops_wo_licks_trials_cells,axis=1)
        ax.plot(m,color=colors[0])
        ax.fill_between(
            range(0, int(range_val / binsize) * 2),
            m - scipy.stats.sem(nrewstops_wo_licks_trials_cells, axis=1, nan_policy='omit'),
            m + scipy.stats.sem(nrewstops_wo_licks_trials_cells, axis=1, nan_policy='omit'),
            alpha=0.5,color=colors[0]
            ) 

        m=np.nanmean(nrewstops_w_licks_trials_cells,axis=1)
        ax.plot(m,color='sienna')
        ax.fill_between(
            range(0, int(range_val / binsize) * 2),
            m - scipy.stats.sem(nrewstops_w_licks_trials_cells, axis=1, nan_policy='omit'),
            m + scipy.stats.sem(nrewstops_w_licks_trials_cells, axis=1, nan_policy='omit'),
            alpha=0.5,color='sienna'
            )                    
        m=np.nanmean(rewstops_trials_cells,axis=1)
        ax.plot(m,color=colors[1])
        ax.fill_between(
            range(0, int(range_val / binsize) * 2),
            m - scipy.stats.sem(rewstops_trials_cells, axis=1, nan_policy='omit'),
            m + scipy.stats.sem(rewstops_trials_cells, axis=1, nan_policy='omit'),
            alpha=0.5,color=colors[1]
            )                    
        ax.set_title(f'{animals[an_eg]}')
        ax.set_ylabel('$\Delta$ F/F')
        ax.spines[['top', 'right']].set_visible(False)    
        ax.axvline(int(range_val / binsize), color='k', linestyle='--')
        # velocity!
        all_cells = []
        for i in vel_nrewstops_wo_licks_trials_per_an[an_eg]: # animal 1 
            for j in i: # sessions?
                all_cells.append(j) # average across cells
        nrewstops_wo_licks_trials_cells = np.hstack(all_cells)

        # all_cells = []
        for i in vel_nrewstops_w_licks_trials_per_an[an_eg]: # animal 1 
            for j in i: # sessions?
                all_cells.append(j) # average across cells
        nrewstops_w_licks_trials_cells = np.hstack(all_cells)

        all_cells = []
        for i in vel_rewstops_trials_per_an[an_eg]: # animal 1 
            for j in i: # sessions?
                all_cells.append(j) # average across cells
        rewstops_trials_cells = np.hstack(all_cells)

        # per animal average
        ax=axes[1]
        m=np.nanmean(nrewstops_wo_licks_trials_cells,axis=1)
        ax.plot(m,color=colors[0])
        ax.fill_between(
            range(0, int(range_val / binsize) * 2),
            m - scipy.stats.sem(nrewstops_wo_licks_trials_cells, axis=1, nan_policy='omit'),
            m + scipy.stats.sem(nrewstops_wo_licks_trials_cells, axis=1, nan_policy='omit'),
            alpha=0.5,color=colors[0]
            )                    

        m=np.nanmean(nrewstops_w_licks_trials_cells,axis=1)
        ax.plot(m,color='sienna')
        ax.fill_between(
            range(0, int(range_val / binsize) * 2),
            m - scipy.stats.sem(nrewstops_w_licks_trials_cells, axis=1, nan_policy='omit'),
            m + scipy.stats.sem(nrewstops_w_licks_trials_cells, axis=1, nan_policy='omit'),
            alpha=0.5,color='sienna'
            )                    
        m=np.nanmean(rewstops_trials_cells,axis=1)
        ax.plot(m,color=colors[1])
        ax.fill_between(
            range(0, int(range_val / binsize) * 2),
            m - scipy.stats.sem(rewstops_trials_cells, axis=1, nan_policy='omit'),
            m + scipy.stats.sem(rewstops_trials_cells, axis=1, nan_policy='omit'),
            alpha=0.5,color=colors[1]
            )                    

        ax.set_ylabel('Velocity (cm/s)')
        ax.spines[['top', 'right']].set_visible(False)    
        ax.axvline(int(range_val / binsize), color='k', linestyle='--')
        # licks!
        all_cells = []
        for i in lick_nrewstops_wo_licks_trials_per_an[an_eg]: # animal 1 
            for j in i: # sessions?
                all_cells.append(j) # average across cells
        nrewstops_wo_licks_trials_cells = np.hstack(all_cells)

        # all_cells = []
        for i in lick_nrewstops_w_licks_trials_per_an[an_eg]: # animal 1 
            for j in i: # sessions?
                all_cells.append(j) # average across cells
        nrewstops_w_licks_trials_cells = np.hstack(all_cells)

        all_cells = []
        for i in lick_rewstops_trials_per_an[an_eg]: # animal 1 
            for j in i: # sessions?
                all_cells.append(j) # average across cells
        rewstops_trials_cells = np.hstack(all_cells)

        # per animal average
        ax=axes[2]
        m=np.nanmean(nrewstops_wo_licks_trials_cells,axis=1)/np.nanmax(nrewstops_w_licks_trials_cells,axis=1)
        sem = np.array([moving_average(s,window_size=5) for s in nrewstops_wo_licks_trials_cells])
        m=moving_average(m, window_size=5)
        ax.plot(m,color=colors[0],label=f'No lick\n {sem.shape[1]} trials')
        ax.fill_between(
            range(0, int(range_val / binsize) * 2),
            m - scipy.stats.sem(sem, axis=1, nan_policy='omit'),
            m + scipy.stats.sem(sem, axis=1, nan_policy='omit'),
            alpha=0.5,color=colors[0]
            )                 

        m=np.nanmean(nrewstops_w_licks_trials_cells,axis=1)/np.nanmax(nrewstops_w_licks_trials_cells,axis=1)
        sem = np.array([moving_average(s,window_size=5) for s in nrewstops_w_licks_trials_cells])
        m=moving_average(m, window_size=5)
        ax.plot(m,color='sienna',label=f'Lick\n {sem.shape[1]} trials')
        ax.fill_between(
            range(0, int(range_val / binsize) * 2),
            m - scipy.stats.sem(sem, axis=1, nan_policy='omit'),
            m + scipy.stats.sem(sem, axis=1, nan_policy='omit'),
            alpha=0.5,color='sienna'
            )                    
        m=np.nanmean(rewstops_trials_cells,axis=1)/np.nanmax(nrewstops_w_licks_trials_cells,axis=1)
        sem = np.array([moving_average(s,window_size=5) for s in rewstops_trials_cells])
        m=moving_average(m, window_size=5)
        ax.plot(m,color=colors[1],label=f'Rewarded\n{sem.shape[1]} trials')
        ax.fill_between(
            range(0, int(range_val / binsize) * 2),
            m - scipy.stats.sem(sem, axis=1, nan_policy='omit'),
            m + scipy.stats.sem(sem, axis=1, nan_policy='omit'),
            alpha=0.5,color=colors[1]
            )                    
        ax.set_xlabel('Time from stop (s)')
        ax.set_ylabel('Norm lick rate')
        ax.axvline(int(range_val / binsize), color='k', linestyle='--')
        ax.spines[['top', 'right']].set_visible(False)    
        ax.set_xticks(np.arange(0, (int(range_val / binsize) * 2) + 1,20))
        ax.set_xticklabels(np.arange(-range_val, range_val + 1, 2))
        ax.legend()
        plt.savefig(os.path.join(dst, f'{animals[an_eg]}_postrew_stops.svg'),bbox_inches='tight')
    except:
        print(animals[an_eg])
#%%
# average activity of all cells
# get post rew / stop activity
secs_post_rew=10# window after stop
mov_start=4
# average of all cells
# subtract from prewidnow
pre=-5
nrewstops_wo_licks_trials_per_an_av = [[np.nanmean(np.hstack(yy)[int((range_val/binsize)+(mov_start/binsize)):int((range_val/binsize)+(secs_post_rew/binsize))],axis=(1,0))-np.nanmean(np.hstack(yy)[int((range_val/binsize)+(pre/binsize)):int((range_val/binsize))],axis=(1,0)) for yy in xx] for xx in nrewstops_wo_licks_trials_per_an]
nrewstops_w_licks_trials_per_an_av = [[np.nanmean(np.hstack(yy)[int((range_val/binsize)+(mov_start/binsize)):int((range_val/binsize)+(secs_post_rew/binsize))],axis=(1,0))-np.nanmean(np.hstack(yy)[int((range_val/binsize)+(pre/binsize)):int((range_val/binsize))],axis=(1,0)) for yy in xx] for xx in nrewstops_w_licks_trials_per_an]
rewstops_trials_per_an_av = [[np.nanmean(np.hstack(yy)[int((range_val/binsize)+(mov_start/binsize)):int((range_val/binsize)+(secs_post_rew/binsize))],axis=(1,0))-np.nanmean(np.hstack(yy)[int((range_val/binsize)+(pre/binsize)):int((range_val/binsize))],axis=(1,0)) for yy in xx] for xx in rewstops_trials_per_an]
# 1 cell per sequence?
# nrewstops_wo_licks_trials_per_an_av = [[np.nanmean(np.hstack(yy)[int((range_val/binsize)+(mov_start/binsize)):int((range_val/binsize)+(secs_post_rew/binsize))],axis=0)[0] for yy in xx] for xx in nrewstops_wo_licks_trials_per_an]
# nrewstops_w_licks_trials_per_an_av = [[np.nanmean(np.hstack(yy)[int((range_val/binsize)+(mov_start/binsize)):int((range_val/binsize)+(secs_post_rew/binsize))],axis=0)[0] for yy in xx] for xx in nrewstops_w_licks_trials_per_an]
# rewstops_trials_per_an_av = [[np.nanmean(np.hstack(yy)[int((range_val/binsize)+(mov_start/binsize)):int((range_val/binsize)+(secs_post_rew/binsize))],axis=0)[0] for yy in xx] for xx in rewstops_trials_per_an]

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
df=df[(df.animal!='e189')&(df.animal!='e145')&(df.animal!='e139')]
# combined nonrew 
# non_reward_df = df[df['trial_type'].isin(['non_rewarded_stops_wo_licks', 'non_reward_stops_w_licks'])]
# non_reward_avg = non_reward_df.groupby('animal')['activity'].mean().reset_index()
# non_reward_avg['trial_type'] = 'non_rewarded_stops'

# # Step 2: Keep rewarded trials as they are
# reward_df = df[df['trial_type'] == 'rewarded_stops']
# reward_avg = reward_df.groupby('animal')['activity'].mean().reset_index()
# reward_avg['trial_type'] = ['rewarded_stops']*len(reward_avg)
# # Step 3: Combine the averaged non-reward and rewarded data
# df = pd.concat([non_reward_avg, reward_avg], ignore_index=True)
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
v# Get unique trial types
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
# fig.suptitle('Stops\nNear post-reward cells')
plt.tight_layout()
plt.savefig(os.path.join(savedst, 'postrew_cells_by_stop_trial_type.svg'),bbox_inches='tight')

#%%
# combined nonrew 
non_reward_df = df[df['trial_type'].isin(['non_rewarded_stops_wo_licks', 'non_reward_stops_w_licks'])]
non_reward_avg = non_reward_df.groupby('animal')['activity'].mean().reset_index()
non_reward_avg['trial_type'] = 'non_rewarded_stops'

# Step 2: Keep rewarded trials as they are
reward_df = df[df['trial_type'] == 'rewarded_stops']
reward_avg = reward_df.groupby('animal')['activity'].mean().reset_index()
reward_avg['trial_type'] = ['rewarded_stops']*len(reward_avg)
# Step 3: Combine the averaged non-reward and rewarded data
df = pd.concat([non_reward_avg, reward_avg], ignore_index=True)
df=df.reset_index()
palette = {'rewarded_stops':'seagreen', 'non_rewarded_stops': 'firebrick'}
#%%
# plot all cells
fig,ax=plt.subplots(figsize=(2.5,5))
s=12
order=['non_rewarded_stops','rewarded_stops']
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
v# Get unique trial types
trial_types = df['trial_type'].unique()
# Create all pairwise comparisons
comparisons = list(itertools.combinations(trial_types, 2))
# Store p-values
p_values = []
test_results = {}
for group1, group2 in comparisons:
    # Extract paired data (same animals)
    paired_data = df.pivot(index="animal", columns="trial_type", values="activity").dropna()
    # Perform paired t-test
    t_stat, p_val = scipy.stats.wilcoxon(paired_data[group1], paired_data[group2])
    p_values.append(p_val)
    test_results[(group1, group2)] = p_val

# Apply Bonferroni correction
reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
# Print results
for (group1, group2), corrected_p in zip(comparisons, pvals_corrected):
    print(f"Paired t-test: {group1} vs {group2}, Corrected p-value: {corrected_p:.4f}")
    
# Get unique trial types and positions
group_names = df["trial_type"].unique()
group_positions = {name: i for i, name in enumerate(group_names)}

# Get max y-value for annotation positioning
y_max = df["activity"].max()-.3
y_offset = (y_max - df["activity"].min()) * 0.4  # Adjust spacing

# Define height adjustment for each comparison
bar_heights = [y_max + (i + 2) * y_offset for i in range(len(comparisons))]

# Iterate through pairwise comparisons
for i, ((group1, group2), corrected_p) in enumerate(zip(comparisons, pvals_corrected)):
    x1, x2 = group_positions[group1], group_positions[group2]
    y = bar_heights[i]  # Assign height for this comparison
    # Draw significance bar
    plt.plot([x1, x1, x2, x2], [y, y + y_offset * 0.2, y + y_offset * 0.2, y], 'k', lw=1.5)
    # Annotate with corrected p-value
    p_text = f"p = {corrected_p:.3g}"
    ax.text((x1 + x2) / 2, y + y_offset * 0.3, p_text, 
             ha='center', va='bottom', fontsize=12)
    ax.text((x1 + x2) / 2, y + y_offset * 0.3, '**', 
             ha='center', va='bottom', fontsize=42)
ax.set_xticklabels(['Unrewarded','Rewarded'])
ax.set_ylabel('Mean $\Delta F/F$ (after-before mov.)')
ax.set_xlabel('Trial type')
fig.suptitle('Types of stops\nPost-reward cells')
plt.savefig(os.path.join(savedst, 'postrew_cells_by_stop_trial_type_combined.svg'),bbox_inches='tight')

# %%

#%% 
# get examples
all_tr_nrewstops_wo_licks_trials_per_an = []
for ii,xx in enumerate(nrewstops_wo_licks_trials_per_an):
    per_day_tr = [np.hstack(yy) for yy in xx if len(xx)>0]
    if len(per_day_tr)>0:
        all_tr_nrewstops_wo_licks_trials_per_an.append(np.hstack(per_day_tr))
# all_tr_nrewstops_w_licks_trials_per_an = [np.hstack([np.hstack(yy) for yy in xx]) for xx in nrewstops_w_licks_trials_per_an]
# all_tr_rewstops_trials_per_an = [np.hstack([np.hstack(yy) for yy in xx]) for xx in rewstops_trials_per_an]


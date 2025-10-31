
"""
zahra
july 2024
get rew-relative cells in different trial conditions

1st probe trial
other 2 probe trials
initial failed trials of an epoch
failed trials after successful trails
1st correct trial
correct trials
"""
#%%

import numpy as np, random, re, os, scipy, pandas as pd, sys, cv2, ot
import matplotlib.pyplot as plt, matplotlib
from itertools import combinations, chain
from scipy.spatial import distance
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.pyr_reward.placecell import intersect_arrays,make_tuning_curves_radians_by_trialtype,\
    consecutive_stretch,make_tuning_curves,make_tuning_curves_trial_by_trial,\
        make_tuning_curves_radians_by_trialtype_behavior, make_tuning_curves_by_trialtype_w_darktime, make_tuning_curves_by_trialtype_w_probes_w_darktime
from projects.opto.behavior.behavior import get_success_failure_trials, smooth_lick_rate
from projects.memory.behavior import get_lick_selectivity
from collections import Counter

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

def cosine_sim_ignore_nan(a, b):
    # Mask where both vectors have valid (non-NaN) values
    mask = ~np.isnan(a) & ~np.isnan(b)
    if np.sum(mask) == 0:
        return np.nan  # No overlapping data to compare
    a_masked = a[mask]
    b_masked = b[mask]
    numerator = np.dot(a_masked, b_masked)
    denominator = np.linalg.norm(a_masked) * np.linalg.norm(b_masked)
    return numerator / denominator if denominator != 0 else np.nan

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def get_rewzones(rewlocs, gainf):
    # note that gainf is multiplied here
    # gainf = 1/scalingf
    # Initialize the reward zone numbers array with zeros
    rewzonenum = np.zeros(len(rewlocs))
    
    # Iterate over each reward location to determine its reward zone
    for kk, loc in enumerate(rewlocs):
        if loc <= 86 * gainf:
            rewzonenum[kk] = 1  # Reward zone 1
        elif 101 * gainf <= loc <= 120 * gainf:
            rewzonenum[kk] = 2  # Reward zone 2
        elif loc >= 135 * gainf:
            rewzonenum[kk] = 3  # Reward zone 3
            
    return rewzonenum


def extract_data_warped_rewcentric(ii,params_pth,animal,day,bins,radian_alignment,
    radian_alignment_saved,goal_cm_window,pdf,epoch_perm,goal_cell_iind,goal_cell_prop,num_epochs,
    goal_cell_null,pvals,total_cells,
    num_iterations=1000):
    """
    changed on 2/6/25 to make it more consistent with splitting the different
    subpopulations
    """
    print(params_pth)
    fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
            'pyr_tc_s2p_cellind', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
            'stat', 'licks'])
    VR = fall['VR'][0][0][()]
    scalingf = VR['scalingFACTOR'][0][0]
    try:
            rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf        
    except:
            rewsize = 10
    ybinned = fall['ybinned'][0]/scalingf
    track_length=180/scalingf    
    forwardvel = fall['forwardvel'][0]    
    changeRewLoc = np.hstack(fall['changeRewLoc'])
    trialnum=fall['trialnum'][0]
    rewards = fall['rewards'][0]
    licks=fall['licks'][0]
    if animal=='e145':
        ybinned=ybinned[:-1]
        forwardvel=forwardvel[:-1]
        changeRewLoc=changeRewLoc[:-1]
        trialnum=trialnum[:-1]
        rewards=rewards[:-1]
        licks=licks[:-1]
    # set vars
    eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
    lasttr=8 # last trials
    bins=90
    # rad = get_radian_position_first_lick_after_rew(eps, ybinned, licks, rewards, rewsize,rewlocs,
    #                 trialnum, track_length) # get radian coordinates
    track_length_rad = track_length*(2*np.pi/track_length)
    bin_size=track_length/bins 
    rz = get_rewzones(rewlocs,1/scalingf)       
    # get average success rate
    rates = []; norm_pos = []
    for ep in range(len(eps)-1):
        eprng = range(eps[ep],eps[ep+1])
        pos = normalize_values(ybinned[eprng], rewlocs[ep]-rewsize/2, rewlocs[ep]+rewsize/2, 
                track_length)
        # values are a list of positions
        # b is the reward start location
        # c is the reward end location
        success, fail, str_trials, ftr_trials, ttr, \
        total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
        rates.append(success/total_trials)
        norm_pos.append(pos)
    rate=np.nanmean(np.array(rates))
    norm_pos=np.concatenate(norm_pos)
    # added to get anatomical info
    # takes time
    fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
    Fc3 = fall_fc3['Fc3']
    dFF = fall_fc3['dFF']
    Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
    dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]
    skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
    Fc3 = Fc3[:, skew>2] # only keep cells with skew greateer than 2
    tcs_correct_abs, coms_correct_abs = make_tuning_curves_warped(eps,rewlocs,norm_pos,Fc3,trialnum,
                rewards,forwardvel,rewsize,bin_size,lasttr=8,bins=90,
                velocity_filter=False)
    # scale window from -1 to 1
    goal_window = ((2*goal_cm_window)/270) # cm converted to warped
    # convert coms from -1 to 1
    coms_correct_abs=np.array([[(2 * (com_cm - track_length/2))/track_length for com_cm in coms_abs] for coms_abs in coms_correct_abs])
    # change to relative value 
    coms_rewrel = coms_correct_abs
    perm = list(combinations(range(len(coms_correct_abs)), 2)) 
    rz_perm = [(int(rz[p[0]]),int(rz[p[1]])) for p in perm]   
    # if 4 ep
    # account for cells that move to the end/front
    # Define a small window around pi (e.g., epsilon)
    epsilon = goal_window # 20 cm
    # Find COMs near pi and shift to -pi
    com_loop_w_in_window = []
    for pi,p in enumerate(perm):
        for cll in range(coms_rewrel.shape[1]):
            com1_rel = coms_rewrel[p[0],cll]
            com2_rel = coms_rewrel[p[1],cll]
            # print(com1_rel,com2_rel,com_diff)
            if ((abs(com1_rel - 1) < epsilon) and 
            (abs(com2_rel + 1) < epsilon)):
                    com_loop_w_in_window.append(cll)
    # get abs value instead
    coms_rewrel[:,com_loop_w_in_window]=abs(coms_rewrel[:,com_loop_w_in_window])
    com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
    com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
    #only get perms with non zero cells
    perm=[p for ii,p in enumerate(perm) if len(com_goal[ii])>0]
    rz_perm=[p for ii,p in enumerate(rz_perm) if len(com_goal[ii])>0]
    com_goal=[com for com in com_goal if len(com)>0]

    print(f'Reward-centric cells total: {[len(xx) for xx in com_goal]}')
    epoch_perm.append([perm,rz_perm]) 
    # get goal cells across all epochs   
    if len(com_goal)>0:
        goal_cells = intersect_arrays(*com_goal); 
    else:
        goal_cells=[]
    # get goal cells across all epochs        
    # s2p_iind_goal_cells = s2p_iind_filter[goal_cells]
    # get per comparison
    goal_cells_p_per_comparison = [len(xx)/len(coms_correct_abs[0]) for xx in com_goal]
    goal_cell_iind.append(goal_cells)
    goal_cell_p=len(goal_cells)/len(coms_correct_abs[0])        
    goal_cell_prop.append([goal_cells_p_per_comparison,goal_cell_p])
    num_epochs.append(len(coms_correct_abs))
    colors = ['k', 'slategray', 'darkcyan', 'darkgoldenrod', 'orchid']
    if len(goal_cells)>0:
        rows = int(np.ceil(np.sqrt(len(goal_cells))))
        cols = int(np.ceil(len(goal_cells) / rows))
        fig, axes = plt.subplots(rows, cols, figsize=(30,30),sharex=True)
        if len(goal_cells) > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        for i,gc in enumerate(goal_cells):            
            for ep in range(len(coms_correct_abs)):
                ax = axes[i]
                ax.plot(tcs_correct_abs[ep,gc,:], label=f'rewloc {rewlocs[ep]}', color=colors[ep])
                ax.axvline((bins/2), color='k')
                ax.set_title(f'cell # {gc}')
                ax.spines[['top','right']].set_visible(False)
        ax.set_xticks(np.arange(0,bins+1,15))
        ax.set_xticklabels(np.round(np.arange(-1, 1+1/3, 1/3),3))
        ax.set_xlabel('Radian position (centered start rew loc)')
        ax.set_ylabel('Fc3')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    # get shuffled iteration
    shuffled_dist = np.zeros((num_iterations))
    # max of 5 epochs = 10 perms
    goal_cell_shuf_ps_per_comp = np.ones((num_iterations,10))*np.nan
    goal_cell_shuf_ps = []
    for i in range(num_iterations):
        # shuffle locations
        rewlocs_shuf = rewlocs #[random.randint(100,250) for iii in range(len(eps))]
        shufs = [list(range(coms_correct_abs[ii].shape[0])) for ii in range(1, len(coms_correct_abs))]
        [random.shuffle(shuf) for shuf in shufs]
        # first com is as ep 1, others are shuffled cell identities
        com_shufs = np.zeros_like(coms_correct_abs); com_shufs[0,:] = coms_correct_abs[0]
        com_shufs[1:1+len(shufs),:] = [coms_correct_abs[ii][np.array(shufs)[ii-1]] for ii in range(1, 1+len(shufs))]
        # OR shuffle cell identities
        # relative to reward
        coms_rewrel = com_shufs
        # account for cells that move to the end/front
        # Find COMs near pi and shift to -pi
        com_loop_w_in_window = []
        for pi,p in enumerate(perm):
            for cll in range(coms_rewrel.shape[1]):
                com1_rel = coms_rewrel[p[0],cll]
                com2_rel = coms_rewrel[p[1],cll]
                # print(com1_rel,com2_rel,com_diff)
                if ((abs(com1_rel - 1) < epsilon) and 
                (abs(com2_rel + 1) < epsilon)):
                        com_loop_w_in_window.append(cll)
        # get abs value instead
        # cont.
        coms_rewrel[:,com_loop_w_in_window]=abs(coms_rewrel[:,com_loop_w_in_window])
        com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
        # get goal cells across all epochs
        com_goal_shuf = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
        com_goal_shuf=[com for com in com_goal_shuf if len(com)>0]
        goal_cells_shuf_p_per_comparison = [len(xx)/len(coms_correct_abs[0]) for xx in com_goal_shuf]
        # get goal cells across all epochs   
        if len(com_goal_shuf)>0:
            goal_cells_shuf = intersect_arrays(*com_goal_shuf); 
        else:
            goal_cells_shuf=[]
        shuffled_dist[i] = len(goal_cells_shuf)/len(coms_correct_abs[0])
        goal_cell_shuf_p=len(goal_cells_shuf)/len(coms_correct_abs[0])
        goal_cell_shuf_ps.append(goal_cell_shuf_p)
        goal_cell_shuf_ps_per_comp[i, :len(goal_cells_shuf_p_per_comparison)] = goal_cells_shuf_p_per_comparison
    # save median of goal cell shuffle
    goal_cell_shuf_ps_per_comp_av = np.nanmedian(goal_cell_shuf_ps_per_comp,axis=0)        
    goal_cell_shuf_ps_av = np.nanmedian(np.array(goal_cell_shuf_ps)[1])
    goal_cell_null.append([goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av])
    p_value = sum(shuffled_dist>goal_cell_p)/num_iterations
    pvals.append(p_value); 
    print(f'{animal}, day {day}: significant goal cells proportion p-value: {p_value}')
    total_cells.append(len(coms_correct_abs[0]))
    radian_alignment[f'{animal}_{day:03d}_index{ii:03d}'] = [tcs_correct_abs, coms_correct_abs,
                    com_goal, goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av]
    
    return radian_alignment,rate,p_value,total_cells,goal_cell_iind,\
        goal_cell_prop,num_epochs,goal_cell_null,epoch_perm,pvals
        
        
def extract_data_prerew(ii,params_pth,
    animal,day,bins,radian_alignment,radian_alignment_saved,goal_cm_window,
    pdf,epoch_perm,goal_cell_iind,goal_cell_prop,num_epochs,goal_cell_null,pvals,
    total_cells,num_iterations=1000):
    """
    changed on 2/6/25 to make it more consistent with splitting the different
    subpopulations
    """
    print(params_pth)
    fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
            'timedFF', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
            'stat', 'timedFF', 'licks'])
    VR = fall['VR'][0][0][()]
    scalingf = VR['scalingFACTOR'][0][0]
    try:
        rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf        
    except:
        rewsize = 10
    ybinned = fall['ybinned'][0]/scalingf
    track_length=180/scalingf    
    forwardvel = fall['forwardvel'][0]    
    changeRewLoc = np.hstack(fall['changeRewLoc'])
    trialnum=fall['trialnum'][0]
    rewards = fall['rewards'][0]
    lick=fall['licks'][0]
    time=fall['timedFF'][0]
    if animal=='e145':
        ybinned=ybinned[:-1]
        forwardvel=forwardvel[:-1]
        changeRewLoc=changeRewLoc[:-1]
        trialnum=trialnum[:-1]
        rewards=rewards[:-1]
        lick=lick[:-1]
        time=time[:-1]
    # set vars
    eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
    lasttr=8 # last trials
    bins=90
    rad = get_radian_position_first_lick_after_rew(eps, ybinned, lick, rewards, rewsize,rewlocs,
                    trialnum, track_length) # get radian coordinates
    track_length_rad = track_length*(2*np.pi/track_length)
    bin_size=track_length_rad/bins 
    rz = get_rewzones(rewlocs,1/scalingf)       
    # get average success rate
    rates = []
    for ep in range(len(eps)-1):
        eprng = range(eps[ep],eps[ep+1])
        success, fail, str_trials, ftr_trials, ttr, \
        total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
        rates.append(success/total_trials)
    rate=np.nanmean(np.array(rates))
    # dark time params
    track_length_dt = 550 # cm estimate based on 99.9% of ypos
    track_length_rad_dt = track_length_dt*(2*np.pi/track_length_dt) # estimate bin for dark time
    bins_dt=150 
    bin_size_dt=track_length_rad_dt/bins_dt # typically 3 cm binswith ~ 475 track length
    # added to get anatomical info
    # takes time
    fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
    Fc3 = fall_fc3['Fc3']
    dFF = fall_fc3['dFF']
    Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
    dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]
    skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
    Fc3 = Fc3[:, skew>2] # only keep cells with skew greateer than 2
    if f'{animal}_{day:03d}_index{ii:03d}' in radian_alignment_saved.keys():
        tcs_correct, coms_correct, tcs_fail, coms_fail, \
        com_goal, goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av = radian_alignment_saved[f'{animal}_{day:03d}_index{ii:03d}']            
    else:# remake tuning curves relative to reward        
        # 9/19/24
        # tc w/ dark time added to the end of track
        tcs_correct, coms_correct, tcs_fail, coms_fail, ybinned_dt, raddt = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,
            rewsize,ybinned,time,lick,
            Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
            bins=bins_dt)  

    goal_window = goal_cm_window*(2*np.pi/track_length) # cm converted to rad
    # change to relative value 
    coms_rewrel = np.array([com-np.pi for com in coms_correct])
    perm = list(combinations(range(len(coms_correct)), 2)) 
    rz_perm = [(int(rz[p[0]]),int(rz[p[1]])) for p in perm]   
    # if 4 ep
    # account for cells that move to the end/front
    # Define a small window around pi (e.g., epsilon)
    epsilon = .7 # 20 cm
    # Find COMs near pi and shift to -pi
    com_loop_w_in_window = []
    for pi,p in enumerate(perm):
        for cll in range(coms_rewrel.shape[1]):
            com1_rel = coms_rewrel[p[0],cll]
            com2_rel = coms_rewrel[p[1],cll]
            # print(com1_rel,com2_rel,com_diff)
            if ((abs(com1_rel - np.pi) < epsilon) and 
            (abs(com2_rel + np.pi) < epsilon)):
                    com_loop_w_in_window.append(cll)
    # get abs value instead
    coms_rewrel[:,com_loop_w_in_window]=abs(coms_rewrel[:,com_loop_w_in_window])
    com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
    com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
    # in addition, com near but after goal
    lowerbound = -np.pi/4
    com_goal_postrew = [[xx for xx in com if ((np.nanmedian(coms_rewrel[:,
        xx], axis=0)>=lowerbound) & (np.nanmedian(coms_rewrel[:,
        xx], axis=0)<0))] if len(com)>0 else [] for com in com_goal]
    #only get perms with non zero cells
    perm=[p for ii,p in enumerate(perm) if len(com_goal_postrew[ii])>0]
    rz_perm=[p for ii,p in enumerate(rz_perm) if len(com_goal_postrew[ii])>0]
    com_goal_postrew=[com for com in com_goal_postrew if len(com)>0]

    print(f'Reward-centric cells total: {[len(xx) for xx in com_goal]}\n\
    Pre-reward cells: {[len(xx) for xx in com_goal_postrew]}')
    assert sum([len(xx) for xx in com_goal])>=sum([len(xx) for xx in com_goal_postrew])
    epoch_perm.append([perm,rz_perm]) 
    # get goal cells across all epochs   
    if len(com_goal_postrew)>0:
        goal_cells = intersect_arrays(*com_goal_postrew); 
    else:
        goal_cells=[]
    # s2p_iind_goal_cells = s2p_iind_filter[goal_cells]
    # get per comparison
    goal_cells_p_per_comparison = [len(xx)/len(coms_correct[0]) for xx in com_goal_postrew]
    goal_cell_iind.append(goal_cells)
    goal_cell_p=len(goal_cells)/len(coms_correct[0])        
    goal_cell_prop.append([goal_cells_p_per_comparison,goal_cell_p])
    num_epochs.append(len(coms_correct))
    colors = ['k', 'slategray', 'darkcyan', 'darkgoldenrod', 'orchid']
    if len(goal_cells)>0:
        rows = int(np.ceil(np.sqrt(len(goal_cells))))
        cols = int(np.ceil(len(goal_cells) / rows))
        fig, axes = plt.subplots(rows, cols, figsize=(30,30),sharex=True)
        if len(goal_cells) > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        for i,gc in enumerate(goal_cells):            
            for ep in range(len(coms_correct)):
                ax = axes[i]
                ax.plot(tcs_correct[ep,gc,:], label=f'rewloc {rewlocs[ep]}', color=colors[ep])
                if len(tcs_fail)>0:
                        ax.plot(tcs_fail[ep,gc,:], label=f'fail rewloc {rewlocs[ep]}', color=colors[ep], linestyle = '--')
                ax.axvline((bins_dt/2), color='k')
                ax.set_title(f'cell # {gc}')
                ax.spines[['top','right']].set_visible(False)
        # ax.set_xticks(np.arange(0,bins+1,20))
        # ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi, np.pi/2.25),3))
        ax.set_xlabel('Radian position (centered start rew loc)')
        ax.set_ylabel('Fc3')
        fig.suptitle(animal)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    # get shuffled iteration
    shuffled_dist = np.zeros((num_iterations))
    # max of 5 epochs = 10 perms
    goal_cell_shuf_ps_per_comp = np.ones((num_iterations,10))*np.nan
    goal_cell_shuf_ps = []
    for i in range(num_iterations):
        # shuffle locations
        rewlocs_shuf = rewlocs #[random.randint(100,250) for iii in range(len(eps))]
        shufs = [list(range(coms_correct[ii].shape[0])) for ii in range(1, len(coms_correct))]
        [random.shuffle(shuf) for shuf in shufs]
        # first com is as ep 1, others are shuffled cell identities
        com_shufs = np.zeros_like(coms_correct); com_shufs[0,:] = coms_correct[0]
        com_shufs[1:1+len(shufs),:] = [coms_correct[ii][np.array(shufs)[ii-1]] for ii in range(1, 1+len(shufs))]
        # OR shuffle cell identities
        # relative to reward
        coms_rewrel = np.array([com-np.pi for ii, com in enumerate(com_shufs)])             
        perm = list(combinations(range(len(coms_correct)), 2)) 
        # account for cells that move to the end/front
        # Find COMs near pi and shift to -pi
        com_loop_w_in_window = []
        for pi,p in enumerate(perm):
            for cll in range(coms_rewrel.shape[1]):
                com1_rel = coms_rewrel[p[0],cll]
                com2_rel = coms_rewrel[p[1],cll]
                # print(com1_rel,com2_rel,com_diff)
                if ((abs(com1_rel - np.pi) < epsilon) and 
                (abs(com2_rel + np.pi) < epsilon)):
                        com_loop_w_in_window.append(cll)
        # get abs value instead
        # cont.
        coms_rewrel[:,com_loop_w_in_window]=abs(coms_rewrel[:,com_loop_w_in_window])
        com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
        # get goal cells across all epochs
        com_goal_shuf = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
        # in addition, com near but after goal
        com_goal_postrew_shuf = [[xx for xx in com if ((np.nanmedian(coms_rewrel[:,
            xx], axis=0)>=lowerbound) & (np.nanmedian(coms_rewrel[:,
            xx], axis=0)<0))] if len(com)>0 else [] for com in com_goal_shuf]
        # check to make sure just a subset
        # otherwise reshuffle
        while not sum([len(xx) for xx in com_goal_shuf])>=sum([len(xx) for xx in com_goal_postrew_shuf]):
            print('redo')
            shufs = [list(range(coms_correct[ii].shape[0])) for ii in range(1, len(coms_correct))]
            [random.shuffle(shuf) for shuf in shufs]
            # first com is as ep 1, others are shuffled cell identities
            com_shufs = np.zeros_like(coms_correct); com_shufs[0,:] = coms_correct[0]
            com_shufs[1:1+len(shufs),:] = [coms_correct[ii][np.array(shufs)[ii-1]] for ii in range(1, 1+len(shufs))]
            # OR shuffle cell identities
            # relative to reward
            coms_rewrel = np.array([com-np.pi for ii, com in enumerate(com_shufs)])             
            perm = list(combinations(range(len(coms_correct)), 2)) 
            # account for cells that move to the end/front
            # Find COMs near pi and shift to -pi
            com_loop_w_in_window = []
            for pi,p in enumerate(perm):
                for cll in range(coms_rewrel.shape[1]):
                    com1_rel = coms_rewrel[p[0],cll]
                    com2_rel = coms_rewrel[p[1],cll]
                    # print(com1_rel,com2_rel,com_diff)
                    if ((abs(com1_rel - np.pi) < epsilon) and 
                    (abs(com2_rel + np.pi) < epsilon)):
                            com_loop_w_in_window.append(cll)
            # get abs value instead
            # cont.
            coms_rewrel[:,com_loop_w_in_window]=abs(coms_rewrel[:,com_loop_w_in_window])
            com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
            # get goal cells across all epochs
            com_goal_shuf = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
            # in addition, com near but after goal
            com_goal_postrew_shuf = [[xx for xx in com if ((np.nanmedian(coms_rewrel[:,
                xx], axis=0)>=lowerbound) & (np.nanmedian(coms_rewrel[:,
                xx], axis=0)<0))] if len(com)>0 else [] for com in com_goal_shuf]

        perm=[p for ii,p in enumerate(perm) if len(com_goal_postrew_shuf[ii])>0]
        com_goal_postrew_shuf=[com for com in com_goal_postrew_shuf if len(com)>0]

        goal_cells_shuf_p_per_comparison = [len(xx)/len(coms_correct[0]) for xx in com_goal_postrew_shuf]
        if len(com_goal_postrew_shuf)>0:
            goal_cells_shuf = intersect_arrays(*com_goal_postrew_shuf); 
        else:
            goal_cells_shuf=[]
        shuffled_dist[i] = len(goal_cells_shuf)/len(coms_correct[0])
        goal_cell_shuf_p=len(goal_cells_shuf)/len(com_shufs[0])
        goal_cell_shuf_ps.append(goal_cell_shuf_p)
        goal_cell_shuf_ps_per_comp[i, :len(goal_cells_shuf_p_per_comparison)] = goal_cells_shuf_p_per_comparison
    # save median of goal cell shuffle
    goal_cell_shuf_ps_per_comp_av = np.nanmedian(goal_cell_shuf_ps_per_comp,axis=0)        
    goal_cell_shuf_ps_av = np.nanmedian(np.array(goal_cell_shuf_ps)[1])
    goal_cell_null.append([goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av])
    p_value = sum(shuffled_dist>goal_cell_p)/num_iterations
    pvals.append(p_value); 
    print(f'{animal}, day {day}: significant goal cells proportion p-value: {p_value}')
    total_cells.append(len(coms_correct[0]))
    radian_alignment[f'{animal}_{day:03d}_index{ii:03d}'] = [tcs_correct, coms_correct, tcs_fail, coms_fail,
                    com_goal, goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av]
    
    return radian_alignment,rate,p_value,total_cells,goal_cell_iind,\
        goal_cell_prop,num_epochs,goal_cell_null,epoch_perm,pvals


def extract_data_rewcentric(ii,params_pth,animal,day,bins,radian_alignment,
    radian_alignment_saved,goal_cm_window,pdf,epoch_perm,goal_cell_iind,goal_cell_prop,num_epochs,
    goal_cell_null,pvals,total_cells,
    num_iterations=1000):
    """
    changed on 2/6/25 to make it more consistent with splitting the different
    subpopulations
    """
    print(params_pth)
    fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
            'timedFF', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
            'stat', 'licks'])
    VR = fall['VR'][0][0][()]
    scalingf = VR['scalingFACTOR'][0][0]
    try:
            rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf        
    except:
            rewsize = 10
    ybinned = fall['ybinned'][0]/scalingf
    track_length=180/scalingf    
    forwardvel = fall['forwardvel'][0]    
    changeRewLoc = np.hstack(fall['changeRewLoc'])
    trialnum=fall['trialnum'][0]
    rewards = fall['rewards'][0]
    licks=fall['licks'][0]
    time=fall['timedFF'][0]
    if animal=='e145':
        ybinned=ybinned[:-1]
        forwardvel=forwardvel[:-1]
        changeRewLoc=changeRewLoc[:-1]
        trialnum=trialnum[:-1]
        rewards=rewards[:-1]
        licks=licks[:-1]
        time=time[:-1]
    # set vars
    eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
    lasttr=8 # last trials
    bins=90
    rad = get_radian_position_first_lick_after_rew(eps, ybinned, licks, rewards, rewsize,rewlocs,
                    trialnum, track_length) # get radian coordinates
    track_length_rad = track_length*(2*np.pi/track_length)
    bin_size=track_length_rad/bins 
    rz = get_rewzones(rewlocs,1/scalingf)       
    # get average success rate
    rates = []
    for ep in range(len(eps)-1):
            eprng = range(eps[ep],eps[ep+1])
            success, fail, str_trials, ftr_trials, ttr, \
            total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
            rates.append(success/total_trials)
    rate=np.nanmean(np.array(rates))
    # dark time params
    track_length_dt = 550 # cm estimate based on 99.9% of ypos
    track_length_rad_dt = track_length_dt*(2*np.pi/track_length_dt) # estimate bin for dark time
    bins_dt=150 
    bin_size_dt=track_length_rad_dt/bins_dt # typically 3 cm binswith ~ 475 track length
    # added to get anatomical info
    # takes time
    fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
    Fc3 = fall_fc3['Fc3']
    dFF = fall_fc3['dFF']
    Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
    dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]
    skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
    Fc3 = Fc3[:, skew>2] # only keep cells with skew greateer than 2
    if f'{animal}_{day:03d}_index{ii:03d}' in radian_alignment_saved.keys():
        tcs_correct, coms_correct, tcs_fail, coms_fail, \
        com_goal, goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av = radian_alignment_saved[f'{animal}_{day:03d}_index{ii:03d}']            
    else:# remake tuning curves relative to reward        
        # 9/19/24
        # tc w/ dark time added to the end of track
        tcs_correct, coms_correct, tcs_fail, coms_fail, rewloc_dt, ybinned_dt = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,
            rewsize,ybinned,time,lick,
            Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
            bins=bins_dt)  

    goal_window = goal_cm_window*(2*np.pi/track_length) # cm converted to rad
    # change to relative value 
    coms_rewrel = np.array([com-np.pi for com in coms_correct])
    perm = list(combinations(range(len(coms_correct)), 2)) 
    rz_perm = [(int(rz[p[0]]),int(rz[p[1]])) for p in perm]   
    # if 4 ep
    # account for cells that move to the end/front
    # Define a small window around pi (e.g., epsilon)
    epsilon = .7 # 20 cm
    # Find COMs near pi and shift to -pi
    com_loop_w_in_window = []
    for pi,p in enumerate(perm):
            for cll in range(coms_rewrel.shape[1]):
                    com1_rel = coms_rewrel[p[0],cll]
                    com2_rel = coms_rewrel[p[1],cll]
                    # print(com1_rel,com2_rel,com_diff)
                    if ((abs(com1_rel - np.pi) < epsilon) and 
                    (abs(com2_rel + np.pi) < epsilon)):
                            com_loop_w_in_window.append(cll)
    # get abs value instead
    coms_rewrel[:,com_loop_w_in_window]=abs(coms_rewrel[:,com_loop_w_in_window])
    com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
    com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
    #only get perms with non zero cells
    perm=[p for ii,p in enumerate(perm) if len(com_goal[ii])>0]
    rz_perm=[p for ii,p in enumerate(rz_perm) if len(com_goal[ii])>0]
    com_goal=[com for com in com_goal if len(com)>0]

    print(f'Reward-centric cells total: {[len(xx) for xx in com_goal]}')
    epoch_perm.append([perm,rz_perm]) 
    # get goal cells across all epochs   
    if len(com_goal)>0:
        goal_cells = intersect_arrays(*com_goal); 
    else:
        goal_cells=[]
    # get goal cells across all epochs        
    # s2p_iind_goal_cells = s2p_iind_filter[goal_cells]
    # get per comparison
    goal_cells_p_per_comparison = [len(xx)/len(coms_correct[0]) for xx in com_goal]
    goal_cell_iind.append(goal_cells)
    goal_cell_p=len(goal_cells)/len(coms_correct[0])        
    goal_cell_prop.append([goal_cells_p_per_comparison,goal_cell_p])
    num_epochs.append(len(coms_correct))
    colors = ['k', 'slategray', 'darkcyan', 'darkgoldenrod', 'orchid']
    if len(goal_cells)>0:
        rows = int(np.ceil(np.sqrt(len(goal_cells))))
        cols = int(np.ceil(len(goal_cells) / rows))
        fig, axes = plt.subplots(rows, cols, figsize=(30,30),sharex=True)
        if len(goal_cells) > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        for i,gc in enumerate(goal_cells):            
            for ep in range(len(coms_correct)):
                ax = axes[i]
                ax.plot(tcs_correct[ep,gc,:], label=f'rewloc {rewlocs[ep]}', color=colors[ep])
                # if len(tcs_fail)>0:
                #         ax.plot(tcs_fail[ep,gc,:], label=f'fail rewloc {rewlocs[ep]}', color=colors[ep], linestyle = '--')
                ax.axvline((bins_dt/2), color='k')
                ax.set_title(f'cell # {gc}')
                ax.spines[['top','right']].set_visible(False)
        # ax.set_xticks(np.arange(0,bins+1,20))
        # ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi, np.pi/2.25),3))
        ax.set_xlabel('Radian position (centered start rew loc)')
        ax.set_ylabel('Fc3')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    # get shuffled iteration
    shuffled_dist = np.zeros((num_iterations))
    # max of 5 epochs = 10 perms
    goal_cell_shuf_ps_per_comp = np.ones((num_iterations,10))*np.nan
    goal_cell_shuf_ps = []
    for i in range(num_iterations):
        # shuffle locations
        rewlocs_shuf = rewlocs #[random.randint(100,250) for iii in range(len(eps))]
        shufs = [list(range(coms_correct[ii].shape[0])) for ii in range(1, len(coms_correct))]
        [random.shuffle(shuf) for shuf in shufs]
        # first com is as ep 1, others are shuffled cell identities
        com_shufs = np.zeros_like(coms_correct); com_shufs[0,:] = coms_correct[0]
        com_shufs[1:1+len(shufs),:] = [coms_correct[ii][np.array(shufs)[ii-1]] for ii in range(1, 1+len(shufs))]
        # OR shuffle cell identities
        # relative to reward
        coms_rewrel = np.array([com-np.pi for ii, com in enumerate(com_shufs)])             
        # account for cells that move to the end/front
        # Find COMs near pi and shift to -pi
        com_loop_w_in_window = []
        for pi,p in enumerate(perm):
            for cll in range(coms_rewrel.shape[1]):
                com1_rel = coms_rewrel[p[0],cll]
                com2_rel = coms_rewrel[p[1],cll]
                # print(com1_rel,com2_rel,com_diff)
                if ((abs(com1_rel - np.pi) < epsilon) and 
                (abs(com2_rel + np.pi) < epsilon)):
                        com_loop_w_in_window.append(cll)
        # get abs value instead
        # cont.
        coms_rewrel[:,com_loop_w_in_window]=abs(coms_rewrel[:,com_loop_w_in_window])
        com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
        # get goal cells across all epochs
        com_goal_shuf = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
        com_goal_shuf=[com for com in com_goal_shuf if len(com)>0]
        goal_cells_shuf_p_per_comparison = [len(xx)/len(coms_correct[0]) for xx in com_goal_shuf]
        # get goal cells across all epochs   
        if len(com_goal_shuf)>0:
            goal_cells_shuf = intersect_arrays(*com_goal_shuf); 
        else:
            goal_cells_shuf=[]
        shuffled_dist[i] = len(goal_cells_shuf)/len(coms_correct[0])
        goal_cell_shuf_p=len(goal_cells_shuf)/len(com_shufs[0])
        goal_cell_shuf_ps.append(goal_cell_shuf_p)
        goal_cell_shuf_ps_per_comp[i, :len(goal_cells_shuf_p_per_comparison)] = goal_cells_shuf_p_per_comparison
    # save median of goal cell shuffle
    goal_cell_shuf_ps_per_comp_av = np.nanmedian(goal_cell_shuf_ps_per_comp,axis=0)        
    goal_cell_shuf_ps_av = np.nanmedian(np.array(goal_cell_shuf_ps)[1])
    goal_cell_null.append([goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av])
    p_value = sum(shuffled_dist>goal_cell_p)/num_iterations
    pvals.append(p_value); 
    print(f'{animal}, day {day}: significant goal cells proportion p-value: {p_value}')
    total_cells.append(len(coms_correct[0]))
    radian_alignment[f'{animal}_{day:03d}_index{ii:03d}'] = [tcs_correct, coms_correct, tcs_fail, coms_fail,
                    com_goal, goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av]
    
    return radian_alignment,rate,p_value,total_cells,goal_cell_iind,\
        goal_cell_prop,num_epochs,goal_cell_null,epoch_perm,pvals
        

def extract_data_farrew(ii,params_pth,animal,day,bins,radian_alignment,
    radian_alignment_saved,goal_cm_window,pdf,epoch_perm,goal_cell_iind,goal_cell_prop,num_epochs,
    goal_cell_null,pvals,total_cells,
    num_iterations=1000):
    """
    changed on 2/6/25 to make it more consistent with splitting the different
    subpopulations
    """
    print(params_pth)
    fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
            'timedFF', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
            'stat', 'licks'])
    VR = fall['VR'][0][0][()]
    scalingf = VR['scalingFACTOR'][0][0]
    try:
            rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf        
    except:
            rewsize = 10
    ybinned = fall['ybinned'][0]/scalingf
    track_length=180/scalingf    
    forwardvel = fall['forwardvel'][0]    
    changeRewLoc = np.hstack(fall['changeRewLoc'])
    trialnum=fall['trialnum'][0]
    rewards = fall['rewards'][0]
    lick = fall['licks'][0]
    time = fall['timedFF'][0]
    if animal=='e145':
        ybinned=ybinned[:-1]
        forwardvel=forwardvel[:-1]
        changeRewLoc=changeRewLoc[:-1]
        trialnum=trialnum[:-1]
        rewards=rewards[:-1]
        lick=lick[:-1]
        time=time[:-1]
    # set vars
    eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
    lasttr=8 # last trials
    bins=90
    rad = get_radian_position_first_lick_after_rew(eps, ybinned, lick, rewards, rewsize,rewlocs,
                    trialnum, track_length) # get radian coordinates
    track_length_rad = track_length*(2*np.pi/track_length)
    bin_size=track_length_rad/bins 
    rz = get_rewzones(rewlocs,1/scalingf)       
    # get average success rate
    rates = []
    for ep in range(len(eps)-1):
        eprng = range(eps[ep],eps[ep+1])
        success, fail, str_trials, ftr_trials, ttr, \
        total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
        rates.append(success/total_trials)
    rate=np.nanmean(np.array(rates))
    # dark time params
    track_length_dt = 550 # cm estimate based on 99.9% of ypos
    track_length_rad_dt = track_length_dt*(2*np.pi/track_length_dt) # estimate bin for dark time
    bins_dt=150 
    bin_size_dt=track_length_rad_dt/bins_dt # typically 3 cm binswith ~ 475 track length
    # added to get anatomical info
    # takes time
    fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
    Fc3 = fall_fc3['Fc3']
    dFF = fall_fc3['dFF']
    Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
    dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]
    skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
    Fc3 = Fc3[:, skew>2] # only keep cells with skew greateer than 2
    if f'{animal}_{day:03d}_index{ii:03d}' in radian_alignment_saved.keys():
        tcs_correct, coms_correct, tcs_fail, coms_fail, \
        com_goal, goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av = radian_alignment_saved[f'{animal}_{day:03d}_index{ii:03d}']            
    else:# remake tuning curves relative to reward        
        # 9/19/24
        # tc w/ dark time added to the end of track
        tcs_correct, coms_correct, tcs_fail, coms_fail, rewloc_dt, ybinned_dt = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,
            rewsize,ybinned,time,lick,
            Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
            bins=bins_dt)  
    goal_window = goal_cm_window*(2*np.pi/track_length) # cm converted to rad
    # change to relative value 
    coms_rewrel = np.array([com-np.pi for com in coms_correct])
    perm = list(combinations(range(len(coms_correct)), 2)) 
    rz_perm = [(int(rz[p[0]]),int(rz[p[1]])) for p in perm]   
    # if 4 ep
    # account for cells that move to the end/front
    # Define a small window around pi (e.g., epsilon)
    epsilon = .7 # 20 cm
    # Find COMs near pi and shift to -pi
    com_loop_w_in_window = []
    for pi,p in enumerate(perm):
        for cll in range(coms_rewrel.shape[1]):
            com1_rel = coms_rewrel[p[0],cll]
            com2_rel = coms_rewrel[p[1],cll]
            # print(com1_rel,com2_rel,com_diff)
            if ((abs(com1_rel - np.pi) < epsilon) and 
            (abs(com2_rel + np.pi) < epsilon)):
                    com_loop_w_in_window.append(cll)
    # get abs value instead
    coms_rewrel[:,com_loop_w_in_window]=abs(coms_rewrel[:,com_loop_w_in_window])
    com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
    com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
    print(f'Reward-centric cells total: {[len(xx) for xx in com_goal]}\n')
    # com away from rew
    #only get perms with non zero cells
    # changed 4/21/25 with new far rew analysis?
    lowerbound = np.pi/3
    com_goal_farrew = [[xx for xx in com if (abs(np.nanmedian(coms_rewrel[:,
        xx], axis=0)>=lowerbound))] if len(com)>0 else [] for com in com_goal]
    perm=[p for ii,p in enumerate(perm) if len(com_goal_farrew[ii])>0]
    rz_perm=[p for ii,p in enumerate(rz_perm) if len(com_goal_farrew[ii])>0]
    com_goal_farrew=[com for com in com_goal_farrew if len(com)>0]
    print(f'Far-reward cells total: {[len(xx) for xx in com_goal_farrew]}')
    epoch_perm.append([perm,rz_perm]) 
    # get goal cells across all epochs        
    if len(com_goal_farrew)>0:
        goal_cells = intersect_arrays(*com_goal_farrew); 
    else:
        goal_cells=[]
    # s2p_iind_goal_cells = s2p_iind_filter[goal_cells]
    # get per comparison
    goal_cells_p_per_comparison = [len(xx)/len(coms_correct[0]) for xx in com_goal_farrew]
    goal_cell_iind.append(goal_cells)
    goal_cell_p=len(goal_cells)/len(coms_correct[0])        
    goal_cell_prop.append([goal_cells_p_per_comparison,goal_cell_p])
    num_epochs.append(len(coms_correct))
    colors = ['k', 'slategray', 'darkcyan', 'darkgoldenrod', 'orchid']
    if len(goal_cells)>0:
        rows = int(np.ceil(np.sqrt(len(goal_cells))))
        cols = int(np.ceil(len(goal_cells) / rows))
        fig, axes = plt.subplots(rows, cols, figsize=(30,30),sharex=True)
        if len(goal_cells) > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        for i,gc in enumerate(goal_cells):            
            for ep in range(len(coms_correct)):
                ax = axes[i]
                ax.plot(tcs_correct[ep,gc,:], label=f'rewloc {rewlocs[ep]}', color=colors[ep])
                if len(tcs_fail)>0:
                        ax.plot(tcs_fail[ep,gc,:], label=f'fail rewloc {rewlocs[ep]}', 
                            color=colors[ep], linestyle = '--')
                ax.axvline((bins/2), color='k')
                ax.set_title(f'cell # {gc}')
                ax.spines[['top','right']].set_visible(False)
        ax.set_xticks(np.arange(0,bins+1,20))
        ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi, np.pi/2.25),3))
        ax.set_xlabel('Radian position (centered start rew loc)')
        ax.set_ylabel('Fc3')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    # get shuffled iteration
    shuffled_dist = np.zeros((num_iterations))
    # max of 5 epochs = 10 perms
    goal_cell_shuf_ps_per_comp = np.ones((num_iterations,10))*np.nan
    goal_cell_shuf_ps = []
    for i in range(num_iterations):
        # shuffle locations
        rewlocs_shuf = rewlocs #[random.randint(100,250) for iii in range(len(eps))]
        shufs = [list(range(coms_correct[ii].shape[0])) for ii in range(1, len(coms_correct))]
        [random.shuffle(shuf) for shuf in shufs]
        # first com is as ep 1, others are shuffled cell identities
        com_shufs = np.zeros_like(coms_correct); com_shufs[0,:] = coms_correct[0]
        com_shufs[1:1+len(shufs),:] = [coms_correct[ii][np.array(shufs)[ii-1]] for ii in range(1, 1+len(shufs))]
        # OR shuffle cell identities
        # relative to reward
        coms_rewrel = np.array([com-np.pi for ii, com in enumerate(com_shufs)])             
        # perm = list(combinations(range(len(coms_correct)), 2)) 
        # account for cells that move to the end/front
        # Find COMs near pi and shift to -pi
        com_loop_w_in_window = []
        for pi,p in enumerate(perm):
            for cll in range(coms_rewrel.shape[1]):
                com1_rel = coms_rewrel[p[0],cll]
                com2_rel = coms_rewrel[p[1],cll]
                # print(com1_rel,com2_rel,com_diff)
                if ((abs(com1_rel - np.pi) < epsilon) and 
                (abs(com2_rel + np.pi) < epsilon)):
                        com_loop_w_in_window.append(cll)
        # get abs value instead
        # cont.
        coms_rewrel[:,com_loop_w_in_window]=abs(coms_rewrel[:,com_loop_w_in_window])
        com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
        # get goal cells across all epochs
        com_goal_shuf = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
        # in addition, com near but after goal
        com_goal_farrew_shuf = [[xx for xx in com if (abs(np.nanmedian(coms_rewrel[:,
            xx], axis=0)>=lowerbound))] if len(com)>0 else [] for com in com_goal_shuf ]
        #only get perms with non zero cells
        com_goal_farrew_shuf=[com for com in com_goal_farrew_shuf if len(com)>0]
        goal_cells_shuf_p_per_comparison = [len(xx)/len(coms_correct[0]) for xx in com_goal_farrew_shuf]
        if len(com_goal_farrew_shuf)>0:
            goal_cells_shuf = intersect_arrays(*com_goal_farrew_shuf); 
        else:
            goal_cells_shuf=[]
        shuffled_dist[i] = len(goal_cells_shuf)/len(coms_correct[0])
        goal_cell_shuf_p=len(goal_cells_shuf)/len(coms_correct[0])
        goal_cell_shuf_ps.append(goal_cell_shuf_p)
        goal_cell_shuf_ps_per_comp[i, :len(goal_cells_shuf_p_per_comparison)] = goal_cells_shuf_p_per_comparison
    # save median of goal cell shuffle
    goal_cell_shuf_ps_per_comp_av = np.nanmedian(goal_cell_shuf_ps_per_comp,axis=0)        
    goal_cell_shuf_ps_av = np.nanmedian(np.array(goal_cell_shuf_ps)[1])
    goal_cell_null.append([goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av])
    p_value = sum(shuffled_dist>goal_cell_p)/num_iterations
    pvals.append(p_value); 
    print(f'{animal}, day {day}: significant goal cells proportion p-value: {p_value}')
    total_cells.append(len(coms_correct[0]))
    radian_alignment[f'{animal}_{day:03d}_index{ii:03d}'] = [tcs_correct, coms_correct, tcs_fail, coms_fail,
            com_goal, goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av]
    
    return radian_alignment,rate,p_value,total_cells,goal_cell_iind,\
        goal_cell_prop,num_epochs,goal_cell_null,epoch_perm,pvals

def extract_data_post_farrew(ii,params_pth,animal,day,bins,radian_alignment,
    radian_alignment_saved,goal_cm_window,pdf,epoch_perm,goal_cell_iind,goal_cell_prop,num_epochs,
    goal_cell_null,pvals,total_cells,
    num_iterations=1000):
    """
    changed on 2/6/25 to make it more consistent with splitting the different
    subpopulations
    """
    print(params_pth)
    fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
            'timedFF', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
            'stat', 'licks'])
    VR = fall['VR'][0][0][()]
    scalingf = VR['scalingFACTOR'][0][0]
    try:
            rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf        
    except:
            rewsize = 10
    ybinned = fall['ybinned'][0]/scalingf
    track_length=180/scalingf    
    forwardvel = fall['forwardvel'][0]    
    changeRewLoc = np.hstack(fall['changeRewLoc'])
    trialnum=fall['trialnum'][0]
    rewards = fall['rewards'][0]
    lick = fall['licks'][0]
    time = fall['timedFF'][0]
    if animal=='e145':
        ybinned=ybinned[:-1]
        forwardvel=forwardvel[:-1]
        changeRewLoc=changeRewLoc[:-1]
        trialnum=trialnum[:-1]
        rewards=rewards[:-1]
        lick=lick[:-1]
        time=time[:-1]
    # set vars
    eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
    lasttr=8 # last trials
    bins=90
    rad = get_radian_position_first_lick_after_rew(eps, ybinned, lick, rewards, rewsize,rewlocs,
                    trialnum, track_length) # get radian coordinates
    track_length_rad = track_length*(2*np.pi/track_length)
    bin_size=track_length_rad/bins 
    rz = get_rewzones(rewlocs,1/scalingf)       
    # get average success rate
    rates = []
    for ep in range(len(eps)-1):
        eprng = range(eps[ep],eps[ep+1])
        success, fail, str_trials, ftr_trials, ttr, \
        total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
        rates.append(success/total_trials)
    rate=np.nanmean(np.array(rates))
    # dark time params
    track_length_dt = 550 # cm estimate based on 99.9% of ypos
    track_length_rad_dt = track_length_dt*(2*np.pi/track_length_dt) # estimate bin for dark time
    bins_dt=150 
    bin_size_dt=track_length_rad_dt/bins_dt # typically 3 cm binswith ~ 475 track length
    # added to get anatomical info
    # takes time
    fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
    Fc3 = fall_fc3['Fc3']
    dFF = fall_fc3['dFF']
    Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
    dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]
    skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
    Fc3 = Fc3[:, skew>2] # only keep cells with skew greateer than 2
    if f'{animal}_{day:03d}_index{ii:03d}' in radian_alignment_saved.keys():
        tcs_correct, coms_correct, tcs_fail, coms_fail, \
        com_goal, goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av = radian_alignment_saved[f'{animal}_{day:03d}_index{ii:03d}']            
    else:# remake tuning curves relative to reward        
        # 9/19/24
        # tc w/ dark time added to the end of track
        tcs_correct, coms_correct, tcs_fail, coms_fail, ybinned_dt, raddt = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,
            rewsize,ybinned,time,lick,
            Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
            bins=bins_dt)  

    goal_window = goal_cm_window*(2*np.pi/track_length) # cm converted to rad
    # change to relative value 
    coms_rewrel = np.array([com-np.pi for com in coms_correct])
    perm = list(combinations(range(len(coms_correct)), 2)) 
    rz_perm = [(int(rz[p[0]]),int(rz[p[1]])) for p in perm]   
    # if 4 ep
    # account for cells that move to the end/front
    # Define a small window around pi (e.g., epsilon)
    epsilon = .7 # 20 cm
    # Find COMs near pi and shift to -pi
    com_loop_w_in_window = []
    for pi,p in enumerate(perm):
        for cll in range(coms_rewrel.shape[1]):
            com1_rel = coms_rewrel[p[0],cll]
            com2_rel = coms_rewrel[p[1],cll]
            # print(com1_rel,com2_rel,com_diff)
            if ((abs(com1_rel - np.pi) < epsilon) and 
            (abs(com2_rel + np.pi) < epsilon)):
                    com_loop_w_in_window.append(cll)
    # get abs value instead
    coms_rewrel[:,com_loop_w_in_window]=abs(coms_rewrel[:,com_loop_w_in_window])
    com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
    com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
    print(f'Reward-centric cells total: {[len(xx) for xx in com_goal]}\n')
    # com away from rew
    #only get perms with non zero cells
    # changed 4/21/25 with new far rew analysis?
    bound = np.pi/4
    com_goal_farrew = [[xx for xx in com if ((np.nanmedian(coms_rewrel[:,
        xx], axis=0)>=bound))] if len(com)>0 else [] for com in com_goal]
    perm=[p for ii,p in enumerate(perm) if len(com_goal_farrew[ii])>0]
    rz_perm=[p for ii,p in enumerate(rz_perm) if len(com_goal_farrew[ii])>0]
    com_goal_farrew=[com for com in com_goal_farrew if len(com)>0]
    print(f'Far-reward cells total: {[len(xx) for xx in com_goal_farrew]}')
    epoch_perm.append([perm,rz_perm]) 
    # get goal cells across all epochs        
    if len(com_goal_farrew)>0:
        goal_cells = intersect_arrays(*com_goal_farrew); 
    else:
        goal_cells=[]
    # s2p_iind_goal_cells = s2p_iind_filter[goal_cells]
    # get per comparison
    goal_cells_p_per_comparison = [len(xx)/len(coms_correct[0]) for xx in com_goal_farrew]
    goal_cell_iind.append(goal_cells)
    goal_cell_p=len(goal_cells)/len(coms_correct[0])        
    goal_cell_prop.append([goal_cells_p_per_comparison,goal_cell_p])
    num_epochs.append(len(coms_correct))
    colors = ['k', 'slategray', 'darkcyan', 'darkgoldenrod', 'orchid']
    if len(goal_cells)>0:
        rows = int(np.ceil(np.sqrt(len(goal_cells))))
        cols = int(np.ceil(len(goal_cells) / rows))
        fig, axes = plt.subplots(rows, cols, figsize=(30,30),sharex=True)
        if len(goal_cells) > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        for i,gc in enumerate(goal_cells):            
            for ep in range(len(coms_correct)):
                ax = axes[i]
                ax.plot(tcs_correct[ep,gc,:], label=f'rewloc {rewlocs[ep]}', color=colors[ep])
                if len(tcs_fail)>0:
                    ax.plot(tcs_fail[ep,gc,:], label=f'fail rewloc {rewlocs[ep]}', 
                        color=colors[ep], linestyle = '--')
                ax.axvline((bins/2), color='k')
                ax.set_title(f'cell # {gc}')
                ax.spines[['top','right']].set_visible(False)
        ax.set_xticks(np.arange(0,bins+1,20))
        ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi, np.pi/2.25),3))
        ax.set_xlabel('Radian position (centered start rew loc)')
        ax.set_ylabel('Fc3')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    # get shuffled iteration
    shuffled_dist = np.zeros((num_iterations))
    # max of 5 epochs = 10 perms
    goal_cell_shuf_ps_per_comp = np.ones((num_iterations,10))*np.nan
    goal_cell_shuf_ps = []
    for i in range(num_iterations):
        # shuffle locations
        rewlocs_shuf = rewlocs #[random.randint(100,250) for iii in range(len(eps))]
        shufs = [list(range(coms_correct[ii].shape[0])) for ii in range(1, len(coms_correct))]
        [random.shuffle(shuf) for shuf in shufs]
        # first com is as ep 1, others are shuffled cell identities
        com_shufs = np.zeros_like(coms_correct); com_shufs[0,:] = coms_correct[0]
        com_shufs[1:1+len(shufs),:] = [coms_correct[ii][np.array(shufs)[ii-1]] for ii in range(1, 1+len(shufs))]
        # OR shuffle cell identities
        # relative to reward
        coms_rewrel = np.array([com-np.pi for ii, com in enumerate(com_shufs)])             
        # perm = list(combinations(range(len(coms_correct)), 2)) 
        # account for cells that move to the end/front
        # Find COMs near pi and shift to -pi
        com_loop_w_in_window = []
        for pi,p in enumerate(perm):
            for cll in range(coms_rewrel.shape[1]):
                com1_rel = coms_rewrel[p[0],cll]
                com2_rel = coms_rewrel[p[1],cll]
                # print(com1_rel,com2_rel,com_diff)
                if ((abs(com1_rel - np.pi) < epsilon) and 
                (abs(com2_rel + np.pi) < epsilon)):
                        com_loop_w_in_window.append(cll)
        # get abs value instead
        # cont.
        coms_rewrel[:,com_loop_w_in_window]=abs(coms_rewrel[:,com_loop_w_in_window])
        com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
        # get goal cells across all epochs
        com_goal_shuf = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
        # in addition, com near but after goal
        com_goal_farrew_shuf = [[xx for xx in com if (np.nanmedian(coms_rewrel[:,
            xx], axis=0)>=bound)] if len(com)>0 else [] for com in com_goal_shuf ]
        #only get perms with non zero cells
        com_goal_farrew_shuf=[com for com in com_goal_farrew_shuf if len(com)>0]
        goal_cells_shuf_p_per_comparison = [len(xx)/len(coms_correct[0]) for xx in com_goal_farrew_shuf]
        if len(com_goal_farrew_shuf)>0:
            goal_cells_shuf = intersect_arrays(*com_goal_farrew_shuf); 
        else:
            goal_cells_shuf=[]
        shuffled_dist[i] = len(goal_cells_shuf)/len(coms_correct[0])
        goal_cell_shuf_p=len(goal_cells_shuf)/len(coms_correct[0])
        goal_cell_shuf_ps.append(goal_cell_shuf_p)
        goal_cell_shuf_ps_per_comp[i, :len(goal_cells_shuf_p_per_comparison)] = goal_cells_shuf_p_per_comparison
    # save median of goal cell shuffle
    goal_cell_shuf_ps_per_comp_av = np.nanmedian(goal_cell_shuf_ps_per_comp,axis=0)        
    goal_cell_shuf_ps_av = np.nanmedian(np.array(goal_cell_shuf_ps)[1])
    goal_cell_null.append([goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av])
    p_value = sum(shuffled_dist>goal_cell_p)/num_iterations
    pvals.append(p_value); 
    print(f'{animal}, day {day}: significant goal cells proportion p-value: {p_value}')
    total_cells.append(len(coms_correct[0]))
    radian_alignment[f'{animal}_{day:03d}_index{ii:03d}'] = [tcs_correct, coms_correct, tcs_fail, coms_fail,
            com_goal, goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av]
    
    return radian_alignment,rate,p_value,total_cells,goal_cell_iind,\
        goal_cell_prop,num_epochs,goal_cell_null,epoch_perm,pvals


def extract_data_nearrew(ii,params_pth,animal,day,bins,radian_alignment,
    radian_alignment_saved,goal_cm_window,pdf,epoch_perm,goal_cell_iind,goal_cell_prop,num_epochs,
    goal_cell_null,pvals,total_cells,
    num_iterations=1000):
    """
    changed on 2/6/25 to make it more consistent with splitting the different
    subpopulations
    """
    print(params_pth)
    fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
        'timedFF', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
        'stat', 'licks'])
    VR = fall['VR'][0][0][()]
    scalingf = VR['scalingFACTOR'][0][0]
    try:
            rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf        
    except:
            rewsize = 10
    ybinned = fall['ybinned'][0]/scalingf
    track_length=180/scalingf    
    forwardvel = fall['forwardvel'][0]    
    changeRewLoc = np.hstack(fall['changeRewLoc'])
    trialnum=fall['trialnum'][0]
    rewards = fall['rewards'][0]
    lick = fall['licks'][0]
    time = fall['timedFF'][0]
    if animal=='e145':
        ybinned=ybinned[:-1]
        forwardvel=forwardvel[:-1]
        changeRewLoc=changeRewLoc[:-1]
        trialnum=trialnum[:-1]
        rewards=rewards[:-1]
        lick=lick[:-1]
        time=time[:-1]
    # set vars
    eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
    lasttr=8 # last trials
    bins=90
    rad = get_radian_position_first_lick_after_rew(eps, ybinned, lick, rewards, rewsize,rewlocs,
                    trialnum, track_length) # get radian coordinates
    track_length_rad = track_length*(2*np.pi/track_length)
    bin_size=track_length_rad/bins 
    rz = get_rewzones(rewlocs,1/scalingf)       
    # get average success rate
    rates = []
    for ep in range(len(eps)-1):
        eprng = range(eps[ep],eps[ep+1])
        success, fail, str_trials, ftr_trials, ttr, \
        total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
        rates.append(success/total_trials)
    rate=np.nanmean(np.array(rates))
    # dark time params
    track_length_dt = 550 # cm estimate based on 99.9% of ypos
    track_length_rad_dt = track_length_dt*(2*np.pi/track_length_dt) # estimate bin for dark time
    bins_dt=150 
    bin_size_dt=track_length_rad_dt/bins_dt # typically 3 cm binswith ~ 475 track length
    # added to get anatomical info
    # takes time
    fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
    Fc3 = fall_fc3['Fc3']
    dFF = fall_fc3['dFF']
    Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
    dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]
    skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
    Fc3 = Fc3[:, skew>2] # only keep cells with skew greateer than 2
    if f'{animal}_{day:03d}_index{ii:03d}' in radian_alignment_saved.keys():
        tcs_correct, coms_correct, tcs_fail, coms_fail, \
        com_goal, goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av = radian_alignment_saved[f'{animal}_{day:03d}_index{ii:03d}']            
    else:# remake tuning curves relative to reward        
        # 9/19/24
        # tc w/ dark time added to the end of track
        tcs_correct, coms_correct, tcs_fail, coms_fail, ybinned_dt, raddt = make_tuning_curves_by_trialtype_w_darktime(eps,
            rewlocs,rewsize,ybinned,time,lick,
            Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
            bins=bins_dt)
        # tcs_correct, coms_correct, tcs_fail, coms_fail = make_tuning_curves_radians_by_trialtype(
        # eps, rewlocs, ybinned, rad, Fc3, trialnum, rewards, forwardvel, rewsize, bin_size)   
        # tcs_correct_dt, coms_correct_dt, tcs_fail_dt, coms_fail_dt, \
        # rewloc_dt, ybinned_dt = make_tuning_curves_by_trialtype_w_darktime(eps,
        #     rewlocs,rewsize,ybinned,time,lick,
        #     Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
        #     bins=bins_dt)
    goal_window = goal_cm_window*(2*np.pi/track_length) # cm converted to rad
    # change to relative value 
    coms_rewrel = np.array([com-np.pi for com in coms_correct])
    perm = list(combinations(range(len(coms_correct)), 2)) 
    rz_perm = [(int(rz[p[0]]),int(rz[p[1]])) for p in perm]   
    # if 4 ep
    # account for cells that move to the end/front
    # Define a small window around pi (e.g., epsilon)
    epsilon = .7 # 20 cm
    # Find COMs near pi and shift to -pi
    com_loop_w_in_window = []
    for pi,p in enumerate(perm):
        for cll in range(coms_rewrel.shape[1]):
            com1_rel = coms_rewrel[p[0],cll]
            com2_rel = coms_rewrel[p[1],cll]
            # print(com1_rel,com2_rel,com_diff)
            if ((abs(com1_rel - np.pi) < epsilon) and 
            (abs(com2_rel + np.pi) < epsilon)):
                    com_loop_w_in_window.append(cll)
    # get abs value instead
    coms_rewrel[:,com_loop_w_in_window]=abs(coms_rewrel[:,com_loop_w_in_window])
    com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
    com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
    # in addition, com near but after goal
    upperbound = np.pi/4
    com_goal_postrew = [[xx for xx in com if ((np.nanmedian(coms_rewrel[:,
        xx], axis=0)<=upperbound) & (np.nanmedian(coms_rewrel[:,
        xx], axis=0)>0))] if len(com)>0 else [] for com in com_goal]
    #only get perms with non zero cells
    perm=[p for ii,p in enumerate(perm) if len(com_goal_postrew[ii])>0]
    rz_perm=[p for ii,p in enumerate(rz_perm) if len(com_goal_postrew[ii])>0]
    com_goal_postrew=[com for com in com_goal_postrew if len(com)>0]

    print(f'Reward-centric cells total: {[len(xx) for xx in com_goal]}\n\
    Post-reward cells: {[len(xx) for xx in com_goal_postrew]}')
    assert sum([len(xx) for xx in com_goal])>=sum([len(xx) for xx in com_goal_postrew])
    epoch_perm.append([perm,rz_perm]) 
    # get goal cells across all epochs   
    if len(com_goal_postrew)>0:
        goal_cells = intersect_arrays(*com_goal_postrew); 
    else:
        goal_cells=[]
    # s2p_iind_goal_cells = s2p_iind_filter[goal_cells]
    # get per comparison
    goal_cells_p_per_comparison = [len(xx)/len(coms_correct[0]) for xx in com_goal_postrew]
    goal_cell_iind.append(goal_cells)
    goal_cell_p=len(goal_cells)/len(coms_correct[0])        
    goal_cell_prop.append([goal_cells_p_per_comparison,goal_cell_p])
    num_epochs.append(len(coms_correct))
    colors = ['k', 'slategray', 'darkcyan', 'darkgoldenrod', 'orchid']
    if len(goal_cells)>0:
        rows = int(np.ceil(np.sqrt(len(goal_cells))))
        cols = int(np.ceil(len(goal_cells) / rows))
        fig, axes = plt.subplots(rows, cols, figsize=(30,30),sharex=True)
        if len(goal_cells) > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        for i,gc in enumerate(goal_cells):            
            for ep in range(len(coms_correct)):
                ax = axes[i]
                ax.plot(tcs_correct[ep,gc,:], label=f'rewloc {rewlocs[ep]}', color=colors[ep])
                if len(tcs_fail)>0:
                    ax.plot(tcs_fail[ep,gc,:], label=f'fail rewloc {rewlocs[ep]}', color=colors[ep], linestyle = '--')
                ax.axvline((bins_dt/2), color='k')
                ax.set_title(f'cell # {gc}')
                ax.spines[['top','right']].set_visible(False)
        # ax.set_xticks(np.arange(0,bins+1,20))
        # ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi, np.pi/2.25),3))
        ax.set_xlabel('Radian position (centered start rew loc)')
        ax.set_ylabel('Fc3')
        fig.tight_layout()
        pdf.savefig(fig)
        # plt.show()
        plt.close(fig)
    # get shuffled iteration
    shuffled_dist = np.zeros((num_iterations))
    # max of 5 epochs = 10 perms
    goal_cell_shuf_ps_per_comp = np.ones((num_iterations,10))*np.nan
    goal_cell_shuf_ps = []
    for i in range(num_iterations):
        shufs = [list(range(coms_correct[ii].shape[0])) for ii in range(1, len(coms_correct))]
        [random.shuffle(shuf) for shuf in shufs]
        # first com is as ep 1, others are shuffled cell identities
        com_shufs = np.zeros_like(coms_correct); com_shufs[0,:] = coms_correct[0]
        com_shufs[1:1+len(shufs),:] = [coms_correct[ii][np.array(shufs)[ii-1]] for ii in range(1, 1+len(shufs))]
        # OR shuffle cell identities
        # relative to reward
        coms_rewrel = np.array([com-np.pi for ii, com in enumerate(com_shufs)])             
        perm = list(combinations(range(len(coms_correct)), 2)) 
        # account for cells that move to the end/front
        # Find COMs near pi and shift to -pi
        com_loop_w_in_window = []
        for pi,p in enumerate(perm):
            for cll in range(coms_rewrel.shape[1]):
                com1_rel = coms_rewrel[p[0],cll]
                com2_rel = coms_rewrel[p[1],cll]
                # print(com1_rel,com2_rel,com_diff)
                if ((abs(com1_rel - np.pi) < epsilon) and 
                (abs(com2_rel + np.pi) < epsilon)):
                        com_loop_w_in_window.append(cll)
        # get abs value instead
        # cont.
        coms_rewrel[:,com_loop_w_in_window]=abs(coms_rewrel[:,com_loop_w_in_window])
        com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
        # get goal cells across all epochs
        com_goal_shuf = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
        # in addition, com near but after goal
        com_goal_postrew_shuf = [[xx for xx in com if ((np.nanmedian(coms_rewrel[:,
            xx], axis=0)<=upperbound) & (np.nanmedian(coms_rewrel[:,
            xx], axis=0)>0))] if len(com)>0 else [] for com in com_goal_shuf]
        # check to make sure just a subset
        # otherwise reshuffle
        while not sum([len(xx) for xx in com_goal_shuf])>=sum([len(xx) for xx in com_goal_postrew_shuf]):
            print('redo')
            shufs = [list(range(coms_correct[ii].shape[0])) for ii in range(1, len(coms_correct))]
            [random.shuffle(shuf) for shuf in shufs]
            # first com is as ep 1, others are shuffled cell identities
            com_shufs = np.zeros_like(coms_correct); com_shufs[0,:] = coms_correct[0]
            com_shufs[1:1+len(shufs),:] = [coms_correct[ii][np.array(shufs)[ii-1]] for ii in range(1, 1+len(shufs))]
            # OR shuffle cell identities
            # relative to reward
            coms_rewrel = np.array([com-np.pi for ii, com in enumerate(com_shufs)])             
            perm = list(combinations(range(len(coms_correct)), 2)) 
            # account for cells that move to the end/front
            # Find COMs near pi and shift to -pi
            com_loop_w_in_window = []
            for pi,p in enumerate(perm):
                for cll in range(coms_rewrel.shape[1]):
                    com1_rel = coms_rewrel[p[0],cll]
                    com2_rel = coms_rewrel[p[1],cll]
                    # print(com1_rel,com2_rel,com_diff)
                    if ((abs(com1_rel - np.pi) < epsilon) and 
                    (abs(com2_rel + np.pi) < epsilon)):
                            com_loop_w_in_window.append(cll)
            # get abs value instead
            # cont.
            coms_rewrel[:,com_loop_w_in_window]=abs(coms_rewrel[:,com_loop_w_in_window])
            com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
            # get goal cells across all epochs
            com_goal_shuf = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
            # in addition, com near but after goal
            com_goal_postrew_shuf = [[xx for xx in com if ((np.nanmedian(coms_rewrel[:,
                xx], axis=0)<=np.pi/2) & (np.nanmedian(coms_rewrel[:,
                xx], axis=0)>0))] if len(com)>0 else [] for com in com_goal_shuf]

        # calc acc corr
        # raccs_all = []
        # for com in com_goal_postrew_shuf:
        #     raccs = []
        #     if len(com)>0:
        #         for cll in com:
        #             nans,x = nan_helper(dFF[1:,cll])
        #             try: # if dff is nans
        #                 dFF[1:,cll][nans] = np.interp(x(nans), x(~nans), dFF[1:,cll][~nans])
        #                 _,racc = scipy.stats.pearsonr(acc, dFF[1:,cll])
        #             except Exception as e:
        #                 racc = np.nan
        #             raccs.append(racc)
        #     raccs_all.append(raccs)
        # com_goal_postrew_shuf = [[xx for jj,xx in enumerate(com) if raccs_all[ii][jj]>thres] 
        #     if len(com)>0 else [] for ii,com in enumerate(com_goal_postrew_shuf)]
        #only get perms with non zero cells
        perm=[p for ii,p in enumerate(perm) if len(com_goal_postrew_shuf[ii])>0]
        com_goal_postrew_shuf=[com for com in com_goal_postrew_shuf if len(com)>0]

        goal_cells_shuf_p_per_comparison = [len(xx)/len(coms_correct[0]) for xx in com_goal_postrew_shuf]
        if len(com_goal_postrew_shuf)>0:
            goal_cells_shuf = intersect_arrays(*com_goal_postrew_shuf); 
        else:
            goal_cells_shuf=[]
        shuffled_dist[i] = len(goal_cells_shuf)/len(coms_correct[0])
        goal_cell_shuf_p=len(goal_cells_shuf)/len(com_shufs[0])
        goal_cell_shuf_ps.append(goal_cell_shuf_p)
        goal_cell_shuf_ps_per_comp[i, :len(goal_cells_shuf_p_per_comparison)] = goal_cells_shuf_p_per_comparison
    # save median of goal cell shuffle
    goal_cell_shuf_ps_per_comp_av = np.nanmedian(goal_cell_shuf_ps_per_comp,axis=0)        
    goal_cell_shuf_ps_av = np.nanmedian(np.array(goal_cell_shuf_ps)[1])
    goal_cell_null.append([goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av])
    p_value = sum(shuffled_dist>goal_cell_p)/num_iterations
    pvals.append(p_value); 
    print(f'{animal}, day {day}: significant goal cells proportion p-value: {p_value}')
    total_cells.append(len(coms_correct[0]))
    radian_alignment[f'{animal}_{day:03d}_index{ii:03d}'] = [tcs_correct, coms_correct, tcs_fail, coms_fail,
                    com_goal, goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av]
    
    return radian_alignment,rate,p_value,total_cells,goal_cell_iind,\
        goal_cell_prop,num_epochs,goal_cell_null,epoch_perm,pvals

def extract_data_pre_farrew(ii,params_pth,animal,day,bins,radian_alignment,
    radian_alignment_saved,goal_cm_window,pdf,epoch_perm,goal_cell_iind,goal_cell_prop,num_epochs,
    goal_cell_null,pvals,total_cells,
    num_iterations=1000):
    """
    changed on 2/6/25 to make it more consistent with splitting the different
    subpopulations
    """
    print(params_pth)
    fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
            'timedFF', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
            'stat', 'licks'])
    VR = fall['VR'][0][0][()]
    scalingf = VR['scalingFACTOR'][0][0]
    try:
            rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf        
    except:
            rewsize = 10
    ybinned = fall['ybinned'][0]/scalingf
    track_length=180/scalingf    
    forwardvel = fall['forwardvel'][0]    
    changeRewLoc = np.hstack(fall['changeRewLoc'])
    trialnum=fall['trialnum'][0]
    rewards = fall['rewards'][0]
    lick = fall['licks'][0]
    time = fall['timedFF'][0]
    if animal=='e145':
        ybinned=ybinned[:-1]
        forwardvel=forwardvel[:-1]
        changeRewLoc=changeRewLoc[:-1]
        trialnum=trialnum[:-1]
        rewards=rewards[:-1]
        lick=lick[:-1]
        time=time[:-1]
    # set vars
    eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
    lasttr=8 # last trials
    bins=90
    rad = get_radian_position_first_lick_after_rew(eps, ybinned, lick, rewards, rewsize,rewlocs,
                    trialnum, track_length) # get radian coordinates
    track_length_rad = track_length*(2*np.pi/track_length)
    bin_size=track_length_rad/bins 
    rz = get_rewzones(rewlocs,1/scalingf)       
    # get average success rate
    rates = []
    for ep in range(len(eps)-1):
        eprng = range(eps[ep],eps[ep+1])
        success, fail, str_trials, ftr_trials, ttr, \
        total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
        rates.append(success/total_trials)
    rate=np.nanmean(np.array(rates))
    # dark time params
    track_length_dt = 550 # cm estimate based on 99.9% of ypos
    track_length_rad_dt = track_length_dt*(2*np.pi/track_length_dt) # estimate bin for dark time
    bins_dt=150 
    bin_size_dt=track_length_rad_dt/bins_dt # typically 3 cm binswith ~ 475 track length
    # added to get anatomical info
    # takes time
    fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
    Fc3 = fall_fc3['Fc3']
    dFF = fall_fc3['dFF']
    Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
    dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]
    skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
    Fc3 = Fc3[:, skew>2] # only keep cells with skew greateer than 2
    if f'{animal}_{day:03d}_index{ii:03d}' in radian_alignment_saved.keys():
        tcs_correct, coms_correct, tcs_fail, coms_fail, \
        com_goal, goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av = radian_alignment_saved[f'{animal}_{day:03d}_index{ii:03d}']            
    else:# remake tuning curves relative to reward        
        # 9/19/24
        # tc w/ dark time added to the end of track
        tcs_correct, coms_correct, tcs_fail, coms_fail, ybinned_dt,raddt = make_tuning_curves_by_trialtype_w_darktime(eps,
            rewlocs,rewsize,ybinned,time,lick,
            Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
            bins=bins_dt)  
    goal_window = goal_cm_window*(2*np.pi/track_length) # cm converted to rad
    # change to relative value 
    coms_rewrel = np.array([com-np.pi for com in coms_correct])
    perm = list(combinations(range(len(coms_correct)), 2)) 
    rz_perm = [(int(rz[p[0]]),int(rz[p[1]])) for p in perm]   
    # if 4 ep
    # account for cells that move to the end/front
    # Define a small window around pi (e.g., epsilon)
    epsilon = .7 # 20 cm
    # Find COMs near pi and shift to -pi
    com_loop_w_in_window = []
    for pi,p in enumerate(perm):
        for cll in range(coms_rewrel.shape[1]):
            com1_rel = coms_rewrel[p[0],cll]
            com2_rel = coms_rewrel[p[1],cll]
            # print(com1_rel,com2_rel,com_diff)
            if ((abs(com1_rel - np.pi) < epsilon) and 
            (abs(com2_rel + np.pi) < epsilon)):
                    com_loop_w_in_window.append(cll)
    # get abs value instead
    coms_rewrel[:,com_loop_w_in_window]=abs(coms_rewrel[:,com_loop_w_in_window])
    com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
    com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
    print(f'Reward-centric cells total: {[len(xx) for xx in com_goal]}\n')
    # com away from rew
    #only get perms with non zero cells
    # changed 4/21/25 with new far rew analysis?
    bound = -np.pi/4
    com_goal_farrew = [[xx for xx in com if ((np.nanmedian(coms_rewrel[:,
        xx], axis=0)<=bound))] if len(com)>0 else [] for com in com_goal]
    perm=[p for ii,p in enumerate(perm) if len(com_goal_farrew[ii])>0]
    rz_perm=[p for ii,p in enumerate(rz_perm) if len(com_goal_farrew[ii])>0]
    com_goal_farrew=[com for com in com_goal_farrew if len(com)>0]
    print(f'Far-reward cells total: {[len(xx) for xx in com_goal_farrew]}')
    epoch_perm.append([perm,rz_perm]) 
    # get goal cells across all epochs        
    if len(com_goal_farrew)>0:
        goal_cells = intersect_arrays(*com_goal_farrew); 
    else:
        goal_cells=[]
    # s2p_iind_goal_cells = s2p_iind_filter[goal_cells]
    # get per comparison
    goal_cells_p_per_comparison = [len(xx)/len(coms_correct[0]) for xx in com_goal_farrew]
    goal_cell_iind.append(goal_cells)
    goal_cell_p=len(goal_cells)/len(coms_correct[0])        
    goal_cell_prop.append([goal_cells_p_per_comparison,goal_cell_p])
    num_epochs.append(len(coms_correct))
    colors = ['k', 'slategray', 'darkcyan', 'darkgoldenrod', 'orchid']
    if len(goal_cells)>0:
        rows = int(np.ceil(np.sqrt(len(goal_cells))))
        cols = int(np.ceil(len(goal_cells) / rows))
        fig, axes = plt.subplots(rows, cols, figsize=(30,30),sharex=True)
        if len(goal_cells) > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        for i,gc in enumerate(goal_cells):            
            for ep in range(len(coms_correct)):
                ax = axes[i]
                ax.plot(tcs_correct[ep,gc,:], label=f'rewloc {rewlocs[ep]}', color=colors[ep])
                if len(tcs_fail)>0:
                        ax.plot(tcs_fail[ep,gc,:], label=f'fail rewloc {rewlocs[ep]}', 
                            color=colors[ep], linestyle = '--')
                ax.axvline((bins/2), color='k')
                ax.set_title(f'cell # {gc}')
                ax.spines[['top','right']].set_visible(False)
        ax.set_xticks(np.arange(0,bins+1,20))
        ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi, np.pi/2.25),3))
        ax.set_xlabel('Radian position (centered start rew loc)')
        ax.set_ylabel('Fc3')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    # get shuffled iteration
    shuffled_dist = np.zeros((num_iterations))
    # max of 5 epochs = 10 perms
    goal_cell_shuf_ps_per_comp = np.ones((num_iterations,10))*np.nan
    goal_cell_shuf_ps = []
    for i in range(num_iterations):
        # shuffle locations
        rewlocs_shuf = rewlocs #[random.randint(100,250) for iii in range(len(eps))]
        shufs = [list(range(coms_correct[ii].shape[0])) for ii in range(1, len(coms_correct))]
        [random.shuffle(shuf) for shuf in shufs]
        # first com is as ep 1, others are shuffled cell identities
        com_shufs = np.zeros_like(coms_correct); com_shufs[0,:] = coms_correct[0]
        com_shufs[1:1+len(shufs),:] = [coms_correct[ii][np.array(shufs)[ii-1]] for ii in range(1, 1+len(shufs))]
        # OR shuffle cell identities
        # relative to reward
        coms_rewrel = np.array([com-np.pi for ii, com in enumerate(com_shufs)])             
        # perm = list(combinations(range(len(coms_correct)), 2)) 
        # account for cells that move to the end/front
        # Find COMs near pi and shift to -pi
        com_loop_w_in_window = []
        for pi,p in enumerate(perm):
            for cll in range(coms_rewrel.shape[1]):
                com1_rel = coms_rewrel[p[0],cll]
                com2_rel = coms_rewrel[p[1],cll]
                # print(com1_rel,com2_rel,com_diff)
                if ((abs(com1_rel - np.pi) < epsilon) and 
                (abs(com2_rel + np.pi) < epsilon)):
                        com_loop_w_in_window.append(cll)
        # get abs value instead
        # cont.
        coms_rewrel[:,com_loop_w_in_window]=abs(coms_rewrel[:,com_loop_w_in_window])
        com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
        # get goal cells across all epochs
        com_goal_shuf = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
        # in addition, com near but after goal
        com_goal_farrew_shuf = [[xx for xx in com if (np.nanmedian(coms_rewrel[:,
            xx], axis=0)<=bound)] if len(com)>0 else [] for com in com_goal_shuf ]
        #only get perms with non zero cells
        com_goal_farrew_shuf=[com for com in com_goal_farrew_shuf if len(com)>0]
        goal_cells_shuf_p_per_comparison = [len(xx)/len(coms_correct[0]) for xx in com_goal_farrew_shuf]
        if len(com_goal_farrew_shuf)>0:
            goal_cells_shuf = intersect_arrays(*com_goal_farrew_shuf); 
        else:
            goal_cells_shuf=[]
        shuffled_dist[i] = len(goal_cells_shuf)/len(coms_correct[0])
        goal_cell_shuf_p=len(goal_cells_shuf)/len(coms_correct[0])
        goal_cell_shuf_ps.append(goal_cell_shuf_p)
        goal_cell_shuf_ps_per_comp[i, :len(goal_cells_shuf_p_per_comparison)] = goal_cells_shuf_p_per_comparison
    # save median of goal cell shuffle
    goal_cell_shuf_ps_per_comp_av = np.nanmedian(goal_cell_shuf_ps_per_comp,axis=0)        
    goal_cell_shuf_ps_av = np.nanmedian(np.array(goal_cell_shuf_ps)[1])
    goal_cell_null.append([goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av])
    p_value = sum(shuffled_dist>goal_cell_p)/num_iterations
    pvals.append(p_value); 
    print(f'{animal}, day {day}: significant goal cells proportion p-value: {p_value}')
    total_cells.append(len(coms_correct[0]))
    radian_alignment[f'{animal}_{day:03d}_index{ii:03d}'] = [tcs_correct, coms_correct, tcs_fail, coms_fail,
            com_goal, goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av]
    
    return radian_alignment,rate,p_value,total_cells,goal_cell_iind,\
        goal_cell_prop,num_epochs,goal_cell_null,epoch_perm,pvals

def trail_type_probe_activity_quant(ii,params_pth,animal,day,bins,radian_alignment,
    radian_alignment_saved,goal_cm_window,pdf,epoch_perm,goal_cell_iind,goal_cell_prop,num_epochs,
    goal_cell_null,pvals,total_cells):
    """
    changed on 2/6/25 to make it more consistent with splitting the different
    subpopulations
    updated 5/26/25 to do all cell types
    """
    print(params_pth)
    fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
            'timedFF', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
            'licks','stat', 'timedFF'])
    VR = fall['VR'][0][0][()]
    scalingf = VR['scalingFACTOR'][0][0]
    try:
        rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf        
    except:
        rewsize = 10
    ybinned = fall['ybinned'][0]/scalingf
    track_length=180/scalingf    
    forwardvel = fall['forwardvel'][0]    
    changeRewLoc = np.hstack(fall['changeRewLoc'])
    trialnum=fall['trialnum'][0]
    rewards = fall['rewards'][0]
    lick = fall['licks'][0]
    time = fall['timedFF'][0]
    if animal=='e145':
        ybinned=ybinned[:-1]
        forwardvel=forwardvel[:-1]
        changeRewLoc=changeRewLoc[:-1]
        trialnum=trialnum[:-1]
        rewards=rewards[:-1]
        lick=lick[:-1]
        time = time[:-1]
    # set vars
    eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
    lasttr=8 # last trials
    bins=90
    rad = get_radian_position_first_lick_after_rew(eps, ybinned, lick, rewards, rewsize,rewlocs,
                    trialnum, track_length) # get radian coordinates
    track_length_rad = track_length*(2*np.pi/track_length)
    bin_size=track_length_rad/bins 
    rz = get_rewzones(rewlocs,1/scalingf)       
    # added to get anatomical info
    # takes time
    fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
    Fc3 = fall_fc3['Fc3']
    dFF = fall_fc3['dFF']
    Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
    dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]
    skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
    Fc3 = Fc3[:, skew>2] # only keep cells with skew greateer than 2
    # dark time params
    track_length_dt = 550 # cm estimate based on 99.9% of ypos
    track_length_rad_dt = track_length_dt*(2*np.pi/track_length_dt) # estimate bin for dark time
    bins_dt=150 
    bin_size_dt=track_length_rad_dt/bins_dt # typically 3 cm binswith ~ 475 track length
    # tc w/ dark time added to the end of track
    tcs_correct, coms_correct, tcs_fail, coms_fail, tcs_probe, coms_probe, ybinned_dt,rad = make_tuning_curves_by_trialtype_w_probes_w_darktime(eps,rewlocs,
        rewsize,ybinned,time,lick,
        Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
        bins=bins_dt)  
    goal_window = goal_cm_window*(2*np.pi/track_length) # cm converted to rad
    # change to relative value 
    coms_rewrel = np.array([com-np.pi for com in coms_correct])
    perm = list(combinations(range(len(coms_correct)), 2)) 
    rz_perm = [(int(rz[p[0]]),int(rz[p[1]])) for p in perm]   
    # account for cells that move to the end/front
    # Define a small window around pi (e.g., epsilon)
    epsilon = .7 # 20 cm
    # Find COMs near pi and shift to -pi
    com_loop_w_in_window = []
    for pi,p in enumerate(perm):
        for cll in range(coms_rewrel.shape[1]):
            com1_rel = coms_rewrel[p[0],cll]
            com2_rel = coms_rewrel[p[1],cll]
            # print(com1_rel,com2_rel,com_diff)
            if ((abs(com1_rel - np.pi) < epsilon) and 
            (abs(com2_rel + np.pi) < epsilon)):
                    com_loop_w_in_window.append(cll)
    # get abs value instead
    coms_rewrel[:,com_loop_w_in_window]=abs(coms_rewrel[:,com_loop_w_in_window])
    com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
    com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
    #only get perms with non zero cells  
    # get both pre and post rew cells at the same time
    cell_types = ['pre', 'post', 'far_pre', 'far_post']
    dfs = []
    tcs_corr = []; tcs_f = []; tcs_probes=[]
    for cell_type in cell_types:
        if cell_type=='far_pre':
            lowerbound = -np.pi/4 # updated 4/21/25
            com_goal_farrew = [[xx for xx in com if (abs(np.nanmedian(coms_rewrel[:,
                xx], axis=0)<=lowerbound))] if len(com)>0 else [] for com in com_goal]
        elif cell_type=='far_post':
            lowerbound = np.pi/4 # updated 4/21/25
            com_goal_farrew = [[xx for xx in com if (abs(np.nanmedian(coms_rewrel[:,
                xx], axis=0)>=lowerbound))] if len(com)>0 else [] for com in com_goal]
        elif cell_type=='post':
            lowerbound = np.pi/4 # updated 4/21/25
            com_goal_farrew = [[xx for xx in com if ((np.nanmedian(coms_rewrel[:,
                xx], axis=0)<=lowerbound) & (np.nanmedian(coms_rewrel[:,
                xx], axis=0)>0))] if len(com)>0 else [] for com in com_goal]
        elif cell_type=='pre':
            lowerbound = -np.pi/4 # updated 4/21/25
            com_goal_farrew = [[xx for xx in com if ((np.nanmedian(coms_rewrel[:,
                xx], axis=0)>=lowerbound) & (np.nanmedian(coms_rewrel[:,
                xx], axis=0)<0))] if len(com)>0 else [] for com in com_goal]
        perm=[p for ii,p in enumerate(perm) if len(com_goal_farrew[ii])>0]
        rz_perm=[p for ii,p in enumerate(rz_perm) if len(com_goal_farrew[ii])>0]
        com_goal_farrew=[com for com in com_goal_farrew if len(com)>0]
        print(f'Far-reward cells total: {[len(xx) for xx in com_goal_farrew]}')
        # get goal cells across all epochs        
        # get far reward cells in any ep
        # CHANGE: 4/24/25
        if len(com_goal_farrew)>0:
            goal_cells = intersect_arrays(*com_goal_farrew); 
            goal_cells = np.unique(np.concatenate(com_goal_farrew))
        else:
            goal_cells=[]    
        # integral
        # get tc 
        correct = scipy.integrate.trapz(tcs_correct[:,goal_cells, :],axis=2)
        # epoch (x) x cells (y)
        incorrect = scipy.integrate.trapz(tcs_fail[:,goal_cells, :],axis=2)
        df=pd.DataFrame()
        df['mean_tc'] = np.concatenate([np.concatenate(correct), 
                            np.concatenate(incorrect)])
        # x 2 for both correct and incorrect
        df['cellid'] = np.concatenate([np.concatenate([np.arange(len(goal_cells))]*correct.shape[0])]*2)
        df['epoch'] = np.concatenate([np.repeat(np.arange(correct.shape[0]),correct.shape[1])]*2)
        df['trial_type'] = np.concatenate([['correct']*len(np.concatenate(correct)),
                        ['incorrect']*len(np.concatenate(incorrect))])
        df['animal']=[animal]*len(df)
        df['day']=[day]*len(df)
        df['cell_type'] = [cell_type]*len(df)
        dfs.append(df)
        tcs_corr.append(tcs_correct[:,goal_cells])
        tcs_f.append(tcs_fail[:,goal_cells])
        tcs_probes.append(tcs_probe[:,:,goal_cells])
    epoch_perm.append([perm,rz_perm]) 
    df=pd.concat(dfs)
    # get mean tuning curve correct vs. incorrect
    return df,tcs_corr,tcs_f,tcs_probes

def trail_type_activity_quant(ii,params_pth,animal,day,bins,radian_alignment,
    radian_alignment_saved,goal_cm_window,pdf,epoch_perm,goal_cell_iind,goal_cell_prop,num_epochs,
    goal_cell_null,pvals,total_cells):
    """
    changed on 2/6/25 to make it more consistent with splitting the different
    subpopulations
    updated 5/26/25 to do all cell types
    """
    print(params_pth)
    fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
            'timedFF', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
            'licks','stat', 'timedFF'])
    VR = fall['VR'][0][0][()]
    scalingf = VR['scalingFACTOR'][0][0]
    try:
        rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf        
    except:
        rewsize = 10
    ybinned = fall['ybinned'][0]/scalingf
    track_length=180/scalingf    
    forwardvel = fall['forwardvel'][0]    
    changeRewLoc = np.hstack(fall['changeRewLoc'])
    trialnum=fall['trialnum'][0]
    rewards = fall['rewards'][0]
    lick = fall['licks'][0]
    time = fall['timedFF'][0]
    if animal=='e145':
        ybinned=ybinned[:-1]
        forwardvel=forwardvel[:-1]
        changeRewLoc=changeRewLoc[:-1]
        trialnum=trialnum[:-1]
        rewards=rewards[:-1]
        lick=lick[:-1]
        time = time[:-1]
    # set vars
    eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
    lasttr=8 # last trials
    bins=90
    rad = get_radian_position_first_lick_after_rew(eps, ybinned, lick, rewards, rewsize,rewlocs,
                    trialnum, track_length) # get radian coordinates
    track_length_rad = track_length*(2*np.pi/track_length)
    bin_size=track_length_rad/bins 
    rz = get_rewzones(rewlocs,1/scalingf)       
    # added to get anatomical info
    # takes time
    fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
    Fc3 = fall_fc3['Fc3']
    dFF = fall_fc3['dFF']
    Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
    dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]
    skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
    Fc3 = Fc3[:, skew>2] # only keep cells with skew greateer than 2
    # dark time params
    track_length_dt = 550 # cm estimate based on 99.9% of ypos
    track_length_rad_dt = track_length_dt*(2*np.pi/track_length_dt) # estimate bin for dark time
    bins_dt=150 
    bin_size_dt=track_length_rad_dt/bins_dt # typically 3 cm binswith ~ 475 track length
    # tc w/ dark time added to the end of track
    tcs_correct, coms_correct, tcs_fail, coms_fail, ybinned_dt, rad = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,
        rewsize,ybinned,time,lick,
        Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
        bins=bins_dt)  
    # get vel tc too
    vel_correct, _, vel_fail, _, ybinned_dt, rad = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,rewsize,ybinned,time,lick,np.array([forwardvel]).T,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
        bins=bins_dt,lasttr=8,velocity_filter=True) 
    vel_correct=np.squeeze(vel_correct)
    vel_fail=np.squeeze(vel_fail)
    goal_window = goal_cm_window*(2*np.pi/track_length) # cm converted to rad
    # change to relative value 
    coms_rewrel = np.array([com-np.pi for com in coms_correct])
    perm = list(combinations(range(len(coms_correct)), 2)) 
    rz_perm = [(int(rz[p[0]]),int(rz[p[1]])) for p in perm]   
    # account for cells that move to the end/front
    # Define a small window around pi (e.g., epsilon)
    epsilon = .7 # 20 cm
    # Find COMs near pi and shift to -pi
    com_loop_w_in_window = []
    for pi,p in enumerate(perm):
        for cll in range(coms_rewrel.shape[1]):
            com1_rel = coms_rewrel[p[0],cll]
            com2_rel = coms_rewrel[p[1],cll]
            # print(com1_rel,com2_rel,com_diff)
            if ((abs(com1_rel - np.pi) < epsilon) and 
            (abs(com2_rel + np.pi) < epsilon)):
                    com_loop_w_in_window.append(cll)
    # get abs value instead
    coms_rewrel[:,com_loop_w_in_window]=abs(coms_rewrel[:,com_loop_w_in_window])
    com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
    com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
    #only get perms with non zero cells  
    # get both pre and post rew cells at the same time
    cell_types = ['pre', 'post', 'far_pre', 'far_post']
    dfs = []
    tcs_corr = []; tcs_f = []
    for cell_type in cell_types:
        if cell_type=='far_pre':
            lowerbound = -np.pi/4 # updated 4/21/25
            com_goal_farrew = [[xx for xx in com if (abs(np.nanmedian(coms_rewrel[:,
                xx], axis=0)<=lowerbound))] if len(com)>0 else [] for com in com_goal]
        elif cell_type=='far_post':
            lowerbound = np.pi/4 # updated 4/21/25
            com_goal_farrew = [[xx for xx in com if (abs(np.nanmedian(coms_rewrel[:,
                xx], axis=0)>=lowerbound))] if len(com)>0 else [] for com in com_goal]
        elif cell_type=='post':
            lowerbound = np.pi/4 # updated 4/21/25
            com_goal_farrew = [[xx for xx in com if ((np.nanmedian(coms_rewrel[:,
                xx], axis=0)<=lowerbound) & (np.nanmedian(coms_rewrel[:,
                xx], axis=0)>0))] if len(com)>0 else [] for com in com_goal]
        elif cell_type=='pre':
            lowerbound = -np.pi/4 # updated 4/21/25
            com_goal_farrew = [[xx for xx in com if ((np.nanmedian(coms_rewrel[:,
                xx], axis=0)>=lowerbound) & (np.nanmedian(coms_rewrel[:,
                xx], axis=0)<0))] if len(com)>0 else [] for com in com_goal]
        perm=[p for ii,p in enumerate(perm) if len(com_goal_farrew[ii])>0]
        rz_perm=[p for ii,p in enumerate(rz_perm) if len(com_goal_farrew[ii])>0]
        com_goal_farrew=[com for com in com_goal_farrew if len(com)>0]
        print(f'Far-reward cells total: {[len(xx) for xx in com_goal_farrew]}')
        # get goal cells across all epochs        
        # get far reward cells in any ep
        # CHANGE: 4/24/25
        if len(com_goal_farrew)>0:
            goal_cells = intersect_arrays(*com_goal_farrew); 
            goal_cells = np.unique(np.concatenate(com_goal_farrew))
        else:
            goal_cells=[]    
        # integral
        # get tc 
        correct = scipy.integrate.trapz(tcs_correct[:,goal_cells, :],axis=2)
        # epoch (x) x cells (y)
        incorrect = scipy.integrate.trapz(tcs_fail[:,goal_cells, :],axis=2)
        df=pd.DataFrame()
        df['mean_tc'] = np.concatenate([np.concatenate(correct), 
                            np.concatenate(incorrect)])
        # x 2 for both correct and incorrect
        df['cellid'] = np.concatenate([np.concatenate([np.arange(len(goal_cells))]*correct.shape[0])]*2)
        df['epoch'] = np.concatenate([np.repeat(np.arange(correct.shape[0]),correct.shape[1])]*2)
        df['trial_type'] = np.concatenate([['correct']*len(np.concatenate(correct)),
                        ['incorrect']*len(np.concatenate(incorrect))])
        df['animal']=[animal]*len(df)
        df['day']=[day]*len(df)
        df['cell_type'] = [cell_type]*len(df)
        dfs.append(df)
        tcs_corr.append(tcs_correct[:,goal_cells])
        tcs_f.append(tcs_fail[:,goal_cells])
    epoch_perm.append([perm,rz_perm]) 
    df=pd.concat(dfs)
    # get mean tuning curve correct vs. incorrect
    return df,tcs_corr,tcs_f,vel_correct,vel_fail

def reward_act_nearrew(ii,params_pth,animal,day,bins,radian_alignment,
    radian_alignment_saved,goal_cm_window,pdf,epoch_perm,goal_cell_iind,goal_cell_prop,num_epochs,
    goal_cell_null,pvals,total_cells,
    num_iterations=1000):
    """
    changed on 2/6/25 to make it more consistent with splitting the different
    subpopulations
    """
    print(params_pth)
    fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
            'pyr_tc_s2p_cellind', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
            'licks','stat', 'timedFF'])
    VR = fall['VR'][0][0][()]
    scalingf = VR['scalingFACTOR'][0][0]
    try:
        rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf        
    except:
        rewsize = 10
    ybinned = fall['ybinned'][0]/scalingf
    track_length=180/scalingf    
    forwardvel = fall['forwardvel'][0]    
    changeRewLoc = np.hstack(fall['changeRewLoc'])
    trialnum=fall['trialnum'][0]
    rewards = fall['rewards'][0]
    lick = fall['licks'][0]
    if animal=='e145':
        ybinned=ybinned[:-1]
        forwardvel=forwardvel[:-1]
        changeRewLoc=changeRewLoc[:-1]
        trialnum=trialnum[:-1]
        rewards=rewards[:-1]
        lick=lick[:-1]
    # set vars
    eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
    lasttr=8 # last trials
    bins=90
    rad = get_radian_position_first_lick_after_rew(eps, ybinned, lick, rewards, rewsize,rewlocs,
                    trialnum, track_length) # get radian coordinates
    track_length_rad = track_length*(2*np.pi/track_length)
    bin_size=track_length_rad/bins 
    rz = get_rewzones(rewlocs,1/scalingf)       
    # get average success rate
    rates = []
    for ep in range(len(eps)-1):
        eprng = range(eps[ep],eps[ep+1])
        success, fail, str_trials, ftr_trials, ttr, \
        total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
        rates.append(success/total_trials)
    rate=np.nanmean(np.array(rates))
    
    # added to get anatomical info
    # takes time
    fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
    Fc3 = fall_fc3['Fc3']
    dFF = fall_fc3['dFF']
    Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
    dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]
    skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
    Fc3 = Fc3[:, skew>2] # only keep cells with skew greateer than 2
    if f'{animal}_{day:03d}_index{ii:03d}' in radian_alignment_saved.keys():
        tcs_correct, coms_correct, tcs_fail, coms_fail, \
        com_goal, goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av = radian_alignment_saved[f'{animal}_{day:03d}_index{ii:03d}']            
    else:# remake tuning curves relative to reward        
        # 9/19/24
        # find correct trials within each epoch!!!!
        tcs_correct, coms_correct, tcs_fail, coms_fail = make_tuning_curves_radians_by_trialtype(eps,rewlocs,ybinned,rad,Fc3,trialnum,
        rewards,forwardvel,rewsize,bin_size)          
    goal_window = goal_cm_window*(2*np.pi/track_length) # cm converted to rad
    # change to relative value 
    coms_rewrel = np.array([com-np.pi for com in coms_correct])
    perm = list(combinations(range(len(coms_correct)), 2)) 
    rz_perm = [(int(rz[p[0]]),int(rz[p[1]])) for p in perm]   
    # if 4 ep
    # account for cells that move to the end/front
    # Define a small window around pi (e.g., epsilon)
    epsilon = .7 # 20 cm
    # Find COMs near pi and shift to -pi
    com_loop_w_in_window = []
    for pi,p in enumerate(perm):
        for cll in range(coms_rewrel.shape[1]):
            com1_rel = coms_rewrel[p[0],cll]
            com2_rel = coms_rewrel[p[1],cll]
            # print(com1_rel,com2_rel,com_diff)
            if ((abs(com1_rel - np.pi) < epsilon) and 
            (abs(com2_rel + np.pi) < epsilon)):
                    com_loop_w_in_window.append(cll)
    # get abs value instead
    coms_rewrel[:,com_loop_w_in_window]=abs(coms_rewrel[:,com_loop_w_in_window])
    com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
    com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
    # in addition, com near but after goal
    upperbound = np.pi/4
    com_goal_postrew = [[xx for xx in com if ((np.nanmedian(coms_rewrel[:,
        xx], axis=0)<=upperbound) & (np.nanmedian(coms_rewrel[:,
        xx], axis=0)>0))] if len(com)>0 else [] for com in com_goal]
    acc = fall['forwardvel'][0][1:]/np.diff(fall['timedFF'][0])
    acc=np.hstack(pd.DataFrame({'acc': acc}).rolling(5).mean().values)    
    nans,x = nan_helper(acc)
    acc[nans] = np.interp(x(nans), x(~nans), acc[~nans])
    # check to make sure just a subset
    # calc acc corr
    raccs_all = []
    for com in com_goal_postrew:
        raccs = []
        if len(com)>0:
            for cll in com:
                try: # if dff is nans
                    dFF[1:,cll][nans] = np.interp(x(nans), x(~nans), dFF[1:,cll][~nans])
                    _,racc = scipy.stats.pearsonr(acc, dFF[1:,cll])
                except Exception as e:
                    racc = np.nan
                raccs.append(racc)
        raccs_all.append(raccs)
    thres=1e-50 # correlation thres
    com_goal_postrew = [[xx for jj,xx in enumerate(com) if raccs_all[ii][jj]>thres] 
        if len(com)>0 else [] for ii,com in enumerate(com_goal_postrew)]
    #only get perms with non zero cells
    perm=[p for ii,p in enumerate(perm) if len(com_goal_postrew[ii])>0]
    rz_perm=[p for ii,p in enumerate(rz_perm) if len(com_goal_postrew[ii])>0]
    com_goal_postrew=[com for com in com_goal_postrew if len(com)>0]
    print(f'Reward-centric cells total: {[len(xx) for xx in com_goal]}\n\
    Post-reward cells: {[len(xx) for xx in com_goal_postrew]}')
    assert sum([len(xx) for xx in com_goal])>=sum([len(xx) for xx in com_goal_postrew])
    epoch_perm.append([perm,rz_perm]) 
    # get goal cells across all epochs   
    if len(com_goal_postrew)>0:
        goal_cells = intersect_arrays(*com_goal_postrew); 
    else:
        goal_cells=[]
    # integral
    # get tc after rew
    correct = scipy.integrate.trapz(tcs_correct[:,goal_cells, int(bins/2):],axis=2)
    # epoch (x) x cells (y)
    incorrect = scipy.integrate.trapz(tcs_fail[:,goal_cells, int(bins/2):],axis=2)
    df=pd.DataFrame()
    df['mean_tc'] = np.concatenate([np.concatenate(correct), 
                        np.concatenate(incorrect)])
    # x 2 for both correct and incorrect
    df['cellid'] = np.concatenate([np.concatenate([np.arange(len(goal_cells))]*correct.shape[0])]*2)
    df['epoch'] = np.concatenate([np.repeat(np.arange(correct.shape[0]),correct.shape[1])]*2)
    df['trial_type'] = np.concatenate([['correct']*len(np.concatenate(correct)),
                    ['incorrect']*len(np.concatenate(incorrect))])
    df['animal']=[animal]*len(df)
    df['day']=[day]*len(df)
    # get mean tuning curve correct vs. incorrect
    return df,tcs_correct[:,goal_cells],tcs_fail[:,goal_cells]

def reward_act_prerew(ii,params_pth,animal,day,bins,radian_alignment,
    radian_alignment_saved,goal_cm_window,pdf,epoch_perm,goal_cell_iind,goal_cell_prop,num_epochs,
    goal_cell_null,pvals,total_cells,
    num_iterations=1000):
    """
    changed on 2/6/25 to make it more consistent with splitting the different
    subpopulations
    """
    print(params_pth)
    fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
            'pyr_tc_s2p_cellind', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
            'licks','stat', 'timedFF'])
    VR = fall['VR'][0][0][()]
    scalingf = VR['scalingFACTOR'][0][0]
    try:
        rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf        
    except:
        rewsize = 10
    ybinned = fall['ybinned'][0]/scalingf
    track_length=180/scalingf    
    forwardvel = fall['forwardvel'][0]    
    changeRewLoc = np.hstack(fall['changeRewLoc'])
    trialnum=fall['trialnum'][0]
    rewards = fall['rewards'][0]
    lick = fall['licks'][0]
    if animal=='e145':
        ybinned=ybinned[:-1]
        forwardvel=forwardvel[:-1]
        changeRewLoc=changeRewLoc[:-1]
        trialnum=trialnum[:-1]
        rewards=rewards[:-1]
        lick=lick[:-1]
    # set vars
    eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
    lasttr=8 # last trials
    bins=90
    rad = get_radian_position_first_lick_after_rew(eps, ybinned, lick, rewards, rewsize,rewlocs,
                    trialnum, track_length) # get radian coordinates
    track_length_rad = track_length*(2*np.pi/track_length)
    bin_size=track_length_rad/bins 
    rz = get_rewzones(rewlocs,1/scalingf)       
    # get average success rate
    rates = []
    for ep in range(len(eps)-1):
        eprng = range(eps[ep],eps[ep+1])
        success, fail, str_trials, ftr_trials, ttr, \
        total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
        rates.append(success/total_trials)
    rate=np.nanmean(np.array(rates))
    
    # added to get anatomical info
    # takes time
    fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
    Fc3 = fall_fc3['Fc3']
    dFF = fall_fc3['dFF']
    Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
    dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]
    skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
    Fc3 = Fc3[:, skew>2] # only keep cells with skew greateer than 2
    # remake tuning curves relative to reward        
    # 9/19/24
    # find correct trials within each epoch!!!!
    # remove fails that are not in between correct trials
    tcs_correct, coms_correct, tcs_fail, coms_fail = make_tuning_curves_radians_by_trialtype(eps,rewlocs,ybinned,rad,Fc3,trialnum,
        rewards,forwardvel,rewsize,bin_size)          
    
    goal_window = goal_cm_window*(2*np.pi/track_length) # cm converted to rad
    # change to relative value 
    coms_rewrel = np.array([com-np.pi for com in coms_correct])
    perm = list(combinations(range(len(coms_correct)), 2)) 
    rz_perm = [(int(rz[p[0]]),int(rz[p[1]])) for p in perm]   
    # if 4 ep
    # account for cells that move to the end/front
    # Define a small window around pi (e.g., epsilon)
    epsilon = .7 # 20 cm
    # Find COMs near pi and shift to -pi
    com_loop_w_in_window = []
    for pi,p in enumerate(perm):
        for cll in range(coms_rewrel.shape[1]):
            com1_rel = coms_rewrel[p[0],cll]
            com2_rel = coms_rewrel[p[1],cll]
            # print(com1_rel,com2_rel,com_diff)
            if ((abs(com1_rel - np.pi) < epsilon) and 
            (abs(com2_rel + np.pi) < epsilon)):
                    com_loop_w_in_window.append(cll)
    # get abs value instead
    coms_rewrel[:,com_loop_w_in_window]=abs(coms_rewrel[:,com_loop_w_in_window])
    com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
    com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
    # in addition, com before goal
    lowerbound = -np.pi/4
    com_goal_postrew = [[xx for xx in com if ((np.nanmedian(coms_rewrel[:,
        xx], axis=0)>=lowerbound) & (np.nanmedian(coms_rewrel[:,
        xx], axis=0)<0))] if len(com)>0 else [] for com in com_goal]
    #only get perms with non zero cells
    perm=[p for ii,p in enumerate(perm) if len(com_goal_postrew[ii])>0]
    rz_perm=[p for ii,p in enumerate(rz_perm) if len(com_goal_postrew[ii])>0]
    com_goal_postrew=[com for com in com_goal_postrew if len(com)>0]

    print(f'Reward-centric cells total: {[len(xx) for xx in com_goal]}\n\
    Pre-reward cells: {[len(xx) for xx in com_goal_postrew]}')
    assert sum([len(xx) for xx in com_goal])>=sum([len(xx) for xx in com_goal_postrew])
    epoch_perm.append([perm,rz_perm]) 
    # get goal cells across all epochs   
    if len(com_goal_postrew)>0:
        goal_cells = intersect_arrays(*com_goal_postrew); 
    else:
        goal_cells=[]

    assert sum([len(xx) for xx in com_goal])>=sum([len(xx) for xx in com_goal_postrew])
    epoch_perm.append([perm,rz_perm]) 
    correct = scipy.integrate.trapz(tcs_correct[:,goal_cells,:int(bins/2)],axis=2)
    # epoch (x) x cells (y)
    incorrect = scipy.integrate.trapz(tcs_fail[:,goal_cells,:int(bins/2)],axis=2)
    df=pd.DataFrame()
    df['mean_tc'] = np.concatenate([np.concatenate(correct), 
                        np.concatenate(incorrect)])
    # x 2 for both correct and incorrect
    df['cellid'] = np.concatenate([np.concatenate([np.arange(len(goal_cells))]*correct.shape[0])]*2)
    df['epoch'] = np.concatenate([np.repeat(np.arange(correct.shape[0]),correct.shape[1])]*2)
    df['trial_type'] = np.concatenate([['correct']*len(np.concatenate(correct)),
                    ['incorrect']*len(np.concatenate(incorrect))])
    df['animal']=[animal]*len(df)
    df['day']=[day]*len(df)
    # get mean tuning curve correct vs. incorrect
    return df,tcs_correct[:,goal_cells],tcs_fail[:,goal_cells]

def reward_act_allrew(ii,params_pth,animal,day,bins,radian_alignment,
    radian_alignment_saved,goal_cm_window,pdf,epoch_perm,goal_cell_iind,goal_cell_prop,num_epochs,
    goal_cell_null,pvals,total_cells,
    num_iterations=1000):
    """
    changed on 2/6/25 to make it more consistent with splitting the different
    subpopulations
    """
    print(params_pth)
    fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
            'pyr_tc_s2p_cellind', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
            'licks','stat', 'timedFF'])
    VR = fall['VR'][0][0][()]
    scalingf = VR['scalingFACTOR'][0][0]
    try:
        rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf        
    except:
        rewsize = 10
    ybinned = fall['ybinned'][0]/scalingf
    track_length=180/scalingf    
    forwardvel = fall['forwardvel'][0]    
    changeRewLoc = np.hstack(fall['changeRewLoc'])
    trialnum=fall['trialnum'][0]
    rewards = fall['rewards'][0]
    lick = fall['licks'][0]
    if animal=='e145':
        ybinned=ybinned[:-1]
        forwardvel=forwardvel[:-1]
        changeRewLoc=changeRewLoc[:-1]
        trialnum=trialnum[:-1]
        rewards=rewards[:-1]
        lick=lick[:-1]
    # set vars
    eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
    lasttr=8 # last trials
    bins=90
    rad = get_radian_position_first_lick_after_rew(eps, ybinned, lick, rewards, rewsize,rewlocs,
                    trialnum, track_length) # get radian coordinates
    track_length_rad = track_length*(2*np.pi/track_length)
    bin_size=track_length_rad/bins 
    rz = get_rewzones(rewlocs,1/scalingf)       
    # get average success rate
    rates = []
    for ep in range(len(eps)-1):
        eprng = range(eps[ep],eps[ep+1])
        success, fail, str_trials, ftr_trials, ttr, \
        total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
        rates.append(success/total_trials)
    rate=np.nanmean(np.array(rates))
    
    # added to get anatomical info
    # takes time
    fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
    Fc3 = fall_fc3['Fc3']
    dFF = fall_fc3['dFF']
    Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
    dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]
    skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
    Fc3 = Fc3[:, skew>2] # only keep cells with skew greateer than 2
    # remake tuning curves relative to reward        
    # 9/19/24
    # find correct trials within each epoch!!!!
    # remove fails that are not in between correct trials
    tcs_correct, coms_correct, tcs_fail, coms_fail = make_tuning_curves_radians_by_trialtype(eps,rewlocs,ybinned,rad,Fc3,trialnum,
        rewards,forwardvel,rewsize,bin_size)          
    
    goal_window = goal_cm_window*(2*np.pi/track_length) # cm converted to rad
    # change to relative value 
    coms_rewrel = np.array([com-np.pi for com in coms_correct])
    perm = list(combinations(range(len(coms_correct)), 2)) 
    rz_perm = [(int(rz[p[0]]),int(rz[p[1]])) for p in perm]   
    # if 4 ep
    # account for cells that move to the end/front
    # Define a small window around pi (e.g., epsilon)
    epsilon = .7 # 20 cm
    # Find COMs near pi and shift to -pi
    com_loop_w_in_window = []
    for pi,p in enumerate(perm):
        for cll in range(coms_rewrel.shape[1]):
            com1_rel = coms_rewrel[p[0],cll]
            com2_rel = coms_rewrel[p[1],cll]
            # print(com1_rel,com2_rel,com_diff)
            if ((abs(com1_rel - np.pi) < epsilon) and 
            (abs(com2_rel + np.pi) < epsilon)):
                    com_loop_w_in_window.append(cll)
    # get abs value instead
    coms_rewrel[:,com_loop_w_in_window]=abs(coms_rewrel[:,com_loop_w_in_window])
    com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
    com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
    com_goal_postrew = com_goal # ALL REW CELLS
    #only get perms with non zero cells
    perm=[p for ii,p in enumerate(perm) if len(com_goal_postrew[ii])>0]
    rz_perm=[p for ii,p in enumerate(rz_perm) if len(com_goal_postrew[ii])>0]
    com_goal_postrew=[com for com in com_goal_postrew if len(com)>0]

    print(f'Reward-centric cells total: {[len(xx) for xx in com_goal]}')
    assert sum([len(xx) for xx in com_goal])>=sum([len(xx) for xx in com_goal_postrew])
    epoch_perm.append([perm,rz_perm]) 
    # get goal cells across all epochs   
    if len(com_goal_postrew)>0:
        goal_cells = intersect_arrays(*com_goal_postrew); 
    else:
        goal_cells=[]
    assert sum([len(xx) for xx in com_goal])>=sum([len(xx) for xx in com_goal_postrew])
    epoch_perm.append([perm,rz_perm]) 
    cs = [[cosine_sim_ignore_nan(tcs_correct[i,j],tcs_fail[i,j])
        for i in range(tcs_correct.shape[0])] for j in goal_cells]
    cs=np.array(cs).T # ep x cells
    # get mean tuning curve correct vs. incorrect
    return tcs_correct[:,goal_cells],tcs_fail[:,goal_cells],coms_rewrel[:,goal_cells],cs

def reward_act_v_probes_allrew(ii,params_pth,animal,day,bins,radian_alignment,
    radian_alignment_saved,goal_cm_window,pdf,epoch_perm,goal_cell_iind,goal_cell_prop,num_epochs,
    goal_cell_null,pvals,total_cells,
    num_iterations=1000):
    """
    changed on 2/6/25 to make it more consistent with splitting the different
    subpopulations
    """
    print(params_pth)
    fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
            'pyr_tc_s2p_cellind', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
            'licks','stat', 'timedFF'])
    VR = fall['VR'][0][0][()]
    scalingf = VR['scalingFACTOR'][0][0]
    try:
        rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf        
    except:
        rewsize = 10
    ybinned = fall['ybinned'][0]/scalingf
    track_length=180/scalingf    
    forwardvel = fall['forwardvel'][0]    
    changeRewLoc = np.hstack(fall['changeRewLoc'])
    trialnum=fall['trialnum'][0]
    rewards = fall['rewards'][0]
    lick = fall['licks'][0]
    if animal=='e145':
        ybinned=ybinned[:-1]
        forwardvel=forwardvel[:-1]
        changeRewLoc=changeRewLoc[:-1]
        trialnum=trialnum[:-1]
        rewards=rewards[:-1]
        lick=lick[:-1]
    # set vars
    eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
    lasttr=8 # last trials
    bins=90
    rad = get_radian_position_first_lick_after_rew(eps, ybinned, lick, rewards, rewsize,rewlocs,
                    trialnum, track_length) # get radian coordinates
    track_length_rad = track_length*(2*np.pi/track_length)
    bin_size=track_length_rad/bins 
    rz = get_rewzones(rewlocs,1/scalingf)       
    # get average success rate
    rates = []
    for ep in range(len(eps)-1):
        eprng = range(eps[ep],eps[ep+1])
        success, fail, str_trials, ftr_trials, ttr, \
        total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
        rates.append(success/total_trials)
    rate=np.nanmean(np.array(rates))
    
    # added to get anatomical info
    # takes time
    fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
    Fc3 = fall_fc3['Fc3']
    dFF = fall_fc3['dFF']
    Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
    dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]
    skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
    Fc3 = Fc3[:, skew>2] # only keep cells with skew greateer than 2
    # remake tuning curves relative to reward        
    # 9/19/24
    # find correct trials within each epoch!!!!
    # remove fails that are not in between correct trials
    tcs_correct, coms_correct, tcs_fail, coms_fail = make_tuning_curves_radians_by_trialtype(eps,rewlocs,ybinned,rad,Fc3,trialnum,
        rewards,forwardvel,rewsize,bin_size)          
    # vs. probe 1 tc
    tcs_probe, coms_probe = make_tuning_curves_probes(eps,rewlocs,ybinned,rad,Fc3,trialnum,
            rewards,forwardvel,rewsize,bin_size, probe=[0,1])
    # compare tc to the probe in the next epoch (based on how trialnum is structred)
    goal_window = goal_cm_window*(2*np.pi/track_length) # cm converted to rad
    # change to relative value 
    coms_rewrel = np.array([com-np.pi for com in coms_correct])
    perm = list(combinations(range(len(coms_correct)), 2)) 
    rz_perm = [(int(rz[p[0]]),int(rz[p[1]])) for p in perm]   
    # if 4 ep
    # account for cells that move to the end/front
    # Define a small window around pi (e.g., epsilon)
    epsilon = .7 # 20 cm
    # Find COMs near pi and shift to -pi
    com_loop_w_in_window = []
    for pi,p in enumerate(perm):
        for cll in range(coms_rewrel.shape[1]):
            com1_rel = coms_rewrel[p[0],cll]
            com2_rel = coms_rewrel[p[1],cll]
            # print(com1_rel,com2_rel,com_diff)
            if ((abs(com1_rel - np.pi) < epsilon) and 
            (abs(com2_rel + np.pi) < epsilon)):
                    com_loop_w_in_window.append(cll)
    # get abs value instead
    coms_rewrel[:,com_loop_w_in_window]=abs(coms_rewrel[:,com_loop_w_in_window])
    com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
    com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
    com_goal_postrew = com_goal # ALL REW CELLS
    # [[xx for xx in com if ((np.nanmedian(coms_rewrel[:,
    #     xx], axis=0)>=lowerbound) & (np.nanmedian(coms_rewrel[:,
    #     xx], axis=0)<0))] if len(com)>0 else [] for com in com_goal]
    #only get perms with non zero cells
    perm=[p for ii,p in enumerate(perm) if len(com_goal_postrew[ii])>0]
    rz_perm=[p for ii,p in enumerate(rz_perm) if len(com_goal_postrew[ii])>0]
    com_goal_postrew=[com for com in com_goal_postrew if len(com)>0]

    print(f'Reward-centric cells total: {[len(xx) for xx in com_goal]}')
    assert sum([len(xx) for xx in com_goal])>=sum([len(xx) for xx in com_goal_postrew])
    epoch_perm.append([perm,rz_perm]) 
    # get goal cells across all epochs   
    if len(com_goal_postrew)>0:
        goal_cells = intersect_arrays(*com_goal_postrew); 
    else:
        goal_cells=[]

    assert sum([len(xx) for xx in com_goal])>=sum([len(xx) for xx in com_goal_postrew])
    epoch_perm.append([perm,rz_perm]) 
    # compare to probe in next ep!
    cs = [[WessersteinDist(tcs_correct[i,j],tcs_probe[i+1,j])
        for i in range(tcs_correct.shape[0]-1)] for j in goal_cells]
    cs=np.array(cs).T # ep x cells
    # get mean tuning curve correct vs. incorrect 
    # remove last epoch before probes for simplicity
    return tcs_correct[:-1,goal_cells],tcs_probe[:-1,goal_cells],coms_rewrel[:-1,goal_cells],cs

def performance_by_trialtype(params_pth, animal,bins=90):
    """
    changed on 2/6/25 to make it more consistent with splitting the different
    subpopulations
    """
    print(params_pth)
    fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
            'pyr_tc_s2p_cellind', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
            'licks','stat', 'timedFF'])
    VR = fall['VR'][0][0][()]
    scalingf = VR['scalingFACTOR'][0][0]
    try:
        rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf        
    except:
        rewsize = 10
    ybinned = fall['ybinned'][0]/scalingf
    track_length=180/scalingf    
    forwardvel = fall['forwardvel'][0]    
    changeRewLoc = np.hstack(fall['changeRewLoc'])
    trialnum=fall['trialnum'][0]
    rewards = fall['rewards'][0]
    lick = fall['licks'][0]
    if animal=='e145':
        ybinned=ybinned[:-1]
        forwardvel=forwardvel[:-1]
        changeRewLoc=changeRewLoc[:-1]
        trialnum=trialnum[:-1]
        rewards=rewards[:-1]
        lick=lick[:-1]
    # set vars
    eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf
    eps = np.append(eps, len(changeRewLoc))
    lasttr=5 # last trials
    earlytr=5
    rad = get_radian_position_first_lick_after_rew(eps, ybinned, lick, rewards, rewsize,rewlocs,
                    trialnum, track_length) # get radian coordinates
    track_length_rad = track_length*(2*np.pi/track_length)
    bin_size=track_length_rad/bins 
    rz = get_rewzones(rewlocs,1/scalingf)       
    # get average success rate
    rates = []; lick_selectivity = []
    for ep in range(len(eps)-1):
        eprng = range(eps[ep],eps[ep+1])
        success, fail, str_trials, ftr_trials, ttr, \
        total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
        rates.append(success/total_trials)
        # get lick selectivity
        mask = np.array([xx in str_trials[-lasttr:] for xx in trialnum[eprng]])
        ls_late = get_lick_selectivity(ybinned[eprng][mask], 
                        trialnum[eprng][mask], lick[eprng][mask], 
                        rewlocs[ep], rewsize,
                        fails_only = False) 
        mask = np.array([xx in ttr[:earlytr] for xx in trialnum[eprng]])
        ls_early = get_lick_selectivity(ybinned[eprng][mask], 
                        trialnum[eprng][mask], lick[eprng][mask], 
                        rewlocs[ep], rewsize,
                        fails_only = False) 
        lick_selectivity.append([ls_early,ls_late])
    rate=np.nanmean(np.array(rates))

    tcs_correct, tcs_fail,_ = make_tuning_curves_radians_by_trialtype_behavior(eps,rewlocs,ybinned,rad,
        lick,trialnum,rewards,forwardvel,rewsize,bin_size)   
    perm = list(combinations(range(len(tcs_correct)), 2)) 
    rz_perm = [(int(rz[p[0]]),int(rz[p[1]])) for p in perm]      
    
    # get mean tuning curve correct vs. incorrect
    return tcs_correct,tcs_fail,rates,rz_perm,lick_selectivity

def licks_by_trialtype(params_pth, animal,bins=90):
    """
    changed on 2/6/25 to make it more consistent with splitting the different
    subpopulations
    """
    print(params_pth)
    fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
            'pyr_tc_s2p_cellind', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
            'licks','stat', 'timedFF'])
    VR = fall['VR'][0][0][()]
    scalingf = VR['scalingFACTOR'][0][0]
    try:
        rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf        
    except:
        rewsize = 10
    ybinned = fall['ybinned'][0]/scalingf
    track_length=180/scalingf    
    forwardvel = fall['forwardvel'][0]    
    changeRewLoc = np.hstack(fall['changeRewLoc'])
    trialnum=fall['trialnum'][0]
    rewards = fall['rewards'][0]
    lick = fall['licks'][0]
    time=fall['timedFF'][0]
    if animal=='e145':
        ybinned=ybinned[:-1]
        forwardvel=forwardvel[:-1]
        changeRewLoc=changeRewLoc[:-1]
        trialnum=trialnum[:-1]
        rewards=rewards[:-1]
        lick=lick[:-1]
        time=time[:-1]
    # set vars
    eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
    lasttr=8 # last trials
    rad = get_radian_position_first_lick_after_rew(eps, ybinned, lick, rewards, rewsize,rewlocs,
                    trialnum, track_length) # get radian coordinates
    track_length_rad = track_length*(2*np.pi/track_length)
    bin_size=track_length_rad/bins 
    rz = get_rewzones(rewlocs,1/scalingf)       
    # get average success rate
    rates = []; lick_rate=[]; vel = []
    for ep in range(len(eps)-1):
        eprng = range(eps[ep],eps[ep+1])
        success, fail, str_trials, ftr_trials, ttr, \
        total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
        rates.append(success/total_trials)
        ## get pre-reward lick rate
        t = time[eprng][(ybinned[eprng]<rewlocs[ep])]
        dt = np.nanmedian(np.diff(t))
        # manually check 0.7 for smoothing
        #correct vs. incorre
        mask = [ii for ii,xx in enumerate(trialnum[eprng]) if xx in str_trials]
        # trials=trialnum[eprng]
        # ypos=[ybinned[eprng][trials==tr] for tr in str_trials]
        # rew=[(rewards==1)[eprng][trials==tr] for tr in str_trials]
        # mask = [y[:int(np.where(rew[i])[0])] for i,y in enumerate(ypos)]
        corr_lr = smooth_lick_rate(lick[eprng][mask][(ybinned[eprng][mask]<(rewlocs[ep]-(rewsize)))], dt)       
        corr_vel = forwardvel[eprng][mask][(ybinned[eprng][mask]<(rewlocs[ep]-(rewsize)))]
        try: # if no failed trials that met criteria
            failed_inbtw = np.array([int(xx)-str_trials[0] for xx in ftr_trials])
            failed_inbtw=np.array(ftr_trials)[failed_inbtw>0]
            # only get those after correct trial
            # failed_after_corr = np.insert(np.diff(failed_inbtw), 0, 10)
            # failed_inbtw=failed_inbtw[failed_after_corr>1]
            mask = [ii for ii,xx in enumerate(trialnum[eprng]) if xx in failed_inbtw]
            # pre-rew
            incorr_lr_pre = smooth_lick_rate(lick[eprng][mask][(ybinned[eprng][mask]<(rewlocs[ep]-(rewsize)))], dt) # not just pre-rew
            incorr_lr = smooth_lick_rate(lick[eprng][mask], dt) # not just pre-rew
            incorr_vel_pre = forwardvel[eprng][mask][(ybinned[eprng][mask]<(rewlocs[ep]-(rewsize)))]
            incorr_vel = forwardvel[eprng][mask]
        except:
            incorr_lr = [np.nan]
            incorr_vel = [np.nan]
        lick_rate.append([corr_lr,incorr_lr_pre,incorr_lr])
        vel.append([corr_vel,incorr_vel,incorr_vel_pre])
    rate=np.nanmean(np.array(rates))
    #lick
    tcs_correct, tcs_fail, tcs_probes = make_tuning_curves_radians_by_trialtype_behavior(eps,rewlocs,ybinned,rad,
        lick,trialnum,rewards,forwardvel,rewsize,bin_size)          
    # vel
    tcs_correct_vel, tcs_fail_vel, tcs_probes_vel = make_tuning_curves_radians_by_trialtype_behavior(eps,rewlocs,ybinned,rad,
        forwardvel,trialnum,rewards,forwardvel,rewsize,bin_size) 
    ## get pre-reward lick rate
    # get mean tuning curve correct vs. incorrect
    return tcs_correct,tcs_fail,tcs_probes,tcs_correct_vel, tcs_fail_vel,tcs_probes_vel, lick_rate,vel

def extract_data_df(ii, params_pth, animal, day, radian_alignment, radian_alignment_saved, 
                    goal_cm_window, pdf, pln):
    """
    changed on 2/6/25 to make it more consistent with splitting the different
    subpopulations
    """
    print(params_pth)
    fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
            'putative_pcs', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
            'licks','stat', 'timedFF'])
    VR = fall['VR'][0][0][()]
    scalingf = VR['scalingFACTOR'][0][0]
    try:
        rewsize = VR['settings']['rewardZone'][0][0][0][0] / scalingf        
    except:
        rewsize = 10
    ybinned = fall['ybinned'][0] / scalingf
    track_length = 180 / scalingf    
    forwardvel = fall['forwardvel'][0]    
    changeRewLoc = np.hstack(fall['changeRewLoc'])
    trialnum = fall['trialnum'][0]
    rewards = fall['rewards'][0]
    lick = fall['licks'][0]
    
    if animal == 'e145':
        ybinned = ybinned[:-1]
        forwardvel = forwardvel[:-1]
        changeRewLoc = changeRewLoc[:-1]
        trialnum = trialnum[:-1]
        rewards = rewards[:-1]
        lick = lick[:-1]
    
    # set vars
    eps = np.where(changeRewLoc > 0)[0]
    rewlocs = changeRewLoc[eps] / scalingf
    eps = np.append(eps, len(changeRewLoc))
    bins = 90
    rad = get_radian_position_first_lick_after_rew(eps, ybinned, lick, rewards, rewsize, rewlocs,trialnum, track_length)  # get radian coordinates
    track_length_rad = track_length * (2 * np.pi / track_length)
    bin_size = track_length_rad / bins 
    rz = get_rewzones(rewlocs, 1 / scalingf)       

    # get average success rate
    rates = []
    for ep in range(len(eps) - 1):
        eprng = range(eps[ep], eps[ep + 1])
        success, fail, str_trials, ftr_trials, ttr, total_trials = get_success_failure_trials(
            trialnum[eprng], rewards[eprng])
        rates.append(success / total_trials)
    rate = np.nanmean(np.array(rates))
    
    # added to get anatomical info
    fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
    Fc3 = fall_fc3['Fc3']
    dFF = fall_fc3['dFF']
    Fc3 = Fc3[:, ((fall['iscell'][:, 0]).astype(bool)) & (~(fall['bordercells'][0]).astype(bool))]
    dFF = dFF[:, ((fall['iscell'][:, 0]).astype(bool)) & (~(fall['bordercells'][0]).astype(bool))]
    skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
    # important for mapping tracked ids
    suite2pind = np.arange(fall['iscell'][:,0].shape[0])
    suite2pind_remain = suite2pind[((fall['iscell'][:, 0]).astype(bool)) & (~(fall['bordercells'][0]).astype(bool))]
    suite2pind_remain = suite2pind_remain[skew>2]
    Fc3 = Fc3[:, skew > 2]  # only keep cells with skew greater than 2
    
    # circularly aligned
    tcs_correct, coms_correct, tcs_fail, coms_fail = make_tuning_curves_radians_by_trialtype(
        eps, rewlocs, ybinned, rad, Fc3, trialnum, rewards, forwardvel, rewsize, bin_size)          
    
    # allocentric ref
    bin_size = track_length / bins 
    tcs_correct_abs, coms_correct_abs, tcs_fail_abs, coms_fail_abs = make_tuning_curves(eps, rewlocs, ybinned, Fc3, trialnum,rewards, forwardvel, rewsize, bin_size)
    assert suite2pind_remain.shape[0]==tcs_correct.shape[1]

    # change to relative value 
    coms_rewrel = np.array([com - np.pi for com in coms_correct])
    perm = list(combinations(range(len(coms_correct)), 2)) 
    rz_perm = [(int(rz[p[0]]), int(rz[p[1]])) for p in perm]   
    
    # Define a small window around pi (e.g., epsilon)
    epsilon = .7  # 20 cm
    # Find COMs near pi and shift to -pi
    com_loop_w_in_window = []
    for pi, p in enumerate(perm):
        for cll in range(coms_rewrel.shape[1]):
            com1_rel = coms_rewrel[p[0], cll]
            com2_rel = coms_rewrel[p[1], cll]
            if (abs(com1_rel - np.pi) < epsilon) and (abs(com2_rel + np.pi) < epsilon):
                com_loop_w_in_window.append(cll)
    
    # get abs value instead
    coms_rewrel[:, com_loop_w_in_window] = abs(coms_rewrel[:, com_loop_w_in_window])
    pc_bool = fall['putative_pcs'][0][0][0]
    pc_bool = pc_bool[skew > 2]  # remember that pc bool removes bordercells
    # make sure dims are same
    if coms_correct_abs.shape[0]!=coms_rewrel.shape[0]:
        if coms_correct_abs.shape[0]>coms_rewrel.shape[0]:
            coms_correct_abs=coms_correct_abs[:coms_rewrel.shape[0],:]
            tcs_correct_abs=tcs_correct_abs[:coms_rewrel.shape[0],:,:]
        else:
            if coms_rewrel.shape[0]>coms_correct_abs.shape[0]:
                coms_rewrel=coms_rewrel[:coms_correct_abs.shape[0],:]
                tcs_correct=tcs_correct[:coms_correct_abs.shape[0],:,:]
    celltrackpth = r'Y:\analysis\celltrack'
    tracked_lut, days= get_tracked_lut(celltrackpth,animal,pln)
    try:
        # change to index!!!! get tracked cell index not s2p index again
        tracked_ind=tracked_lut[day][suite2pind_remain].index.values
    except:
        tracked_ind=np.ones_like(suite2pind_remain)*np.nan
    # add them back into the matrix as just pc_bool=0
    df = pd.DataFrame()
    df['reward_relative_circular_com'] = np.concatenate(coms_rewrel)
    df['allocentric_com'] = np.concatenate(coms_correct_abs)
    df['cellid'] = np.concatenate([list(np.arange(len(coms_rewrel[0])))]*len(coms_rewrel))
    df['tracked_cellid'] = np.concatenate([tracked_ind]*len(coms_rewrel))

    df['reward_location'] = np.concatenate([[rewlocs[ii]] * len(coms_rewrel[ii]) for ii in range(len(coms_rewrel))])
    df['epoch'] = np.concatenate([[ii + 1] * len(comr) for ii, comr in enumerate(coms_rewrel)])
    df['animal'] = [animal] * len(df)
    df['recording_day'] = [day] * len(df)
    df['spatially_tuned'] = np.concatenate([pc_bool] * len(coms_rewrel))
    
    # get average activity in tuning curve 
    df['mean_tuning_curve'] = np.concatenate(np.nanmean(tcs_correct, axis=2))
    df['max_tuning_curve'] = np.concatenate(np.nanmax(tcs_correct, axis=2))
    
    # get trial by trial tuning
    trialstates, licks, tcs, coms = make_tuning_curves_trial_by_trial(eps, rewlocs, lick, ybinned, rad, Fc3,
                                                trialnum, rewards, forwardvel, rewsize, bin_size)
    # amplitude greater than 0.1
    trialsactive = [[[xx > 0.1 for xx in cll] for cll in np.nanmax(tcs[ep], axis=2)] for ep in range(len(tcs))]
    df['percent_trials_active'] = np.concatenate([[np.nansum(xx) / len(xx) for xx in ep] for ep in trialsactive[:coms_rewrel.shape[0]]])
    
    return df

def acc_corr_cells(forwardvel, timedFF, pln, dFF, eps):
    acccells_per_ep = []
    for ep in range(len(eps)-1):
        eprng = np.arange(eps[ep], eps[ep+1])
        # get acceleration correlated cells across all trials
        # get goal cells across all epochs      
        acc = np.diff(forwardvel[eprng])/np.diff(timedFF[eprng])
        accdf = pd.DataFrame({'acc': acc})
        acc = np.hstack(accdf.rolling(100).mean().fillna(0).values)
        # cells correlated with acc
        # Calculate phase-shifted correlation
        max_shift = int(np.ceil(31.25/(pln+1)))  # You can adjust the max shift based on your data
        # ~ 1 s phase shifts
        rshiftmax = []
        for i in range(dFF.shape[1]):
            dff = dFF[eprng[:-1],i]
            dff[np.isnan(dff)]=0 # nan to 0
            r=phase_shifted_correlation(acc, dff, max_shift)
            rshiftmax.append(np.max(r))
        # only get top 10% for now
        acccells = np.where(np.array(rshiftmax)>np.nanquantile(rshiftmax,.90))[0]
        acccells_per_ep.append(acccells)        
        
    return acccells_per_ep

def phase_shifted_correlation(acceleration, neural_activity, max_shift):
    """
    Calculate phase-shifted correlation between acceleration and neural activity.
    
    Parameters:
        acceleration (np.array): The acceleration data.
        neural_activity (np.array): The neural activity data.
        max_shift (int): The maximum shift (in samples) to apply for phase-shifting.
        
    Returns:
        shifts (np.array): Array of shift values.
        correlations (np.array): Correlation values for each shift.
    """
    
    # Ensure the signals have the same length
    assert len(acceleration) == len(neural_activity), "Signals must have the same length"
    
    shifts = np.arange(-max_shift, max_shift + 1, 5)
    correlations = np.zeros(len(shifts))
    
    for i, shift in enumerate(shifts):
        if shift < 0:
            shifted_neural_activity = np.roll(neural_activity, shift)
            shifted_neural_activity[shift:] = 0
        else:
            shifted_neural_activity = np.roll(neural_activity, shift)
            shifted_neural_activity[:shift] = 0
        
        # Calculate the correlation for this shift
        correlation, _ = scipy.stats.pearsonr(acceleration, shifted_neural_activity)
        correlations[i] = correlation
    return correlations
    
def perireward_binned_activity_early_late(dFF, rewards, timedFF, trialnum, range_val, binsize,
                    early_trial=2, late_trial=5):
    """Adapts code to align dFF or pose data to rewards within a certain window on a per-trial basis,
    only considering trials with trialnum > 3. Calculates activity for the first 5 and last 5 trials separately.

    Args:
        dFF (_type_): _description_
        rewards (_type_): _description_
        timedFF (_type_): _description_
        trialnum (_type_): array denoting the trial number per frame
        range_val (_type_): _description_
        binsize (_type_): _description_

    Returns:
        dict: Contains mean and normalized activity for first and last 5 trials.
    """
    # Filter rewards for trialnum > 3
    Rewindx = np.where(rewards & (trialnum > 3))[0]
    
    # Calculate separately for first 5 trials
    first_trials = Rewindx[:early_trial]
    last_trials = Rewindx[-late_trial:]
    
    def calculate_activity(TrialIndexes):
        rewdFF = np.ones((int(np.ceil(range_val * 2 / binsize)), len(TrialIndexes)))*np.nan
        for rr in range(0, len(TrialIndexes)):
            current_trial = trialnum[TrialIndexes[rr]]
            rewtime = timedFF[TrialIndexes[rr]]
            currentrewchecks = np.where((timedFF > rewtime - range_val) & 
                                        (timedFF <= rewtime + range_val) & 
                                        (trialnum == current_trial))[0]
            currentrewcheckscell = consecutive_stretch(currentrewchecks)
            currentrewcheckscell = [xx for xx in currentrewcheckscell if len(xx) > 0]
            currentrewcheckscell = np.array(currentrewcheckscell)
            currentrewardlogical = np.array([sum(TrialIndexes[rr] == x).astype(bool) for x in currentrewcheckscell])
            val = 0
            for bin_val in range(int(np.ceil(range_val * 2 / binsize))):
                val = bin_val + 1
                currentidxt = np.where((timedFF > (rewtime - range_val + (val * binsize) - binsize)) & 
                                       (timedFF <= rewtime - range_val + val * binsize) &
                                    (trialnum == current_trial))[0]
                checks = consecutive_stretch(currentidxt)
                checks = [list(xx) for xx in checks]
                if len(checks[0]) > 0:
                    currentidxlogical = np.array([np.isin(x, currentrewcheckscell[currentrewardlogical][0]) \
                                    for x in checks])
                    for i, cidx in enumerate(currentidxlogical):
                        cidx = [bool(xx) for xx in cidx]
                        if sum(cidx) > 0:
                            checkidx = np.array(np.array(checks)[i])[np.array(cidx)]
                            rewdFF[bin_val, rr] = np.nanmean(dFF[checkidx])

        meanrewdFF = np.nanmean(rewdFF, axis=1)
        normmeanrewdFF = (meanrewdFF - np.min(meanrewdFF)) / (np.max(meanrewdFF) - np.min(meanrewdFF))
        normrewdFF = np.array([(xx - np.min(xx)) / ((np.max(xx) - np.min(xx))) for xx in rewdFF.T])

        return normmeanrewdFF, meanrewdFF, normrewdFF, rewdFF
    
    normmeanrewdFF_first, meanrewdFF_first, normrewdFF_first, rewdFF_first = calculate_activity(first_trials)
    normmeanrewdFF_last, meanrewdFF_last, normrewdFF_last, rewdFF_last = calculate_activity(last_trials)

    return {
        'first_5': {
            'normmeanrewdFF': normmeanrewdFF_first,
            'meanrewdFF': meanrewdFF_first,
            'normrewdFF': normrewdFF_first,
            'rewdFF': rewdFF_first
        },
        'last_5': {
            'normmeanrewdFF': normmeanrewdFF_last,
            'meanrewdFF': meanrewdFF_last,
            'normrewdFF': normrewdFF_last,
            'rewdFF': rewdFF_last
        }
    }

def perireward_binned_activity(dFF, rewards, timedFF, 
        trialnum, range_val, binsize):
    """adaptation of gerardo's code to align IN BOTH TIME AND POSITION, dff or pose data to 
    rewards within a certain window on a per-trial basis, only considering trials with trialnum > 3

    Args:
        dFF (_type_): _description_
        rewards (_type_): _description_
        timedFF (_type_): _description_
        trialnum (_type_): array denoting the trial number per frame
        range_val (_type_): _description_
        binsize (_type_): _description_

    Returns:
        _type_: _description_
    """
    Rewindx = np.where(rewards & (trialnum > 3))[0]  # Filter rewards for trialnum > 3
    rewdFF = np.ones((int(np.ceil(range_val * 2 / binsize)), len(Rewindx)))*np.nan

    for rr in range(0, len(Rewindx)):
        current_trial = trialnum[Rewindx[rr]]
        rewtime = timedFF[Rewindx[rr]]
        currentrewchecks = np.where((timedFF > rewtime - range_val) & 
                                    (timedFF <= rewtime + range_val) & 
                                    (trialnum == current_trial))[0]
        currentrewcheckscell = consecutive_stretch(currentrewchecks)  # Get consecutive stretch of reward ind
        # Check for missing vals
        currentrewcheckscell = [xx for xx in currentrewcheckscell if len(xx) > 0]
        currentrewcheckscell = np.array(currentrewcheckscell)  # Reformat for Python
        currentrewardlogical = np.array([sum(Rewindx[rr] == x).astype(bool) for x in currentrewcheckscell])
        val = 0
        for bin_val in range(int(np.ceil(range_val * 2 / binsize))):
            val = bin_val + 1
            currentidxt = np.where((timedFF > (rewtime - range_val + (val * binsize) - binsize)) & 
                                   (timedFF <= rewtime - range_val + val * binsize) &
                                (trialnum == current_trial))[0]
            checks = consecutive_stretch(currentidxt)
            checks = [list(xx) for xx in checks]
            if len(checks[0]) > 0:
                currentidxlogical = np.array([np.isin(x, currentrewcheckscell[currentrewardlogical][0]) \
                                for x in checks])
                for i, cidx in enumerate(currentidxlogical):
                    cidx = [bool(xx) for xx in cidx]
                    if sum(cidx) > 0:
                        checkidx = np.array(np.array(checks)[i])[np.array(cidx)]
                        rewdFF[bin_val, rr] = np.nanmean(dFF[checkidx])

    meanrewdFF = np.nanmean(rewdFF, axis=1)
    normmeanrewdFF = (meanrewdFF - np.min(meanrewdFF)) / (np.max(meanrewdFF) - np.min(meanrewdFF))
    normrewdFF = np.array([(xx - np.min(xx)) / ((np.max(xx) - np.min(xx))) for xx in rewdFF.T])
    
    return normmeanrewdFF, meanrewdFF, normrewdFF, rewdFF

def get_radian_position(eps,ybinned,rewlocs,track_length,rewsize):
    rad = [] # get radian coordinates
    # same as giocomo preprint - worked with gerardo
    for i in range(len(eps)-1):
        y = ybinned[eps[i]:eps[i+1]]
        rew = rewlocs[i]-rewsize/2
        # convert to radians and align to rew
        rad.append((((((y-rew)*2*np.pi)/track_length)+np.pi)%(2*np.pi))-np.pi)
    rad = np.concatenate(rad)
    return rad

import numpy as np

def get_radian_position_first_lick_after_rew(eps, ybinned, licks, reward, rewsize,rewlocs,
                    trialnum, track_length):
    """
    Computes radian position aligned to the first lick after reward.

    Parameters:
    - eps: List of trial start indices.
    - ybinned: 1D array of position values.
    - licks: 1D binary array (same length as ybinned) indicating lick events.
    - reward: 1D binary array (same length as ybinned) indicating reward delivery.
    - track_length: Total length of the circular track.

    Returns:
    - rad: 1D array of radian positions aligned to the first lick after reward.
    """
    rad = []  # Store radian coordinates
    for i in range(len(eps) - 1):
        # Extract data for the current trial
        y_trial = ybinned[eps[i]:eps[i+1]]
        licks_trial = licks[eps[i]:eps[i+1]]
        reward_trial = reward[eps[i]:eps[i+1]]
        trialnum_trial = trialnum[eps[i]:eps[i+1]]
        unique_trials = np.unique(trialnum[eps[i]:eps[i+1]])  # Get unique trial numbers
        for trial in unique_trials:
            # Extract data for the current trial
            trial_mask = trialnum_trial == trial  # Boolean mask for the current trial
            y = y_trial[trial_mask]
            licks_trial_ = licks_trial[trial_mask]
            reward_trial_ = reward_trial[trial_mask]
            # Find the reward location in this trial
            reward_indices = np.where(reward_trial_ > 0)[0]  # Indices where reward occurs
            if len(reward_indices) == 0:
                try:
                    y_rew = np.where((y<(rewlocs[i]+rewsize*1.5)) & (y>(rewlocs[i]-rewsize*1.5)))[0][0]
                    reward_idx=y_rew
                except Exception as e: # if trial is empty??
                    reward_idx=int(len(y)/2) # put in random middle place of trials
            else:
                reward_idx = reward_indices[0]  # First occurrence of reward
            # Find the first lick after the reward
            lick_indices_after_reward = np.where((licks_trial_ > 0) & (np.arange(len(licks_trial_)) > reward_idx))[0]
            if len(lick_indices_after_reward) > 0:
                first_lick_idx = lick_indices_after_reward[0]  # First lick after reward
            else:
                # if animal did not lick after reward/no reward was given
                first_lick_idx=reward_idx
            # Convert positions to radians relative to the first lick
            first_lick_pos = y[first_lick_idx]
            rad.append((((((y - first_lick_pos) * 2 * np.pi) / track_length) + np.pi) % (2 * np.pi)) - np.pi)

    if len(rad) > 0:
        rad = np.concatenate(rad)
        return rad
    else:
        return np.array([])  # Return empty array if no valid trials

def get_goal_cells(rz, goal_window, coms_correct, cell_type = 'all'):    
    # change to relative value 
    coms_rewrel = np.array([com-np.pi for com in coms_correct])
    perm = list(combinations(range(len(coms_correct)), 2)) 
    # if 4 ep
    # account for cells that move to the end/front
    # Define a small window around pi (e.g., epsilon)
    epsilon = .7 # 20 cm
    # Find COMs near pi and shift to -pi
    com_loop_w_in_window = []
    for pi,p in enumerate(perm):
        for cll in range(coms_rewrel.shape[1]):
            com1_rel = coms_rewrel[p[0],cll]
            com2_rel = coms_rewrel[p[1],cll]
            # print(com1_rel,com2_rel,com_diff)
            if ((abs(com1_rel - np.pi) < epsilon) and 
            (abs(com2_rel + np.pi) < epsilon)):
                    com_loop_w_in_window.append(cll)
    # get abs value instead
    coms_rewrel[:,com_loop_w_in_window]=abs(coms_rewrel[:,com_loop_w_in_window])
    com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
    com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
    # TODO: implement per cell type
    # if cell_type=='all'
    com_goal_postrew = com_goal
    perm=[p for ii,p in enumerate(perm) if len(com_goal_postrew[ii])>0]
    rz_perm = [(int(rz[p[0]]),int(rz[p[1]])) for p in perm] 
    rz_perm=[p for jj,p in enumerate(rz_perm) if len(com_goal_postrew[jj])>0]
    # remove empty epochs
    # save non empty epochs
    com_goal_postrew=[com for com in com_goal_postrew if len(com)>0]

    if len(com_goal_postrew)>0:
        goal_cells = intersect_arrays(*com_goal_postrew); 
    else:
        goal_cells=[]
    return goal_cells, com_goal_postrew, perm, rz_perm

def get_goal_cells_time(rz, goal_window, coms_correct, cell_type = 'all'):    
    # change to relative value 
    coms_rewrel = np.array([com-np.pi for com in coms_correct])
    perm = list(combinations(range(len(coms_correct)), 2)) 
    # if 4 ep
    com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
    com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
    # TODO: implement per cell type
    # if cell_type=='all'
    com_goal_postrew = com_goal
    perm=[p for ii,p in enumerate(perm) if len(com_goal_postrew[ii])>0]
    rz_perm = [(int(rz[p[0]]),int(rz[p[1]])) for p in perm] 
    rz_perm=[p for jj,p in enumerate(rz_perm) if len(com_goal_postrew[jj])>0]
    com_goal_postrew=[com for com in com_goal_postrew if len(com)>0]

    if len(com_goal_postrew)>0:
        goal_cells = intersect_arrays(*com_goal_postrew); 
    else:
        goal_cells=[]
    return goal_cells, com_goal_postrew, perm, rz_perm


def goal_cell_shuffle(coms_correct, goal_window, perm, num_iterations = 1000):
    # also give perm from original non zero perm combinations
    # get shuffled iterations
    shuffled_dist = np.zeros((num_iterations))
    # max of 5 epochs = 10 perms
    goal_cell_shuf_ps_per_comp = np.ones((num_iterations,10))*np.nan
    goal_cell_shuf_ps = []
    for i in range(num_iterations):
        shufs = [list(range(coms_correct[ii].shape[0])) for ii in range(1, len(coms_correct))]
        [random.shuffle(shuf) for shuf in shufs]
        # first com is as ep 1, others are shuffled cell identities
        com_shufs = np.zeros_like(coms_correct); com_shufs[0,:] = coms_correct[0]
        com_shufs[1:1+len(shufs),:] = [coms_correct[ii][np.array(shufs)[ii-1]] for ii in range(1, 1+len(shufs))]
        # OR shuffle cell identities relative to reward
        coms_rewrel = np.array([com-np.pi for ii, com in enumerate(com_shufs)])             
        # perm = list(combinations(range(len(coms_correct)), 2))     
        com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
        # get goal cells across all epochs
        com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
        com_goal_postrew=com_goal
        goal_cells_shuf_p_per_comparison = [len(xx)/len(coms_correct[0]) for xx in com_goal]
        perm=[p for ii,p in enumerate(perm) if len(com_goal_postrew[ii])>0]
        com_goal_postrew=[com for com in com_goal_postrew if len(com)>0]
        if len(com_goal_postrew)>0:
            goal_cells_shuf = intersect_arrays(*com_goal_postrew); 
        else:
            goal_cells_shuf=[]
        shuffled_dist[i] = len(goal_cells_shuf)/len(coms_correct[0])
        goal_cell_shuf_p=len(goal_cells_shuf)/len(com_shufs[0])
        goal_cell_shuf_ps.append(goal_cell_shuf_p)
        goal_cell_shuf_ps_per_comp[i, :len(goal_cells_shuf_p_per_comparison)] = goal_cells_shuf_p_per_comparison
    
    return goal_cell_shuf_ps_per_comp, goal_cell_shuf_ps, shuffled_dist

def goal_cell_shuffle_time(coms_rew, coms_place,coms_correct, goal_window, perm, place_window=20,
            num_iterations = 1000):
    """
    exclude reward cells
    """
    # also give perm from original non zero perm combinations
    # get shuffled iterations
    shuffled_dist = np.zeros((num_iterations))
    # max of 5 epochs = 10 perms
    goal_cell_shuf_ps_per_comp = np.ones((num_iterations,10))*np.nan
    goal_cell_shuf_ps = []
    for i in range(num_iterations):
        shufs = [list(range(coms_correct[ii].shape[0])) for ii in range(1, len(coms_correct))]
        [random.shuffle(shuf) for shuf in shufs]
        # first com is as ep 1, others are shuffled cell identities
        com_shufs = np.zeros_like(coms_correct); com_shufs[0,:] = coms_correct[0]
        com_shufs[1:1+len(shufs),:] = [coms_correct[ii][np.array(shufs)[ii-1]] for ii in range(1, 1+len(shufs))]
        # OR shuffle cell identities relative to reward
        coms_rewrel = np.array([com-np.pi for ii, com in enumerate(com_shufs)])             
        # perm = list(combinations(range(len(coms_correct)), 2))     
        com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
        # get goal cells across all epochs
        com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
        com_goal_postrew=com_goal
        goal_cells_shuf_p_per_comparison = [len(xx)/len(coms_correct[0]) for xx in com_goal]
        perm=[p for ii,p in enumerate(perm) if len(com_goal_postrew[ii])>0]
        com_goal_postrew=[com for com in com_goal_postrew if len(com)>0]
        if len(com_goal_postrew)>0:
            goal_cells_shuf = intersect_arrays(*com_goal_postrew); 
        else:
            goal_cells_shuf=[]
        ############################
        # do same for rew cells
        shufs = [list(range(coms_rew[ii].shape[0])) for ii in range(1, len(coms_rew))]
        [random.shuffle(shuf) for shuf in shufs]
        # first com is as ep 1, others are shuffled cell identities
        com_shufs = np.zeros_like(coms_rew); com_shufs[0,:] = coms_rew[0]
        com_shufs[1:1+len(shufs),:] = [coms_rew[ii][np.array(shufs)[ii-1]] for ii in range(1, 1+len(shufs))]
        # OR shuffle cell identities relative to reward
        coms_rewrel = np.array([com-np.pi for ii, com in enumerate(com_shufs)])             
        # perm = list(combinations(range(len(coms_rew)), 2))     
        com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
        # get goal cells across all epochs
        com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
        com_goal_postrew=com_goal
        goal_cells_shuf_p_per_comparison = [len(xx)/len(coms_rew[0]) for xx in com_goal]
        perm=[p for ii,p in enumerate(perm) if len(com_goal_postrew[ii])>0]
        com_goal_postrew=[com for com in com_goal_postrew if len(com)>0]
        if len(com_goal_postrew)>0:
            rew_cells = intersect_arrays(*com_goal_postrew); 
        else:
            rew_cells=[]
        ############################
        # do same for place cells
        shufs = [list(range(coms_place[ii].shape[0])) for ii in range(1, len(coms_place))]
        [random.shuffle(shuf) for shuf in shufs]
        # first com is as ep 1, others are shuffled cell identities
        com_shufs = np.zeros_like(coms_place); com_shufs[0,:] = coms_place[0]
        com_shufs[1:1+len(shufs),:] = [coms_rew[ii][np.array(shufs)[ii-1]] for ii in range(1, 1+len(shufs))]
        # get cells that maintain their coms across at least 2 epochs
        perm = list(combinations(range(len(com_shufs)), 2))     
        com_per_ep = np.array([(com_shufs[perm[jj][0]]-com_shufs[perm[jj][1]]) for jj in range(len(perm))])        
        compc = [np.where((comr<place_window) & (comr>-place_window))[0] for comr in com_per_ep]
        # get cells across all epochs that meet crit
        if len(compc)>0:
            pcs_all = intersect_arrays(*compc)
        else:
            pcs_all=[]
        # remove rew and place cells
        ############################
        goal_cells_shuf = [xx for xx in goal_cells_shuf if xx not in rew_cells and xx not in pcs_all]
        shuffled_dist[i] = len(goal_cells_shuf)/len(coms_correct[0])
        goal_cell_shuf_p=len(goal_cells_shuf)/len(com_shufs[0])
        goal_cell_shuf_ps.append(goal_cell_shuf_p)
        goal_cell_shuf_ps_per_comp[i, :len(goal_cells_shuf_p_per_comparison)] = goal_cells_shuf_p_per_comparison

    return goal_cell_shuf_ps_per_comp, goal_cell_shuf_ps, shuffled_dist


def get_trialtypes(trialnum, rewards, ybinned, coms_correct, eps):
    
    per_ep_trialtypes = []
    
    for i in range(len(coms_correct)):
        eprng = np.arange(eps[i],eps[i+1])
        eprng = eprng[ybinned[eprng]>2] # exclude dark time
        trialnum_ep = np.array(trialnum)[eprng]        
        rewards_ep = np.array(rewards)[eprng]
        unique_trials = np.array([xx for xx in np.unique(trialnum_ep) if np.sum(trialnum_ep==xx)>100])
        
        init_fails = [] # initial failed trials
        first_correct = []
        correct_trials_besides_first = []  # success trials
        inbtw_fails = []  # failed trials

        for tt, trial in enumerate(unique_trials):
            if trial >= 3:  # Exclude probe trials
                trial_indices = trialnum_ep == trial
                if np.any(rewards_ep[trial_indices] == 1):                    
                    if trial>3:
                        correct_trials_besides_first.append(trial)
                    else:
                        first_correct.append(trial)
                elif trial==3:                    
                    init_fails.append(trial)
                else:
                    inbtw_fails.append(trial)
                                
        total_trials = np.sum(unique_trials)
        per_ep_trialtypes.append([init_fails, first_correct, correct_trials_besides_first, 
                inbtw_fails, total_trials])
        
    return per_ep_trialtypes
    

def get_days_from_cellreg_log_file(txtpth):
    # Specify the path to your text file
    # Read the file content into a string
    with open(txtpth, 'r') as file:
        data = file.read()

    # Split the data into lines
    lines = data.strip().split('\n')

    # Regular expression pattern to extract session number and day number
    pattern = r'Session (\d+) - .*_day(\d+)_'

    # List to hold the extracted session and day numbers
    sessions = []; days = []

    # Extract session and day numbers using regex
    for line in lines:
        match = re.search(pattern, line)
        if match:
            session_number = match.group(1)
            day_number = match.group(2)
            sessions.append(int(session_number))
            days.append(int(day_number))

    return sessions, days

def find_log_file(pth):
    """for cell track logs

    Args:
        pth (_type_): _description_
    """
    # Find the first file that matches the criteria
    matching_file = None
    for filename in os.listdir(pth):
        if filename.startswith('logFile') and filename.endswith('.txt'):
            matching_file = filename
            break
    
    return matching_file

def get_tracking_vars_wo_dff(params_pth):                
    print(params_pth)
    fall = scipy.io.loadmat(params_pth, variable_names=['changeRewLoc', 
        'ybinned', 'VR', 'forwardvel', 
        'trialnum', 'rewards', 'iscell', 'bordercells', 'dFF'])
    # to remove skew cells
    dFF = fall['dFF']
    suite2pind = np.arange(fall['iscell'][:,0].shape[0])
    dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
    suite2pind_remain = suite2pind[((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
    # we need to find cells to map back to suite2p indexes
    skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
    suite2pind_remain = suite2pind_remain[skew>2]
    VR = fall['VR'][0][0][()]
    scalingf = VR['scalingFACTOR'][0][0]
    # mainly for e145
    try:
            rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf        
    except:
            rewsize = 10
    ybinned = fall['ybinned'][0]/scalingf;track_length=180/scalingf    
    forwardvel = fall['forwardvel'][0]    
    changeRewLoc = np.hstack(fall['changeRewLoc']); trialnum=fall['trialnum'][0]
    rewards = fall['rewards'][0]
    # set vars
    eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf
    eps = np.append(eps, len(changeRewLoc))
    
    return dFF, suite2pind_remain, VR, scalingf, rewsize, ybinned, forwardvel, changeRewLoc,\
        rewards, eps, rewlocs, track_length


def get_tracking_vars(params_pth):                
    print(params_pth)
    fall = scipy.io.loadmat(params_pth, variable_names=['changeRewLoc', 
        'ybinned', 'VR', 'forwardvel', 
        'trialnum', 'rewards', 'iscell', 'bordercells', 'dFF','putative_pcs'])
    pcs = np.vstack(np.array(fall['putative_pcs'][0]))
    # to remove skew cells
    dFF = fall['dFF']
    suite2pind = np.arange(fall['iscell'][:,0].shape[0])
    dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool) & ~(fall['bordercells'][0].astype(bool)))]
    suite2pind_remain = suite2pind[((fall['iscell'][:,0]).astype(bool) & ~(fall['bordercells'][0].astype(bool)))]
    # we need to find cells to map back to suite2p indexes
    skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
    pc_bool = np.sum(pcs,axis=0)>0
    suite2pind_remain = suite2pind_remain[((skew>2)&pc_bool)]
    VR = fall['VR'][0][0][()]
    scalingf = VR['scalingFACTOR'][0][0]
    # mainly for e145
    try:
            rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf        
    except:
            rewsize = 10
    ybinned = fall['ybinned'][0]/scalingf;track_length=180/scalingf    
    forwardvel = fall['forwardvel'][0]    
    changeRewLoc = np.hstack(fall['changeRewLoc']); trialnum=fall['trialnum'][0]
    rewards = fall['rewards'][0]
    # set vars
    eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf
    eps = np.append(eps, len(changeRewLoc))
    
    return dFF, suite2pind_remain, VR, scalingf, rewsize, ybinned, forwardvel, changeRewLoc,\
        rewards, eps, rewlocs, track_length

def get_tracked_lut(celltrackpth, animal, pln):
    
    tracked_lut = scipy.io.loadmat(os.path.join(celltrackpth, 
    rf"{animal}_daily_tracking_plane{pln}\Results\commoncells_once_per_week.mat"))
    tracked_lut = tracked_lut['commoncells_once_per_week'].astype(int)
    # CHANGE INDEX TO MATCH SUITE2P INDEX!! -1!!!
    tracked_lut = tracked_lut-1
    # find day match with session        
    txtpth = os.path.join(celltrackpth, rf"{animal}_daily_tracking_plane{pln}\Results")
    txtpth = os.path.join(txtpth, find_log_file(txtpth))
    sessions, days = get_days_from_cellreg_log_file(txtpth)    
    tracked_lut = pd.DataFrame(tracked_lut, columns = days)

    return tracked_lut, days

def get_shuffled_goal_cell_indices(rewlocs, coms_correct, goal_window, suite2pind_remain,
                num_iterations = 1000):
    # get shuffled iterations
    shuffled_dist = np.zeros((num_iterations))
    # max of 5 epochs = 10 perms
    goal_cells_shuf_s2pind = []; coms_rewrels = []
    goal_cell_shuf_ps_per_comp = np.ones((num_iterations,10))*np.nan; goal_cell_shuf_ps = []
    for i in range(num_iterations):
        # shuffle locations
        rewlocs_shuf = rewlocs #[random.randint(100,250) for iii in range(len(eps))]
        shufs = [list(range(coms_correct[ii].shape[0])) for ii in range(1, 
                len(coms_correct))]
        [random.shuffle(shuf) for shuf in shufs]
        # first com is as ep 1, others are shuffled cell identities
        com_shufs = np.zeros_like(coms_correct)
        com_shufs[0,:] = coms_correct[0]
        com_shufs[1:1+len(shufs),:] = [coms_correct[ii][np.array(shufs)[ii-1]] for ii in range(1, 
                                            1+len(shufs))]
        # OR shuffle cell identities
        # relative to reward
        coms_rewrel = np.array([com-np.pi for ii, com in enumerate(com_shufs)])             
        perm = list(combinations(range(len(coms_correct)), 2))     
        com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) 
                    for jj in range(len(perm))])        
        # get goal cells across all epochs
        com_goal = [np.where((comr<goal_window) & 
                (comr>-goal_window))[0] for comr in com_remap]     
        # # (com near goal)
        # com_goal = [[xx for xx in com if ((np.nanmedian(coms_rewrel[:,
        #     xx], axis=0)<=np.pi/4) & (np.nanmedian(coms_rewrel[:,
        #     xx], axis=0)>0))] for com in com_goal if len(com)>0]
        goal_cells = intersect_arrays(*com_goal)
        if len(goal_cells)>0:
            goal_cells_s2p_ind = suite2pind_remain[goal_cells]
        else:
            goal_cells_s2p_ind = []
        goal_cells_shuf_s2pind.append(goal_cells_s2p_ind)
        coms_rewrels.append(coms_rewrel)
    return goal_cells_shuf_s2pind, coms_rewrels

import numpy as np

def normalize_values(values, b, c, track_length,dark_time_pos=3):
    """
    bo's function to normalize values from -1 to 1, with 0 at reward loc
    """
    # values are a list of positions
    # b is the reward start location
    # c is the reward end location
    values = np.array(values)
    normalized = np.zeros_like(values, dtype=float)
    # I remove dark time (position smaller than 5)
    mask1 = (values >= dark_time_pos) & (values <= b)
    normalized[mask1] = np.interp(values[mask1], [dark_time_pos, b], [-1, 0])
    # Values in (b, c) becomes 0, where I collapse all the reward zone into one point.
    mask2 = (values > b) & (values < c)
    normalized[mask2] = 0
    # Normalize values in [c, 180] to [0, 1], considering a 180 track, you can simply change 180 to 270 
    mask3 = (values >= c) & (values <= track_length)
    normalized[mask3] = np.interp(values[mask3], [c, track_length], [0, 1])

    return normalized
# Test
# a_values = [10, 20, 50, 100, 120, 150]  # Example sorted values
# b = 30
# c = 90
# normalized_values = normalize_values(a_values, b, c)
# print(normalized_values)

 
def get_shuffled_goal_cell_indices_com_away_from_goal(rewlocs, coms_correct, goal_window, suite2pind_remain,
                num_iterations = 1000):
    # get shuffled iterations
    shuffled_dist = np.zeros((num_iterations))
    # max of 5 epochs = 10 perms
    goal_cells_shuf_s2pind = []; coms_rewrels = []
    goal_cell_shuf_ps_per_comp = np.ones((num_iterations,10))*np.nan; goal_cell_shuf_ps = []
    for i in range(num_iterations):
        if i%1000==0: print(f'shuffle number: {i}')
        # shuffle locations
        rewlocs_shuf = rewlocs #[random.randint(100,250) for iii in range(len(eps))]
        shufs = [list(range(coms_correct[ii].shape[0])) for ii in range(1, 
                len(coms_correct))]
        [random.shuffle(shuf) for shuf in shufs]
        # first com is as ep 1, others are shuffled cell identities
        com_shufs = np.zeros_like(coms_correct)
        com_shufs[0,:] = coms_correct[0]
        com_shufs[1:1+len(shufs),:] = [coms_correct[ii][np.array(shufs)[ii-1]] for ii in range(1, 
                                            1+len(shufs))]
        # OR shuffle cell identities
        # relative to reward
        coms_rewrel = np.array([com-np.pi for ii, com in enumerate(com_shufs)])             
        perm = list(combinations(range(len(coms_correct)), 2))     
        com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) 
                    for jj in range(len(perm))])        
        # get goal cells across all epochs
        com_goal = [np.where((comr<goal_window) & 
                (comr>-goal_window))[0] for comr in com_remap]     
        # (com away from goal)
        com_goal_pr = [[xx for xx in com if ((np.nanmedian(coms_rewrel[:,
            xx], axis=0)<=np.pi/2) & (np.nanmedian(coms_rewrel[:,
            xx], axis=0)>0))] for com in com_goal if len(com)>0]
        goal_cells = intersect_arrays(*com_goal)
        goal_cells_pr = intersect_arrays(*com_goal_pr)
        goal_cells = [xx for xx in goal_cells if xx not in goal_cells_pr]
        if len(goal_cells)>0:
            goal_cells_s2p_ind = suite2pind_remain[goal_cells]
        else:
            goal_cells_s2p_ind = []
        goal_cells_shuf_s2pind.append(goal_cells_s2p_ind)
        coms_rewrels.append(coms_rewrel)
    return goal_cells_shuf_s2pind, coms_rewrels

def get_reward_cells_that_are_tracked(tracked_lut, goal_cells_s2p_ind, 
        animal, day,  suite2pind_remain):
    tracked_rew_cell_ind = [ii for ii,xx in enumerate(tracked_lut[day].values) if xx in goal_cells_s2p_ind]
    tracked_cells_that_are_rew_pyr_id = tracked_lut[day].values[tracked_rew_cell_ind]
    rew_cells_that_are_tracked_iind = np.array([np.where(suite2pind_remain==xx)[0][0] for xx in tracked_cells_that_are_rew_pyr_id])
    
    return tracked_rew_cell_ind, rew_cells_that_are_tracked_iind
                
def create_mask_from_coordinates(coordinates, image_shape):
    """Creates a mask from a list of coordinates.

    Args:
        coordinates: A list of (x, y) coordinates defining the region to mask.
        image_shape: A tuple (height, width) defining the shape of the image.

    Returns:
        A numpy array representing the mask, where 1 indicates the masked region.
    """

    mask = np.zeros(image_shape, dtype=np.uint8)
    height,width=image_shape

    # Create a polygon from the coordinates
    polygon = np.array(coordinates, dtype=np.int32)
    polygon = polygon.reshape((-1, 1, 2))

    # Fill the polygon with 1s
    cv2.fillPoly(mask, [polygon], 1)
    # Find contours of the filled mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask to draw the contours
    contour_mask = np.zeros((height, width), dtype=np.uint8)

    # Draw the detected contours on the mask
    cv2.drawContours(contour_mask, contours, -1, (255, 255, 255), 1)
    # Calculate the moments of the contours to find the center
    M = cv2.moments(contour_mask)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    return mask,contour_mask,(cX,cY)
# Function to compute pairwise distances
def pairwise_distances(points):
    n = len(points)
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            dist = distance.euclidean(points[i], points[j])
            distances[i, j] = dist
            distances[j, i] = dist
    
    return distances


def per_trial_dff(reward_cell_type,ii,params_pth,radian_alignment_saved,
    animal,day,bins,goal_cm_window=20):
    """
    changed on 2/6/25 to make it more consistent with splitting the different
    subpopulations
    """
    print(params_pth)
    fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
            'pyr_tc_s2p_cellind', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
            'stat', 'timedFF', 'licks'])
    VR = fall['VR'][0][0][()]
    scalingf = VR['scalingFACTOR'][0][0]
    try:
        rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf        
    except:
        rewsize = 10
    ybinned = fall['ybinned'][0]/scalingf
    track_length=180/scalingf    
    forwardvel = fall['forwardvel'][0]    
    changeRewLoc = np.hstack(fall['changeRewLoc'])
    trialnum=fall['trialnum'][0]
    rewards = fall['rewards'][0]
    lick=fall['licks'][0]
    if animal=='e145':
        ybinned=ybinned[:-1]
        forwardvel=forwardvel[:-1]
        changeRewLoc=changeRewLoc[:-1]
        trialnum=trialnum[:-1]
        rewards=rewards[:-1]
        lick=lick[:-1]
    # set vars
    eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
    lasttr=8 # last trials
    bins=90
    rad = get_radian_position_first_lick_after_rew(eps, ybinned, lick, rewards, rewsize,rewlocs,
                    trialnum, track_length) # get radian coordinates
    track_length_rad = track_length*(2*np.pi/track_length)
    bin_size=track_length_rad/bins 
    rz = get_rewzones(rewlocs,1/scalingf)   
    # added to get anatomical info
    # takes time
    fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
    Fc3 = fall_fc3['Fc3']
    dFF = fall_fc3['dFF']
    Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
    dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]
    skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
    Fc3 = Fc3[:, skew>2] # only keep cells with skew greateer than 2
    dFF = dFF[:, skew>2] 
    if f'{animal}_{day:03d}_index{ii:03d}' in radian_alignment_saved.keys():
        tcs_correct, coms_correct, tcs_fail, coms_fail, \
        com_goal, goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av = radian_alignment_saved[f'{animal}_{day:03d}_index{ii:03d}']            
    else:# remake tuning curves relative to reward        
        # 9/19/24
        # find correct trials within each epoch!!!!
        tcs_correct, coms_correct, tcs_fail, coms_fail = make_tuning_curves_radians_by_trialtype(eps,rewlocs,ybinned,rad,Fc3,trialnum,
        rewards,forwardvel,rewsize,bin_size)          
    # fall_stat = scipy.io.loadmat(params_pth, variable_names=['stat','ops'])
    # ops = fall_stat['ops']
    # stat = fall_stat['stat']
    # meanimg=np.squeeze(ops)[()]['meanImg']
    # s2p_iind = np.arange(stat.shape[1])
    # s2p_iind_filter = s2p_iind[(fall['iscell'][:,0]).astype(bool)]
    # s2p_iind_filter = s2p_iind_filter[skew>2]
    goal_window = goal_cm_window*(2*np.pi/track_length) # cm converted to rad
    # change to relative value 
    coms_rewrel = np.array([com-np.pi for com in coms_correct])
    perm = list(combinations(range(len(coms_correct)), 2)) 
    rz_perm = [(int(rz[p[0]]),int(rz[p[1]])) for p in perm]   
    # if 4 ep
    # account for cells that move to the end/front
    # Define a small window around pi (e.g., epsilon)
    epsilon = .7 # 20 cm
    # Find COMs near pi and shift to -pi
    com_loop_w_in_window = []
    for pi,p in enumerate(perm):
        for cll in range(coms_rewrel.shape[1]):
            com1_rel = coms_rewrel[p[0],cll]
            com2_rel = coms_rewrel[p[1],cll]
            # print(com1_rel,com2_rel,com_diff)
            if ((abs(com1_rel - np.pi) < epsilon) and 
            (abs(com2_rel + np.pi) < epsilon)):
                    com_loop_w_in_window.append(cll)
    # get abs value instead
    coms_rewrel[:,com_loop_w_in_window]=abs(coms_rewrel[:,com_loop_w_in_window])
    com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
    com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
    # in addition, com near but after goal
    lowerbound = -np.pi/4
    if reward_cell_type=='pre':
        com_goal_postrew = [[xx for xx in com if ((np.nanmedian(coms_rewrel[:,
            xx], axis=0)>=lowerbound) & (np.nanmedian(coms_rewrel[:,
            xx], axis=0)<0))] if len(com)>0 else [] for com in com_goal]
    else:
        com_goal_postrew = [[xx for xx in com if ((np.nanmedian(coms_rewrel[:,
            xx], axis=0)>=0) & (np.nanmedian(coms_rewrel[:,
            xx], axis=0)<abs(lowerbound)))] if len(com)>0 else [] for com in com_goal]
    #only get perms with non zero cells
    perm=[p for ii,p in enumerate(perm) if len(com_goal_postrew[ii])>0]
    rz_perm=[p for ii,p in enumerate(rz_perm) if len(com_goal_postrew[ii])>0]
    com_goal_postrew=[com for com in com_goal_postrew if len(com)>0]
    assert sum([len(xx) for xx in com_goal])>=sum([len(xx) for xx in com_goal_postrew])
    epoch_perm=[perm,rz_perm]
    # get goal cells across all epochs   
    if len(com_goal_postrew)>0:
        goal_cells = intersect_arrays(*com_goal_postrew); 
    else:
        goal_cells=[]
    # get goal cells dff per trial (per cell)
    # eg
    # cells = [f'cell{gc:04d}' for gc in goal_cells]
    # ep1 = [[v for k,v in trial_dff.items() if cll in k and 'ep1' in k] for cll in cells] 
    # Flatten the list and count occurrences
    counter = Counter(elem for sublist in com_goal_postrew for elem in set(sublist))
    # Get elements that appear in at least two different sublists
    com_goal_subset = [key for key, count in counter.items() if count >= len(eps)-2]
    # get goal cells dff per trial (per cell)
    # keep both dedicated and loosely dedicated cells for now
    trial_dff = {}; trialstates={}
    for gc in goal_cells:    
        for ep in range(len(eps)-1):
            eprng = np.arange(eps[ep],eps[ep+1])
            success, fail, str_trials, ftr_trials, ttr, \
            total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
            goal_cells_Fc = Fc3[eprng, gc]        
            trials = trialnum[eprng]
            for trial in np.unique(trials):
                if trial>3:
                    trial_dff[f"cell{gc:04d}_trial{trial:03d}_ep{ep+1}"] = np.nanmean(goal_cells_Fc[trials==trial])
            trialstate=np.zeros_like(ttr[ttr>3])
            trialstate[[xx in str_trials for xx in ttr[ttr>3]]]=1
            trialstates[f'ep{ep+1}']=trialstate

    return trial_dff,trialstates,com_goal,com_goal_subset,goal_cells,\
            epoch_perm


# %%

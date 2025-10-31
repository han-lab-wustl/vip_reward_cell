"""
jan 2025
generate circular statistics
"""

import scipy, numpy as np
from projects.pyr_reward.rewardcell import get_radian_position_first_lick_after_rew,get_rewzones,\
    normalize_values
from projects.opto.behavior.behavior import get_success_failure_trials
from projects.pyr_reward.placecell import intersect_arrays,make_tuning_curves_radians_by_trialtype,\
    consecutive_stretch,make_tuning_curves_radians_trial_by_trial,make_tuning_curves,\
        make_tuning_curves_warped
from itertools import combinations, chain

def compute_circular_stats(tuning_curve, positions, track_length):
    """
    Computes the circular mean and resultant vector length (a measure of variance)
    for a tuning curve along a circular track.

    Parameters:
    - tuning_curve: Array of firing rates at different positions.
    - positions: Array of positions along the track.
    - track_length: Total length of the circular track.

    Returns:
    - mean_angle (radians): Circular mean of the firing field.
    - resultant_vector_length (R): Measure of concentration (ranges from 0 to 1).
    """
    # Convert positions to angles (radians)
    angles = (2 * np.pi * positions) / track_length
    # Normalize tuning curve (acts as probability distribution)
    weights = tuning_curve / np.sum(tuning_curve)
    # Compute weighted circular mean
    x_mean = np.sum(weights * np.cos(angles))
    y_mean = np.sum(weights * np.sin(angles))
    mean_angle = np.arctan2(y_mean, x_mean)  # Circular mean in radians
    # Compute resultant vector length (R) as a measure of variance
    R = np.sqrt(x_mean**2 + y_mean**2)
    
    return mean_angle, R

def compute_circular_stats_rad(tuning_curve, positions):
    """
    Computes the circular mean and resultant vector length (a measure of variance)
    for a tuning curve along a circular track.

    Parameters:
    - tuning_curve: Array of firing rates at different positions.
    - positions: Array of positions along the track.
    - track_length: Total length of the circular track.

    Returns:
    - mean_angle (radians): Circular mean of the firing field.
    - resultant_vector_length (R): Measure of concentration (ranges from 0 to 1).
    """
    # Convert positions to angles (radians)
    angles = positions
    # Normalize tuning curve (acts as probability distribution)
    weights = tuning_curve / np.sum(tuning_curve)
    # Compute weighted circular mean
    x_mean = np.sum(weights * np.cos(angles))
    y_mean = np.sum(weights * np.sin(angles))
    mean_angle = np.arctan2(y_mean, x_mean)  # Circular mean in radians
    # Compute resultant vector length (R) as a measure of variance
    R = np.sqrt(x_mean**2 + y_mean**2)
    
    return mean_angle, R

def get_circular_data(ii,params_pth,animal,day,bins,radian_alignment,
    radian_alignment_saved,goal_cm_window,pdf,epoch_perm,goal_cell_iind,goal_cell_prop,num_epochs,
    goal_cell_null,pvals,total_cells,
    num_iterations=1000):
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
    
    fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
    Fc3 = fall_fc3['Fc3']
    dFF = fall_fc3['dFF']
    Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
    dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]
    skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
    Fc3 = Fc3[:, skew>2] # only keep cells with skew greater than 2
    # get tuning curves trial by trial and get calculate radians
    if f'{animal}_{day:03d}_index{ii:03d}' in radian_alignment_saved.keys():
        tcs_correct, coms_correct, tcs_fail, coms_fail, \
        com_goal, goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av = radian_alignment_saved[f'{animal}_{day:03d}_index{ii:03d}']            
    else:# remake tuning curves relative to reward        
        # 9/19/24
        # find correct trials within each epoch!!!!
        tcs_correct, coms_correct, tcs_fail, coms_fail = make_tuning_curves_radians_by_trialtype(eps,rewlocs,ybinned,rad,Fc3,trialnum,
        rewards,forwardvel,rewsize,bin_size)      
    # allocentric ref
    bin_size=track_length/bins 
    tcs_correct_abs, coms_correct_abs = make_tuning_curves(eps,rewlocs,ybinned,
        Fc3,trialnum,rewards,forwardvel,
        rewsize,bin_size)
    # binsize = 2/90 bc track is essentially size 2 (-1 to 1)
    tcs_correct_abs_warped, coms_correct_abs_warped = make_tuning_curves_warped(eps,rewlocs,
            norm_pos,Fc3,trialnum,
            rewards,forwardvel,rewsize,2/bins,lasttr=8,bins=90,
            velocity_filter=False)
    # check to see if there is activity in all 3 epochs
    clls_to_keep=[]
    maxthres=.2
    for ep in range(len(tcs_correct)):
        act = np.nanmax(tcs_correct[ep],axis=1)
        clls_to_keep.append(np.where(act>maxthres)[0])
    clls_to_keep=intersect_arrays(*clls_to_keep)
    # norm to -1 to 1
    coms_correct_abs_warped = np.array([com-1 for com in coms_correct_abs_warped])
    tcs_abs_mean = np.nansum(tcs_correct_abs[:,clls_to_keep,:],axis=0)
    com_abs_mean = np.nanmean(coms_correct_abs[:,clls_to_keep],axis=0)
    # keep all cells
    com_abs_mean=coms_correct_abs[:,clls_to_keep]
    tcs_warped_mean = np.nansum(tcs_correct_abs_warped[:,clls_to_keep,:],axis=0)
    com_warped_mean = np.nanmean(coms_correct_abs_warped[:,clls_to_keep],axis=0)
    com_warped_mean=coms_correct_abs_warped[:,clls_to_keep]
    # tc mean across epochs    
    tc_mean = np.nansum(tcs_correct[:,clls_to_keep,:],axis=0)
    # first get goal cells
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
    #average after looping
    com_mean_rewrel = np.nanmean(coms_rewrel[:,clls_to_keep],axis=0)   
    com_mean_rewrel=coms_rewrel[:,clls_to_keep] 
    rad_binned = np.linspace(0, 2*np.pi, bins)
    # compute circular statistics
    meanangles_rad = []; rvals_rad = []
    for cll in range(tc_mean.shape[0]):
        tc = tc_mean[cll,:]
        mean_ang, r = compute_circular_stats_rad(tc, rad_binned)
        meanangles_rad.append(mean_ang); rvals_rad.append(r)
    # allocentric ref
    ypos_binned = np.linspace(0, track_length, bins)
    meanangles_abs = []; rvals_abs = []
    for cll in range(tcs_abs_mean.shape[0]):
        tc = tcs_abs_mean[cll,:]
        mean_ang, r = compute_circular_stats(tc, ypos_binned, track_length)
        meanangles_abs.append(mean_ang); rvals_abs.append(r)
    # warped ref
    warped_binned = np.linspace(-1, 1, bins)
    meanangles_warped = []; rvals_warped = []
    for cll in range(tcs_warped_mean.shape[0]):
        tc = tcs_warped_mean[cll,:]
        mean_ang, r = compute_circular_stats(tc, warped_binned, 2) # track length of 2
        meanangles_warped.append(mean_ang); rvals_warped.append(r)

    return meanangles_abs,rvals_abs,meanangles_rad,rvals_rad,meanangles_warped,rvals_warped,\
        tc_mean,com_mean_rewrel,tcs_abs_mean,com_abs_mean,tcs_warped_mean,com_warped_mean,\
        tcs_correct[:,clls_to_keep,:],tcs_correct_abs[:,clls_to_keep,:]
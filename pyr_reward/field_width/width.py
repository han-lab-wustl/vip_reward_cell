

import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
from scipy.ndimage import gaussian_filter1d
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
# plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.pyr_reward.placecell import make_tuning_curves_radians_by_trialtype, intersect_arrays,\
    make_tuning_curves_by_trialtype_w_darktime,make_tuning_curves
from projects.pyr_reward.rewardcell import get_radian_position,create_mask_from_coordinates,pairwise_distances,extract_data_rewcentric,\
    get_radian_position_first_lick_after_rew, get_rewzones
from projects.pyr_reward.placecell import get_tuning_curve, calc_COM_EH, make_tuning_curves_by_trialtype_w_darktime, make_tuning_curves_time_trial_by_trial_w_darktime, intersect_arrays
from projects.opto.behavior.behavior import get_success_failure_trials, smooth_lick_rate
from scipy.ndimage import gaussian_filter1d
import numpy as np
from scipy.ndimage import gaussian_filter1d
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def count_fields(tuning_curve, smooth_sigma=3, min_height=0, min_distance=5, plot=False):
    """
    Counts the number of place fields in a tuning curve using peak detection.

    Parameters:
        tuning_curve (1D array): The activity across spatial bins.
        smooth_sigma (float): Smoothing parameter for Gaussian filter.
        min_height (float): Minimum height of a peak to count as a field.
        min_distance (int): Minimum distance (in bins) between peaks.
        plot (bool): If True, plots the tuning curve and detected fields.

    Returns:
        n_fields (int): Number of detected fields.
        peaks (array): Indices of detected peaks.
    """
    smoothed = gaussian_filter1d(tuning_curve, sigma=smooth_sigma)
    peaks, _ = find_peaks(smoothed, height=min_height, distance=min_distance)
    
    if plot:
        plt.plot(smoothed, label='Smoothed TC')
        plt.scatter(peaks, smoothed[peaks], color='red', label='Detected Fields')
        plt.title(f'Detected Fields: {len(peaks)}')
        plt.xlabel('Spatial Bin')
        plt.ylabel('Activity')
        plt.legend()
        plt.show()
    
    return len(peaks), peaks

def compute_field_width(Fc3_trial, ybinned_trial, bins=150, smooth_sigma=1.5, threshold_ratio=0.5):
    """
    Compute place field width for each neuron in a single trial.
    
    Parameters:
    - Fc3_trial: np.ndarray of shape (T, N), neural activity over time (T) for N cells
    - ybinned_trial: np.ndarray of shape (T,), position over time
    - bins: number of spatial bins
    - smooth_sigma: std for Gaussian smoothing
    - threshold_ratio: threshold for field width (e.g., 0.5 = half max)
    
    Returns:
    - widths: np.ndarray of shape (N,), field widths in cm (or same units as ybinned)
              NaN for neurons with no activity
    """
    T, N = Fc3_trial.shape
    widths = np.full(N, np.nan)
    bin_edges = np.linspace(np.nanmin(ybinned_trial), np.nanmax(ybinned_trial), bins+1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    tcs = []
    for cell in range(N):
        # Get activity and position
        activity = Fc3_trial[:, cell]
        if np.all(np.isnan(activity)) or np.nanmax(activity) == 0:
            continue

        # Bin activity
        binned_activity = np.zeros(bins)
        counts = np.zeros(bins)
        for i in range(T):
            pos = ybinned_trial[i]
            if np.isnan(pos) or np.isnan(activity[i]):
                continue
            bin_idx = np.digitize(pos, bin_edges) - 1
            if bin_idx < 0 or bin_idx >= bins:
                continue
            binned_activity[bin_idx] += activity[i]
            counts[bin_idx] += 1
        with np.errstate(divide='ignore', invalid='ignore'):
            tuning_curve = np.divide(binned_activity, counts)
        
        # Smooth and normalize
        tuning_curve = gaussian_filter1d(tuning_curve, sigma=smooth_sigma, mode='wrap')
        if np.nanmax(tuning_curve) == 0:
            continue
        norm_curve = tuning_curve / np.nanmax(tuning_curve)

        # Find field width: count bins above threshold
        field_bins = np.where(norm_curve >= threshold_ratio)[0]
        if len(field_bins) == 0:
            continue

        # Handle wrap-around fields
        diff_bins = np.diff(np.concatenate([[field_bins[-1]-bins], field_bins]))
        gaps = np.where(diff_bins > 1)[0]
        if len(gaps) > 0:
            # split into separate fields
            split_fields = np.split(field_bins, gaps + 1)
            main_field = max(split_fields, key=len)
        else:
            main_field = field_bins

        # Width = number of bins × bin size
        bin_size = np.nanmean(np.diff(bin_edges))
        widths[cell] = len(main_field) * bin_size
        tcs.append(tuning_curve)
    return widths,tcs


def check_reward_in_bouts(lick_bout_starts, lick_bout_ends, reward, time):
    """
    Check whether each lick bout contains a reward event.

    Parameters:
        lick_bout_starts (list or np.ndarray): Start times of lick bouts.
        lick_bout_ends (list or np.ndarray): End times of lick bouts.
        reward (np.ndarray): Binary vector, 1 if reward occurs at that time.
        time (np.ndarray): Time vector corresponding to reward.

    Returns:
        np.ndarray of booleans: True if reward is present in the bout.
    """
    reward_in_bout = []
    reward_times = time[reward == 1]

    for start, end in zip(lick_bout_starts, lick_bout_ends):
        in_bout = np.any((reward_times >= start) & (reward_times <= end))
        reward_in_bout.append(in_bout)

    return np.array(reward_in_bout)

def detect_lick_bouts(lickrate, time, threshold=1.0, min_duration=0.2, smooth_sigma_sec=0.2):
    """
    Detects lick bouts from a lick rate vector using a low-pass filter.
    
    Parameters:
        lickrate (np.ndarray): Instantaneous lick rate vector.
        time (np.ndarray): Time vector corresponding to lickrate.
        threshold (float): Minimum lick rate to define a bout (licks/sec).
        min_duration (float): Minimum bout duration (in seconds).
        smooth_sigma_sec (float): Standard deviation of Gaussian filter (in seconds).
        
    Returns:
        bout_starts (list of float): Start times of lick bouts.
        bout_ends (list of float): End times of lick bouts.
    """
    dt = np.median(np.diff(time))  # Time step
    sigma_samples = smooth_sigma_sec / dt

    # Smooth the lickrate
    # skip smoothing
    lickrate_smooth = lickrate#gaussian_filter1d(lickrate.astype(float), sigma=sigma_samples)

    # Thresholding
    above_thresh = lickrate_smooth > threshold
    min_samples = int(min_duration / dt)

    bout_starts = []
    bout_ends = []

    i = 0
    while i < len(above_thresh):
        if above_thresh[i]:
            bout_start_idx = i
            while i < len(above_thresh) and above_thresh[i]:
                i += 1
            bout_end_idx = i - 1
            if (bout_end_idx - bout_start_idx + 1) >= min_samples:
                bout_starts.append(time[bout_start_idx])
                bout_ends.append(time[bout_end_idx])
        else:
            i += 1

    return bout_starts, bout_ends

def check_reward_in_bouts(lick_bout_starts, lick_bout_ends, reward, time):
    """
    Check whether each lick bout contains a reward event.

    Parameters:
        lick_bout_starts (list or np.ndarray): Start times of lick bouts.
        lick_bout_ends (list or np.ndarray): End times of lick bouts.
        reward (np.ndarray): Binary vector, 1 if reward occurs at that time.
        time (np.ndarray): Time vector corresponding to reward.

    Returns:
        np.ndarray of booleans: True if reward is present in the bout.
    """
    reward_in_bout = []
    reward_times = time[reward == 1]

    for start, end in zip(lick_bout_starts, lick_bout_ends):
        in_bout = np.any((reward_times >= start) & (reward_times <= end))
        reward_in_bout.append(in_bout)

    return np.array(reward_in_bout)

def circular_arc_length(start_idx, end_idx, nbins):
    """
    Return the number of bins in the forward arc from start_idx to end_idx
    on a circle of length nbins.  Always ≥1 and ≤nbins.
    """
    # if end comes after start, simple
    if end_idx >= start_idx:
        return end_idx - start_idx + 1
    # if end wrapped around past zero
    else:
        return (nbins - start_idx) + (end_idx + 1)
import numpy as np

def circular_fwhm(tc, bin_size):
    """
    Compute full‐width at half‐maximum on a circular tuning curve tc.
    Returns (width, left_cross, right_cross) where width is in same units
    as bin_size.
    """
    nbins = len(tc)
    peak = np.argmax(tc)
    half = tc[peak]*.2 # 20%

    # mask of bins ≥ half-max
    mask = tc >= half
    if not mask.any():
        return 0.0, None, None
    if mask.all():
        # everywhere above half-max
        return nbins * bin_size, 0, nbins - 1

    # double for wrap detection
    mm = np.concatenate([mask, mask])
    d = np.diff(mm.astype(int))
    starts = np.where(d == 1)[0] + 1
    ends   = np.where(d == -1)[0]

    # pick the run that contains the peak (in either copy)
    candidate = None
    for s, e in zip(starts, ends):
        if (s <= peak < e) or (s <= peak + nbins < e):
            candidate = (s, e)
            break
    if candidate is None:
        # fallback to longest run
        lengths = [e - s + 1 for s, e in zip(starts, ends)]
        idx = np.argmax(lengths)
        candidate = (starts[idx], ends[idx])

    s, e = candidate
    # map back into [0, nbins)
    left  = s  % nbins
    right = e  % nbins

    # compute forward arc length
    length_bins = circular_arc_length(left, right, nbins)
    return length_bins * bin_size, left, right
import numpy as np

def eq_rectangular_width(tc, bin_size):
    """
    Area under tc ÷ peak height → width in same units as bin_size.
    """
    peak = np.nanmax(tc)
    if peak <= 0:
        return 0.0
    area = np.trapz(tc, dx=bin_size)
    return area / peak

def get_pre_post_field_widths(params_pth,animal,day,ii,goal_window_cm=20,bins=90):
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
    lick=fall['licks'][0]
    time=fall['timedFF'][0]
    if animal=='e145':
        ybinned=ybinned[:-1]
        forwardvel=forwardvel[:-1]
        changeRewLoc=changeRewLoc[:-1]
        trialnum=trialnum[:-1]
        rewards=rewards[:-1]
        lick=licks[:-1]
        time=time[:-1]
    # set vars
    eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
    rz = get_rewzones(rewlocs,1/scalingf)       
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
    # tc w/ dark time added to the end of track
    tcs_correct, coms_correct, tcs_fail, coms_fail, ybinned_dt, rad = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,
        rewsize,ybinned,time,lick,
        Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
        bins=bins_dt)  
    goal_window = goal_window_cm*(2*np.pi/track_length) # cm converted to rad
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
    # all cells before 0
    com_goal_postrew = [[xx for xx in com if (np.nanmedian(coms_rewrel[:,
        xx], axis=0)<0)] if len(com)>0 else [] for com in com_goal]
    # get goal cells across all epochs        
    if len(com_goal_postrew)>0:
        goal_cells = intersect_arrays(*com_goal_postrew); 
    else:
        goal_cells=[]
    goal_all = np.unique(np.concatenate(com_goal_postrew)).astype(int)     
    goal_cells_prepost=[];goal_cells_prepost.append(goal_all)   
    # example:
    plt.close('all')
    # rectangular method
    alldf=[]
    widths_per_ep = []
    peak_per_ep = []
    # adjusted binsize to cm?
    # TODO - test only on dedicated cells?
    bin_size = track_length_dt/bins_dt
    for ep in range(len(tcs_correct)):
        sigma = 1   # in bin-units; start small (0.5–2 bins)
        tcs_smoothed = gaussian_filter1d(tcs_correct[ep,goal_all], sigma=sigma, axis=1)
        r = [circular_fwhm(tc, bin_size) for tc in tcs_smoothed]
        w = [xx[0] for xx in r]
        # for large fields?
        w = [eq_rectangular_width(tc, bin_size) for tc in tcs_smoothed]
        # convert to cm
        # w = np.array(w)
        widths_per_ep.append(w)
        peak_per_ep.append([np.quantile(tc,.75) for tc in tcs_correct[ep,goal_all]])
    widths_per_ep=np.array(widths_per_ep)
    df = pd.DataFrame()
    df['width_cm'] = np.concatenate(widths_per_ep)
    df['75_quantile'] = np.concatenate(peak_per_ep)
    df['cellid'] = np.concatenate([goal_all]*len(widths_per_ep))
    df['epoch'] = np.concatenate([[f'epoch{ep+1}_rz{int(rz[ep])}']*len(widths_per_ep[ep]) for ep in range(len(tcs_correct))])        
    df['animal'] = [animal]*len(df)
    df['rewloc'] = np.concatenate([[f'epoch{ep+1}_rewloc{rewlocs[ep]}']*len(widths_per_ep[ep]) for ep in range(len(tcs_correct))])  
    df['day'] = [day]*len(df)
    df['cell_type'] = ['Pre']*len(df)
    try: # eg., suppress after validation
        ii=0
        plt.figure()
        plt.plot(tcs_correct[:,goal_all[ii],:].T)
        plt.title(f"{df.loc[df.cellid==goal_all[ii], 'width_cm'].values}")
        plt.show()
    except Exception as e:
        print(e)
    alldf.append(df)
    
    ################ post reward
    com_goal_postrew = [[xx for xx in com if (np.nanmedian(coms_rewrel[:,
        xx], axis=0)>0)] if len(com)>0 else [] for com in com_goal]
    # get goal cells across all epochs        
    if len(com_goal_postrew)>0:
        goal_cells = intersect_arrays(*com_goal_postrew); 
    else:
        goal_cells=[]
    goal_all = np.unique(np.concatenate(com_goal_postrew)).astype(int)        
    # dedicated
    # goal_all = goal_cells
    widths_per_ep = []
    peak_per_ep = []
    for ep in range(len(tcs_correct)):
        sigma = 1   # in bin-units; start small (0.5–2 bins)
        tcs_smoothed = gaussian_filter1d(tcs_correct[ep,goal_all], sigma=sigma, axis=1)
        r = [circular_fwhm(tc, bin_size) for tc in tcs_smoothed]
        w = [xx[0] for xx in r]
        w = [eq_rectangular_width(tc, bin_size) for tc in tcs_smoothed]
        widths_per_ep.append(w)
        peak_per_ep.append([np.quantile(tc,.75) for tc in tcs_correct[ep,goal_all]])
    widths_per_ep=np.array(widths_per_ep)
    df = pd.DataFrame()
    df['width_cm'] = np.concatenate(widths_per_ep)
    df['75_quantile'] = np.concatenate(peak_per_ep)
    df['cellid'] = np.concatenate([goal_all]*len(widths_per_ep))
    df['epoch'] = np.concatenate([[f'epoch{ep+1}_rz{int(rz[ep])}']*len(widths_per_ep[ep]) for ep in range(len(tcs_correct))])        
    df['rewloc'] = np.concatenate([[f'epoch{ep+1}_rewloc{rewlocs[ep]}']*len(widths_per_ep[ep]) for ep in range(len(tcs_correct))])  
    df['animal'] = [animal]*len(df)
    df['day'] = [day]*len(df)
    df['cell_type'] = ['Post']*len(df)
    goal_cells_prepost.append(goal_all)   
    # try:
    #     ii=0
    #     plt.figure()
    #     plt.plot(tcs_correct[:,goal_all[ii],:].T)
    #     plt.title(f"{df.loc[df.cellid==goal_all[ii], 'width_cm'].values}")
    # except Exception as e:
    #     print(e)
    # plt.show()
    # add add pre and post dfs
    alldf.append(df)
    ################################### PLACE
    tcs_correct_abs, coms_correct_abs,_,__ = make_tuning_curves(eps,rewlocs,ybinned,
        Fc3,trialnum,rewards,forwardvel,
        rewsize,bin_size)
    # get cells that maintain their coms across at least 2 epochs
    place_window = 20 # cm converted to rad                
    perm = list(combinations(range(len(coms_correct_abs)), 2))     
    com_per_ep = np.array([(coms_correct_abs[perm[jj][0]]-coms_correct_abs[perm[jj][1]]) for jj in range(len(perm))])        
    compc = [np.where((comr<place_window) & (comr>-place_window))[0] for comr in com_per_ep]
    # get cells across all epochs that meet crit
    pcs = np.unique(np.concatenate(compc))
    pcs = [xx for xx in pcs if xx not in np.concatenate(goal_cells_prepost)]
    # goal_all = goal_cells
    widths_per_ep = []
    peak_per_ep = []
    for ep in range(len(tcs_correct)):
        sigma = 1   # in bin-units; start small (0.5–2 bins)
        tcs_smoothed = gaussian_filter1d(tcs_correct[ep,pcs], sigma=sigma, axis=1)
        r = [circular_fwhm(tc, bin_size) for tc in tcs_smoothed]
        w = [xx[0] for xx in r]
        w = [eq_rectangular_width(tc, bin_size) for tc in tcs_smoothed]
        widths_per_ep.append(w)
        peak_per_ep.append([np.quantile(tc,.75) for tc in tcs_correct[ep,pcs]])
    widths_per_ep=np.array(widths_per_ep)
    df = pd.DataFrame()
    df['width_cm'] = np.concatenate(widths_per_ep)
    df['75_quantile'] = np.concatenate(peak_per_ep)
    df['cellid'] = np.concatenate([pcs]*len(widths_per_ep))
    df['epoch'] = np.concatenate([[f'epoch{ep+1}_rz{int(rz[ep])}']*len(widths_per_ep[ep]) for ep in range(len(tcs_correct))])        
    df['rewloc'] = np.concatenate([[f'epoch{ep+1}_rewloc{rewlocs[ep]}']*len(widths_per_ep[ep]) for ep in range(len(tcs_correct))])  
    df['animal'] = [animal]*len(df)
    df['day'] = [day]*len(df)
    df['cell_type'] = ['Place']*len(df)    
    alldf.append(df)
    # lick
    pre_lick_locs = []
    pre_lick_times = []
    lick_rates = []
    epoch_labels = []    
    pre_velocities = []

    # Define pre-reward window in seconds and/or cm
    pre_window_s = 12  # seconds before reward
    for ep in range(len(eps)-1):
        eprng = range(eps[ep], eps[ep+1])
        trials = np.unique(trialnum[eprng])
        trialnum_ep = trialnum[eprng]
        tr_range = range(eps[ep], eps[ep+1])
        trials = np.unique(trialnum[tr_range])
        rewards_ep = rewards[eprng]
        lick_ep = lick[eprng]
        success, fail, str_trials, ftr_trials, ttr, \
        total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
        # only use last 8 correct trials
        lasttr=8
        for tr in str_trials[-lasttr:]:
            tr_idx = (trialnum_ep == tr)
            if np.sum(rewards_ep[tr_idx]) == 0:
                continue  # no reward delivered
            time_tr = time[eprng][tr_idx]-time[eprng][tr_idx][0]
            reward_time = time_tr[(rewards_ep[tr_idx] > 0)][0]  # first reward frame
            lick_mask = (lick_ep[tr_idx] > 0) & (time_tr < reward_time) & (time_tr > reward_time - pre_window_s)
            # Get velocity in pre-reward window
            vel_tr = forwardvel[eprng][tr_idx]
            pre_vel_mask = (time_tr < reward_time) & (time_tr > reward_time - pre_window_s)
            if np.any(pre_vel_mask):
                avg_vel = np.nanmean(vel_tr[pre_vel_mask])
            else:
                avg_vel = np.nan
            pre_velocities.append(avg_vel)

            if np.any(lick_mask):
                # Locations (in cm)
                lick_locs = ybinned[eprng][tr_idx][lick_mask]
                lick_times = time_tr[lick_mask]
                pre_lick_locs.append((np.min(lick_locs), np.max(lick_locs)))  # first and last
                pre_lick_times.append((np.min(lick_times), np.max(lick_times)))
                lick_rates.append(np.sum(lick_mask) / pre_window_s)
            else:
                pre_lick_locs.append((np.nan, np.nan))
                pre_lick_times.append((np.nan, np.nan))
                lick_rates.append(np.nan)
            # ... your lick processing here ...
            epoch_labels.append(f'epoch{ep+1}_rz{int(rz[ep])}')
            
    # Convert to DataFrame for easier analysis
    lick_df = pd.DataFrame({
        'first_lick_loc_cm': [x[0] for x in pre_lick_locs],
        'last_lick_loc_cm': [x[1] for x in pre_lick_locs],
        'first_lick_time': [x[0] for x in pre_lick_times],
        'last_lick_time': [x[1] for x in pre_lick_times],
        'lick_rate_hz': lick_rates,
        'avg_velocity_cm_s': pre_velocities,
        'animal': [animal]*len(pre_lick_locs),
        'day': [day]*len(pre_lick_locs),
        'epoch': epoch_labels
    })

    # Optional: return this alongside widths
    return pd.concat(alldf), lick_df


def get_beh_in_field(params_pth,animal,day,ii,goal_window_cm=20,bins=90):
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
    lick=fall['licks'][0]
    time=fall['timedFF'][0]
    if animal=='e145':
        ybinned=ybinned[:-1]
        forwardvel=forwardvel[:-1]
        changeRewLoc=changeRewLoc[:-1]
        trialnum=trialnum[:-1]
        rewards=rewards[:-1]
        lick=licks[:-1]
        time=time[:-1]
    # set vars
    eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
    rz = get_rewzones(rewlocs,1/scalingf)       
    # get average success rate
    rates = []
    for ep in range(len(eps)-1):
        eprng = range(eps[ep],eps[ep+1])
        success, fail, str_trials, ftr_trials, ttr, \
        total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
        rates.append(success/total_trials)
    rate=np.nanmean(np.array(rates))
    # get lick rate 
    fr = 1/31.25
    if animal=='z9' or animal=='e190':
        fr=2/31.25
    lick_rate = smooth_lick_rate(lick, fr)
    rad = get_radian_position_first_lick_after_rew(eps, ybinned, lick, rewards, rewsize, rewlocs,trialnum, track_length) # get radian coordinates
    fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
    Fc3 = fall_fc3['Fc3']
    dFF = fall_fc3['dFF']
    Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
    dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]
    skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
    Fc3 = Fc3[:, skew>2] # only keep cells with skew greateer than 2
    # normal tc        
    # tc w/ dark time
    track_length_dt = 550 # cm estimate based on 99.9% of ypos
    track_length_rad_dt = track_length_dt*(2*np.pi/track_length_dt) # estimate bin for dark time
    bins_dt=150 
    bin_size_dt=track_length_rad_dt/bins_dt # typically 3 cm binswith ~ 475 track length
    tcs_correct, coms_correct, tcs_fail, coms_fail, ybinned_dt,rad = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,rewsize,ybinned,time,lick,Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,bins=bins_dt)    # allocentric
    # takes time
    bin_size=3
    tcs_correct_abs, coms_correct_abs,tcs_fail_abs, coms_fail_abs = make_tuning_curves(eps,rewlocs,ybinned,
    Fc3,trialnum,rewards,forwardvel,
    rewsize,bin_size) # last 8 trials
    goal_window = goal_window_cm*(2*np.pi/track_length) # cm converted to rad
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
    # all cells before 0 near reward
    com_goal_postrew = [[xx for xx in com if ((np.nanmedian(coms_rewrel[:,xx], axis=0)<0) & (np.nanmedian(coms_rewrel[:,xx], axis=0)>-np.pi/4))] if len(com)>0 else [] for com in com_goal]
    # get goal cells across all epochs        
    if len(com_goal_postrew)>0:
        goal_cells = intersect_arrays(*com_goal_postrew); 
    else:
        goal_cells=[]
    goal_all = np.unique(np.concatenate(com_goal_postrew)).astype(int)   
    # place     
    place_window = 20 # cm converted to rad                
    perm = list(combinations(range(len(coms_correct_abs)), 2))     
    com_per_ep = np.array([(coms_correct_abs[perm[jj][0]]-coms_correct_abs[perm[jj][1]]) for jj in range(len(perm))])        
    compc = [np.where((comr<place_window) & (comr>-place_window))[0] for comr in com_per_ep]
    # get cells across all epochs that meet crit
    pcs = np.unique(np.concatenate(compc)).astype(int)       
    # exclude goal cells that are place and place that are goal
    # get all goal
    pre_post_goal_all = np.unique(np.concatenate(com_goal)).astype(int)   
    pcs = [xx for xx in pcs if xx not in pre_post_goal_all]
    goal_all = [xx for xx in goal_all if xx not in pcs]    
    ########################## COLLECT RESULTS ##########################
    # place
    plt.close('all')
    # rectangular method
    alldf=[]
    widths_per_ep = []
    field_velocities_per_ep = []
    field_lr_per_ep = []
    # adjusted binsize to cm?
    # TODO - test only on dedicated cells?
    # PLACE
    bin_size = 3
    cells_quant = []
    for ep in range(len(tcs_correct_abs)):
        field_velocities=[];field_lrs=[]
        sigma = 1   # in bin-units; start small (0.5–2 bins)
        tcs_smoothed = gaussian_filter1d(tcs_correct_abs[ep,pcs], sigma=sigma, axis=1)
        # remove multiple field
        field_counts = np.array([count_fields(tc, smooth_sigma=5,min_distance=40,plot=False)[0] for tc in tcs_smoothed])
        # for large fields?
        w = [eq_rectangular_width(tc, bin_size)*3 for tc in tcs_smoothed[field_counts==1]]        
        widths_per_ep.append(w)
        # 4. Get position and velocity
        yb = ybinned[eps[ep]:eps[ep+1]]
        vel = forwardvel[eps[ep]:eps[ep+1]]
        trl = trialnum[eps[ep]:eps[ep+1]]
        lr = lick_rate[eps[ep]:eps[ep+1]]
        for cidx, (com, width) in enumerate(zip(coms_correct_abs[ep,np.array(pcs)[field_counts==1]], w)):
            lower = com - width / 2
            upper = com + width / 2
            # handle circular wrapping
            in_field_mask = ((yb > lower) & (yb < upper)) if lower < upper else \
                            ((yb > lower) | (yb < upper))
            vel_in_field = vel[in_field_mask]
            lr_in_field = lr[in_field_mask]
            if len(vel_in_field) > 0:
                field_vel = np.nanmean(vel_in_field)
                field_lr= np.nanmean(lr_in_field)
            else:
                field_vel = np.nan
                field_lr= np.nan
            field_velocities.append(field_vel)
            field_lrs.append(field_lr)
        cells_quant.append(np.array(pcs)[field_counts==1])
        field_velocities_per_ep.append(field_velocities)
        field_lr_per_ep.append(field_lrs)
    df = pd.DataFrame()
    df['width_cm'] = np.concatenate(widths_per_ep)
    df['cellid'] = np.concatenate(cells_quant)
    df['epoch'] = np.concatenate([[f'epoch{ep+1}_rz{int(rz[ep])}']*len(widths_per_ep[ep]) for ep in range(len(tcs_correct_abs))])        
    df['animal'] = [animal]*len(df)
    df['rewloc'] = np.concatenate([[f'epoch{ep+1}_rewloc{rewlocs[ep]}']*len(widths_per_ep[ep]) for ep in range(len(tcs_correct_abs))])  
    df['day'] = [day]*len(df)
    df['cell_type'] = ['Place']*len(df)
    df['vel_in_field_cm_s'] = np.concatenate(field_velocities_per_ep)
    df['lick_rate_in_field'] = np.concatenate(field_lr_per_ep)

    #add
    alldf.append(df)
    # pre
    # example:
    plt.close('all')
    widths_per_ep = []
    field_velocities_per_ep = []
    field_lr_per_ep = []
    cells_quant=[]
    bin_size=3
    # adjusted binsize to cm?
    for ep in range(len(tcs_correct)):
        field_velocities=[];field_lrs=[]
        sigma = 1   # in bin-units; start small (0.5–2 bins)
        tcs_smoothed = gaussian_filter1d(tcs_correct[ep,goal_all], sigma=sigma, axis=1)
        # remove multiple field
        field_counts = np.array([count_fields(tc, smooth_sigma=5,min_distance=40,plot=False)[0] for tc in tcs_smoothed])
        # for large fields?
        w = [eq_rectangular_width(tc, bin_size)*bin_size for tc in tcs_smoothed[field_counts==1]]        # cm
        widths_per_ep.append(w)
        # 4. Get position and velocity
        yb = np.concatenate(rad)[eps[ep]:eps[ep+1]]
        # add to make 2pi
        yb=yb+np.pi
        vel = forwardvel[eps[ep]:eps[ep+1]]
        trl = trialnum[eps[ep]:eps[ep+1]]
        lr = lick_rate[eps[ep]:eps[ep+1]]
        for cidx, (com, width) in enumerate(zip(coms_correct[ep,np.array(goal_all)[field_counts==1]], w)):
            # convert to rad
            width_n = width*(2*np.pi/track_length_dt)
            lower = com - width_n / 2
            upper = com + width_n / 2
            # handle circular wrapping
            in_field_mask = ((yb > lower) & (yb < upper)) if lower < upper else \
                            ((yb > lower) | (yb < upper))
            vel_in_field = vel[in_field_mask]
            lr_in_field = lr[in_field_mask]
            if len(vel_in_field) > 0:
                field_vel = np.nanmean(vel_in_field)
                field_lr= np.nanmean(lr_in_field)
            else:
                field_vel = np.nan
                field_lr= np.nan
            field_velocities.append(field_vel)
            field_lrs.append(field_lr)
        cells_quant.append(np.array(goal_all)[field_counts==1])
        field_velocities_per_ep.append(field_velocities)
        field_lr_per_ep.append(field_lrs)
    df = pd.DataFrame()
    df['width_cm'] = np.concatenate(widths_per_ep)
    df['cellid'] = np.concatenate(cells_quant)
    df['epoch'] = np.concatenate([[f'epoch{ep+1}_rz{int(rz[ep])}']*len(widths_per_ep[ep]) for ep in range(len(tcs_correct_abs))])        
    df['animal'] = [animal]*len(df)
    df['rewloc'] = np.concatenate([[f'epoch{ep+1}_rewloc{rewlocs[ep]}']*len(widths_per_ep[ep]) for ep in range(len(tcs_correct_abs))])  
    df['day'] = [day]*len(df)
    df['cell_type'] = ['Place']*len(df)
    df['vel_in_field_cm_s'] = np.concatenate(field_velocities_per_ep)
    df['lick_rate_in_field'] = np.concatenate(field_lr_per_ep)
    try: # eg., suppress after validation
        ii=df.cellid.unique()[4]
        plt.figure()
        plt.plot(tcs_correct[:,ii,:].T)
        plt.title(f"width={df.loc[df.cellid==ii, 'width_cm'].values.astype(int)} \n lick_rate={df.loc[df.cellid==ii, 'lick_rate_in_field'].values.astype(int)}  \n velocity={df.loc[df.cellid==ii, 'vel_in_field_cm_s'].values.astype(int)}")
        plt.show()
    except Exception as e:
        print(e)

    #add
    alldf.append(df)
    # post reward
    com_goal_postrew = [[xx for xx in com if (np.nanmedian(coms_rewrel[:,xx], axis=0)>0)] if len(com)>0 else [] for com in com_goal]
    # get goal cells across all epochs        
    if len(com_goal_postrew)>0:
        goal_cells = intersect_arrays(*com_goal_postrew); 
    else:
        goal_cells=[]
    goal_all = np.unique(np.concatenate(com_goal_postrew)).astype(int)     
    goal_all = [xx for xx in goal_all if xx not in pcs]
    # dedicated
    # goal_all = goal_cells
    widths_per_ep = []
    peak_per_ep = []
    field_velocities_per_ep = []
    field_lr_per_ep=[]
    # adjusted binsize to cm?
    # TODO - test only on dedicated cells?
    for ep in range(len(tcs_correct)):
        field_velocities=[];field_lrs=[]
        sigma = 1   # in bin-units; start small (0.5–2 bins)
        tcs_smoothed = gaussian_filter1d(tcs_correct[ep,goal_all], sigma=sigma, axis=1)
        r = [circular_fwhm(tc, bin_size) for tc in tcs_smoothed]
        w = [xx[0] for xx in r]
        # for large fields?
        w = [eq_rectangular_width(tc, bin_size) for tc in tcs_smoothed]
        # convert to cm
        # w = np.array(w)
        widths_per_ep.append(w)
        # 4. Get position and velocity
        yb = ybinned_dt[eps[ep]:eps[ep+1]]
        vel = forwardvel[eps[ep]:eps[ep+1]]
        trl = trialnum[eps[ep]:eps[ep+1]]
        lr = lick_rate[eps[ep]:eps[ep+1]]
        for cidx, (com, width) in enumerate(zip(coms_correct[ep,goal_all], w)):
            lower = com - width / 2
            upper = com + width / 2
            # handle circular wrapping
            in_field_mask = ((yb > lower) & (yb < upper)) if lower < upper else \
                            ((yb > lower) | (yb < upper))
            vel_in_field = vel[in_field_mask]
            lr_in_field = lr[in_field_mask]
            if len(vel_in_field) > 0:
                field_vel = np.nanmean(vel_in_field)
                field_lr= np.nanmean(lr_in_field)
            else:
                field_vel = np.nan
                field_lr= np.nan
            field_velocities.append(field_vel)
            field_lrs.append(field_lr)
        peak_per_ep.append([np.quantile(tc,.75) for tc in tcs_correct[ep,goal_all]])
        field_velocities_per_ep.append(field_velocities)
        field_lr_per_ep.append(field_lrs)
    widths_per_ep=np.array(widths_per_ep)
    df = pd.DataFrame()
    df['width_cm'] = np.concatenate(widths_per_ep)
    df['75_quantile'] = np.concatenate(peak_per_ep)
    df['cellid'] = np.concatenate([goal_all]*len(widths_per_ep))
    df['epoch'] = np.concatenate([[f'epoch{ep+1}_rz{int(rz[ep])}']*len(widths_per_ep[ep]) for ep in range(len(tcs_correct))])        
    df['animal'] = [animal]*len(df)
    df['rewloc'] = np.concatenate([[f'epoch{ep+1}_rewloc{rewlocs[ep]}']*len(widths_per_ep[ep]) for ep in range(len(tcs_correct))])  
    df['day'] = [day]*len(df)
    df['cell_type'] = ['Post']*len(df)
    df['vel_in_field_cm_s'] = np.concatenate(field_velocities_per_ep)
    df['lick_rate_in_field'] = np.concatenate(field_lr_per_ep)
    # add add pre and post dfs
    alldf.append(df)

    # lick
    pre_lick_locs = []
    pre_lick_times = []
    lick_rates = []
    epoch_labels = []    
    pre_velocities = []

    # Define pre-reward window in seconds and/or cm
    pre_window_s = 12  # seconds before reward
    for ep in range(len(eps)-1):
        eprng = range(eps[ep], eps[ep+1])
        trials = np.unique(trialnum[eprng])
        trialnum_ep = trialnum[eprng]
        tr_range = range(eps[ep], eps[ep+1])
        trials = np.unique(trialnum[tr_range])
        rewards_ep = rewards[eprng]
        lick_ep = lick[eprng]
        success, fail, str_trials, ftr_trials, ttr, \
        total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
        # only use last 8 correct trials
        lasttr=8
        for tr in str_trials[-lasttr:]:
            tr_idx = (trialnum_ep == tr)
            if np.sum(rewards_ep[tr_idx]) == 0:
                continue  # no reward delivered
            time_tr = time[eprng][tr_idx]-time[eprng][tr_idx][0]
            reward_time = time_tr[(rewards_ep[tr_idx] > 0)][0]  # first reward frame
            lick_mask = (lick_ep[tr_idx] > 0) & (time_tr < reward_time) & (time_tr > reward_time - pre_window_s)
            # Get velocity in pre-reward window
            vel_tr = forwardvel[eprng][tr_idx]
            pre_vel_mask = (time_tr < reward_time) & (time_tr > reward_time - pre_window_s)
            if np.any(pre_vel_mask):
                avg_vel = np.nanmean(vel_tr[pre_vel_mask])
            else:
                avg_vel = np.nan
            pre_velocities.append(avg_vel)

            if np.any(lick_mask):
                # Locations (in cm)
                lick_locs = ybinned[eprng][tr_idx][lick_mask]
                lick_times = time_tr[lick_mask]
                pre_lick_locs.append((np.min(lick_locs), np.max(lick_locs)))  # first and last
                pre_lick_times.append((np.min(lick_times), np.max(lick_times)))
                lick_rates.append(np.sum(lick_mask) / pre_window_s)
            else:
                pre_lick_locs.append((np.nan, np.nan))
                pre_lick_times.append((np.nan, np.nan))
                lick_rates.append(np.nan)
            # ... your lick processing here ...
            epoch_labels.append(f'epoch{ep+1}_rz{int(rz[ep])}')
            
    # Convert to DataFrame for easier analysis
    lick_df = pd.DataFrame({
        'first_lick_loc_cm': [x[0] for x in pre_lick_locs],
        'last_lick_loc_cm': [x[1] for x in pre_lick_locs],
        'first_lick_time': [x[0] for x in pre_lick_times],
        'last_lick_time': [x[1] for x in pre_lick_times],
        'lick_rate_hz': lick_rates,
        'avg_velocity_cm_s': pre_velocities,
        'animal': [animal]*len(pre_lick_locs),
        'day': [day]*len(pre_lick_locs),
        'epoch': epoch_labels
    })

    # Optional: return this alongside widths
    return pd.concat(alldf), lick_df
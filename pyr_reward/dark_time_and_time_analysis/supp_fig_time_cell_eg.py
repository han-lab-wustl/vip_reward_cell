
"""
zahra
pure time cells (not rew or place)
"""
#%%
import numpy as np, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
# plt.rc('font', size=20)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.memory.behavior import consecutive_stretch
from projects.pyr_reward.placecell import get_tuning_curve, calc_COM_EH, make_tuning_curves_by_trialtype_w_darktime,\
    make_time_tuning_curves_radians, make_tuning_curves, intersect_arrays
from projects.pyr_reward.rewardcell import get_radian_position,\
    get_radian_position_first_lick_after_rew, get_rewzones, get_goal_cells, goal_cell_shuffle,\
        goal_cell_shuffle_time, get_goal_cells_time
from projects.opto.behavior.behavior import get_success_failure_trials
from projects.pyr_reward.dark_time_and_time_analysis.time import filter_cells_by_field_selectivity
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
saveddataset = r"Z:\saved_datasets\time_tuning_circ.p"
# with open(saveddataset, "rb") as fp: #unpickle
        # radian_alignment_saved = pickle.load(fp)
#%%
bins=90
goal_window_cm=20
# iterate through all animals
ii=188    
day = conddf.days.values[ii]
animal = conddf.animals.values[ii]
pln=0
# check if its the last 4 days of animal behavior
andf = conddf[(conddf.animals==animal) &( conddf.optoep<2)]
lastdays = andf.days.values[-4:]
params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
print(params_pth)
fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 'timedFF',
         'putative_pcs', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
         'stat', 'licks'])
pcs = np.vstack(np.array(fall['putative_pcs'][0]))
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
animals_w_2_planes = ['z9', 'e190']
animals_w_3_planes = ['e139', 'e145']
# framerate
fr = 31.25
if animal in animals_w_2_planes: fr/2
if animal in animals_w_3_planes: fr/3
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
rz = get_rewzones(rewlocs,1/scalingf)       
# get average success rate
rates = []
for ep in range(len(eps)-1):
   eprng = range(eps[ep],eps[ep+1])
   success, fail, str_trials, ftr_trials, ttr, \
   total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
   rates.append(success/total_trials)
rate=np.nanmean(np.array(rates))
rad = get_radian_position_first_lick_after_rew(eps, ybinned, licks, rewards, rewsize,rewlocs,
            trialnum, track_length) # get radian coordinates

fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
Fc3 = fall_fc3['Fc3']
dFF = fall_fc3['dFF']
Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & ~(fall['bordercells'][0].astype(bool)))]
dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool) & ~(fall['bordercells'][0].astype(bool)))]
skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
#if pc in all but 1
# looser restrictions
pc_bool = np.sum(pcs,axis=0)>0
Fc3 = Fc3[:,((skew>2)&pc_bool)] # only keep cells with skew 
#################### tc w/ time ###################
tcs_correct_time, coms_correct_time, tcs_fail_time, coms_fail_time, trial_times = make_time_tuning_curves_radians(eps, 
         time, Fc3, trialnum, rewards, licks, ybinned, rewlocs, rewsize, forwardvel, lasttr=8, bins=bins, velocity_filter=True)
# fig.tight_layout()
# normal tc
track_length_rad = track_length*(2*np.pi/track_length)
bin_size=track_length_rad/bins 
################### get rew cells ###################
# dark time params
track_length_dt = 550 # cm estimate based on 99.9% of ypos
track_length_rad_dt = track_length_dt*(2*np.pi/track_length_dt) # estimate bin for dark time
bins_dt=150 
bin_size_dt=track_length_rad_dt/bins_dt # typically 3 cm binswith ~ 475 track length
# tc w/ dark time added to the end of track
tcs_correct, coms_correct, tcs_fail, coms_fail, ybinned_dt, raddt = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,
rewsize,ybinned,time,licks,
Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
bins=bins_dt)  
# get abs dist tuning for rew cells
# binsize = 3 for place
################### get place cells ###################
tcs_correct_abs, coms_correct_abs,_,__ = make_tuning_curves(eps,rewlocs,ybinned,
         Fc3,trialnum,rewards,forwardvel,
         rewsize,3)
goal_window = goal_window_cm*(2*np.pi/track_length) # cm converted to rad
goal_cells, com_goal_postrew, perm, rz_perm = get_goal_cells(rz, goal_window, coms_correct, cell_type = 'all')
goal_cells_p_per_comparison = [len(xx)/len(coms_correct[0]) for xx in com_goal_postrew]           
# get goal cells aligned to time
# same % as goal window
max_trial_times = np.nanmax(np.array([tr.shape[1]/fr for tr in trial_times]))
time_window = max_trial_times*.074 #s/rad
# 5/19/25: halved window of time compared to distance?
# 5/20/25:back to original window
time_window = time_window*(2*np.pi/max_trial_times) # s converted to rad        
goal_cells_time, com_goal_postrew_time, perm_time, rz_perm_time = get_goal_cells_time(rz, time_window, coms_correct_time)
goal_cells_p_per_comparison_time = [len(xx)/len(coms_correct[0]) for xx in com_goal_postrew_time]            
################### get place cells ###################
# get cells that maintain their coms across at least 2 epochs
place_window = 20 # cm converted to rad                
perm = list(combinations(range(len(coms_correct_abs)), 2))     
com_per_ep = np.array([(coms_correct_abs[perm[jj][0]]-coms_correct_abs[perm[jj][1]]) for jj in range(len(perm))])        
compc = [np.where((comr<place_window) & (comr>-place_window))[0] for comr in com_per_ep]
# get cells across all epochs that meet crit
pcs = np.unique(np.concatenate(compc))
pcs_all = intersect_arrays(*compc)
place_cells_p_per_comparison = [len(xx)/len(coms_correct_abs[0]) for xx in compc]           
# get indices
# remove rew and place cells
pure_time = [xx for xx in goal_cells_time if (xx not in goal_cells) and (xx not in pcs_all)]
# also for per comparison
pure_time_cells_p_per_comparison = [[xx for xx in com_time if xx not in compc[ll] and xx not in com_goal_postrew[ll]] for ll,com_time in enumerate(com_goal_postrew_time)]
#%%
# remove vw/ vel filter
tcs_correct_time, coms_correct_time, tcs_fail_time, coms_fail_time, trial_times = make_time_tuning_curves_radians(eps, 
         time, Fc3, trialnum, rewards, licks, ybinned, rewlocs, rewsize, forwardvel, lasttr=8, bins=bins, velocity_filter=False)
#%%
import numpy as np
import matplotlib.pyplot as plt

def moving_average(x, window_size=5):
    return np.convolve(x, np.ones(window_size) / window_size, mode='same')

def normalize(x):
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-10)

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(8,6), sharey=True)
colors = ['k', 'slategray', 'darkcyan', 'darkgoldenrod']
plt.rc('font', size=13)

for ep in range(len(tcs_correct_abs)):
    # Reward-distance (track-aligned)
    ax = axes[0, 0]
    trace = normalize(tcs_correct_abs[ep, goal_cells[1]])
    ax.plot(moving_average(trace), color=colors[ep],label=f'Epoch {ep+1}')
    ax.axvline(rewlocs[ep] / 3, linestyle='--', color=colors[ep])
    if ep == 0:
        ax.set_ylabel('Norm. $\Delta$ F/F')
    ax.set_title('Reward-distance\nTrack-aligned')
    ax.set_xticks([0, 90])
    ax.set_xticklabels([0, 270])
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('Track position (cm)')
    ax.legend()

    # Distance-aligned
    ax = axes[0, 1]
    trace = normalize(tcs_correct[ep, goal_cells[1]])
    ax.plot(moving_average(trace), color=colors[ep])
    ax.axvline(75, linestyle='--', color='gray')
    ax.set_title('Distance-aligned')
    ax.set_xticks([0, 75, 150])
    ax.set_xticklabels(['-$\\pi$', 0, '$\\pi$'])
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('Reward-centric ($\Theta$)')

    # Time-aligned
    ax = axes[0, 2]
    trace = normalize(tcs_correct_time[ep, goal_cells[1]])
    ax.plot(moving_average(trace), color=colors[ep])
    ax.axvline(45, linestyle='--', color='gray')
    ax.set_title('Time-aligned')
    ax.set_xticks([0, 45, 90])
    ax.set_xticklabels(['-$\\pi$', 0, '$\\pi$'])
    ax.spines[['top', 'right']].set_visible(False)

    # Reward-time
    ax = axes[1, 0]
    trace = normalize(tcs_correct_abs[ep, pure_time[1]])
    ax.plot(moving_average(trace), color=colors[ep])
    ax.axvline(rewlocs[ep] / 3, linestyle='--', color=colors[ep])
    ax.set_title('Reward-time\nTrack-aligned')
    ax.set_xticks([0, 90])
    ax.set_xticklabels([0, 270])
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('Track position (cm)')

    # Distance-aligned (time-pure)
    ax = axes[1, 1]
    trace = normalize(tcs_correct[ep, pure_time[1]])
    ax.plot(moving_average(trace), color=colors[ep])
    ax.axvline(75, linestyle='--', color='gray')
    ax.set_title('Distance-aligned')
    ax.set_xticks([0, 75, 150])
    ax.set_xticklabels(['-$\\pi$', 0, '$\\pi$'])
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('Reward-centric ($\Theta$)')

    # Time-aligned (time-pure)
    ax = axes[1, 2]
    trace = normalize(tcs_correct_time[ep, pure_time[1]])
    ax.plot(moving_average(trace), color=colors[ep])
    ax.axvline(45, linestyle='--', color='gray')
    ax.set_title('Time-aligned')
    ax.set_xticks([0, 45, 90])
    ax.set_xticklabels(['-$\\pi$', 0, '$\\pi$'])
    ax.spines[['top', 'right']].set_visible(False)
# Uniform y-axis limit after normalization
for ax in axes.flat:
    ax.set_ylim([0, 1])
plt.tight_layout()

plt.savefig(os.path.join(savedst,'time_cell_eg.svg'),bbox_inches='tight')
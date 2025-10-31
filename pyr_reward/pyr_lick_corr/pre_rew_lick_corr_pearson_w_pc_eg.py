
"""
zahra
lick corr trial by trial
split into corr vs incorr
shuffle: circ shuffle and calculate tuning curve
repeat x 100
"""
#%%
import numpy as np, scipy, matplotlib.pyplot as plt, sys, pandas as pd
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
# import condition df
from projects.pyr_reward.placecell import make_tuning_curves_time_trial_by_trial
from projects.pyr_reward.placecell import make_tuning_curves_radians_by_trialtype, intersect_arrays,\
    make_tuning_curves_by_trialtype_w_darktime,make_tuning_curves
from projects.pyr_reward.rewardcell import get_radian_position,create_mask_from_coordinates,pairwise_distances,extract_data_rewcentric,\
    get_radian_position_first_lick_after_rew, get_rewzones, cosine_sim_ignore_nan
from projects.pyr_reward.placecell import make_tuning_curves_abs_w_probe_types, intersect_arrays
from projects.opto.behavior.behavior import get_success_failure_trials, smooth_lick_rate
from scipy.stats import spearmanr

conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_rewardcentric_all.p"
with open(saveddataset, "rb") as fp: #unpickle
        radian_alignment_saved = pickle.load(fp)
savepth = os.path.join(savedst, 'lickcorr.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)

def safe_pearsonr(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    # Mask out NaN or inf
    mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
    x = x[mask]
    y = y[mask]

    # Need at least 2 valid points and non-constant
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0

    r, _ = scipy.stats.pearsonr(x, y)
    if np.isnan(r):
        return 0.0
    return r

def shuffle_licks_by_trial_1s_segments(licks, trialnum, fps=30, shuffle_rounds=3):
   """
   Shuffle licks within 1-second segments per trial, repeating shuffle to increase randomness.

   Parameters:
      licks (np.ndarray): 1D binary lick array
      trialnum (np.ndarray): 1D trial number array
      fps (int): frames per second
      shuffle_rounds (int): how many times to reshuffle each segment

   Returns:
      np.ndarray: shuffled lick array
   """
   shuffled = np.zeros_like(licks)
   segment_len = int(fps)
   trials = np.unique(trialnum)

   for tr in trials:
      idxs = np.where(trialnum == tr)[0]
      trial_licks = licks[idxs]
      n = len(trial_licks)

      for start in range(0, n, segment_len):
         end = min(start + segment_len, n)
         segment = trial_licks[start:end]
         for _ in range(shuffle_rounds):
               if np.any(segment):
                  shift = np.random.randint(1, len(segment))  # avoid 0
                  segment = np.roll(segment, shift)
         shuffled[idxs[start:end]] = segment

   return shuffled

def permute_licks_by_trial_1s_segments(licks, trialnum, fps=30):
   """
   Fully permute lick positions within 1-second segments per trial.

   Parameters:
      licks (np.ndarray): 1D binary lick array
      trialnum (np.ndarray): 1D trial number array
      fps (int): frames per second

   Returns:
      np.ndarray: shuffled lick array
   """
   shuffled = np.zeros_like(licks)
   segment_len = int(fps/4)
   trials = np.unique(trialnum)

   for tr in trials:
      idxs = np.where(trialnum == tr)[0]
      trial_licks = licks[idxs]
      n = len(trial_licks)

      for start in range(0, n, segment_len):
         end = min(start + segment_len, n)
         segment = trial_licks[start:end]
         permuted = np.random.permutation(segment)
         shuffled[idxs[start:end]] = permuted

   return shuffled

def circular_shuffle_with_shift_range(licks, n_shuffles=100, fps=30, min_shift_ms=200, max_shift_s=1.0):
   """
   Perform circular shuffling of the lick time series with random shifts between min and max.

   Parameters:
      licks (np.ndarray): 1D binary lick array
      n_shuffles (int): number of shuffled versions to generate
      fps (int): sampling rate in Hz
      min_shift_ms (float): minimum shift in milliseconds (e.g. 200)
      max_shift_s (float): maximum shift in seconds (e.g. 1.0)

   Returns:
      np.ndarray: 2D array of shape (n_shuffles, len(licks))
   """
   licks = np.asarray(licks)
   assert licks.ndim == 1, "licks must be a 1D array"
   n = len(licks)

   min_shift = int(np.ceil(min_shift_ms / 1000 * fps))
   max_shift = int(np.floor(max_shift_s * fps))

   if min_shift >= n or max_shift >= n:
      raise ValueError("Shift range is too large for the time series length.")

   shuffles = np.zeros((n_shuffles, n), dtype=int)
   for i in range(n_shuffles):
      shift = np.random.randint(min_shift, max_shift + 1)
      shuffles[i] = np.roll(licks, shift)

   return shuffles
#%%
# initialize var
# radian_alignment_saved = {} # overwrite
bins = 150
cm_window=20
df_all=[]

# iterate through all animals
ii=169
day = conddf.days.values[ii]
animal = conddf.animals.values[ii]
if animal=='e145' or animal=='e139': pln=2 
else: pln=0
params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
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
rz = get_rewzones(rewlocs,1/scalingf)       
rates = []; 
for ep in range(len(eps)-1):
   eprng = range(eps[ep],eps[ep+1])
   ypos=ybinned[eprng]
   success, fail, str_trials, ftr_trials, ttr, \
   total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
   rates.append(success/total_trials)
rate=np.nanmean(np.array(rates))
# t = time_[mask][(ybinned_<rewloc)[mask]]
dt = np.nanmedian(np.diff(time))
lick_bin=lick.copy()
lick= smooth_lick_rate(lick,dt)
# lick_shufs=100
# lick_no_consp_shuf = circular_shuffle_with_shift_range(lick, n_shuffles=lick_shufs, fps=1/dt, min_shift_ms=1000, max_shift_s=30.0)
# vel_no_rew_shuf = circular_shuffle_with_shift_range(forwardvel, n_shuffles=lick_shufs, fps=1/dt, min_shift_ms=1000, max_shift_s=30.0)
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
goal_window = 20*(2*np.pi/track_length) 
tcs_correct, coms_correct, tcs_fail, coms_fail, ybinned_dt, rad = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,
   rewsize,ybinned,time,lick,
   Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
   bins=bins_dt)  
bin_size=3
# position tc with different trial types
tcs_correct_abs, coms_correct_abs, tcs_fail_abs, coms_fail_abs, tcs_probe, coms_probe, tcs_bigmiss, coms_bigmiss, tcs_no_lick, coms_no_lick, tcs_precorr, coms_precorr= make_tuning_curves_abs_w_probe_types(eps,rewlocs,ybinned,Fc3,trialnum,lick,rewards,forwardvel,rewsize,bin_size)
# get cells that maintain their coms across at least 2 epochs
##################################### PLACE 
place_window = 20 # cm converted to rad                
perm = list(combinations(range(len(coms_correct_abs)), 2))     
com_per_ep = np.array([(coms_correct_abs[perm[jj][0]]-coms_correct_abs[perm[jj][1]]) for jj in range(len(perm))])        
compc = [np.where((comr<place_window) & (comr>-place_window))[0] for comr in com_per_ep]
# get cells across all epochs that meet crit
pcs = np.unique(np.concatenate(compc))
pcs_all = intersect_arrays(*compc)
##################################### REWARD
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
######################## near pre reward only
bound=np.pi/4
com_goal_postrew = [[xx for xx in com if (((np.nanmedian(coms_rewrel[:,
xx], axis=0)<0)&(np.nanmedian(coms_rewrel[:,
xx], axis=0)>-bound) | ((np.nanmedian(coms_rewrel[:,
xx], axis=0)>0)&(np.nanmedian(coms_rewrel[:,
xx], axis=0)<bound))))] if len(com)>0 else [] for com in com_goal]
# pre rew
com_goal_prerew = [[xx for xx in com if (((np.nanmedian(coms_rewrel[:,
xx], axis=0)<0)&(np.nanmedian(coms_rewrel[:,
xx], axis=0)>-bound)))] if len(com)>0 else [] for com in com_goal]

# get goal cells across all epochs        
if len(com_goal_postrew)>0:
   goal_cells = intersect_arrays(*com_goal_postrew); 
else:
   goal_cells=[]
# here i take any cell that is a goal cell in any given epoch combo
pre_goal_cells =np.unique(np.concatenate(com_goal_prerew)).astype(int)  
goal_cells = np.unique(np.concatenate(com_goal_postrew)).astype(int)  
all_goal_cell = np.unique(np.concatenate(com_goal)).astype(int)     
# pcs that are not goal cells
pcs = [xx for xx in pcs if xx not in goal_cells]   
# lick and vel tc with different trial types
lick_correct_abs, _, lick_fail, _, lick_probe, _, lick_bigmiss, _, lick_no_lick, _, lick_precorr, _= make_tuning_curves_abs_w_probe_types(eps,rewlocs,ybinned,np.array([lick]).T,trialnum,lick,rewards,forwardvel,rewsize,bin_size)
vel_correct_abs, _, vel_fail, _, vel_probe, _, vel_bigmiss, _, vel_no_lick, _, vel_precorr, _= make_tuning_curves_abs_w_probe_types(eps,rewlocs,ybinned,np.array([forwardvel]).T,trialnum,lick,rewards,forwardvel,rewsize,bin_size)
plt.rc('font', size=14)

#%%
colors=['k','slategray','darkcyan','darkgoldenrod']
cell_color=['cornflowerblue','indigo']
lbls = ['$\Delta F/F$','$\Delta F/F$','(Hz)', '(cm/s)','$\Delta F/F$','$\Delta F/F$']
title = ['Pre-reward, Single cell', 'Pre-reward, Population','Lick','Velocity','Place, Single cell', 'Place, Population']
fig,axes=plt.subplots(nrows=6,ncols=4,sharex=True,sharey='row',figsize=(7,9))
tcs = [tcs_fail_abs[:,pre_goal_cells[1],:],tcs_fail_abs[:,pre_goal_cells,:],np.squeeze(lick_fail),np.squeeze(vel_fail),tcs_fail_abs[:,pcs[1],:],tcs_fail_abs[:,pcs,:],]
for ep in range(len(tcs_correct_abs)):
   for jj,tc in enumerate(tcs):
      ax=axes[jj,ep]
      if jj<2: color=cell_color[0]
      elif jj>3: color=cell_color[1]
      else: color=colors[ep]
      # population
      if jj==1 or jj==5: 
         tcorg=tc.copy()
         tc = np.nanmean(tcorg[ep],axis=0)
         ax.plot(tc,color=color)
         sem = scipy.stats.sem(tcorg[ep],axis=0)
         ax.fill_between(range(tc.shape[0]),tc-sem,tc+sem,alpha=0.5,color=color)
      else: 
         ax.plot(tc[ep].T,color=color)
      ax.axvline(rewlocs[ep]/3,linestyle='--',color=colors[ep])      
      ax.set_xticks([0,90])
      ax.set_xticklabels([0,270])
      if ep==0: 
         ax.set_ylabel(lbls[jj])
         if jj==5: ax.set_xlabel('Track position (cm)')
         ax.set_title(title[jj],fontsize=14)
         
      ax.spines[['top','right']].set_visible(False)
fig.suptitle(f'mouse {animal[1:]}, session {day}\nPlace and reward cells\n Incorrect trials')
plt.tight_layout()
# plt.savefig(os.path.join(savedst,'place_lick_vel_eg.svg'), bbox_inches='tight')

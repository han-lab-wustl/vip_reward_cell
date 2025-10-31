
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
    get_radian_position_first_lick_after_rew, get_rewzones, wilcoxon_r
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

def make_corr_df(tcs, lick_abs, vel_abs,coms_rewrel,  trial_type, condition, animal, day, goal_cells):
   """ nan rho is tc is nan """
   if len(tcs)>len(lick_abs): # remove extra eps
      tcs=tcs[:-1,:,:]
   lick_tc_cs = [
      [safe_pearsonr(tcs[ep, cll, :], lick_abs[ep][0]) if ~np.isnan(tcs[ep, cll, :]).any() else np.nan for cll in goal_cells]
      for ep in range(len(tcs))
   ]
   vel_tc_cs = [
      [safe_pearsonr(tcs[ep, cll, :], vel_abs[ep][0]) if ~np.isnan(tcs[ep, cll, :]).any() else np.nan for cll in goal_cells]
      for ep in range(len(tcs))
   ]
   df = pd.DataFrame({
      "cellid": np.concatenate([goal_cells] * len(tcs)),
      "com": np.hstack(coms_rewrel[:,goal_cells]), # checked
      "cs_lick_v_tc": np.concatenate(lick_tc_cs),
      "cs_vel_v_tc": np.concatenate(vel_tc_cs),
      "trial_type": [trial_type] * (len(goal_cells) * len(tcs)),
      "cell_type": ["pre"] * (len(goal_cells) * len(tcs)),   # adjust if you classify cells
      "condition": [condition] * (len(goal_cells) * len(tcs)),
      "animal": [animal] * (len(goal_cells) * len(tcs)),
      "day": [day] * (len(goal_cells) * len(tcs)),
   })
   return df

def main(params_pth,animal,day):
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
   com_goal_postrew = [[xx for xx in com if ((np.nanmedian(coms_rewrel[:,
   xx], axis=0)<0)&(np.nanmedian(coms_rewrel[:,
   xx], axis=0)>-bound))] if len(com)>0 else [] for com in com_goal]
   # get goal cells across all epochs        
   if len(com_goal_postrew)>0:
      goal_cells = intersect_arrays(*com_goal_postrew); 
   else:
      goal_cells=[]
   # here i take any cell that is a goal cell in any given epoch combo
   goal_cells = np.unique(np.concatenate(com_goal_postrew)).astype(int)     
   all_goal_cells = np.unique(np.concatenate(com_goal)).astype(int)    
   # pcs that are not goal cells
   pcs = [xx for xx in pcs if xx not in all_goal_cells]   
   # lick and vel tc with different trial types
   lick_correct_abs, _, lick_fail, _, lick_probe, _, lick_bigmiss, _, lick_no_lick, _, lick_precorr, _= make_tuning_curves_abs_w_probe_types(eps,rewlocs,ybinned,np.array([lick]).T,trialnum,lick,rewards,forwardvel,rewsize,bin_size)
   vel_correct_abs, _, vel_fail, _, vel_probe, _, vel_bigmiss, _, vel_no_lick, _, vel_precorr, _= make_tuning_curves_abs_w_probe_types(eps,rewlocs,ybinned,np.array([forwardvel]).T,trialnum,lick,rewards,forwardvel,rewsize,bin_size)
   # Helper to build df for one trial type
   dfs=[]

   ########## correct trials      
   dfs.append(make_corr_df(tcs_correct_abs, lick_correct_abs, vel_correct_abs,coms_rewrel,
                           trial_type="correct", condition="real",
                           animal=animal, day=day, goal_cells=pcs))
   ########## incorrect trials  
   dfs.append(make_corr_df(tcs_fail_abs, lick_fail, vel_fail,coms_rewrel,
                  trial_type="incorrect", condition="real",
                     animal=animal, day=day, goal_cells=pcs))
   # bigmiss
   dfs.append(make_corr_df(tcs_bigmiss, lick_bigmiss, vel_bigmiss,coms_rewrel,
            trial_type="bigmiss", condition="real",
               animal=animal, day=day, goal_cells=pcs))
   # probe 1,2,3
   for probe in range(3):
      dfs.append(make_corr_df(tcs_probe[:,probe], lick_probe[:,probe], vel_probe[:,probe],coms_rewrel,
            trial_type=f"probe_{probe}", condition="real",
               animal=animal, day=day, goal_cells=pcs))
   # precorr
   dfs.append(make_corr_df(tcs_precorr, lick_precorr, vel_precorr,coms_rewrel,
      trial_type="pre_first_correct", condition="real",
         animal=animal, day=day, goal_cells=pcs))
   # no lick
   dfs.append(make_corr_df(tcs_no_lick, lick_no_lick, vel_no_lick,coms_rewrel,
      trial_type="no_lick", condition="real",
         animal=animal, day=day, goal_cells=pcs))
   # correct shuffle
   n_shufs=50
   # remove high lr
   lick_no_consp = lick_bin.copy()
   lick_no_consp[forwardvel<2]=0
   vel_no_rew = forwardvel.copy()
   vel_no_rew[forwardvel<2]=0
   for shuf in range(n_shufs):
      ################## lick shuffle
      lick_no_consp_shuf = np.squeeze(circular_shuffle_with_shift_range(lick_no_consp, n_shuffles=1, fps=1/dt, min_shift_ms=1000, max_shift_s=30.0))
      # convert to rate
      lick_no_consp_shuf=smooth_lick_rate(lick_no_consp_shuf,dt)
      vel_no_rew_shuf = np.squeeze(circular_shuffle_with_shift_range(vel_no_rew, n_shuffles=1, fps=1/dt, min_shift_ms=1000, max_shift_s=30.0))
      # test
      # if shuf==5:
      #    plt.figure()
      #    plt.plot(lick[5000:10000])
      #    plt.plot(lick_no_consp_shuf[5000:10000])
      lick_correct_abs, _, lick_fail, _, lick_probe, _, lick_bigmiss, _, lick_no_lick, _, lick_precorr, _= make_tuning_curves_abs_w_probe_types(eps,rewlocs,ybinned,np.array([lick_no_consp_shuf]).T,trialnum,lick,rewards,forwardvel,rewsize,bin_size)
      vel_correct_abs, _, vel_fail, _, vel_probe, _, vel_bigmiss, _, vel_no_lick, _, vel_precorr, _= make_tuning_curves_abs_w_probe_types(eps,rewlocs,ybinned,np.array([vel_no_rew_shuf]).T,trialnum,lick,rewards,forwardvel,rewsize,bin_size)
      dfs.append(make_corr_df(tcs_correct_abs, lick_correct_abs, vel_correct_abs,coms_rewrel,trial_type="correct", condition=f"shuffle_{shuf}",
                     animal=animal, day=day, goal_cells=pcs))

   df=pd.concat(dfs)
   return df
#%%
# initialize var
# radian_alignment_saved = {} # overwrite
bins = 150
cm_window=20
df_all=[]
###################### reward cell only
# iterate through all animals
for ii in range(len(conddf)):
   day = conddf.days.values[ii]
   animal = conddf.animals.values[ii]
   if (animal!='e217') & (conddf.optoep.values[ii]<2):
      if animal=='e145' or animal=='e139': pln=2 
      else: pln=0
      params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
      df=main(params_pth,animal,day)
      df_all.append(df)
      # test
      df=df.reset_index()
      plt.figure()
      sns.barplot(y='cs_lick_v_tc',x='trial_type',hue='condition',data=df[((df.condition=='real') | (df.condition=='shuffle_1'))])
      # com vs. cs
      plt.figure()
      sns.scatterplot(y='cs_lick_v_tc',x='com',data=df[((df.condition=='real') & (df.trial_type=='correct'))])
      plt.title(f'{animal}_{day}')
      plt.show()
      # dfs.append(df)
      

#%%
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats
from statsmodels.stats.multitest import multipletests

plt.rc('font', size=16)

# --- Build dataframe ---
bigdf = pd.concat(df_all)
#%%
# pre-reward cell vs. com
# per animal
tempdf=bigdf[~bigdf.condition.str.contains('shuffle') & (bigdf.trial_type=='correct')].reset_index()
ans=tempdf.animal.unique()
ans=ans[2:]
dims = int(np.ceil(np.sqrt(len(ans))))
fig,axes=plt.subplots(ncols=dims,nrows=dims-1,sharey=True,figsize=(9,8))
axes=axes.flatten()
for a,an in enumerate(ans):
   ax=axes[a]
   clls = len(tempdf[(tempdf.animal==an)])
   days = len(tempdf[(tempdf.animal==an)].day.unique())
   numclls=int(clls/days)
   sns.scatterplot(x='com',y='cs_lick_v_tc',data=tempdf[(tempdf.animal==an)],hue='day',alpha=.1,ax=ax,legend=False,rasterized=True)
   ax.set_title(f'{an[1:]}\n{days} sessions, {numclls} cells/session',fontsize=10)
   ax.axvline(0,color='k',linestyle='--')
   ax.axhline(0,color='dimgrey',linestyle='--')   
   if a==0: ax.set_ylabel(r'Lick Pearson $\rho$')
   else: ax.set_ylabel('')
   sns.despine(ax=ax)
   if a==9: ax.set_xlabel('Reward-centric COM ($\Theta$)')
   else: ax.set_xlabel('')
   ax.set_xticks([-3.1,0,3.1])
   ax.set_xticklabels(['-$\pi$',0,'$\pi$'])
fig.suptitle('Correlation of individual pre-reward cells and lick rate')
plt.tight_layout()
plt.savefig(os.path.join(savedst,'single_place_lick_pearson.svg'), bbox_inches='tight')
#%%
# velocity?
# pre-reward cell vs. com
# per animal
tempdf=bigdf[~bigdf.condition.str.contains('shuffle') & (bigdf.trial_type=='correct')].reset_index()
ans=tempdf.animal.unique()
ans=ans[2:]
dims = int(np.ceil(np.sqrt(len(ans))))
fig,axes=plt.subplots(ncols=dims,nrows=dims-1,sharey=True,figsize=(9,8))
axes=axes.flatten()
for a,an in enumerate(ans):
   ax=axes[a]
   clls = len(tempdf[(tempdf.animal==an)])
   days = len(tempdf[(tempdf.animal==an)].day.unique())
   numclls=int(clls/days)
   sns.scatterplot(x='com',y='cs_vel_v_tc',data=tempdf[(tempdf.animal==an)],hue='day',alpha=.1,ax=ax,legend=False,rasterized=True)
   ax.set_title(f'{an[1:]}\n{days} sessions, {numclls} cells/session',fontsize=10)
   ax.axvline(0,color='k',linestyle='--')
   ax.axhline(0,color='dimgrey',linestyle='--')   
   if a==0: ax.set_ylabel(r'Velocity Pearson $\rho$')
   else: ax.set_ylabel('')
   sns.despine(ax=ax)
   if a==9: ax.set_xlabel('Reward-centric COM ($\Theta$)')
   else: ax.set_xlabel('')
   ax.set_xticks([-3.1,0,3.1])
   ax.set_xticklabels(['-$\pi$',0,'$\pi$'])
fig.suptitle('Correlation of individual pre-reward cells and velocity')
plt.tight_layout()
plt.savefig(os.path.join(savedst,'single_place_vel_pearson.svg'), bbox_inches='tight')

#%%


def add_sigstar(ax, xloc, pval, t, height, starsize=14):
    """Add annotation with corrected p-value"""
    if pval < 0.001:
        star = '***'
    elif pval < 0.01:
        star = '**'
    elif pval < 0.05:
        star = '*'
    else:
        star = 'ns'
    if star!='ns': starsize=25
    ax.text(
        xloc, height, star,
        ha='center', va='bottom', fontsize=starsize, color='k'
    )
    ax.text(
        xloc, height-height*.2, f'r={t:.3g}\np={pval:.3g}',
        ha='center', va='bottom', fontsize=8, color='k'
    )

palette = ['seagreen','firebrick','rosybrown','lightslategray']
# only get near rew corr?
bigdf=bigdf[(bigdf.com>-np.pi/4)&(bigdf.com<0)]
df = bigdf.groupby(['animal','condition','trial_type']).mean(numeric_only=True).reset_index()
df = df.dropna()

# Split
df_real    = df[(df["condition"] == "real") & ~(df["trial_type"].str.startswith("probe"))].copy()
df_shuffle = df[df["condition"].str.startswith("shuffle")].copy()
df_probe   = df[df["trial_type"].str.startswith("probe")].copy()
df_probe   = df[df["trial_type"]=='probe_1'].copy()

# Average shuffle
df_shuffle = (
    df_shuffle
    .groupby(["animal", "trial_type", "cellid", "day"], as_index=False)
    .agg({"cs_lick_v_tc": "mean", "cs_vel_v_tc": "mean"})
)
df_shuffle["condition"] = "shuffle"

# Average probe into single category
df_probe = (
    df_probe
    .groupby(["animal", "trial_type", "cellid", "day"], as_index=False)
    .agg({"cs_lick_v_tc": "mean", "cs_vel_v_tc": "mean"})
)
df_probe["trial_type"] = "probe"
df_probe["condition"] = "real"

# Combine
df = pd.concat([df_real, df_shuffle, df_probe], ignore_index=True)
# remove bigmiss and no lick
df = df[df.trial_type!='bigmiss']
df = df[df.trial_type!='no_lick']
df=df[(df.animal!='z16')& (df.animal!='e139')& (df.animal!='e145')]
# --- Plot ---
fig, axes = plt.subplots(ncols=2, figsize=(7,4), sharey=False)
order = ['correct','incorrect','pre_first_correct', 'probe', 'no_lick']

for ax, metric, ylabel in zip(
   axes,
   ['cs_lick_v_tc', 'cs_vel_v_tc'],
   ['Lick Pearson $\\rho$', 'Velocity Pearson $\\rho$']
):
   # Bars
   sns.barplot(
      x='trial_type', y=metric, order=order,
      data=df[df.condition=='real'], fill=False,
      palette=palette, ax=ax, errorbar='se'
   )
   sns.barplot(
      x='trial_type', y=metric, order=order,
      data=df[df.condition=='shuffle'], color='grey',
      alpha=0.5, errorbar=None, ax=ax
   )
   sns.barplot(
      x='trial_type', y=metric,order=order,
      data=df[df.condition=='shuffle'], color='grey',
      label='shuffle', alpha=0.5, err_kws={'color': 'grey'},
      errorbar=None, ax=ax
   )
   ax.set_ylabel(ylabel)
   ax.set_xlabel('')
   ax.set_xticklabels(['Correct','Incorrect','Initial\nIncorrect','First Probe','No lick'], rotation=20)
   sns.despine(ax=ax)

# --- Paired lines across trial types (real only) ---
   animals = df[df.condition=='real']['animal'].unique()
   for animal in animals:
      sub = (
         df[(df.condition=='real') & (df.animal==animal)]
         .groupby("trial_type")[metric]
         .mean()                           # collapse duplicates
         .reindex(order)                   # enforce order
      )
      if sub.notna().any():
         ax.plot(
               range(len(order)),
               sub.values,
               linewidth=1.5, alpha=0.5,
               color='gray'
         )

   # --- Stats with multiple correction ---
   pvals = []
   for ttype in order:
      real_vals = df[(df.condition=='real') & (df.trial_type==ttype)][metric]
      shuf_vals = df[(df.condition=='shuffle') & (df.trial_type=='correct')][metric]
      if len(real_vals) > 0 and len(shuf_vals) > 0:
         stat, p = wilcoxon_r(real_vals.values, shuf_vals.values)
         pvals.append((ttype,stat, p))
      else:
         pvals.append((ttype,stat, np.nan))

   # Apply FDR correction
   ttypes, rs, raw_pvals = zip(*pvals)
   mask = ~np.isnan(raw_pvals)
   corrected = np.full(len(raw_pvals), np.nan)
   if any(mask):
      reject, pvals_corr, _, _ = multipletests(np.array(raw_pvals)[mask], method="fdr_bh")
      corrected[mask] = pvals_corr

   # Place stars
   ymax = df[metric].max()
   for i, (ttype, p_corr) in enumerate(zip(ttypes, corrected)):
      if np.isnan(p_corr): 
         continue
      add_sigstar(ax, i, p_corr,pvals[i][1], ymax - 0.01)

fig.suptitle(rf'Place cells (COM=$-\pi/4$ to $0$)')
plt.tight_layout()
plt.savefig(os.path.join(savedst,'place_lick_vel_pearson.svg'), bbox_inches='tight')

#%%
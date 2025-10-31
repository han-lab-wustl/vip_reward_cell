#%%
"""
8/31/25
- train in 70% of opto trials, all trials in rest of epochs
- test on 30% of opto trials only
- no min window for changepoint
"""
import matplotlib.pyplot as plt
import ruptures as rpt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import poisson
from scipy.special import logsumexp
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import torch
from torch.utils.data import TensorDataset, DataLoader
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error, mean_absolute_error

import numpy as np, sys
import scipy.io, scipy.interpolate, scipy.stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from hmmlearn import hmm
import random
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.opto.behavior.behavior import smooth_lick_rate
from projects.pyr_reward.placecell import make_tuning_curves_by_trialtype_w_darktime, intersect_arrays,make_tuning_curves
from projects.opto.behavior.behavior import smooth_lick_rate
import matplotlib.backends.backend_pdf, matplotlib as mpl
from sklearn.preprocessing import StandardScaler
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
# plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"

bins = 90
goal_window_cm = 20
conddf=pd.read_csv(r'Z:\condition_df\conddf_performance_chrimson.csv')
savedst = r"C:\Users\Han\Desktop\goal_decoding"

# conddf = conddf[(conddf.optoep>1)]
iis = np.arange(len(conddf))  # Animal indices
iis = [ii for ii in iis if ii!=202 and ii!=40 and ii!=129 and ii!=164 and ii!=199]
dct = {}
iis=np.array(iis)

def estimate_tuning(fc3, ybinned, goal_zone, cm_per_bin=1, bin_size_cm=3, n_goals=3):
   trials, time, N = fc3.shape
   # Re-bin positions to 3 cm bins
   pos3cm = (ybinned // (bin_size_cm // cm_per_bin)).astype(int)
   n_pos_bins = pos3cm.max() + 1  # auto-detect number of 3 cm bins
   tuning = np.zeros((N, n_pos_bins, n_goals))
   for i in range(N):  # loop over cells
      for g in range(n_goals):  # loop over goals
         mask = np.ravel(goal_zone[:, None] == g)  # trial x time
         spikes = fc3[mask, :,i]
         positions = pos3cm[mask]
         for x in range(n_pos_bins):
               inds = positions == x
               if inds.sum() > 0:
                  tuning[i, x, g] = np.nanmean(spikes[inds])
   return tuning

# Decoder
def decode_trial(trial_fc, trial_ybin, goal, tuning):
   T = trial_fc.shape[0]
   n_pos_bins = tuning.shape[1]
   log_post = np.full((T, n_pos_bins, n_goals), -np.inf)
   log_prior = np.log(np.ones((n_pos_bins, n_goals)) / (n_pos_bins * n_goals))

   for t in range(T):
      obs = trial_fc[t]
      log_likelihood = np.zeros((n_pos_bins, n_goals))
      for x in range(n_pos_bins):
         for g in range(n_goals):
            lam = tuning[:, x, g]
            lam = np.clip(lam, 1e-3, None)
            log_likelihood[x, g] = np.sum(obs * np.log(lam) - lam)

      if t == 0:
         log_post[t] = log_prior + log_likelihood
      else:
         prev_log_post = log_post[t - 1]
         trans_pos = gaussian_filter1d(np.exp(prev_log_post), sigma=np.sqrt(var_pos), axis=0)
         trans_goal = (1 - p_stay_goal) / (n_goals - 1)
         for g in range(n_goals):
            sticky = p_stay_goal * trans_pos[:, g] + trans_goal * np.sum(trans_pos, axis=1)
            log_post[t, :, g] = np.log(sticky + 1e-10) + log_likelihood[:, g]

      log_post[t] -= logsumexp(log_post[t])

   return np.exp(log_post)

def process_trial(
   trial,
   fc3,
   ybinned,
   goal_zone,
   tuning,
   ep_trials,
   rewlocs,
   rewsize,
   time,
   lick_trial,
   decode_trial_fn,
   pdf,
   min_frac=0.1,
   make_plot=False
):
   """
   Process a single trial with PELT changepoint detection, automatically
   picking penalty so smallest segment >= min_frac * trial length.
   
   Parameters
   ----------
   trial : int
      Trial index to process.
   fc3 : ndarray
      Calcium data, shape (trials, time, cells).
   ybinned : ndarray
      Position data, shape (trials, time).
   goal_zone : ndarray
      Goal zone labels for each trial.
   tuning : ndarray
      Tuning curves or model input for decoder.
   ep_trials : ndarray
      Epoch index for each trial.
   rewlocs : ndarray
      Reward location per epoch.
   rewsize : float
      Size of reward zone in same units as ybinned.
   time : ndarray
      Time vector for a trial.
   lick_trial : ndarray
      Lick rate data, shape (trials, time).
   decode_trial_fn : callable
      Function that takes (fc3_trial, ybinned_trial, goal_zone_trial, tuning)
      and returns posterior probabilities [time × position × goal].
   min_frac : float
      Minimum segment length as fraction of trial length.
   """

   # Decode trial
   ybin_trial = ybinned[trial].copy()
   if ybin_trial[0]>200: ybin_trial[0]=1.5
   fc3_trial = fc3[trial].copy()
   fc3_trial=fc3_trial[ybin_trial>3]
   ybin_trial=ybin_trial[ybin_trial>3]
   post = decode_trial_fn(fc3_trial, ybin_trial, goal_zone[trial], tuning)
   post_goal = post.sum(axis=1)  # marginalize over position
   post_pos = post.sum(axis=2)   # not used below

   ep = ep_trials[trial]
   # 10 cm before rewzone start
   rewloc_start = rewlocs[ep]-( rewsize/2)

   # Find index before reward location
   ypos_temp = ybin_trial.copy()
   ypos_temp[ypos_temp == 0] = 1e6
   rewloc_ind_candidates = np.where(ypos_temp < rewloc_start)[0]
   if len(rewloc_ind_candidates) == 0:
      return {
         "trial": trial,
         "correct": None,
         "time_before_change": None,
         "time_to_rew": None,
         "predicted": None,
         "penalty_used": None,
         "fig": None,
         'changepoint_ind': None,
         'frames': None,
         'position_mae': None
      }
   rewloc_ind = rewloc_ind_candidates[-1]

   # MAP goal trace
   # rewloc_ind = np.where(ybinned[trial]==0)[0][0]
   goal_trace = np.argmax(post_goal, axis=1) + 1
   pos_pred = np.argmax(post_pos, axis=1)
   # smooth pos
   pos_preddf = pd.DataFrame({'pos_pred': pos_pred})
   pos_pred = np.hstack(pos_preddf.rolling(10).mean().values)
   pos_mae = np.nanmean(np.abs(ybin_trial[ybin_trial<rewloc_start]-(pos_pred*3)[ybin_trial<rewloc_start]))
   trial_len = len(goal_trace[:rewloc_ind])
   min_seg_len = int(min_frac * trial_len)

   # Automatically find penalty
   low_pen, high_pen = 1, 1000
   chosen_pen = None
   for _ in range(15):  # binary search steps
      pen = (low_pen + high_pen) / 2
      bkps = np.array(rpt.Pelt(model="l2").fit(goal_trace).predict(pen=pen))
      seg_lens = np.diff([0] + bkps.tolist())
      if len(seg_lens) == 0:
         break
      if min(seg_lens) < min_seg_len:
         low_pen = pen  # too many short segments
      else:
         chosen_pen = pen
         high_pen = pen  # try smaller penalty
   if chosen_pen is None:
      chosen_pen = high_pen  # fallback

   # Final changepoint detection
   bkps = np.array(rpt.Pelt(model="l2").fit(goal_trace).predict(pen=chosen_pen))
   changepoint = bkps[bkps < rewloc_ind]
   # Predict goal zone from last changepoint before reward
   if len(changepoint) > 0:
      pred_goal_zone_cp = np.ceil(np.nanmedian(goal_trace[changepoint[-1]:rewloc_ind]))
      time_before_change = (rewloc_ind - changepoint[-1]) * np.nanmedian(np.diff(time))
   else:
      pred_goal_zone_cp = np.ceil(np.nanmedian(goal_trace[:rewloc_ind]))
      time_before_change = rewloc_ind * np.nanmedian(np.diff(time))

   real_goal_zone = goal_zone[trial] + 1
   correct = pred_goal_zone_cp == real_goal_zone
   time_to_rew = rewloc_ind * np.nanmedian(np.diff(time))
   # get fraction of time
   time_before_change=time_before_change/time_to_rew
   fig = None
   if make_plot:
      fig, ax1 = plt.subplots()
      ax1.plot(goal_trace[:rewloc_ind], label="Goal Trace")
      for cp in changepoint:
         ax1.axvline(cp, color='r', linestyle='--', alpha=0.7)
      if len(lick_trial[trial][:rewloc_ind])>0:
         ax1.plot(
            (lick_trial[trial][:rewloc_ind] / np.nanmax(lick_trial[trial][:rewloc_ind])) * 3,
            label='Lick rate'
         )
      # Instead, add small colored rectangles outside right edge:
      ylim = ax1.get_ylim()
      # Coordinates just beyond right edge
      x_rect = len(goal_trace[:rewloc_ind]) + 1
      width = 2  # small horizontal bar width
      # Add real reward zone patch
      ax1.add_patch(
         plt.Rectangle(
            (x_rect, goal_zone[trial] + 1 - 0.25),  # x, y bottom left
            width,
            0.5,  # height
            color='k',
            alpha=0.6,
            label='Real reward zone'
         )
      )
      # Add predicted reward zone patch
      ax1.add_patch(
         plt.Rectangle(
            (x_rect-(x_rect/3) + width + 0.1, pred_goal_zone_cp - 0.25),
            width,
            0.5,
            color='cyan',
            alpha=0.8,
            label='Predicted reward zone'
         )
      )
      # Add text labels next to rectangles
      ax1.text(x_rect + width/2, goal_zone[trial] + 1,
               "Real", va='center', ha='center', fontsize=9, color='k')
      ax1.text(x_rect-(x_rect/3) + width + 0.1 + width/2, pred_goal_zone_cp,
               "Pred", va='center', ha='center', fontsize=9, color='k')
      ax1.set_xlabel("Time in trial (Hz)")
      ax1.set_ylabel("Goal Trace")
      ax1.legend(loc="upper left")
      ax2 = ax1.twinx()
      ax2.plot(ybin_trial[:rewloc_ind], label="Y binned", color='g', alpha=0.6)
      ax2.set_ylabel("Y Position (binned)")
      ax2.legend(loc="upper right")

      plt.title(f"Trial {trial}")
      plt.tight_layout()

   return {
      "trial": trial,
      "correct": correct,
      "time_before_change": time_before_change if correct else None,
      "time_to_rew": time_to_rew,
      "predicted": [pred_goal_zone_cp, real_goal_zone],
      "penalty_used": chosen_pen,
      "fig": fig,      
      'changepoint_ind': changepoint,
      'frames': rewloc_ind,
      'position_mae': pos_mae
   }
from matplotlib.backends.backend_pdf import PdfPages

def run_trials_and_save_pdf(
   trial_list,
   fc3,
   ybinned,
   goal_zone,
   tuning,
   ep_trials,
   rewlocs,
   rewsize,
   time,
   lick_trial,
   decode_trial_fn,
   min_frac=0.05,
   pdf_filename="trial_plots.pdf"
):
   """Run process_trial in parallel and save selected trial plots to PDF."""
   results = Parallel(n_jobs=6)(
      delayed(process_trial)(
         trial,
         fc3,
         ybinned,
         goal_zone,
         tuning,
         ep_trials,
         rewlocs,
         rewsize,
         time,
         lick_trial,
         decode_trial_fn,
         min_frac,
         make_plot=True  # only these trials make plots
      )
      for trial in trial_list
   )

   with PdfPages(pdf_filename) as pdf:
      for res in results:
         if res is not None and res["fig"] is not None:
            pdf.savefig(res["fig"])
            plt.close(res["fig"])  # free memory
   return results

def get_rewzones(rewlocs, gainf):
   """
   spliy into 6!
   """
   # Initialize the reward zone numbers array with zeros
   rewzonenum = np.zeros(len(rewlocs))
   bounds = np.array([[40,86],[86,120],[120,160]])*gainf
   # early v. late prediction
   # half_bound = [bound[1]/2 for bound in bounds]
   # new_bounds = [[bounds[ii][0],half_bound[ii]] for ii,nb in enumerate(half_bound)]
   # new_bounds2 = [[half_bound[ii],bounds[ii][1]] for ii,nb in enumerate(half_bound)]
   # all_new_bounds = np.array(new_bounds + new_bounds2)
   all_new_bounds=np.sort(bounds,axis=0)
   # Iterate over each reward location to determine its reward zone
   for kk, loc in enumerate(rewlocs):
      for rz,rewzones in enumerate(all_new_bounds):
         if rewzones[0] <= loc <= rewzones[1]:
            rewzonenum[kk] = rz+1  # Reward zone 1
            
   return rewzonenum


def get_success_failure_trials(trialnum, reward):
   """
   Counts the number of success and failure trials.

   Parameters:
   trialnum : array-like, list of trial numbers
   reward : array-like, list indicating whether a reward was found (1) or not (0) for each trial

   Returns:
   success : int, number of successful trials
   fail : int, number of failed trials
   str : list, successful trial numbers
   ftr : list, failed trial numbers
   ttr : list, trial numbers excluding probes
   total_trials : int, total number of trials excluding probes
   """
   trialnum = np.array(trialnum)
   reward = np.array(reward)
   unique_trials = np.unique(trialnum)
   
   success = 0
   fail = 0
   str_trials = []  # success trials
   ftr_trials = []  # failure trials
   probe_trials = []

   for trial in unique_trials:
      if trial >= 3:  # Exclude probe trials
         trial_indices = trialnum == trial
         if np.any(reward[trial_indices] == 1):
               success += 1
               str_trials.append(trial)
         else:
               fail += 1
               ftr_trials.append(trial)
      else:
         probe_trials.append(trial)
   
   total_trials = np.sum(unique_trials)
   ttr = unique_trials  # trials excluding probes

   return success, fail, str_trials, ftr_trials, probe_trials, ttr, total_trials
# iis=iis[iis>2]
#%%
# iis=iis[ii s>199] # control v inhib x ex
dct={}
for ii in iis:
   # ---------- Load animal info ---------- #
   day = conddf.days.values[ii]
   animal = conddf.animals.values[ii]
   in_type = conddf.in_type.values[ii]
   plane = 2 if animal in ['e145', 'e139'] else 0
   params_pth = f"Y:/analysis/fmats/{animal}/days/{animal}_day{day:03d}_plane{plane}_Fall.mat"
   print(params_pth)

   # ---------- Load required variables ---------- #
   keys = ['coms', 'changeRewLoc', 'ybinned', 'VR', 
         'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
         'timedFF', 'stat', 'licks']
   fall = scipy.io.loadmat(params_pth, variable_names=keys)
   VR = fall['VR'][0][0]
   scalingf = VR['scalingFACTOR'][0][0]
   rewsize = VR['settings']['rewardZone'][0][0][0][0] / scalingf if 'rewardZone' in VR['settings'].dtype.names else 10
   track_length=180/scalingf    
   # ---------- Preprocess variables ---------- #
   ybinned = fall['ybinned'][0] / scalingf
   forwardvel = fall['forwardvel'][0]
   trialnum = fall['trialnum'][0]
   rewards = fall['rewards'][0]
   licks = fall['licks'][0]
   time = fall['timedFF'][0]
   if animal == 'e145':  # Trim 1 sample
      trim_len = len(ybinned) - 1
      ybinned = ybinned[:trim_len]
      forwardvel = forwardvel[:trim_len]
      trialnum = trialnum[:trim_len]
      rewards = rewards[:trim_len]
      licks = licks[:trim_len]
      time = time[:trim_len]

   # ---------- Define epochs ---------- #
   changeRewLoc = np.hstack(fall['changeRewLoc'])
   eps = np.where(changeRewLoc > 0)[0]
   rewlocs = changeRewLoc[eps] / scalingf
   eps = np.append(eps, len(changeRewLoc))
   dt=np.nanmedian(np.diff(time))
   # Pick training/testing epochs
   optoep = conddf.optoep.values[ii]
   
   if optoep>1 or (in_type=='vip' and optoep==0):
      eptest = optoep if optoep >= 2 else random.randint(2, 3)
      if len(eps) < 4: eptest = 2
      ep_train = eptest - 2
      lick_rate = smooth_lick_rate(licks, np.nanmedian(np.diff(time)))
      # ---------- Load fluorescence data ---------- #
      fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
      Fc3 = fall_fc3['Fc3']
      dFF = fall_fc3['dFF']
      if in_type=='vip' or animal=='z17':
         Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
         dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
      else:
         Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
         dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]
      skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
      skewthres=1.2
      Fc3 = Fc3[:, skew>skewthres] # only keep cells with skew greateer than 2
      lick_position=np.zeros_like(licks)
      lick_position[licks>0] = ybinned[licks>0]
      ybinned_rel = []
      # Assume these exist:
      # Fc3: shape (time, n_cells)
      # trialnum: shape (time,) indicating trial number
      # lick_position: shape (time,) with lick position at each timepoint
      trial_X =[]
      trial_y =[]
      lick_rel = []
      trial_pos = []
      trial_states = []
      strind = []
      flind = []
      probeind = []
      lick_rate_trial = []
      ep_trials = []
      all_trial_num=[]
      # 50 msec time bin
      # no binning for now
      bin_size = int(0.05 * 1/np.nanmedian(np.diff(time)))  # number of frames in 100ms
      rzs= get_rewzones(rewlocs,1/scalingf)
      for ep in range(len(eps)-1):
         eprng = np.arange(eps[ep],eps[ep+1])
         unique_trials = np.unique(trialnum[eprng])
         success, fail, str_trials, ftr_trials, probe_trials, ttr, total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
         strials_ind = np.array([si for si,st in enumerate(ttr) if ttr[si] in str_trials])
         ttrind=np.arange(len(ttr))
         if ep>0: 
            strials_ind=strials_ind+all_trial_num[-1][-1]+1
            ftrials_ind=ftrials_ind+all_trial_num[-1][-1]+1
            ttrind=ttrind+all_trial_num[-1][-1]+1
         probe_ind = np.array([si for si,st in enumerate(ttr) if ttr[si] in probe_trials])      

         ftrials_ind = np.array([si for si,st in enumerate(ttr) if ttr[si] in ftr_trials])      
         strind.append(strials_ind)
         flind.append(ftrials_ind)      
         all_trial_num.append(ttrind)
         lick_position_rel = lick_position[eprng]-(rewlocs[ep]+rewsize/2)
         ypos = ybinned[eprng]
         lick_position_rel = lick_position_rel.astype(float)
         lick_position_rel[lick_rate[eprng]>9]=np.nan
         lick_position_rel[licks[eprng]==0]=np.nan
         for tt,tr in enumerate(unique_trials):
            tr_mask = trialnum[eprng] == tr
            fc_trial = Fc3[eprng][tr_mask, :]                  # shape (t, n_cells)
            # remove later activity
            ypos_tr = ypos[tr_mask]
            fc_trial[(ypos_tr<3)]=0
            lick_trial = lick_rate[eprng][tr_mask]         # shape (t,)
            lick_trial[(ypos_tr<3)]=0
            ypos_tr[(ypos_tr<3)]=0
            if fc_trial.shape[0] >= 10:#; dont exclude probes for now
               fc_trial_binned = fc_trial
               trial_X.append(fc_trial_binned)
               if tr<3 and ep>0:
                  trial_y.append(rzs[ep-1])
                  probeind.append(tt+all_trial_num[-2][-1]+1)
                  print(ep,tr,tt+all_trial_num[-2][-1]+1)
               else:
                  trial_y.append(rzs[ep])
               if tr<3 and ep==0: 
                  print(ep,tr,tt)
                  probeind.append(tt)
               # trial_pos.append(ypos_tr[(ypos_tr<((rewlocs[ep]-rewsize/2)))])
               trial_pos.append(ypos_tr)
               lick_rate_trial.append(lick_trial)
               ep_trials.append(ep)
               
      # trial_y = get_rewzones(trial_y, 1/scalingf)
      max_time = np.nanmax([len(xx) for xx in trial_X])
      strind=np.concatenate(strind)
      flind = np.concatenate(flind)
      probeind=np.array(probeind)
      trial_fc_org = np.zeros((len(trial_X),max_time,trial_X[0].shape[1]))
      for trind,trx in enumerate(trial_X):
         trial_fc_org[trind,:len(trx)]=trx
         
      X = np.stack(trial_fc_org)   # shape (n_trials, time, n_cells)
      # Reshape to (n_trials*time, n_cells) for cell-wise standardization
      n_trials, t, n_cells = X.shape
      trial_pos_ = np.zeros((len(trial_pos),max_time))
      for trind,trx in enumerate(trial_pos):
         trial_pos_[trind,:len(trx)]=trx

      trial_lick = np.zeros((len(lick_rate_trial),max_time))
      for trind,trx in enumerate(lick_rate_trial):
         trial_lick[trind,:len(trx)]=trx
         
      # Sample data: (trials x time x cells)
      # only correct trials?
      n_trials, T, N = trial_fc_org.shape
      fc3 = trial_fc_org
      ybinned = trial_pos_
      lick_trial = trial_lick
      goal_zone = np.array(trial_y)-1

      n_pos_bins = 90
      n_goals = 3
      dt = np.nanmedian(np.diff(time))  # 10 ms per bin
      var_pos = 20  # cm^2
      p_stay_goal = .9 ** dt

      all_indices=np.arange(fc3.shape[0])
      # only for opto ep split into 70/30
      ep_trials=np.array(ep_trials)
      opto_trials = all_indices[ep_trials==eptest-1]
      # Split indices instead of the data directly
      train_idx, test_idx = train_test_split(opto_trials, test_size=0.3, random_state=42)
      # add back other ep to training data
      train_idx = np.append(train_idx, all_indices[ep_trials!=eptest-1])
      # Now use the indices to subset your data
      fc3_train, fc3_test = fc3[train_idx], fc3[test_idx]
      ybinned_train, ybinned_test = ybinned[train_idx], ybinned[test_idx]
      goal_zone_train, goal_zone_test = goal_zone[train_idx], goal_zone[test_idx]
      lick_trial_train, lick_trial_test = lick_trial[train_idx], lick_trial[test_idx]
      # training ; use held out trials?
      tuning = estimate_tuning(fc3_train, ybinned_train, goal_zone_train)

      # Parallel execution
      subset_trials = np.sort(test_idx)  # choose trials you want figures for
      results = run_trials_and_save_pdf(
         subset_trials,
         fc3,
         ybinned,
         goal_zone,
         tuning,
         ep_trials,
         rewlocs,
         rewsize,
         time,
         lick_trial,
         decode_trial,
         min_frac=0,
         pdf_filename=os.path.join(savedst,f"{animal}_{day}_selected_trials.pdf")
      )
      # Filter out None results (failed trials)
      results = [r for r in results if r is not None]

      # Unpack results
      correct = [r["trial"] for r in results if r["correct"]]
      time_before_change = [r["time_before_change"] for r in results if r["correct"]]
      time_to_rew = [r["time_to_rew"] for r in results]
      predicted = [r["predicted"] for r in results]      # # Plot goal change 
      cps = [r['changepoint_ind'] for r in results]
      num_frames = [r['frames'] for r in results]
      pos_mae = [r['position_mae'] for r in results]
      
      # opto ind 
      opto_idx = subset_trials
      prev_rew_zone = rzs[eptest-2]
      if len(opto_idx)==0:
         continue   
      opto_s = [xx for xx in opto_idx if xx in strind]
      opto_f = [xx for xx in opto_idx if xx in flind]
      opto_p = [xx for xx in opto_idx if xx in probeind]
      opto_correct_s = [xx for xx in correct if xx in opto_s]
      opto_correct_f = [xx for xx in correct if xx in opto_f]
      opto_correct_p = [xx for xx in correct if xx in opto_p]
      if len(opto_s)>0:
         opto_s_rate = len(opto_correct_s)/len(opto_s)
      else: opto_s_rate=np.nan
      if len(opto_f)>0:
         opto_f_rate = len(opto_correct_f)/len(opto_f)
      else: opto_f_rate=np.nan
      if len(opto_p)>0:
         opto_p_rate = len(opto_correct_p)/len(opto_p)
      else: opto_p_rate=np.nan
      
      opto_time_before_predict_s = np.nanmean([xx for ii,xx in enumerate(time_before_change) if correct[ii] in opto_s])
      opto_time_before_predict_f = np.nanmean([xx for ii,xx in enumerate(time_before_change) if correct[ii] in opto_f])
      opto_time_before_predict_p = np.nanmean([xx for ii,xx in enumerate(time_before_change) if correct[ii] in opto_p])
      
      opto_idx = [hh for hh,xx in enumerate(test_idx) if ep_trials[xx]==eptest-1]   
      opto_time_to_rew = np.nanmean(np.array(time_to_rew)[opto_idx])

      print('####################################')
      print(f'opto correct prediction rate: {opto_s_rate*100:.2g}%')
      print(f'opto incorrect prediction rate: {opto_f_rate*100:.2g}%')
      print(f'opto probe prediction rate: {opto_p_rate*100:.2g}%')
      print(f'opto prediction latency (correct trials): {opto_time_before_predict_s:.2g}s')
      print(f'opto prediction latency (incorrect trials): {opto_time_before_predict_f:.2g}s')
      print(f'opto prediction latency (probe trials): {opto_time_before_predict_p:.2g}s')
      print(f'opto average time to rew: {opto_time_to_rew:.2g}s')

      dct[f'{animal}_{day}']=[opto_s_rate, opto_f_rate, opto_p_rate, opto_time_before_predict_s, opto_time_before_predict_f, opto_time_before_predict_p, opto_time_to_rew,
         predicted,rzs,eps,opto_idx,strind,flind,probeind,cps,num_frames,pos_mae]
# last few to find patterns in prediction accuracy
# %%
df=pd.DataFrame()
plt.rc('font', size=18)          # controls default text sizes

df['opto_s_rate'] = [v[0] for k, v in dct.items()]
df['opto_f_rate'] = [v[1] for k, v in dct.items()]
df['opto_p_rate'] = [v[2] for k, v in dct.items()]
df['opto_time_before_predict_s'] = [v[3] for k, v in dct.items()]
df['opto_time_before_predict_f'] = [v[4] for k, v in dct.items()]
df['opto_time_before_predict_p'] = [v[5] for k, v in dct.items()]
df['opto_time_to_rew'] = [v[6] for k, v in dct.items()]
df['pos_mae']=[np.nanmean(v[16]) for k, v in dct.items()] # position mean abs error

df['animals'] = [k.split('_')[0] for k, v in dct.items()]
df['days'] = [int(k.split('_')[1]) for k, v in dct.items()]
# average correct/incorrect
df['opto_s_rate'] = df[['opto_s_rate', 'opto_f_rate']].mean(axis=1)
df['opto_s_rate']=df['opto_s_rate']*100

# df['opto_time_before_predict_s'] = df[['opto_time_before_predict_s', 'opto_time_before_predict_f']].mean(axis=1)
df['opto_time_before_predict_s']=df['opto_time_before_predict_s']*100
df_shuffle = pd.read_csv(r'Z:\saved_datasets\decoding_shuffle_202509.csv')

cdf = conddf.copy()
df = pd.merge(df, cdf, on=['animals', 'days'], how='inner')
# df=df[df.opto_s_rate>0]

df['type']=[xx if 'vip' in xx else 'ctrl' for xx in df.in_type]

df=df[(df.animals!='e189')&(df.animals!='e190')]
# remove outlier days

df=df[~((df.animals=='e216')&(df.days.isin([55,57])))]
df=df[~((df.animals=='e218')&(df.days.isin([55])))]

# df.to_csv(r'Z:\saved_datasets\bayesian_goal_decoding_70_30_split.csv', index=None)
pl=['slategray','red','darkgoldenrod']
order = ['ctrl','vip','vip_ex']

fig,axes=plt.subplots(ncols=2)
ax=axes[0]
var='opto_s_rate'
df=df.groupby(['animals','days','type']).mean(numeric_only=True)
dfan=df.groupby(['animals','type']).mean(numeric_only=True)
sns.barplot(x='type',y=var,data=df,fill=False,ax=ax,palette=pl)
sns.stripplot(x='type',y=var,data=df,ax=ax,alpha=0.3,palette=pl)
sns.stripplot(x='type',y=var,data=dfan,ax=ax,s=10,alpha=.7,palette=pl)
sns.barplot(data=df_shuffle, # correct shift
        x='type', y=var,color='grey', 
        label='shuffle', alpha=0.4, err_kws={'color': 'grey'},errorbar=None,ax=ax)

ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'], rotation=20)
df = df.reset_index()
group_counts = df.groupby("type")[var].count()
# Add text annotations above each bar
for i, t in enumerate(order):
    n = group_counts.get(t, 0)
    y = df[df['type'] == t][var].mean()  # bar height (mean)
    ax.text(i, y + 0.05*y, f"n={n}", ha='center', va='bottom', fontsize=10)
ax.set_xlabel('')
# sns.barplot(x='condition',y='f_rate',data=df)
# Group by animal, type, and condition to get per-animal means
def add_sig_bar(ax, x1, x2, y, h, p,t):
    """
    Draws a significance bar with asterisks between x1 and x2 at height y+h.
    """
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='k')
    if p < 0.001:
        stars = '***'
    elif p < 0.01:
        stars = '**'
    elif p < 0.05:
        stars = '*'
    else:
        stars = 'n.s.'
    ax.text((x1+x2)/2, y+h, f'p={p:.3g}', ha='center', va='bottom',fontsize=10)
    ax.text((x1+x2)/2, y-h*4, f't={t:.3g}', ha='center', va='bottom',fontsize=10)
# Example: compare ctrl vs vip and ctrl vs vipex
df = df.reset_index()
ymax = df[var].max()
h = ymax * 0.1  # bracket height step

pairs = [('ctrl','vip'), ('ctrl','vip_ex')]
for i,(a,b) in enumerate(pairs):
   x1 = df.loc[df.type==a, var].dropna().values
   x2 = df.loc[df.type==b, var].dropna().values
   t, p = scipy.stats.ranksums(x1, x2)
   print(a,b,p)
   add_sig_bar(ax,
               order.index(a),
               order.index(b),
               ymax*1.05 + i*h,
               h*0.2,  # short vertical tick
               p,t)
ax.set_ylabel('Accuracy (%)')

var='opto_time_before_predict_s'
ax=axes[1]# df=df.groupby(['animals','days','type']).mean(numeric_only=True)
sns.barplot(x='type',y=var,data=df,fill=False,ax=ax,palette=pl)
sns.stripplot(x='type',y=var,data=df,ax=ax,alpha=0.3,hue='type',palette=pl)
sns.stripplot(x='type',y=var,data=dfan,ax=ax,s=10,alpha=.7,hue='type',palette=pl)
df_grouped = df.reset_index()
dfan=dfan.reset_index()
# Add lines per animal
ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'], rotation=20)
ax.set_xlabel('')
group_counts = df.groupby("type")[var].count()
# Add text annotations above each bar
for i, t in enumerate(order):
    n = group_counts.get(t, 0)
    y = df[df['type'] == t][var].mean()  # bar height (mean)
    ax.text(i, y + 0.05*y, f"n={n}", ha='center', va='bottom', fontsize=10)

# Final tweaks
ax.set_ylabel('% Time in correct prediction')
# Example: compare ctrl vs vip and ctrl vs vipex
df = df.reset_index()
order = ['ctrl','vip','vip_ex']
ymax = df[var].max()
h = ymax * 0.1  # bracket height step

pairs = [('ctrl','vip'), ('ctrl','vip_ex')]
for i,(a,b) in enumerate(pairs):
   x1 = df.loc[df.type==a, var].dropna().values
   x2 = df.loc[df.type==b, var].dropna().values
   t, p = scipy.stats.ranksums(x1, x2)
   print(a,b,p)
   add_sig_bar(ax,
               order.index(a),
               order.index(b),
               ymax*1.05 + i*h,
               h*0.2,  # short vertical tick
               p,t)

sns.despine()
fig.suptitle('Goal decoding performance (LED on)')
plt.tight_layout()
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
# plt.savefig(os.path.join(savedst, 'goal_decoding_opto.svg'), bbox_inches='tight')
#%
# pos decoding
# %%
 
fig,ax=plt.subplots(figsize=(3.2,4))
var='pos_mae'
df=df.groupby(['animals','days','type']).mean(numeric_only=True)
dfan=df.groupby(['animals','type']).mean(numeric_only=True).reset_index()
sns.barplot(x='type',y=var,data=df,fill=False,ax=ax,palette=pl)
sns.stripplot(x='type',y=var,data=df,ax=ax,alpha=0.3,palette=pl)
sns.stripplot(x='type',y=var,data=dfan,ax=ax,s=10,alpha=.7,palette=pl)

ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'], rotation=20)
df = df.reset_index()
group_counts = df.groupby("type")[var].count()
# Add text annotations above each bar
for i, t in enumerate(order):
    n = group_counts.get(t, 0)
    y = df[df['type'] == t][var].mean()  # bar height (mean)
    ax.text(i, y + 0.05*y, f"n={n}", ha='center', va='bottom', fontsize=10)
ax.set_xlabel('')
# sns.barplot(x='condition',y='f_rate',data=df)
# Group by animal, type, and condition to get per-animal means
def add_sig_bar(ax, x1, x2, y, h, p,t):
    """
    Draws a significance bar with asterisks between x1 and x2 at height y+h.
    """
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='k')
    if p < 0.001:
        stars = '***'
    elif p < 0.01:
        stars = '**'
    elif p < 0.05:
        stars = '*'
    else:
        stars = 'n.s.'
    ax.text((x1+x2)/2, y+h, f'p={p:.3g}', ha='center', va='bottom',fontsize=10)
    ax.text((x1+x2)/2, y-h*4, f't={t:.3g}', ha='center', va='bottom',fontsize=10)
# Example: compare ctrl vs vip and ctrl vs vipex
ymax = df[var].max()-10
h = ymax * 0.1  # bracketheight step

pairs = [('ctrl','vip'), ('ctrl','vip_ex')]
for i,(a,b) in enumerate(pairs):
   x1 = df.loc[df.type==a, var].dropna().values
   x2 = df.loc[df.type==b, var].dropna().values
   t, p = scipy.stats.ranksums(x1, x2)
   print(a,b,p)
   add_sig_bar(ax,
               order.index(a),
               order.index(b),
               ymax*1.05 + i*h,
               h*0.2,  # short vertical tick
               p,t)
ax.set_ylabel('Mean absolute error (cm)')
sns.despine()
# fig.suptitle('Position decoding performance (LED on)')
plt.tight_layout()
# plt.savefig(os.path.join(savedst, 'pos_decoding_opto.svg'), bbox_inches='tight')



#%%
"""
zahra
2025
dff by trial type
probes 1,2,3
pre-first correct incorrects
no lick incorrects
close incorrects
big miss incorrects
"""
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
from projects.pyr_reward.placecell import make_tuning_curves_by_trialtype_w_probes_w_darktime
from projects.pyr_reward.rewardcell import get_radian_position_first_lick_after_rew, get_rewzones, intersect_arrays
from projects.opto.behavior.behavior import smooth_lick_rate
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper\panels_main_figures'
#%%
goal_cm_window=20 # to search for rew cells
lasttr=8 #  last trials
bins=90
# iterate through all animals
tcs_allcelltypes = []
coms_allcelltypes=[]
lick_all=[]
vel_all=[]
epoch_perm=[]
for ii in range(len(conddf)):
   day = conddf.days.values[ii]
   animal = conddf.animals.values[ii]
   if (animal!='e217') & (conddf.optoep.values[ii]<2):
      if animal=='e145' or animal=='e139': pln=2 
      else: pln=0
      params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
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
      track_length_rad = track_length*(2*np.pi/track_length)
      bin_size=track_length_rad/bins 
      rz =  get_rewzones(rewlocs,1/scalingf)  
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
      tcs_correct, coms_correct, tcs_fail, coms_fail, tcs_probe, coms_probe, tcs_bigmiss, coms_bigmiss, tcs_no_lick, coms_no_lick, tcs_precorr, coms_precorr, ybinned_dt,relpos_all_ep = make_tuning_curves_by_trialtype_w_probes_w_darktime(eps,rewlocs,
         rewsize,ybinned,time,lick,
         Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
         bins=bins_dt)        
      dt = np.nanmedian(np.diff(time))
      lick_rate=smooth_lick_rate(lick,dt)
      # lick and velocity tc
      lick_tcs_correct, lick_coms_correct, lick_tcs_fail, lick_coms_fail, lick_tcs_probe, lick_coms_probe, lick_tcs_bigmiss, lick_coms_bigmiss, lick_tcs_no_lick, lick_coms_no_lick, lick_tcs_precorr, lick_coms_precorr, ybinned_dt,relpos_all_ep = make_tuning_curves_by_trialtype_w_probes_w_darktime(eps,rewlocs,
      rewsize,ybinned,time,lick,np.array([lick_rate]).T,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
      bins=bins_dt)  
      vel_tcs_correct, vel_coms_correct, vel_tcs_fail, vel_coms_fail, vel_tcs_probe, vel_coms_probe, vel_tcs_bigmiss, vel_coms_bigmiss, vel_tcs_no_lick, vel_coms_no_lick, vel_tcs_precorr, vel_coms_precorr, ybinned_dt,relpos_all_ep = make_tuning_curves_by_trialtype_w_probes_w_darktime(eps,rewlocs,
      rewsize,ybinned,time,lick,np.array([forwardvel]).T,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
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
      tcs=[]; coms=[]
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
         tcs.append([tcs_correct[:,goal_cells], tcs_fail[:,goal_cells], tcs_probe[:,:,goal_cells], tcs_bigmiss[:,goal_cells],  tcs_no_lick[:,goal_cells], tcs_precorr[:,goal_cells]])
         coms.append([coms_correct[:,goal_cells],  coms_fail[:,goal_cells],  coms_probe[:,:,goal_cells], tcs_bigmiss[:,goal_cells], coms_bigmiss[:,goal_cells], coms_no_lick[:,goal_cells], coms_precorr[:,goal_cells]])
      # look at 1to2 or 3to1 transitions
      epoch_perm.append([perm,rz_perm,rewlocs]) 
      tcs_allcelltypes.append(tcs)
      coms_allcelltypes.append(coms)
      lick_all.append([lick_tcs_correct, lick_tcs_fail,lick_tcs_probe,  lick_tcs_bigmiss, lick_tcs_no_lick,lick_tcs_precorr])
      vel_all.append([vel_tcs_correct, vel_tcs_fail, vel_tcs_probe,  vel_tcs_bigmiss, vel_tcs_no_lick, vel_tcs_precorr])


#%%
# split trial types
tcs_correct_all=[[yy[0] for yy in xx] for xx in tcs_allcelltypes]
tcs_fail_all=[[yy[1] for yy in xx] for xx in tcs_allcelltypes]
tcs_probes_all=[[yy[2] for yy in xx] for xx in tcs_allcelltypes]
tcs_bigmiss_all=[[yy[3] for yy in xx] for xx in tcs_allcelltypes]
tcs_no_lick_all=[[yy[4] for yy in xx] for xx in tcs_allcelltypes]
tcs_precorr_all=[[yy[5] for yy in xx] for xx in tcs_allcelltypes]
# lick
lick_tcs_correct_all=[np.squeeze(xx[0]) for xx in lick_all]
lick_tcs_fail_all=[np.squeeze(xx[1]) for xx in lick_all]
lick_tcs_probes_all=[np.squeeze(xx[2]) for xx in lick_all]
lick_tcs_bigmiss_all=[np.squeeze(xx[3]) for xx in lick_all]
lick_tcs_no_lick_all=[np.squeeze(xx[4]) for xx in lick_all]
lick_tcs_precorr_all=[np.squeeze(xx[5]) for xx in lick_all]
#vel
vel_tcs_correct_all=[np.squeeze(xx[0]) for xx in vel_all]
vel_tcs_fail_all=[np.squeeze(xx[1]) for xx in vel_all]
vel_tcs_probes_all=[np.squeeze(xx[2]) for xx in vel_all]
vel_tcs_bigmiss_all=[np.squeeze(xx[3]) for xx in vel_all]
vel_tcs_no_lick_all=[np.squeeze(xx[4]) for xx in vel_all]
vel_tcs_precorr_all=[np.squeeze(xx[5]) for xx in vel_all]

# per day per animal
import scipy.stats
# Normalize each row to [0, 1]
def normalize_rows_0_to_1(arr):
   row_max = np.nanmax(arr, axis=1, keepdims=True)
   # Identify rows where max is NaN (i.e., all values were NaN)
   bad_rows = np.isnan(row_max).flatten()
   row_max[bad_rows] = 1  # avoid division by zero
   normed = arr / row_max
   normed[bad_rows] = 0.001  # set all-NaN rows to 0
   return normed

plt.rc('font', size=16)

# --- Settings ---
animals = [xx for ii, xx in enumerate(conddf.animals.values) if (xx != 'e217') and (conddf.optoep.values[ii] < 2)]
animals_test = np.unique(animals)
animals_test = [ 'e145', 'e186', 'e189', 'e190', 'e200', 'e201', 'e216',
       'e218', 'z8', 'z9']
# animals_test = ['z9']

cell_types = ['pre', 'post', 'far_pre', 'far_post']
bins = 150
# recalc tc
dff_correct_per_type = []
dff_fail_per_type = []
cs_per_type = []
dff_probe_per_type=[]
cs_probe_per_type=[]
cs_probe_correct_per_type=[]
# --- Loop through cell types ---
for cll, cell_type in enumerate(cell_types):
   dff_correct_per_an = []
   dff_fail_per_an = []
   dff_probe_per_an = []
   cs_per_an = []
   cs_probe_per_an = []
   cs_probe_correct_per_an = []

   for animal in animals_test:
      # --- Initialize containers ---
      tcs_correct, tcs_fail = [], []
      tcs_probe =[]  # probe 0, 1, 2 traces

      if 'pre' in cell_type:
         activity_window = 'pre'
         win = slice(bins // 3, bins // 2)
      else:
         activity_window = 'post'
         win = slice(bins // 2, bins)
      # LICK
      lick_corr = np.vstack([xx for ii, xx in enumerate(lick_tcs_correct_all) if animals[ii] == animal])
      lick_fail = np.vstack([xx for ii, xx in enumerate(lick_tcs_fail_all) if animals[ii] == animal])
      lick_prob1 = np.vstack([xx[:, 0] for ii, xx in enumerate(lick_tcs_probes_all) if animals[ii] == animal])
      lick_prob2 = np.vstack([xx[:, 1] for ii, xx in enumerate(lick_tcs_probes_all) if animals[ii] == animal])
      lick_prob3 = np.vstack([xx[:, 2] for ii, xx in enumerate(lick_tcs_probes_all) if animals[ii] == animal])
      lick_bigmiss = np.vstack([xx for ii, xx in enumerate(lick_tcs_bigmiss_all) if animals[ii] == animal])
      lick_nolick = np.vstack([xx for ii, xx in enumerate(lick_tcs_no_lick_all) if animals[ii] == animal])
      lick_precorr = np.vstack([xx for ii, xx in enumerate(lick_tcs_precorr_all) if animals[ii] == animal])

      # VELOCITY
      vel_corr = np.vstack([xx for ii, xx in enumerate(vel_tcs_correct_all) if animals[ii] == animal])
      vel_fail = np.vstack([xx for ii, xx in enumerate(vel_tcs_fail_all) if animals[ii] == animal])
      vel_prob1 = np.vstack([xx[:, 0] for ii, xx in enumerate(vel_tcs_probes_all) if animals[ii] == animal])
      vel_prob2 = np.vstack([xx[:, 1] for ii, xx in enumerate(vel_tcs_probes_all) if animals[ii] == animal])
      vel_prob3 = np.vstack([xx[:, 2] for ii, xx in enumerate(vel_tcs_probes_all) if animals[ii] == animal])
      vel_bigmiss = np.vstack([xx for ii, xx in enumerate(vel_tcs_bigmiss_all) if animals[ii] == animal])
      vel_nolick = np.vstack([xx for ii, xx in enumerate(vel_tcs_no_lick_all) if animals[ii] == animal])
      vel_precorr = np.vstack([xx for ii, xx in enumerate(vel_tcs_precorr_all) if animals[ii] == animal])

      # --- Get correct trial data ---
      for ii, xx in enumerate(tcs_correct_all):
         if animals[ii] == animal:
            tc = xx[cll]
            if tc.shape[1] > 0:
               tc_avg = np.nanmean(tc,axis=0) #do not average across epochs??!
               if tc_avg.ndim == 1: tc_avg = np.expand_dims(tc_avg, 0)
               tcs_correct.append(tc_avg)

      # --- Get fail trial data ---
      for ii, xx in enumerate(tcs_fail_all):
         if animals[ii] == animal:
            tc = xx[cll]
            if tc.shape[1] > 0:
               tc_avg = np.nanmean(tc,axis=0)
               if tc_avg.ndim == 1: tc_avg = np.expand_dims(tc_avg, 0)
               tcs_fail.append(tc_avg)
      #probe
      for ii, xx in enumerate(tcs_probes_all):
         if animals[ii] == animal:
            tc = xx[cll]
            if tc.shape[2] > 0:
               # average across ep
               tc_avg = [np.nanmean(tc[:,k,:,:],axis=0) for k in range(3)]
               tcs_probe.append(tc_avg)
      tcs_bigmiss, tcs_nolick, tcs_precorr = [], [], []
      # Big miss
      for ii, xx in enumerate(tcs_bigmiss_all):
         if animals[ii] == animal:
            tc = xx[cll]
            if tc.shape[1] > 0:
               tc_avg = np.nanmean(tc, axis=0)
               if tc_avg.ndim == 1: tc_avg = np.expand_dims(tc_avg, 0)
               tcs_bigmiss.append(tc_avg)

      # No lick
      for ii, xx in enumerate(tcs_no_lick_all):
         if animals[ii] == animal:
            tc = xx[cll]
            if tc.shape[1] > 0:
               tc_avg = np.nanmean(tc, axis=0)
               if tc_avg.ndim == 1: tc_avg = np.expand_dims(tc_avg, 0)
               tcs_nolick.append(tc_avg)

      # Pre-correct
      for ii, xx in enumerate(tcs_precorr_all):
         if animals[ii] == animal:
            tc = xx[cll]
            if tc.shape[1] > 0:
               tc_avg = np.nanmean(tc, axis=0)
               if tc_avg.ndim == 1: tc_avg = np.expand_dims(tc_avg, 0)
               tcs_precorr.append(tc_avg)

      # --- Stack and sort ---
      tc_corr = np.vstack(tcs_correct)
      tc_fail = np.vstack(tcs_fail)
      tc_prob1=np.vstack([xx[0] for xx in tcs_probe])
      tc_prob2=np.vstack([xx[1] for xx in tcs_probe])
      tc_prob3=np.vstack([xx[2] for xx in tcs_probe])
      # Remove rows where all bins are NaN in correct trials (sets reference for sorting)
      # valid_rows = ~np.all(np.isnan(tc_fail), axis=1)
      # tc_fail = tc_fail[valid_rows]
      # tc_prob1 = tc_prob1[valid_rows]
      # tc_prob2 = tc_prob2[valid_rows]
      # tc_prob3 = tc_prob3[valid_rows]
      # Normalize
      tc_corr_norm = normalize_rows_0_to_1(tc_corr)
      valid_rows = ~np.all(np.isnan(tc_corr_norm), axis=1)
      tc_corr_norm = tc_corr_norm[valid_rows]
      sort_idx = np.argsort(np.nanargmax(tc_corr_norm, axis=1))
      # Sort all trial types using the same cell order
      tc_corr_sorted = tc_corr_norm[sort_idx]
      valid_rows = ~np.all(np.isnan(tc_fail), axis=1)
      tc_fail_sorted = normalize_rows_0_to_1(tc_fail)[sort_idx][valid_rows]
      # correct for mean calc
      tc_fail_sorted[np.isnan(tc_fail_sorted)]=0
      tc_fail=tc_fail[valid_rows]
      tc_fail[np.isnan(tc_fail)]=0
      valid_rows = ~np.all(np.isnan(tc_prob1), axis=1)
      tc_prob1_sorted = normalize_rows_0_to_1(tc_prob1)[sort_idx][valid_rows]
      tc_prob1_sorted[np.isnan(tc_prob1_sorted)]=0
      tc_prob1=tc_prob1[valid_rows]
      tc_prob1[np.isnan(tc_prob1)]=0
      valid_rows = ~np.all(np.isnan(tc_prob2), axis=1)      
      tc_prob2_sorted = normalize_rows_0_to_1(tc_prob2)[sort_idx][valid_rows]
      tc_prob2_sorted[np.isnan(tc_prob2_sorted)]=0
      tc_prob2=tc_prob2[valid_rows]
      tc_prob2[np.isnan(tc_prob2)]=0
      valid_rows = ~np.all(np.isnan(tc_prob3), axis=1)
      tc_prob3_sorted = normalize_rows_0_to_1(tc_prob3)[sort_idx][valid_rows]
      tc_prob3_sorted[np.isnan(tc_prob3_sorted)]=0
      tc_prob3=tc_prob3[valid_rows]
      tc_prob3[np.isnan(tc_prob3)]=0
      tc_bigmiss = np.vstack(tcs_bigmiss)
      tc_nolick = np.vstack(tcs_nolick)
      tc_precorr = np.vstack(tcs_precorr)
      valid_rows = ~np.all(np.isnan(tc_bigmiss), axis=1)
      tc_bigmiss_sorted = normalize_rows_0_to_1(tc_bigmiss)[sort_idx][valid_rows]
      tc_bigmiss_sorted[np.isnan(tc_bigmiss_sorted)]=0
      tc_bigmiss=tc_bigmiss[valid_rows]
      tc_bigmiss[np.isnan(tc_bigmiss)]=0
      valid_rows = ~np.all(np.isnan(tc_nolick), axis=1)
      tc_nolick_sorted = normalize_rows_0_to_1(tc_nolick)[sort_idx][valid_rows]
      tc_nolick_sorted[np.isnan(tc_nolick_sorted)]=0
      tc_nolick=tc_nolick[valid_rows]
      tc_nolick[np.isnan(tc_nolick)]=0
      valid_rows = ~np.all(np.isnan(tc_precorr), axis=1)
      tc_precorr_sorted = normalize_rows_0_to_1(tc_precorr)[sort_idx][valid_rows]
      tc_precorr_sorted[np.isnan(tc_precorr_sorted)]=0
      tc_precorr=tc_precorr[valid_rows]
      tc_precorr[np.isnan(tc_precorr)]=0
      # --- dF/F from activity window ---
      dff_correct = np.nanmean(tc_corr[:, win], axis=1)
      dff_fail = np.nanmean(tc_fail[:, win], axis=1)
      dff_correct_per_an.append(dff_correct)
      dff_fail_per_an.append(dff_fail)

      # --- Cosine similarity (correct vs fail) ---
      # cs = [cosine_sim_ignore_nan(tc_corr[i], tc_fail[i]) for i in range(tc_corr.shape[0])]
      # cs_per_an.append(np.array(cs))
      fig, axes_org = plt.subplots(nrows=4, ncols=8, figsize=(20,10),   sharex=True,constrained_layout=True,sharey='row',gridspec_kw={'height_ratios': [3, 1, 1, 1]})
      axes = axes_org.flatten()
      titles = ['Correct', 'Incorrect', 'Probe 1', 'Probe 2', 'Probe 3','Pre-Correct Fails','Big Miss', 'No Lick']

      data_to_plot = [tc_corr_sorted, tc_fail_sorted,
                     tc_prob1_sorted, tc_prob2_sorted, tc_prob3_sorted,tc_precorr_sorted,tc_bigmiss_sorted, tc_nolick_sorted]
      for i, ax in enumerate(axes[:8]):
         im = ax.imshow(data_to_plot[i], aspect='auto', vmin=0, vmax=1, cmap='viridis')
         ax.axvline(bins // 2, color='w', linestyle='--',linewidth=3)
         ax.set_title(titles[i])
         ax.set_xticks([0, bins // 2, bins])
         ax.set_xticklabels(['-$\\pi$', 0, '$\\pi$'])
         if i % 3 == 0:
            ax.set_ylabel('Cells (sorted)')
         trace_data = [tc_corr, tc_fail,
                     tc_prob1, tc_prob2, tc_prob3,tc_precorr,tc_bigmiss, tc_nolick]
         colors = ['seagreen', 'firebrick', 'royalblue', 'goldenrod', 'purple', 'k', 'gray', 'dodgerblue']
         a=0.1
         for i in range(8):
            ax = axes[8 + i]
            if not i == 0:
               ax.sharey(axes[8])
            tc = trace_data[i]
            if len(tc) == 0: continue            
            sem = scipy.stats.sem(tc, axis=0, nan_policy='omit')
            m = np.nanmean(tc,axis=0)
            ax.plot(m, color=colors[i])
            ax.fill_between(np.arange(len(m)), m - sem, m + sem, color=colors[i], alpha=a)
            ax.axvline(bins // 2, color='k', linestyle='--',linewidth=2)
            ax.set_xticks([0, bins // 2, bins])
            ax.set_xticklabels(['-$\\pi$', 0, '$\\pi$'])
            ax.set_title(f'{titles[i]} Mean')
         lick_data = [lick_corr, lick_fail, lick_prob1, lick_prob2, lick_prob3,lick_precorr,lick_bigmiss, lick_nolick]

         for i in range(8):
            ax = axes[16 + i]
            trace = lick_data[i]
            m = np.nanmean(trace, axis=0)
            sem = scipy.stats.sem(trace, axis=0, nan_policy='omit')
            ax.plot(m, color=colors[i])
            ax.fill_between(np.arange(len(m)), m - sem, m + sem, color=colors[i], alpha=a)
            ax.axvline(bins // 2, color='k', linestyle='--',linewidth=2)
            ax.set_xticks([0, bins // 2, bins])
            ax.set_xticklabels(['-$\\pi$', 0, '$\\pi$'])
            ax.set_title(f'{titles[i]} Lick')
         vel_data = [vel_corr, vel_fail, vel_prob1, vel_prob2, vel_prob3,vel_precorr,vel_bigmiss, vel_nolick]
         for i in range(8):
            ax = axes[24 + i]
            trace = vel_data[i]
            m = np.nanmean(trace, axis=0)
            sem = scipy.stats.sem(trace, axis=0, nan_policy='omit')
            ax.plot(m, color=colors[i])
            ax.fill_between(np.arange(len(m)), m - sem, m + sem, color=colors[i], alpha=a)
            ax.axvline(bins // 2, color='k', linestyle='--')
            ax.set_xticks([0, bins // 2, bins])
            ax.set_xticklabels(['-$\\pi$', 0, '$\\pi$'])
            ax.set_title(f'{titles[i]} Velocity')
      fig.suptitle(f'{animal}, {cell_type}')
            # plt.close(fig)

      #   # Store results
      #   dff_correct_per_type.append(dff_correct_per_an)
      #   dff_fail_per_type.append(dff_fail_per_an)
      # #   dff_probe_per_type.append(dff_probe_per_an)
      #   cs_per_type.append(cs_per_an)
      # #   cs_probe_per_type.append(cs_probe_per_an)
      # #   cs_probe_correct_per_type.append(cs_probe_correct_per_an)
      import scipy.stats
      # --- New Figure ---
      fig_overlay, axes_overlay = plt.subplots(3, 3, figsize=(8, 8),sharex=True, sharey=True)
      axes_overlay=axes_overlay.flatten()
      for i in range(8):
         ax = axes_overlay[i]
         tc = trace_data[i]     # shape: (trials, bins)
         lick = lick_data[i]
         vel = vel_data[i]

         # --- Mean and SEM ---
         m_tc = np.nanmean(tc, axis=0)
         sem_tc = scipy.stats.sem(tc, axis=0, nan_policy='omit')

         m_lick = np.nanmean(lick, axis=0)
         sem_lick = scipy.stats.sem(lick, axis=0, nan_policy='omit')

         m_vel = np.nanmean(vel, axis=0)
         sem_vel = scipy.stats.sem(vel, axis=0, nan_policy='omit')

         # --- Z-score normalization (standardize each signal) ---
         def zscore(x):
            return (x - np.nanmean(x)) / np.nanstd(x)

         m_tc_z = zscore(m_tc)
         sem_tc_z = sem_tc / np.nanstd(m_tc)

         m_lick_z = zscore(m_lick)
         sem_lick_z = sem_lick / np.nanstd(m_lick)

         m_vel_z = zscore(m_vel)
         sem_vel_z = sem_vel / np.nanstd(m_vel)

         # --- Plot ---
         ax.plot(m_tc_z, color='cornflowerblue', label='Neural')
         ax.fill_between(np.arange(len(m_tc_z)), m_tc_z - sem_tc_z, m_tc_z + sem_tc_z, color='cornflowerblue', alpha=0.15)
         ax.plot(m_lick_z, color='k', label='Lick')
         ax.fill_between(np.arange(len(m_lick_z)), m_lick_z - sem_lick_z, m_lick_z + sem_lick_z, color='k', alpha=0.2)
         ax.plot(m_vel_z, color='gray', label='Velocity')
         ax.fill_between(np.arange(len(m_vel_z)), m_vel_z - sem_vel_z, m_vel_z + sem_vel_z, color='gray', alpha=0.2)
         ax.axvline(bins // 2, color='k', linestyle='--')
         ax.set_xticks([0, bins // 2, bins])
         ax.set_xticklabels(['-$\\pi$', 0, '$\\pi$'])
         ax.set_title(f'{titles[i]}')

         if i == 0:
            ax.legend(loc='upper right', fontsize=10)
      axes_overlay[8].axis('off')
      fig_overlay.suptitle(f'{animal},{cell_type}', fontsize=22)
      # fig_overlay.tight_layout(rect=[0, 0, 1, 0.93])
      plt.show()

#%%

# recalculate tc
dfsall = []
for cll,celltype in enumerate(cell_types):
    animals_unique = animals_test
    df=pd.DataFrame()
    correct = np.concatenate([np.concatenate(xx) for xx in dff_correct_per_type[cll]])
    incorrect = np.concatenate([np.concatenate(xx) if len(xx)>0 else [] for xx in dff_fail_per_type[cll]])
    df['mean_dff'] = np.concatenate([correct,incorrect])
    df['trial_type']=np.concatenate([['correct']*len(correct),['incorrect']*len(incorrect)])
    ancorr = np.concatenate([[animals_unique[ii]]*len(np.concatenate(xx)) for ii,xx in enumerate(dff_correct_per_type[cll])])
    anincorr = np.concatenate([[animals_unique[ii]]*len(np.concatenate(xx)) if len(xx)>0 else [] for ii,xx in enumerate(dff_fail_per_type[cll])])
    df['animal'] = np.concatenate([ancorr, anincorr])
    df['cell_type'] = [celltype]*len(df)
    dfsall.append(df)
    
# average
bigdf = pd.concat(dfsall)
bigdf=bigdf.groupby(['animal', 'trial_type', 'cell_type']).mean(numeric_only=True)
bigdf=bigdf.reset_index()
bigdf=bigdf[bigdf.animal!='e189']
s=12
cell_order = cell_types
fig,ax = plt.subplots(figsize=(6,4))
sns.stripplot(x='cell_type', y='mean_dff', data=bigdf,hue='trial_type',
        dodge=True,palette={'correct':'seagreen', 'incorrect': 'firebrick'},
        s=s,alpha=0.7,    order=cell_order)
sns.barplot(x='cell_type', y='mean_dff', data=bigdf,hue='trial_type',
        fill=False,palette={'correct':'seagreen', 'incorrect': 'firebrick'},
            order=cell_order)

ax.spines[['top','right']].set_visible(False)
ax.set_ylabel('Mean $\Delta F/F$ rel. to rew.')
ax.set_xlabel('Reward cell type')
ax.legend_.remove()
ax.set_xticklabels(['Pre', 'Post', 'Far pre', 'Far post'])
# Use the last axis to get handles/labels
handles, labels = ax.get_legend_handles_labels()
# Create a single shared legend with title "Trial type"
ax.legend(
    handles, labels,
    loc='upper left',
    bbox_to_anchor=(1.02, 1),
    borderaxespad=0.,
    title='Trial type'
)
xpos = {ct: i for i, ct in enumerate(cell_order)}

# Draw dim gray connecting lines between paired trial types
for animal in bigdf['animal'].unique():
    for ct in cell_order:
        sub = bigdf[(bigdf['animal'] == animal) & (bigdf['cell_type'] == ct)]
        if len(sub) == 2:  # both trial types present
            # Get x locations for dodge-separated points
            x_base = xpos[ct]
            offsets = [-0.2, 0.2]  # match sns stripplot dodge
            y_vals = sub.sort_values('trial_type')['mean_dff'].values
            x_vals = [x_base + offset for offset in offsets]
            ax.plot(x_vals, y_vals, color='dimgray', alpha=0.5, linewidth=1)

# ans = bigdf.animal.unique()
# for i in range(len(ans)):
#     for j,tr in enumerate(np.unique(bigdf.cell_type.values)):
#         testdf= bigdf[(bigdf.animal==ans[i]) & (bigdf.cell_type==tr)]
#         ax = sns.lineplot(x='trial_type', y='mean_dff', 
#         data=testdf,
#         errorbar=None, color='dimgray', linewidth=2, alpha=0.7,ax=ax)
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM

# 1) Two-way repeated measures ANOVA
aov = AnovaRM(
    data=bigdf,
    depvar='mean_dff',
    subject='animal',
    within=['trial_type','cell_type']
).fit()
print(aov)    # F-stats and p-values for main effects and interaction

# 2) Post-hoc paired comparisons: correct vs incorrect within each cell_type
posthoc = []
for ct in cell_order:
    sub = bigdf[bigdf['cell_type']==ct]
    cor = sub[sub['trial_type']=='correct']['mean_dff']
    inc = sub[sub['trial_type']=='incorrect']['mean_dff']
    t, p_unc = scipy.stats.ttest_rel(cor, inc)
    posthoc.append({
        'cell_type': ct,
        't_stat':    t,
        'p_uncorrected': p_unc
    })

posthoc = pd.DataFrame(posthoc)
# Bonferroni
posthoc['p_bonferroni'] = np.minimum(posthoc['p_uncorrected'] * len(posthoc), 1.0)
print(posthoc)
# map cell_type → x-position
xpos = {ct: i for i, ct in enumerate(cell_order)}
for _, row in posthoc.iterrows():
    x = xpos[row['cell_type']]
    y = bigdf[
        (bigdf['cell_type']==row['cell_type'])
    ]['mean_dff'].quantile(.7) + 0.1  # just above the tallest bar
    p = row['p_bonferroni']
    stars = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
    ax.text(x, y, stars, ha='center', va='bottom', fontsize=42)
    if p>0.05:
        ax.text(x, y, f'p={p:.2g}', ha='center', va='bottom', fontsize=12)
# Assuming `axes` is a list of subplots and `ax` is the one with the legend (e.g., the last one)

# Example interpretation (fill in with your numbers)
# trial_type (Num DF = 1, Den DF = 9, F = 12.3, p = 0.006)
# -- There is a significant main effect of trial type: across all cell types, mean ΔF/F is different on correct vs. incorrect trials.

# cell_type (Num DF = 3, Den DF = 27, F = 8.7, p < 0.001)
# -- There is a significant main effect of cell type: some cell types have higher overall ΔF/F than others, regardless of trial outcome.

# trial_type × cell_type (Num DF = 3, Den DF = 27, F = 4.2, p = 0.014)
# -- The interaction is significant: the difference between correct vs. incorrect ΔF/F depends on which cell type you look at.

# Because the interaction is significant, you should then examine post-hoc tests (e.g., the paired comparisons you ran) to see for each cell type whether correct vs. incorrect is significant.
# plt.savefig(os.path.join(savedst, 'allcelltype_trialtype.svg'),bbox_inches='tight')
#%%
# quantify cosine sim
# TODO: get COM per cell
dfsall = []
for cll,celltype in enumerate(cell_types):
    animals_unique = animals_test
    df=pd.DataFrame()
    cs = cs_per_type[cll]
    df['cosine_sim'] = np.concatenate(cs)
    # df['com'] = np.nanmax(tc) for tc
    ancorr = np.concatenate([[animals_unique[ii]]*len(xx) for ii,xx in enumerate(cs)])
    df['animal'] = ancorr
    df['cell_type'] = [celltype]*len(df)
    dfsall.append(df)
    
# average
bigdf = pd.concat(dfsall)
bigdf_avg = bigdf.groupby(['animal', 'cell_type'])['cosine_sim'].mean().reset_index()
bigdf = bigdf.groupby(['animal', 'cell_type']).mean().reset_index()
# Step 2: Check if data is balanced
pivoted = bigdf_avg.pivot(index='animal', columns='cell_type', values='cosine_sim')
if pivoted.isnull().any().any():
    print("⚠️ Warning: Data is unbalanced — some animals are missing data for some cell types.")
else:
    print("✅ Data is balanced.")

# Step 3: One-way repeated measures ANOVA
aov = AnovaRM(
    data=bigdf_avg,
    depvar='cosine_sim',
    subject='animal',
    within=['cell_type']
).fit()
print(aov)

# Step 4: Post-hoc paired comparisons between all cell types
posthoc = []
pvals = []

comb = [ ('post', 'far_post'),
 ('pre', 'far_pre')]
for ct1, ct2 in comb:
    sub1 = pivoted[ct1]
    sub2 = pivoted[ct2]
    t, p_unc = scipy.stats.wilcoxon(sub1, sub2)
    posthoc.append({
        'comparison': f"{ct1} vs {ct2}",
        'W-statistic': t,
        'p_uncorrected': p_unc
    })
    pvals.append(p_unc)

from statsmodels.stats.multitest import fdrcorrection
# Step 5: FDR correction
rej, p_fdr = fdrcorrection(pvals, alpha=0.05)
for i, fdr_p in enumerate(p_fdr):
    posthoc[i]['p_fdr'] = fdr_p
    posthoc[i]['significant'] = rej[i]
import matplotlib.pyplot as plt
import seaborn as sns

# Re-map cell type names to x-axis positions
cell_order = ['pre', 'post', 'far_pre', 'far_post']  # or whatever order you're using
xpos = {ct: i for i, ct in enumerate(cell_order)}
color='saddlebrown'
# Plot
fig,ax = plt.subplots(figsize=(5,6))
sns.stripplot(data=bigdf, x='cell_type', y='cosine_sim',
    order=cell_order,color=color,
    alpha=0.7, size=10,)
sns.barplot(data=bigdf, x='cell_type', y='cosine_sim',color=color,fill=False,
            order=cell_order, errorbar='se')
ax.set_xticklabels(['Pre', 'Post', 'Far pre', 'Far post'],rotation=30)
# Annotate pairwise comparisons
height = bigdf['cosine_sim'].max() + 0.02
step = 0.02
# Add lines connecting same-animal points across cell types
for animal, subdf in bigdf_avg.groupby('animal'):
    subdf = subdf.set_index('cell_type').reindex(cell_order)
    xs = [xpos[ct] for ct in subdf.index]
    ys = subdf['cosine_sim'].values
    ax.plot(xs, ys, marker='o', color='gray', linewidth=1, alpha=0.5, zorder=0)

posthoc=pd.DataFrame(posthoc)
for i, row in posthoc.iterrows():
    if row['significant']:
        group1, group2 = row['comparison'].split(' vs ')
        x1, x2 = xpos[group1], xpos[group2]
        y = height + i * step* 4

        # Connecting line
        ax.plot([x1, x1, x2, x2], [y, y + step / 2, y + step / 2, y], lw=1.2, color='black')

        # Stars
        p = row['p_fdr']
        stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        ax.text((x1 + x2) / 2, y + step / 2 + 0.002, stars, ha='center', va='bottom', fontsize=46)

# Axis formatting
ax.set_ylabel("Cosine similarity\n Correct vs. Incorrect tuning curve")
ax.set_xlabel("Cell type")
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
fig.suptitle('Tuning properties')
plt.savefig(os.path.join(savedst, 'allcelltype_corr_v_incorr_cs.svg'),bbox_inches='tight')

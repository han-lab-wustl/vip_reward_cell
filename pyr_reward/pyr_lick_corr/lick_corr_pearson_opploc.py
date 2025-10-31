
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

#%%
# initialize var
# radian_alignment_saved = {} # overwrite
plt.close('all')
bins = 150
cm_window=20
sessions = []
iis = np.arange(len(conddf))
# iis=np.arange(40,170)
# iterate through all animals
for ii in iis:
   day = conddf.days.values[ii]
   animal = conddf.animals.values[ii]
   if (animal!='e217') & (conddf.optoep.values[ii]<2):
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
      # bin_size=3
      # # position tc with different trial types
      # tcs_correct_abs, coms_correct_abs, tcs_fail_abs, coms_fail_abs, tcs_probe, coms_probe, tcs_bigmiss, coms_bigmiss, tcs_no_lick, coms_no_lick, tcs_precorr, coms_precorr= make_tuning_curves_abs_w_probe_types(eps,rewlocs,ybinned,Fc3,trialnum,lick,rewards,forwardvel,rewsize,bin_size)
      # # get cells that maintain their coms across at least 2 epochs
      # ##################################### PLACE 
      # place_window = 20 # cm converted to rad                
      # perm = list(combinations(range(len(coms_correct_abs)), 2))     
      # com_per_ep = np.array([(coms_correct_abs[perm[jj][0]]-coms_correct_abs[perm[jj][1]]) for jj in range(len(perm))])        
      # compc = [np.where((comr<place_window) & (comr>-place_window))[0] for comr in com_per_ep]
      # # get cells across all epochs that meet crit
      # pcs = np.unique(np.concatenate(compc))
      # pcs_all = intersect_arrays(*compc)
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
      ######################## pre reward only
      bound=np.pi/4
      com_goal_postrew = [[xx for xx in com if ((np.nanmedian(coms_rewrel[:,
      xx], axis=0)<0) & (np.nanmedian(coms_rewrel[:,
      xx], axis=0)>-bound))] if len(com)>0 else [] for com in com_goal]
      # get goal cells across all epochs        
      if len(com_goal_postrew)>0:
         goal_cells = intersect_arrays(*com_goal_postrew); 
      else:
         goal_cells=[]
      # here i take any cell that is a goal cell in any given epoch combo
      goal_cells = np.unique(np.concatenate(com_goal_postrew)).astype(int)    
      # exclude post cells
      com_goal_postrew = [[xx for xx in com if ((np.nanmedian(coms_rewrel[:,
      xx], axis=0)>0))] if len(com)>0 else [] for com in com_goal]
      # get goal cells across all epochs        
      if len(com_goal_postrew)>0:
         post_goal_cells = np.unique(np.concatenate(com_goal_postrew)).astype(int)    
      else:
         post_goal_cells=[]
      goal_cells = [xx for xx in goal_cells if xx not in post_goal_cells]   
      # pcs that are not goal cells
      # get dark time / far from rew loc licks
      # remove consumption licks
      lick_bin[forwardvel<5]=0
      lick=smooth_lick_rate(lick_bin,dt)
      dff_lick_far=[]; licks_far=[]; vel_far=[];
      pos_far=[]
      dff_lick_corr=[]; licks_corr=[]; vel_corr=[];
      pos_corr=[]
      dff=Fc3[:,goal_cells].mean(axis=1)
      for ep in range(len(eps)-1):
         eprng=np.arange(eps[ep],eps[ep+1])
         trials=trialnum[eprng]
         success, fail, str_trials, ftr_trials, ttr, \
         total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
         rewloc=rewlocs[ep]
         trials_lick_far=[];trials_lick_corr=[]
         for trial in np.unique(trials):
            if trial in ftr_trials: # only failed trials?
               trial_mask = trials==trial
               lick_tr = lick_bin[eprng][trial_mask]
               ypos_tr=ybinned[eprng][trial_mask]
               # check opposite rew loc
               opploc = ((rewloc/scalingf-((rewsize/scalingf)/2)+90)%180)*scalingf
               # if opploc<rewloc:
               #    lick_far = lick_tr[(ypos_tr<opploc)]
               #    # within 
               #    lick_near = lick_tr[(ypos_tr>opploc)]
               if rewloc<opploc:
                  lick_far = lick_tr[(ypos_tr>opploc)]
                  lick_near = lick_tr[(ypos_tr<opploc)]
                  # 100% bigger than near licks?  
                  if sum(lick_far)>sum(lick_near)*1.5 and ybinned[eprng][trial_mask][-1]>250: # more licks in wrong pos? + nearly completed trial
                     trials_lick_far.append(trial)
                     dff_lick_far.append(dff[eprng][trial_mask])
                     licks_far.append(lick[eprng][trial_mask])
                     vel_far.append(forwardvel[eprng][trial_mask])
                     pos_far.append(ybinned[eprng][trial_mask])
            if trial in str_trials:
               trial_mask = trials==trial
               lick_tr = lick_bin[eprng][trial_mask]
               ypos_tr=ybinned[eprng][trial_mask]
               # check opposite rew loc
               opploc = ((rewloc*scalingf-((rewsize*scalingf)/2)+90)%180)/scalingf               
               # if opploc<rewloc:
               #    lick_far = lick_tr[(ypos_tr<opploc)]
               #    # within 
               #    lick_near = lick_tr[(ypos_tr>opploc)]
               if rewloc<opploc: # only post-reward licks
                  lick_far = lick_tr[(ypos_tr>opploc)]
                  lick_near = lick_tr[(ypos_tr<opploc)]
                  if (sum(lick_far))<sum(lick_near): # less licks in wrong pos
                     trials_lick_corr.append(trial)
                     # average of pre-reward cells
                     dff_lick_corr.append(dff[eprng][trial_mask])
                     licks_corr.append(lick[eprng][trial_mask])
                     vel_corr.append(forwardvel[eprng][trial_mask])
                     pos_corr.append(ybinned[eprng][trial_mask])
         # print(rewloc,opploc)
      print(f'########\n{animal}, {day}, trials: {len(dff_lick_far)}\n')
      print(f'########\n{animal}, {day}, correct trials: {len(dff_lick_corr)}\n')

      if len(dff_lick_far)>0:
         maxlen = max([len(xx) for xx in dff_lick_far])   
         dff_lick_far_arr=np.ones((maxlen,len(dff_lick_far)))*np.nan
         for d,dff in enumerate(dff_lick_far):
            dff_lick_far_arr[:len(dff),d]=dff
         lick_far_arr=np.ones((maxlen,len(licks_far)))*np.nan
         for d,dff in enumerate(licks_far):
            lick_far_arr[:len(dff),d]=dff 
         maxlen = max([len(xx) for xx in dff_lick_far])   
         vel_far_arr=np.ones((maxlen,len(vel_far)))*np.nan
         for d,dff in enumerate(vel_far):
            vel_far_arr[:len(dff),d]=dff
         pos_far_arr=np.ones((maxlen,len(pos_far)))*np.nan
         for d,dff in enumerate(pos_far):
            pos_far_arr[:len(dff),d]=dff
      if len(dff_lick_corr)>0:# all correct
         maxlen = max([len(xx) for xx in dff_lick_corr])   
         dff_lick_corr_arr=np.ones((maxlen,len(dff_lick_corr)))*np.nan
         for d,dff in enumerate(dff_lick_corr):
            dff_lick_corr_arr[:len(dff),d]=dff
         lick_corr_arr=np.ones((maxlen,len(licks_corr)))*np.nan
         for d,dff in enumerate(licks_corr):
            lick_corr_arr[:len(dff),d]=dff       
         vel_corr_arr=np.ones((maxlen,len(vel_corr)))*np.nan
         for d,dff in enumerate(vel_corr):
            vel_corr_arr[:len(dff),d]=dff
         pos_corr_arr=np.ones((maxlen,len(pos_corr)))*np.nan
         for d,dff in enumerate(pos_corr):
            pos_corr_arr[:len(dff),d]=dff

      fig,axes=plt.subplots(ncols=1+len(dff_lick_far),figsize=((7+7*(len(dff_lick_far)*.7)),4))      
      try:
         ax=axes[0]
      except:
         ax=axes
      ax.plot(np.nanmean(vel_corr_arr,axis=1)/30,label='Velocity')
      ax.plot(np.nanmean(lick_corr_arr,axis=1),label='Lick rate')
      ax.plot(np.nanmean(pos_corr_arr,axis=1)/135,label='Position')
      ax2 = ax.twinx()   
      ax2.plot(np.nanmean(dff_lick_corr_arr, axis=1),color='k',label='Pre-reward cells')
      ax.set_title('Correct trials average')
      ax.legend()
      ax2.legend()
      if len(dff_lick_far)>0:
         for tr in range(pos_far_arr.shape[1]):
            ax=axes[1+tr]
            ax.plot(vel_far_arr[:,tr]/30)
            ax.plot(lick_far_arr[:,tr])
            ax.plot(pos_far_arr[:,tr]/135)
            ax2 = ax.twinx()         
            ax2.plot(dff_lick_far_arr[:,tr],color='k')
            ax.set_title('Far licks, incorrect trial')
      plt.tight_layout()
      plt.show()



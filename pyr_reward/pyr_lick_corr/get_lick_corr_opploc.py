
"""
zahra
licks in opposite location (after passing pre-reward zone)
only limit to epoch in which these licks are occuring
oct 2025
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
from projects.pyr_reward.placecell import make_tuning_curves_radians_by_trialtype, intersect_arrays,make_tuning_curves_by_trialtype_w_darktime,make_tuning_curves
from projects.pyr_reward.rewardcell import get_radian_position,create_mask_from_coordinates,pairwise_distances,extract_data_rewcentric, get_radian_position_first_lick_after_rew, get_rewzones, cosine_sim_ignore_nan,wilcoxon_r
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
df_all = []
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
      # position tc with different trial types
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
      # pcs = [xx for xx in pcs if xx not in goal_cells and xx not in post_goal_cells]
      # get dark time / far from rew loc licks
      # remove consumption licks
      lick_bin[forwardvel<5]=0
      lick=smooth_lick_rate(lick_bin,dt)
      dff_lick_far=[]; licks_far=[]; vel_far=[];
      pos_far=[]
      dff_lick_corr=[]; licks_corr=[]; vel_corr=[];
      pos_corr=[]
      dff_far_only=[];dff_far_only_no_lick=[];dff_near_no_lick=[];dff_near_far_lick=[]
      dff=Fc3[:,goal_cells].mean(axis=1)
      trials_lick_far=[];trials_lick_corr=[]
      for ep in range(len(eps)-1):
         eprng=np.arange(eps[ep],eps[ep+1])
         trials=trialnum[eprng]
         success, fail, str_trials, ftr_trials, ttr, \
         total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
         rewloc=rewlocs[ep]
         if rz[ep]==1: # only area 1
            for trial in np.unique(trials):
               if trial in str_trials:
                  trial_mask = trials==trial
                  lick_tr = lick_bin[eprng][trial_mask]
                  ypos_tr=ybinned[eprng][trial_mask]
                  # check opposite rew loc
                  opploc = ((rewloc*scalingf-((rewsize*scalingf)/2)+30)%180)/scalingf               
                  # if opploc<rewloc:
                  #    lick_far = lick_tr[(ypos_tr<opploc)]
                  #    # within 
                  #    lick_near = lick_tr[(ypos_tr>opploc)]
                  lick_far = lick_tr[(ypos_tr>opploc)]
                  lick_near = lick_tr[(ypos_tr<opploc)]
                  bound=30
                  if (sum(lick_tr[(ypos_tr>(rewloc+bound))]))>0: # less licks in wrong pos
                     trials_lick_far.append(trial)
                     # average of pre-reward cells
                     dff_lick_far.append(dff[eprng][trial_mask])
                     licks_far.append(lick[eprng][trial_mask])
                     vel_far.append(forwardvel[eprng][trial_mask])
                     pos_far.append(ybinned[eprng][trial_mask])
                     f=dff[eprng][trial_mask]
                     ypos=ybinned[eprng][trial_mask]
                     vel=forwardvel[eprng][trial_mask]
                     licktr=lick[eprng][trial_mask]
                     f[ypos<(rewloc+bound)]=np.nan
                     vel[ypos<(rewloc+bound)]=np.nan
                     licktr[ypos<(rewloc+bound)]=np.nan
                     dff_far_only.append([f,vel,licktr])
                     f=dff[eprng][trial_mask]
                     f[ypos>(rewloc+bound)]=np.nan
                     dff_near_far_lick.append(f)
                  elif sum(lick_tr[(ypos_tr>(rewloc+30))])==0:
                     # average of pre-reward cells
                     trials_lick_corr.append(trial)
                     dff_lick_corr.append(dff[eprng][trial_mask])
                     licks_corr.append(lick[eprng][trial_mask])
                     vel_corr.append(forwardvel[eprng][trial_mask])
                     pos_corr.append(ybinned[eprng][trial_mask])
                     f=dff[eprng][trial_mask]
                     ypos=ybinned[eprng][trial_mask]
                     vel=forwardvel[eprng][trial_mask]
                     licktr=lick[eprng][trial_mask]
                     f[ypos<(rewloc+bound)]=np.nan
                     vel[ypos<(rewloc+bound)]=np.nan
                     licktr[ypos<(rewloc+bound)]=np.nan
                     dff_far_only_no_lick.append([f,vel,licktr])
                     f=dff[eprng][trial_mask]
                     f[ypos>(rewloc+bound)]=np.nan
                     dff_near_no_lick.append(f)
         # print(rewloc,opploc)
      print(f'########\n{animal}, {day}, trials: {len(dff_lick_far)}\n')
      print(f'########\n{animal}, {day}, correct trials: {len(dff_lick_corr)}\n')
      # eg
      # trials=[3,4,12,13,15]
      # non_lick_trials=[5,  6,  7,  8,  9, 10, 11, 14, 16,
      #  17, 18, 19, 20, 21, 22, 23]
      # tcs,com=make_tuning_curves_abs_specify_trials(eps,2,rewlocs,ybinned,Fc3[:,goal_cells],trialnum,lick,rewards,forwardvel,rewsize,bin_size,trials,lasttr=8,bins=90,eptrials=3,
      #       velocity_filter=False)
      # tcs_nl,com_nl=make_tuning_curves_abs_specify_trials(eps,2,rewlocs,ybinned,Fc3[:,goal_cells],trialnum,lick,rewards,forwardvel,rewsize,bin_size,non_lick_trials,lasttr=8,bins=90,eptrials=3,
      # velocity_filter=False)
      # # for licks
      # tcs_lick,_=make_tuning_curves_abs_specify_trials(eps,2,rewlocs,ybinned,np.array([lick]).T,trialnum,lick,rewards,forwardvel,rewsize,bin_size,trials,lasttr=8,bins=90,eptrials=3,
      # velocity_filter=False)
      # tcs_nl_lick,_=make_tuning_curves_abs_specify_trials(eps,2,rewlocs,ybinned,np.array([lick]).T,trialnum,lick,rewards,forwardvel,rewsize,bin_size,non_lick_trials,lasttr=8,bins=90,eptrials=3,velocity_filter=False)
      # # for vel
      # tcs_vel,_=make_tuning_curves_abs_specify_trials(eps,2,rewlocs,ybinned,np.array([forwardvel]).T,trialnum,lick,rewards,forwardvel,rewsize,bin_size,trials,lasttr=8,bins=90,eptrials=3,
      # velocity_filter=False)
      # tcs_nl_vel,_=make_tuning_curves_abs_specify_trials(eps,2,rewlocs,ybinned,np.array([forwardvel]).T,trialnum,lick,rewards,forwardvel,rewsize,bin_size,non_lick_trials,lasttr=8,bins=90,eptrials=3,velocity_filter=False)
      # def normalize_rows(x):
      #    x_min = np.min(x, axis=1, keepdims=True)
      #    x_max = np.max(x, axis=1, keepdims=True)
      #    return (x - x_min) / (x_max - x_min + 1e-10)
      # fig,axes=plt.subplots(nrows=4,ncols=2,height_ratios=[2,1,1,1],sharex=True,sharey='row',figsize=(4,6))
      # arrs = [tcs_nl,tcs]
      # coms = [com_nl,com]
      # lickarr=[tcs_nl_lick,tcs_lick]
      # velarr=[tcs_nl_vel,tcs_vel]
      # lbls=['No','Yes']
      # for i in range(2):
      #    ax=axes[0,i]
      #    ax.imshow(normalize_rows(arrs[i][np.argsort(com_nl)]),aspect='auto')
      #    ax.set_yticks([0,len(arrs[i])-1])
      #    ax.set_yticklabels([1,len(arrs[i])])
      #    if i==0: ax.set_ylabel('Pre-reward cell # (sorted)')
      #    ax.set_title(lbls[i])
      #    ax.axvline(rewlocs[2]/bin_size, linewidth=2,color='w',linestyle='--')
      #    ax=axes[1,i]
      #    m=np.nanmean(arrs[i],axis=0)
      #    sem=scipy.stats.sem(arrs[i],axis=0,nan_policy='omit')
      #    ax.plot(m,color='cornflowerblue')
      #    ax.fill_between(range(len(arrs[i].T)),m-sem,m+sem,alpha=0.2,color='cornflowerblue')
      #    ax.spines[['top','right']].set_visible(False)
      #    ax.axvline(rewlocs[2]/bin_size, linewidth=2,color='k',linestyle='--')
      #    if i==0: ax.set_ylabel('Mean $\Delta F/F$')
      #    ax=axes[2,i]
      #    ax.plot(lickarr[i][0],color='slategray')
      #    ax.spines[['top','right']].set_visible(False)
      #    ax.axvline(rewlocs[2]/bin_size, linewidth=2,color='k',linestyle='--')
      #    ax.set_xticks([0,90])
      #    ax.set_xticklabels([0,270])
      #    if i==0: ax.set_ylabel('Lick rate (Hz)')
      #    ax=axes[3,i]
      #    ax.plot(velarr[i][0],color='k')
      #    ax.spines[['top','right']].set_visible(False)
      #    ax.axvline(rewlocs[2]/bin_size, linewidth=2,color='k',linestyle='--')
      #    ax.set_xticks([0,90])
      #    ax.set_xticklabels([0,270])
      #    if i==0: ax.set_ylabel('Velocity (cm/s)')
      #    ax.set_xlabel('Track position (cm)')
      # fig.suptitle('mouse 16, session 10\nLicks after passing reward zone?')
      # plt.savefig(os.path.join(savedst, 'current_v_old_tc_fig3_eg.svg'), bbox_inches='tight')
      # ########## place
      # tcs,com=make_tuning_curves_abs_specify_trials(eps,2,rewlocs,ybinned,Fc3[:,pcs],trialnum,lick,rewards,forwardvel,rewsize,bin_size,trials)
      # tcs_nl,com_nl=make_tuning_curves_abs_specify_trials(eps,2,rewlocs,ybinned,Fc3[:,pcs],trialnum,lick,rewards,forwardvel,rewsize,bin_size,non_lick_trials)

      # fig,axes=plt.subplots(nrows=4,ncols=2,height_ratios=[2,1,1,1],sharex=True,sharey='row',figsize=(4,6))
      # arrs = [tcs_nl,tcs]
      # coms = [com_nl,com]
      # lickarr=[tcs_nl_lick,tcs_lick]
      # velarr=[tcs_nl_vel,tcs_vel]
      # lbls=['No','Yes']
      # for i in range(2):
      #    ax=axes[0,i]
      #    ax.imshow(normalize_rows(arrs[i][np.argsort(com_nl)]),aspect='auto')
      #    ax.set_yticks([0,len(arrs[i])-1])
      #    ax.set_yticklabels([1,len(arrs[i])])
      #    if i==0: ax.set_ylabel('Placee cell # (sorted)')
      #    ax.set_title(lbls[i])
      #    ax.axvline(rewlocs[2]/bin_size, linewidth=2,color='w',linestyle='--')
      #    ax=axes[1,i]
      #    m=np.nanmean(arrs[i],axis=0)
      #    sem=scipy.stats.sem(arrs[i],axis=0,nan_policy='omit')
      #    ax.plot(m,color='indigo')
      #    ax.fill_between(range(len(arrs[i].T)),m-sem,m+sem,alpha=0.2,color='indigo')
      #    ax.spines[['top','right']].set_visible(False)
      #    ax.axvline(rewlocs[2]/bin_size, linewidth=2,color='k',linestyle='--')
      #    if i==0: ax.set_ylabel('Mean $\Delta F/F$')
      #    ax=axes[2,i]
      #    ax.plot(lickarr[i][0],color='slategray')
      #    ax.spines[['top','right']].set_visible(False)
      #    ax.axvline(rewlocs[2]/bin_size, linewidth=2,color='k',linestyle='--')
      #    if i==0: ax.set_ylabel('Lick rate (Hz)')
      #    ax=axes[3,i]
      #    ax.plot(velarr[i][0],color='k')
      #    ax.spines[['top','right']].set_visible(False)
      #    ax.axvline(rewlocs[2]/bin_size, linewidth=2,color='k',linestyle='--')
      #    ax.set_xticks([0,90])
      #    ax.set_xticklabels([0,270])
      #    if i==0: ax.set_ylabel('Velocity (cm/s)')
      #    ax.set_xlabel('Track position (cm)')
      # fig.suptitle('mouse 16, session 10\nLicks after passing reward zone?')
      # plt.savefig(os.path.join(savedst, 'current_v_old_place_tc_fig3_eg.svg'), bbox_inches='tight')

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
         # per trial average
         dff_far_only_f = [xx[0] for xx in dff_far_only]
         dff_far_only_v = [xx[1] for xx in dff_far_only]
         dff_far_only_l = [xx[2] for xx in dff_far_only]
         dff_far_only_f = [np.nanmean(xx[0]) for xx in dff_far_only]
         dff_far_only_v = [np.nanmean(xx[1]) for xx in dff_far_only]
         dff_far_only_l = [np.nanmean(xx[2]) for xx in dff_far_only]
         
         dff_near_far_lick = [np.nanmean(xx) for xx in dff_near_far_lick]
         # pearson 
         dff_far_only_rho = [safe_pearsonr(xx[0],xx[2]) for xx in dff_far_only]
         dff_far_only_vel_rho = [safe_pearsonr(xx[0],xx[1]) for xx in dff_far_only]
      else:
         dff_far_only_f=[];dff_far_only_rho=[];dff_far_only_vel_rho=[];dff_far_only_v=[]
         dff_far_only_l=[];
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
         # per trial average
         dff_far_no_lick_f = [xx[0] for xx in dff_far_only_no_lick]
         dff_far_no_lick_v = [xx[1] for xx in dff_far_only_no_lick]
         dff_far_no_lick_l = [xx[2] for xx in dff_far_only_no_lick]
         dff_far_no_lick_f = [np.nanmean(xx[0]) for xx in dff_far_only_no_lick]
         dff_far_no_lick_v = [np.nanmean(xx[1]) for xx in dff_far_only_no_lick]
         dff_far_no_lick_l = [np.nanmean(xx[2]) for xx in dff_far_only_no_lick]
         dff_far_no_lick_rho = [safe_pearsonr(xx[0],xx[2]) for xx in dff_far_only_no_lick]
         dff_far_no_vel_rho = [safe_pearsonr(xx[0],xx[1]) for xx in dff_far_only_no_lick]
         dff_near_no_lick=[np.nanmean(xx) for xx in dff_near_no_lick]
         # near and far dff
         df=pd.DataFrame()
         df['mean_dff_far']=np.concatenate([dff_far_no_lick_f,dff_far_only_f])
         df['trial_num'] = np.concatenate([trials_lick_corr,trials_lick_far])
         df['mean_dff_near']=np.concatenate([dff_near_no_lick,dff_near_far_lick])
         df['condition']=np.concatenate([['nofarlick']*len(dff_far_no_lick_f),['farlick']*len(dff_far_only_f)])
         df['lick_rho']=np.concatenate([dff_far_no_lick_rho,dff_far_only_rho])
         df['vel_rho']=np.concatenate([dff_far_no_vel_rho,dff_far_only_vel_rho])
         df['animal']=[animal]*len(df)
         df['day']=[day]*len(df)
         plt.figure()
         sns.barplot(x='condition',y='mean_dff_far',data=df)
         plt.figure()
         sns.scatterplot(x='trial_num',y='mean_dff_far',hue='condition',data=df)
         ######### SAVE
         df_all.append(df)
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


#%%
bigdf=pd.concat(df_all)

# only include days with both trial types
newdf=bigdf.copy()
newnewdf=[]
ans=newdf.animal.unique()
for an in ans:
   andf=newdf[newdf.animal==an]
   dys = andf.day.unique()
   for dy in dys:
      dydf = andf[andf.day==dy]
      if len(dydf.condition.unique())==2:
         newnewdf.append(dydf)
bigdf=pd.concat(newnewdf)
#%%
# summary fig
pl=['cornflowerblue','cadetblue']
fig,ax=plt.subplots(figsize=(4,3))
pltdf=bigdf.reset_index()
pltdf=pltdf[pltdf.trial_num<25]
pltdf['mean_dff_far']=pltdf['mean_dff_far']*100
sns.lineplot(x='trial_num',y='mean_dff_far',hue='condition',data=pltdf,palette=pl)
ax.spines[['top','right']].set_visible(False)
ax.set_ylabel(r'Opp. loc. mean % $\Delta F/F$')
ax.set_xlabel(r'Trial #')
fig.suptitle('Reward')
plt.tight_layout()
plt.savefig(os.path.join(savedst, 'current_v_old_summary_meandff.svg'), bbox_inches='tight')
# save raw data
pltdf.to_csv(r'C:\Users\Han\Documents\MATLAB\vip_reward_cell\fig3\fig3f.csv')
#%%
# quant
bigdf=bigdf.groupby(['animal','condition']).mean(numeric_only=True).reset_index()
bigdf=bigdf[(bigdf.animal!='e139')]

         
def add_pair_lines(ax, data, x='condition', y='', order=None, id_col='animal', color='gray', alpha=0.5):
   """Draw paired lines connecting conditions for each animal."""
   for a, sub in data.groupby(id_col):
      if order is not None:
         sub = sub.set_index(x).loc[order].reset_index()
      if sub.shape[0] == 2:  # only draw if both conditions exist
         ax.plot([0,1], sub[y], color=color, alpha=alpha, zorder=0,linewidth=1.5)
         
def add_sig(ax, data, y, order, id_col='animal', ypos=None,h=.1):
   """Run Wilcoxon signed-rank test and add significance annotation to ax."""
   # wide pivot for paired comparison
   wide = data.pivot(index=id_col, columns='condition', values=y)
   wide = wide[order].dropna()  # ensure both conditions present
   stat, p = wilcoxon_r(wide[order[0]], wide[order[1]])
   # y position for the annotatio
   if ypos is None:
      ypos = data[y].max() * 1.1
   ax.text(0.5, ypos, f"cohen's r={stat:.3g}\np={p:.3g}", ha='center', va='bottom',fontsize=8)
   return stat, p

bigdf['mean_dff_near']=bigdf['mean_dff_near']*100
bigdf['mean_dff_far']=bigdf['mean_dff_far']*100
plt.rc('font', size=14)  
pl=['cornflowerblue','cadetblue']
order=['nofarlick','farlick']
lbls=['No', 'Yes']
fig,axes=plt.subplots(ncols=2,figsize=(4.5,4),sharex=True)
ax=axes[0]
sns.barplot(x='condition',y='lick_rho',data=bigdf,fill=False,palette=pl,order=order,ax=ax)
add_pair_lines(ax, bigdf, x='condition', y='lick_rho', order=order)
add_sig(ax, bigdf, y='lick_rho', order=order)
ax.spines[['top','right']].set_visible(False)
ax.set_xticklabels(lbls)
ax.set_ylabel(r'Lick Pearson $\rho$')
ax.set_xlabel('Licks after passing reward zone?')
ax=axes[1]
sns.barplot(x='condition',y='vel_rho',data=bigdf,fill=False,palette=pl,order=order,ax=ax)
ax.spines[['top','right']].set_visible(False)
add_pair_lines(ax, bigdf, x='condition', y='vel_rho', order=order)
add_sig(ax, bigdf, y='vel_rho', order=order)
ax.set_xticklabels(lbls)
ax.set_xlabel('')
ax.set_ylabel(r'Velocity Pearson $\rho$')
plt.tight_layout()
plt.savefig(os.path.join(savedst, 'current_v_old_rho.svg'), bbox_inches='tight')

fig,axes=plt.subplots(ncols=2,figsize=(4,4),sharey=True)
ax=axes[0]
sns.barplot(x='condition',y='mean_dff_near',data=bigdf,fill=False,palette=pl,order=order,ax=ax)
ax.spines[['top','right']].set_visible(False)
add_sig(ax, bigdf, y='mean_dff_near', order=order)
add_pair_lines(ax, bigdf, x='condition', y='mean_dff_near', order=order)
ax.set_xticklabels(lbls)
ax.set_xlabel('Licks after passing reward zone?')
ax.set_ylabel(r'Mean % $\Delta F/F$')
ax.set_title('Current reward zone\n(Area 1)')
ax=axes[1]
sns.barplot(x='condition',y='mean_dff_far',data=bigdf,fill=False,palette=pl,order=order,ax=ax)
ax.set_xticklabels(lbls)
ax.set_xlabel('')
add_sig(ax, bigdf, y='mean_dff_far', order=order)
add_pair_lines(ax, bigdf, x='condition', y='mean_dff_far', order=order)
ax.spines[['top','right']].set_visible(False)
ax.set_ylabel(r'Mean $\Delta F/F$')
ax.set_title('After reward zone')
fig.suptitle('Reward')
plt.tight_layout()
plt.savefig(os.path.join(savedst, 'current_v_old_meandff.svg'), bbox_inches='tight')

bigdf.to_csv(r'C:\Users\Han\Documents\MATLAB\vip_reward_cell\fig3\fig3g.csv')
# %%

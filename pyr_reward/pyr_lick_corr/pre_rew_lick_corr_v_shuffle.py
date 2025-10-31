
"""
zahra
lick corr trial by trial
split into corr vs incorr
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
from projects.pyr_reward.placecell import get_tuning_curve, calc_COM_EH, make_tuning_curves_by_trialtype_w_darktime, make_tuning_curves_time_trial_by_trial_w_darktime, intersect_arrays
from projects.opto.behavior.behavior import get_success_failure_trials, smooth_lick_rate
from scipy.stats import spearmanr

conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_rewardcentric_all.p"
with open(saveddataset, "rb") as fp: #unpickle
        radian_alignment_saved = pickle.load(fp)
savepth = os.path.join(savedst, 'lickcorr.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)

import numpy as np
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
import numpy as np

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
# cm_window = [10,20,30,40,50,
# 60,70,80] # cm
# iterate through all animals
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
      # get average success rate
      rates = []; lick_no_consp_ep=[];shuffle_lick_ep=[]
      vel_no_rew=[]
      for ep in range(len(eps)-1):
         eprng = range(eps[ep],eps[ep+1])
         lick_ep = lick[eprng]
         ypos=ybinned[eprng]
         lick_ep[(ypos>(rewlocs[ep]-rewsize/2)) & (ypos<(rewlocs[ep]+rewsize/2))]=0
         vel_ep=forwardvel[eprng]
         vel_ep[(ypos>(rewlocs[ep]-rewsize/2)) & (ypos<(rewlocs[ep]+rewsize/2))]=0
         lick_no_consp_ep.append(lick_ep)
         vel_no_rew.append(vel_ep)
         success, fail, str_trials, ftr_trials, ttr, \
         total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
         rates.append(success/total_trials)
      rate=np.nanmean(np.array(rates))
      lick_no_consp=np.concatenate(lick_no_consp_ep)
      vel_no_rew=np.concatenate(vel_no_rew)
      # t = time_[mask][(ybinned_<rewloc)[mask]]
      dt = np.nanmedian(np.diff(time))
      lr= smooth_lick_rate(lick_no_consp,dt)
      # remove high lr
      lick_no_consp[lr>8]=0
      vel_no_rew[lr>8]=0
      ################## lick shuffle
      lick_shufs=100
      lick_no_consp_shuf = circular_shuffle_with_shift_range(lick_no_consp, n_shuffles=lick_shufs, fps=1/dt, min_shift_ms=1000, max_shift_s=30.0)
      vel_no_rew_shuf = circular_shuffle_with_shift_range(vel_no_rew, n_shuffles=lick_shufs, fps=1/dt, min_shift_ms=1000, max_shift_s=30.0)
      # vel_no_rew_shuf=vel_no_rew_shuf[np.random.randint(lick_shufs)]

      # trial by trial does not work
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
      tcs_correct, coms_correct, tcs_fail, coms_fail, ybinned_dt, rad = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,
         rewsize,ybinned,time,lick,
         Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
         bins=bins_dt)  
      bin_size=3
      # abs position
      tcs_correct_abs, coms_correct_abs,tcs_fail_abs, coms_fail_abs= make_tuning_curves(eps,rewlocs,ybinned,Fc3,trialnum,rewards,forwardvel,rewsize,bin_size)
      # get cells that maintain their coms across at least 2 epochs
      place_window = 20 # cm converted to rad                
      perm = list(combinations(range(len(coms_correct_abs)), 2))     
      com_per_ep = np.array([(coms_correct_abs[perm[jj][0]]-coms_correct_abs[perm[jj][1]]) for jj in range(len(perm))])        
      compc = [np.where((comr<place_window) & (comr>-place_window))[0] for comr in com_per_ep]
      # get cells across all epochs that meet crit
      pcs = np.unique(np.concatenate(compc))
      pcs_all = intersect_arrays(*compc)

      lick_correct_abs, _,lick_fail_abs,__ = make_tuning_curves(eps,rewlocs,ybinned,np.array([lick,lick]).T,trialnum,rewards,forwardvel,rewsize,bin_size)
      ######## shuffle
      lick_abs_shuf=[]
      for shuf in range(lick_shufs):
         lick_correct_abs_shuf, _,lick_fail_abs_shuf,__ = make_tuning_curves(eps,rewlocs,ybinned,np.array([lick_no_consp_shuf[shuf]]).T,trialnum,rewards,forwardvel,rewsize,bin_size)
         lick_abs_shuf.append([lick_correct_abs_shuf,lick_fail_abs_shuf])
      # average tc
      lick_correct_abs_shuf=[xx[0] for xx in lick_abs_shuf]
      lick_correct_abs_shuf=np.squeeze(np.nanmean(lick_correct_abs_shuf,axis=0))
      lick_fail_abs_shuf=[xx[1] for xx in lick_abs_shuf]
      lick_fail_abs_shuf=np.squeeze(np.nanmean(lick_fail_abs_shuf,axis=0))
      ######## vel
      vel_correct_abs, _,vel_fail_abs,__ = make_tuning_curves(eps,rewlocs,ybinned,np.array([forwardvel,forwardvel]).T,trialnum,rewards,forwardvel,rewsize,bin_si ze)
      ######## shuffle
      vel_abs_shuf=[]
      for shuf in range(lick_shufs):
         vel_correct_abs_shuf, _,vel_fail_abs_shuf,__ = make_tuning_curves(eps,rewlocs,ybinned,np.array([vel_no_rew_shuf[shuf]]).T,trialnum,rewards,forwardvel,rewsize,bin_size)
         vel_abs_shuf.append([vel_correct_abs_shuf,vel_fail_abs_shuf])
      # average tc
      vel_correct_abs_shuf=[xx[0] for xx in vel_abs_shuf]
      vel_correct_abs_shuf=np.squeeze(np.nanmean(vel_correct_abs_shuf,axis=0))
      vel_fail_abs_shuf=[xx[1] for xx in vel_abs_shuf]
      vel_fail_abs_shuf=np.squeeze(np.nanmean(vel_fail_abs_shuf,axis=0))
      
      goal_window = 20*(2*np.pi/track_length) # cm converted to rad
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
      goal_cells = np.unique(np.concatenate(com_goal_postrew)).astype(int)     
      # pcs that are not goal cells
      pcs = [xx for xx in pcs if xx not in goal_cells]   
      ########## correct trials      
      lick_tc_cs = [[spearmanr(tcs_correct_abs[ep,cll,:], lick_correct_abs[ep][0])[0] for cll in goal_cells] for ep in range(len(tcs_correct_abs))]
      vel_tc_cs = [[spearmanr(tcs_correct_abs[ep,cll,:], vel_correct_abs[ep][0])[0] for cll in goal_cells] for ep in range(len(tcs_correct_abs))]
      # save this 
      dfs=[]
      df = pd.DataFrame()
      df['cellid']=np.concatenate([goal_cells]*len(tcs_correct_abs))
      df['cs_lick_v_tc']=np.concatenate(lick_tc_cs)
      df['cs_vel_v_tc']=np.concatenate(vel_tc_cs)
      df['trial_type']=['correct']*len(df)
      df['cell_type']=['pre']*len(df)
      df['animal']=[animal]*len(df)
      df['condition']=['real']*len(df)
      df['day']=[day]*len(df)
      dfs.append(df)
      ########## incorrect trials      
      lick_tc_cs = [[spearmanr(tcs_fail_abs[ep,cll,:], lick_fail_abs[ep][0])[0] for cll in goal_cells] for ep in range(len(tcs_correct_abs))]
      vel_tc_cs = [[spearmanr(tcs_fail_abs[ep,cll,:], vel_fail_abs[ep][0])[0] for cll in goal_cells] for ep in range(len(tcs_correct_abs))]
      # save this 
      df = pd.DataFrame()
      df['cellid']=np.concatenate([goal_cells]*len(tcs_correct_abs))
      df['cs_lick_v_tc']=np.concatenate(lick_tc_cs)
      df['cs_vel_v_tc']=np.concatenate(vel_tc_cs)
      df['trial_type']=['incorrect']*len(df)
      df['cell_type']=['pre']*len(df)
      df['condition']=['real']*len(df)
      df['animal']=[animal]*len(df)
      df['day']=[day]*len(df)
      dfs.append(df)
      ############################# vs. shuffle
      lick_tc_cs = [[spearmanr(tcs_correct_abs[ep,cll,:], lick_correct_abs_shuf[ep])[0] for cll in goal_cells] for ep in range(len(tcs_correct_abs))]
      vel_tc_cs = [[spearmanr(tcs_correct_abs[ep,cll,:], vel_correct_abs_shuf[ep])[0] for cll in goal_cells] for ep in range(len(tcs_correct_abs))]
      # save this 
      df = pd.DataFrame()
      df['cellid']=np.concatenate([goal_cells]*len(tcs_correct_abs))
      df['cs_lick_v_tc']=np.concatenate(lick_tc_cs)
      df['cs_vel_v_tc']=np.concatenate(vel_tc_cs)
      df['trial_type']=['correct']*len(df)
      df['cell_type']=['pre']*len(df)
      df['condition']=['shuffle']*len(df)
      df['animal']=[animal]*len(df)
      df['day']=[day]*len(df)
      dfs.append(df)
      ################# incorrect
      lick_tc_cs = [[spearmanr(tcs_fail_abs[ep,cll,:], lick_fail_abs_shuf[ep])[0] for cll in goal_cells] for ep in range(len(tcs_correct_abs))]
      vel_tc_cs = [[spearmanr(tcs_fail_abs[ep,cll,:], vel_fail_abs_shuf[ep])[0] for cll in goal_cells] for ep in range(len(tcs_correct_abs))]
      # save this 
      df = pd.DataFrame()
      df['cellid']=np.concatenate([goal_cells]*len(tcs_correct_abs))
      df['cs_lick_v_tc']=np.concatenate(lick_tc_cs)
      df['cs_vel_v_tc']=np.concatenate(vel_tc_cs)
      df['trial_type']=['incorrect']*len(df)
      df['cell_type']=['pre']*len(df)
      df['condition']=['shuffle']*len(df)
      df['animal']=[animal]*len(df)
      df['day']=[day]*len(df)
      dfs.append(df)

      df=pd.concat(dfs)

      # test
      plt.figure()
      sns.barplot(x='condition',y='cs_lick_v_tc',data=df.reset_index())
      plt.show()
      plt.figure()
      sns.barplot(x='condition',y='cs_vel_v_tc',data=df.reset_index())
      plt.show()
      df_all.append(df)

#%%
from statsmodels.stats.multitest import multipletests
plt.rc('font', size=20)
# get all cells width cm 
bigdf = pd.concat(df_all)
s=10;a=0.7
palette=[sns.color_palette('Dark2')[0],'gray']
order=['real','shuffle']
df = bigdf.groupby(['animal','cell_type','trial_type','condition']).mean(numeric_only=True)
df=df.reset_index()
typ='correct'
df=df[df.trial_type==typ]
typs=['correct','incorrect']
# df=df[df.trial_type=='incorrect']
df=df.dropna()
df=df[(df.animal!='e189') & (df.animal!='e139')& (df.animal!='e145')]
df=df[df.cell_type!='post_place']
fig, axes = plt.subplots(ncols=2,figsize=(7,6))

ax=axes[0]
sns.barplot(x='condition',y='cs_lick_v_tc',data=df,fill=False,palette=palette,order=order,ax=ax,errorbar='se')
# sns.stripplot(x='cell_type',y='cs_lick_v_tc',data=df,s=s,palette=palette,order=order,alpha=a,ax=ax)
# Draw lines per animal for Lick rho
pivot_lick = df.pivot(index='animal', columns='condition', values='cs_lick_v_tc')
for animal, row in pivot_lick.iterrows():
    if all(ct in row for ct in order):
        x = list(range(len(order)))
        y = [row[ct] for ct in order]
        ax.plot(x, y, color='gray', alpha=0.5, linewidth=1.5)

groups = [g['cs_lick_v_tc'].values for _, g in df.groupby('condition')]
kw_stat, kw_p = scipy.stats.kruskal(*groups)
print(f"Kruskal–Wallis H = {kw_stat:.4f}, p = {kw_p:.4g}")
# --- 2) Post-hoc pairwise Mann–Whitney U tests ---
comparisons = list(combinations(order, 2))
pvals = []

for a, b in comparisons:
    da = df[df['condition'] == a]['cs_lick_v_tc']
    db = df[df['condition'] == b]['cs_lick_v_tc']
    stat, p = scipy.stats.wilcoxon(da, db, alternative='two-sided')
    pvals.append(p)

# --- 3) FDR correction ---
reject, pvals_fdr, _, _ = multipletests(pvals, method='fdr_bh')

# --- 5) Annotate significant comparisons ---
y_max = df['cs_lick_v_tc'].max()-.1
y_step = 0.1 * y_max
start_y = y_max + y_step

for i, ((a, b), pval, sig) in enumerate(zip(comparisons, pvals_fdr, reject)):
    x1 = order.index(a)
    x2 = order.index(b)
    y = start_y + i * y_step
    ax.plot([x1, x1, x2, x2], [y, y + y_step/2, y + y_step/2, y], c='k', lw=1.5)
    if pval < 0.001:
        label = '***'
    elif pval < 0.01:
        label = '**'
    elif pval < 0.05:
        label = '*'
    else:
        label = 'ns'
    ax.text((x1 + x2) / 2, y + y_step * 0.6, label, ha='center', va='bottom', fontsize=25)

ax.set_ylabel('Lick Spearman $\\rho$')
# Final formatting
ax.set_xlabel('')
ax.set_xticklabels(['Real', 'Lick shuffle'],rotation=20)
ax.set_title(f'Kruskal–Wallis H={kw_stat:.2f}, p = {kw_p:.3g}',fontsize=12)
sns.despine()
plt.tight_layout()


# vel
ax=axes[1]
a=0.7
sns.barplot(x='condition',y='cs_vel_v_tc',data=df,fill=False,palette=palette,order=order,errorbar='se')
# sns.stripplot(x='cell_type',y='cs_vel_v_tc',data=df,s=s,palette=palette,order=order,alpha=a)
# Draw lines per animal for Velocity rho
pivot_vel = df.pivot(index='animal', columns='condition', values='cs_vel_v_tc')
for animal, row in pivot_vel.iterrows():
    if all(ct in row for ct in order):
        x = list(range(len(order)))
        y = [row[ct] for ct in order]
        ax.plot(x, y, color='gray', alpha=0.5, linewidth=1.5)

groups = [g['cs_vel_v_tc'].values for _, g in df.groupby('condition')]
kw_stat, kw_p = scipy.stats.kruskal(*groups)
print(f"Kruskal–Wallis H = {kw_stat:.4f}, p = {kw_p:.4g}")
# --- 2) Post-hoc pairwise Mann–Whitney U tests ---
comparisons = list(combinations(order, 2))
pvals = []

for a, b in comparisons:
    da = df[df['condition'] == a]['cs_vel_v_tc']
    db = df[df['condition'] == b]['cs_vel_v_tc']
    stat, p = scipy.stats.wilcoxon(da, db, alternative='two-sided')
    pvals.append(p)

# --- 3) FDR correction ---
reject, pvals_fdr, _, _ = multipletests(pvals, method='fdr_bh')

# --- 5) Annotate significant comparisons ---
y_max = df['cs_vel_v_tc'].max()-.1
y_step = 0.2 * y_max
start_y = y_max + y_step

for i, ((a, b), pval, sig) in enumerate(zip(comparisons, pvals_fdr, reject)):
    x1 = order.index(a)
    x2 = order.index(b)
    y = start_y + i * y_step
    ax.plot([x1, x1, x2, x2], [y, y + y_step/2, y + y_step/2, y], c='k', lw=1.5)
    if pval < 0.001:
        label = '***'
    elif pval < 0.01:
        label = '**'
    elif pval < 0.05:
        label = '*'
    else:
        label = 'ns'
    ax.text((x1 + x2) / 2, y + y_step * 0.6, label, ha='center', va='bottom', fontsize=25)

ax.set_ylabel('Velocity Spearman $\\rho$')
# Final formatting
ax.set_xlabel('')
ax.set_xticklabels(['Real', 'Velocity shuffle'],rotation=20)
ax.set_title(f'Kruskal–Wallis H={kw_stat:.2f}, p = {kw_p:.3g}',fontsize=12)
sns.despine()
# plt.tight_layout()
fig.suptitle('Pre-reward cells')
plt.tight_layout()

plt.savefig(os.path.join(savedst,f'shuffle_lick_vel_rho.svg'),bbox_inches='tight')

# %%
plt.rc('font', size=18)
# --- Prepare pre-reward data ---
df = bigdf.groupby(['animal','condition','trial_type']).mean(numeric_only=True)
df=df.reset_index()
df=df[df.animal!='z16']

df_pre = df.dropna(subset=['cs_lick_v_tc', 'cs_vel_v_tc'])
# Group by animal and compute mean Spearman correlations
df_agg = df_pre.groupby(['animal','trial_type','condition'])[['cs_lick_v_tc', 'cs_vel_v_tc']].mean().reset_index()
df_agg = df_agg.rename(columns={'cs_lick_v_tc': 'Lick', 'cs_vel_v_tc': 'Velocity'})
# Melt to long format for plotting
df_long = df_agg.melt(id_vars=['animal','condition','trial_type'], var_name='Variable', value_name='Spearman_rho')
df_long = df_long.copy()
df_long.loc[df_long['condition'] == 'shuffle', 'trial_type'] = 'shuffle'

# --- Plot ---
pl=['seagreen','firebrick','gray']
order=['correct','incorrect','shuffle']
fig,ax=plt.subplots(figsize=(5,5))
sns.barplot(data=df_long, x='Variable', y='Spearman_rho',hue='trial_type', hue_order=order,errorbar='se', palette=pl, fill=False,ax=ax)

xpos = {
    ('Lick', 'correct'): -0.2,
    ('Lick', 'incorrect'): 0,
    ('Lick', 'shuffle'): 0.2,
    ('Velocity', 'correct'): .8,
    ('Velocity', 'incorrect'): 1,
    ('Velocity', 'shuffle'): 1.2,
}

# --- Draw connecting lines between correct and incorrect for each behavior ---
for beh in ['Lick', 'Velocity']:
    df_pivot =df_long[df_long.Variable==beh].groupby(['animal', 'trial_type', 'Variable']).mean(numeric_only=True).reset_index()
    for an in df_pivot.animal.unique():
        row=df_pivot[df_pivot.animal==an]
        x1 = xpos[(beh, 'correct')]
        x2 = xpos[(beh, 'incorrect')]
        x3 = xpos[(beh, 'shuffle')]
        y1 = row.loc[row.trial_type=='correct', 'Spearman_rho'].values[0]
        y2 = row.loc[row.trial_type=='incorrect', 'Spearman_rho'].values[0]
        y3 = row.loc[row.trial_type=='shuffle', 'Spearman_rho'].values[0]
        ax.plot([x1, x2,x3], [y1, y2,y3], color='gray', alpha=0.5, linewidth=1.5)
            
# --- Wilcoxon signed-rank test ---
# Loop through both 'Lick' and 'Velocity'
for var in ['Lick', 'Velocity']:
    df_pivot =df_long[df_long.Variable==var].groupby(['animal','condition' ,'trial_type', 'Variable']).mean(numeric_only=True).reset_index()
    df_sub = df_pivot[df_pivot['animal'].isin(df_long['animal'].unique())]
    df_c = df_sub.loc[(df_sub['condition'] == 'real') & (df_sub['trial_type'] == 'correct'), 'Spearman_rho'].values
    df_i = df_sub.loc[(df_sub['condition'] == 'real') & (df_sub['trial_type'] == 'incorrect'), 'Spearman_rho'].values
    # av of shuffle
    df_pivot =df_long[df_long.Variable==var].groupby(['animal','trial_type', 'Variable']).mean(numeric_only=True).reset_index()
    df_sub = df_pivot[df_pivot['animal'].isin(df_long['animal'].unique())]
    df_s = df_sub.loc[(df_sub['trial_type'] == 'shuffle'), 'Spearman_rho'].values

    # Correct vs Incorrect
    stat1, pval1 = scipy.stats.wilcoxon(df_c, df_i)
    # Incorrect vs Shuffle
    stat2, pval2 = scipy.stats.wilcoxon(df_i, df_s)

    # --- Annotate plot ---
    xpos = {'Lick': 0, 'Velocity': 1}
    x0 = xpos[var]

    # Draw lines
    def draw_bar(x1, x2, y, h, pval):
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c='k')
        if pval < 0.001:
            stars = '***'
        elif pval < 0.01:
            stars = '**'
        elif pval < 0.05:
            stars = '*'
        else:
            stars = 'n.s.'
        ax.text((x1 + x2) * .5, y + h + 0.005, stars, ha='center', va='bottom', fontsize=30)

    height = df_long[df_long['Variable'] == var]['Spearman_rho'].max()
    offset = 0.03
    draw_bar(x0 - 0.25, x0, height + offset, 0.03, pval1)   # correct vs incorrect
    draw_bar(x0, x0 + 0.25, height + offset + 0.05, 0.03, pval2)   # incorrect vs shuffle

# Final formatting
ax.set_ylabel("Spearman $\\rho$")
ax.set_xlabel("")
ax.set_title("Pre-reward cells")
sns.despine()
plt.tight_layout()

plt.savefig(os.path.join(savedst, f'correctvincorrect_shuffle_lick_vel_prereward.svg'), bbox_inches='tight')

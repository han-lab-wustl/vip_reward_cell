
"""
zahra
get tuning curves with dark time
reward cell p vs. behavior
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
plt.rc('font', size=20)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.memory.behavior import consecutive_stretch
from projects.pyr_reward.placecell import get_tuning_curve, calc_COM_EH, make_tuning_curves_radians_by_trialtype,make_tuning_curves
from projects.pyr_reward.rewardcell import get_radian_position,\
    get_radian_position_first_lick_after_rew, get_rewzones, get_goal_cells, goal_cell_shuffle, \
    intersect_arrays
from projects.opto.behavior.behavior import get_success_failure_trials, get_lick_selectivity,smooth_lick_rate
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_rewardcentric_all.p"
with open(saveddataset, "rb") as fp: #unpickle
        radian_alignment_saved = pickle.load(fp)

#%%
####################################### RUN CODE #######################################
# initialize var
radian_alignment_saved = {} # overwrite
bins = 90
datadct = {}
cm_window = 20 # cm

# iterate through all animals
for ii in range(len(conddf)):
   day = conddf.days.values[ii]
   animal = conddf.animals.values[ii]
   in_type = conddf.in_type.values[ii]
   if ('vip' not in in_type) & (conddf.optoep.values[ii]<2):        
      if animal=='e145' or animal=='e139': pln=2 
      else: pln=0
      params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
      print(params_pth)
      fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 'timedFF',
               'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
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
      licks=fall['licks'][0]
      time=fall['timedFF'][0]
      if animal=='e145':
         ybinned=ybinned[:-1]
         forwardvel=forwardvel[:-1]
         changeRewLoc=changeRewLoc[:-1]
         trialnum=trialnum[:-1]
         rewards=rewards[:-1]
         licks=licks[:-1]
         time=time[:-1]
      dt= np.nanmedian(np.diff(time))
      # set vars
      eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
      track_length_rad = track_length*(2*np.pi/track_length)
      bin_size=track_length_rad/bins 
      rz = get_rewzones(rewlocs,1/scalingf)       
      # get average success rate
      lasttr = 8
      rates = []; ls_all = []; lr_all = []
      for ep in range(len(eps)-1):
         eprng = range(eps[ep],eps[ep+1])
         success, fail, str_trials, ftr_trials, ttr, \
         total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
         rates.append(success/total_trials)
         # lick rate and selecitivty
         mask = np.array([xx in str_trials[-lasttr:] for xx in trialnum[eprng]])
         ls = get_lick_selectivity(ybinned[eprng][mask], trialnum[eprng][mask], licks[eprng][mask], rewlocs[ep], rewsize,
                  fails_only = False)
         lr = smooth_lick_rate(licks[eprng][mask], dt)
         ls_all.append(np.nanmean(ls))
         lr_all.append(np.nanmean(lr))
      rad = get_radian_position_first_lick_after_rew(eps, ybinned, licks, rewards, rewsize,rewlocs, trialnum, track_length) # get radian coordinates
      # added to get anatomical info
      # takes time
      fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
      Fc3 = fall_fc3['Fc3']
      dFF = fall_fc3['dFF']
      Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
      dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]
      skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
      Fc3 = Fc3[:, skew>2] # only keep cells with skew greateer than 2
      # get abs dist tuning 
      bin_size=3
      tcs_correct_abs, coms_correct_abs,_,__ = make_tuning_curves(eps,rewlocs,ybinned,
      Fc3,trialnum,rewards,forwardvel,
      rewsize,bin_size)
      perm_real = list(combinations(range(len(coms_correct_abs)), 2)) 
      rz_perm_real = [(int(rz[p[0]]),int(rz[p[1]])) for p in perm_real]   
      # get cells that maintain their coms across at least 2 epochs
      place_window = 20 # cm converted to rad                
      perm = list(combinations(range(len(coms_correct_abs)), 2))     
      com_per_ep = np.array([(coms_correct_abs[perm[jj][0]]-coms_correct_abs[perm[jj][1]]) for jj in range(len(perm))])        
      compc = [np.where((comr<place_window) & (comr>-place_window))[0] for comr in com_per_ep]

      rates_per_perm = [rates[perm[1]]-rates[perm[0]] for perm in perm_real]
      goal_per_perm = [len(xx)/len(coms_correct_abs[0]) for xx in compc]
      ls_per_perm = [ls_all[perm[1]]-ls_all[perm[0]] for perm in perm_real]
      lr_per_perm= [lr_all[perm[0]]-lr_all[perm[1]] for perm in perm_real]
      for ii,xx in enumerate(rates_per_perm):
         print(f'rate difference: {xx}, place cell prop: {goal_per_perm[ii]}')
      datadct[f'{animal}_{day}'] = [rates_per_perm,ls_per_perm,lr_per_perm,goal_per_perm]

####################################### RUN CODE #######################################
#%%
plt.rc('font', size=18)          # controls default text sizes
df=pd.DataFrame()
df['rates'] = np.concatenate([v[0] for k,v in datadct.items()])
df['rates']=df['rates']*100
df['lick_selectivity'] = np.concatenate([v[1] for k,v in datadct.items()])
df['lick_rate'] = np.concatenate([v[2] for k,v in datadct.items()])
df['reward_cell'] = np.concatenate([v[3] for k,v in datadct.items()])
df['reward_cell']=df['reward_cell']*100
df['animals'] = np.concatenate([[k.split('_')[0]]*len(v[0]) for k,v in datadct.items()])
df['days'] = np.concatenate([[k.split('_')[1]]*len(v[0]) for k,v in datadct.items()])
df=df[df.reward_cell>0]
df = df.dropna()

# df=df[df.reward_cell<0]
# df=df[(df.animals!='e139')]
#%%
# --- Scatterplot with hue per animal ---
fig,axes = plt.subplots(ncols=3,figsize=(11,4),sharey=True)
metrics = ['rates', 'lick_selectivity', 'lick_rate']
lbl=['$\Delta$ % Correct trials', '$\Delta$ Lick selectivity', '$\Delta$ Lick rate']
for ii,m in enumerate(metrics):
    ax=axes[ii]
    r, p = scipy.stats.pearsonr(df[m], df['reward_cell'])
    n = len(df)
    if ii==2: legend=True
    else: legend=False
    sns.scatterplot(
        data=df, x=m, y='reward_cell',
        hue='animals', palette='tab10', s=60, ax=ax,legend=legend,alpha=0.5
    )
    # Optional: regression line on top of all points
    sns.regplot(
        data=df, x=m, y='reward_cell',
        scatter=False, color='black', ci=None,ax=ax
    )
    # --- Annotate stats ---
    ax.text(
        0.05, 0.95,
        f"r = {r:.3g}\np = {p:.3g}\nn = {n}",
        transform=ax.transAxes,
        verticalalignment='top',fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )
    ax.spines[['top','right']].set_visible(False)
    # plt.xlabel("% Correct trials")
    if ii==2:
        ax.legend(title="Animal", bbox_to_anchor=(1.05, 1), loc='upper left',fontsize=12)
    else: ax.set_ylabel("")
    ax.set_xlabel(lbl[ii])
    ax.set_ylabel("Place cell %")

fig.suptitle('Place cell % vs. performance metrics')
plt.tight_layout()
plt.savefig(os.path.join(savedst, "performance_v_placecell.svg"))

#%%
# --- Correlation stats ---
df=df.groupby(['animals']).mean(numeric_only=True).reset_index()
# --- Scatterplot with hue per animal ---
fig,axes = plt.subplots(ncols=3,figsize=(11,4),sharey=True)
metrics = ['rates', 'lick_selectivity', 'lick_rate']
lbl=['$\Delta$ % Correct trials', '$\Delta$ Lick selectivity', '$\Delta$ Lick rate']
for ii,m in enumerate(metrics):
    ax=axes[ii]
    r, p = scipy.stats.pearsonr(df[m], df['reward_cell'])
    n = len(df)
    if ii==2: legend=True
    else: legend=False
    sns.scatterplot(
        data=df, x=m, y='reward_cell',
        hue='animals', palette='tab10', s=60, ax=ax,legend=legend,
    )
    # Optional: regression line on top of all points
    sns.regplot(
        data=df, x=m, y='reward_cell',
        scatter=False, color='black', ci=None,ax=ax
    )
    # --- Annotate stats ---
    ax.text(
        0.05, 0.95,
        f"r = {r:.3g}\np = {p:.3g}\nn = {n}",
        transform=ax.transAxes,
        verticalalignment='top',fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )
    ax.spines[['top','right']].set_visible(False)
    # plt.xlabel("% Correct trials")
    if ii==2:
        ax.legend(title="Animal", bbox_to_anchor=(1.05, 1), loc='upper left',fontsize=12)
    else: ax.set_ylabel("")
    ax.set_xlabel(lbl[ii])
    ax.set_ylabel("Place cell %")

fig.suptitle('Place cell % vs. performance metrics, per mouse')
plt.tight_layout()
plt.savefig(os.path.join(savedst, "performance_v_placecell_per_mouse.svg"))

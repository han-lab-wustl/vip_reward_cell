

"""
zahra
nov 2024
quantify reward-relative cells post reward
"""
#%%
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
from projects.pyr_reward.placecell import make_tuning_curves_radians_by_trialtype, intersect_arrays, consecutive_stretch,make_tuning_curves, \
    make_velocity_tuning_curves
from projects.opto.behavior.behavior import get_success_failure_trials
from projects.pyr_reward.rewardcell import get_radian_position, extract_data_nearrew, perireward_binned_activity, get_rewzones
from projects.dopamine_receptor.drd import get_moving_time_v3, get_stops_licks
from projects.pyr_reward.rewardcell import perireward_binned_activity_early_late, perireward_binned_activity, get_radian_position_first_lick_after_rew
from projects.memory.behavior import get_behavior_tuning_curve
from projects.pyr_reward.placecell import make_tuning_curves_by_trialtype_w_darktime

# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'

goal_window_cm=20 # to search for rew cells
saveddataset = rf'Z:\saved_datasets\radian_tuning_curves_nearreward_cell_bytrialtype_nopto_{goal_window_cm}cm_window.p'
with open(saveddataset, "rb") as fp: #unpickle
    radian_alignment_saved = pickle.load(fp)
#%%
plt.close('all')
bins=90
dfs=[]
for ii in range(len(conddf)):
   day = conddf.days.values[ii]
   animal = conddf.animals.values[ii]
   if (animal!='e217') & (conddf.optoep.values[ii]<2):
      if animal=='e145' or animal=='e139': pln=2 
      else: pln=0
      params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
      print(params_pth)
      fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 'timedFF',
               'pyr_tc_s2p_cellind', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
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
      # set vars
      eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
      diff =np.insert(np.diff(eps), 0, 1e15)
      eps=eps[diff>2000]
      track_length_rad = track_length*(2*np.pi/track_length)
      bin_size=track_length_rad/bins 
      rz = get_rewzones(rewlocs,1/scalingf)       
      
      # takes time
      fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
      Fc3 = fall_fc3['Fc3']
      dFF = fall_fc3['dFF']
      Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
      dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]
      skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
      ######### place
      # looser restrictions
      # pc_bool = np.sum(pcs,axis=0)>=1
      Fc3 = Fc3[:,((skew>2))] # only keep cells with skew greateer than 2
      dFF = dFF[:,((skew>2))]
      # if no cells pass these crit
      if Fc3.shape[1]==0:
         Fc3 = fall_fc3['Fc3']
         Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
         pc_bool = np.sum(pcs,axis=0)>=1
         Fc3 = Fc3[:,((skew>1.2))]
         dFF = dFF[:,((skew>1.2))]
      bin_size=3 # cm
      # get abs dist tuning 
      tcs_correct_abs, coms_correct_abs,_,__ = make_tuning_curves(eps,rewlocs,ybinned,
      Fc3,trialnum,rewards,forwardvel,
      rewsize,bin_size)
      # get cells that maintain their coms across at least 2 epochs
      place_window = 20 # cm converted to rad                
      perm = list(combinations(range(len(coms_correct_abs)), 2))     
      com_per_ep = np.array([(coms_correct_abs[perm[jj][0]]-coms_correct_abs[perm[jj][1]]) for jj in range(len(perm))])        
      compc = [np.where((comr<place_window) & (comr>-place_window))[0] for comr in com_per_ep]
      # get cells across all epochs that meet crit
      pcs_all = intersect_arrays(*compc)
      coms_correct_abs_rewrel=np.array([com-rewlocs[kk] for kk, com in enumerate(coms_correct_abs)])
      ######### place END
      ######### rew
      # tc w/ dark time
      track_length_dt = 550 # cm estimate based on 99.9% of ypos
      track_length_rad_dt = track_length_dt*(2*np.pi/track_length_dt) # estimate bin for dark time
      bins_dt=150 
      bin_size_dt=track_length_rad_dt/bins_dt # typically 3 cm binswith ~ 475 track length
      tcs_correct, coms_correct, tcs_fail_dt, coms_fail_dt, ybinned_dt, rad = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,rewsize,ybinned,time,licks,
         Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
         bins=bins_dt)

      track_length=270
      goal_window = goal_window_cm*(2*np.pi/track_length) 
      # change to relative value 
      coms_rewrel = np.array([com-np.pi for com in coms_correct])
      # only get cells near reward        
      perm = list(combinations(range(len(coms_correct)), 2))     
      com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])                
      # tuning curves that are close to each other across epochs
      com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
      # in addition, com near but after goal
      # do same quantification for both pre and post rew cells 
      cell_types = ['pre', 'post', 'place']
      velocity = fall['forwardvel'][0]
      veldf = pd.DataFrame({'velocity': velocity})
      velocity = np.hstack(veldf.rolling(5).mean().values)
      # velocity - ndarray: velocity of the animal
      # thres - float: Threshold speed in cm/s
      # Fs - int: Number of frames minimum to be considered stopped
      # ftol - int: Frame tolerance for merging stop periods
      moving_middle,stop = get_moving_time_v3(velocity,2,40,20)
      pre_win_framesALL, post_win_framesALL=31.25*5,31.25*5
      nonrew_stop_without_lick, nonrew_stop_with_lick, rew_stop_without_lick, \
      rew_stop_with_lick,mov_success_tmpts=get_stops_licks(moving_middle, stop, 
                  pre_win_framesALL, post_win_framesALL,\
               velocity, (rewards==1).astype(int), licks, 
               max_reward_stop=31.25*5)    
      # get different stops
      nonrew_stop_without_lick_per_plane = np.zeros_like(changeRewLoc)
      nonrew_stop_without_lick_per_plane[nonrew_stop_without_lick.astype(int)] = 1
      nonrew_stop_with_lick_per_plane = np.zeros_like(changeRewLoc)
      nonrew_stop_with_lick_per_plane[nonrew_stop_with_lick.astype(int)] = 1
      movement_starts=mov_success_tmpts.astype(int)
      rew_per_plane = np.zeros_like(changeRewLoc)
      rew_per_plane[rew_stop_with_lick.astype(int)] = 1
      move_start = np.zeros_like(changeRewLoc)
      move_start[movement_starts.astype(int)] = 1
      range_val=8;binsize=0.1
      # per cell
      celltydf = []
      bound= np.pi/4
      for cell_type in cell_types:
         if cell_type=='post':
            com_goal_subtype = [[xx for xx in com if ((np.nanmedian(coms_rewrel[:,
               xx], axis=0)<=bound) & (np.nanmedian(coms_rewrel[:,
               xx], axis=0)>0))] for com in com_goal if len(com)>0]
         elif cell_type=='pre': # pre
            com_goal_subtype = [[xx for xx in com if ((np.nanmedian(coms_rewrel[:,
               xx], axis=0)>=-bound) & (np.nanmedian(coms_rewrel[:,
               xx], axis=0)<0))] for com in com_goal if len(com)>0]
         elif cell_type=='place':
            # post place
            com_goal_subtype = [[xx for xx in com if np.nanmedian(coms_correct_abs_rewrel,axis=0)[xx]>0] for com in compc if len(com)>0]
            
         # get goal cells across all epochs        
         goal_cells = intersect_arrays(*com_goal_subtype) if len(com_goal_subtype)>0 else []
         goal_cell_iind = goal_cells
         if cell_type!='place':
            tc = tcs_correct
         else: 
            tc = tcs_correct_abs
         # instead of latencies quantify dff
         gc_latencies_mov=[];gc_latencies_rew=[];cellid=[]
         # get latencies based on average of trials
         for gc in goal_cell_iind:
               # _, meanvelrew, __, velrew = perireward_binned_activity(velocity, move_start, 
               #         fall['timedFF'][0], fall['trialnum'][0], range_val,binsize)
               # _, meanlickrew, __, lickrew = perireward_binned_activity(fall['licks'][0], move_start, 
               #     fall['timedFF'][0], fall['trialnum'][0], range_val,binsize)
               _, meanrew, __, rewall = perireward_binned_activity(dFF[:,gc], rewards==1, 
                  time, trialnum, range_val,binsize)
               if np.nanmax(meanrew)>.5: # only get highly active cells?
                  _, meanrstops, __, rewrstops = perireward_binned_activity(dFF[:,gc], move_start, 
                  time, trialnum, range_val,binsize)
                  # quantify dff
                  transient_around_rew = np.nanmean(meanrew[int(range_val/binsize)-int(0/binsize):int(range_val/binsize)+int(2/binsize)])
                  gc_latencies_rew.append(transient_around_rew)
                  transient_after_rew = np.nanmean(meanrstops[int(range_val/binsize)-int(1/binsize):int(range_val/binsize)+int(1/binsize)])
                  gc_latencies_mov.append(transient_after_rew)
                  cellid.append(gc)
         # concat by cell
         df=pd.DataFrame()
         df['dff']=np.concatenate([gc_latencies_rew,gc_latencies_mov])
         df['behavior']=np.concatenate([['Reward']*len(gc_latencies_rew),
                           ['Movement Start']*len(gc_latencies_mov)])
         df['animal']=[animal]*len(df)
         df['day']=[day]*len(df)
         df['cellid']=np.concatenate([cellid]*2)
         df['cell_type'] = [cell_type]*len(df)
         print(cell_type, len(df))
         celltydf.append(df)
      dfs.append(pd.concat(celltydf))

#%%
#plot all cells
plt.rc('font', size=20)
df=pd.concat(dfs)
df = df.reset_index()
# df=df[df.animal=='e201']
# df=dfs[0]
fig,ax=plt.subplots(figsize=(8,5))
sns.stripplot(x='behavior',y='dff',data=df,hue='animal',s=8,alpha=0.3,dodge=True)
sns.boxplot(x='behavior',y='dff',data=df,hue='animal',fill=False,showfliers=False,whis=0)
ax.axhline(0,color='k',linestyle='--')
# for an in df.animal.unique():
#     dfan = df[df.animal==an]
#     for dy in dfan.day.unique():
#         dfdy = dfan[dfan.day==dy]
#         for celliid in dfdy.cellid.unique():
#             sns.lineplot(x='behavior',y='latency (s)',data=dfdy[dfdy.cellid==celliid],
#                 alpha=0.1,color='gray')

# sns.barplot(x='behavior',y='latency (s)',data=df,fill=False)
ax.spines[['top','right']].set_visible(False)
#%%
plt.rc('font', size=20)
s=10
fig,axes=plt.subplots(ncols=2, figsize=(8,5),sharex=True,width_ratios=[1.3,1])
ax=axes[0]
df=df[(df.animal!='e139') & (df.animal!='e145') & (df.animal!='e189')]
# only get last day
day_per_animal = [df.loc[(df.animal==an), 'day'].unique()[-1] for an in df.animal.unique()]
df_day = pd.concat([df[(df.animal==an) & (df.day==day_per_animal[ii])] for ii,an in enumerate(df.animal.unique())])
celln_per_animal = [[len(df_day[(df_day.animal==an)&(df_day.cell_type==ct)]) for ii,an in enumerate(df.animal.unique())] for ct in cell_types ]
order=['Reward', 'Movement Start']
hue_order=['pre', 'post']
sns.violinplot(x='behavior',y='dff',data=df_day,order=order,hue_order=hue_order,ax=ax,legend=False,
        dodge=True,hue='cell_type',palette='Dark2')
comparisons=[];pvals=[]
for ct in cell_types:
    d1 = df_day[(df_day.behavior == 'Movement Start') & (df_day.cell_type == ct)]['dff']
    d2 = df_day[(df_day.behavior == 'Reward') & (df_day.cell_type == ct)]['dff']
    if len(d1) > 0 and len(d2) > 0:
        stat, p = scipy.stats.wilcoxon(d1, d2)
        pvals.append(p)
        comparisons.append((ct, 'Reward', 'Movement Start'))
from statsmodels.stats.multitest import multipletests
# --- Correct p-values using FDR ---
reject, pvals_corrected, _, _ = multipletests(pvals, method='fdr_bh')
# --- Plot comparisons with corrected p-values ---
y_max = df_day['dff'].max()
y_start = y_max
y_step = .6
fs = 30
behavior_positions = {'Reward': 0, 'Movement Start': 1}
for i, ((ct, b1, b2), p_corr, sig) in enumerate(zip(comparisons, pvals_corrected, reject)):
    x1 = behavior_positions[b1] + (-0.2 if ct == 'pre' else 0.2)
    x2 = behavior_positions[b2] + (-0.2 if ct == 'pre' else 0.2)
    y = y_start + i * y_step
    h=0.05
    ax.plot([x1, x1, x2, x2], [y-h, y, y, y-h], lw=1.5, c='black')
    if p_corr<0.05:
        star='*'
    if p_corr<0.01:
        star='**'
    if p_corr<0.001:
        star='***'
    if p_corr>0.05: star=''
    ax.text((x1 + x2)/2, y + 0.01, f"{star}", 
            ha='center', va='bottom', fontsize=46)
    ax.text((x1 + x2)/2, y + 0.05, f"p={p_corr:.3g}", 
            ha='center', va='bottom', fontsize=12)

# per animal
dfagg = df.groupby(['animal','behavior','cell_type']).mean(numeric_only=True)
hue_order=['pre', 'post','place']
order = ['Reward','Movement Start']
ax.set_xticklabels(ax.get_xticklabels(), rotation=10, ha="right")
# sns.barplot(x='behavior',y='latency (s)',data=df,fill=False)
ax.spines[['top','right']].set_visible(False)
ax.set_ylabel('Mean $\Delta F/F$')
ax.set_xlabel('')
 
ax=axes[1]
sns.stripplot(x='behavior',y='dff',data=dfagg,order=order,s=s,alpha=0.7,hue_order=hue_order,ax=ax,
        dodge=True,hue='cell_type',palette='Dark2')
sns.barplot(x='behavior',y='dff',hue='cell_type',order=order,hue_order=hue_order,ax=ax,
            data=dfagg,fill=False,palette='Dark2')
dfagg=dfagg.reset_index()
xpos = [[-0.2, 0.8], [0.2, 1.2]]
for ii, cell_type in enumerate(cell_types):
    for an in dfagg.animal.unique():
        dfan = dfagg[(dfagg.animal == an) & (dfagg.cell_type == cell_type)]
        for dy in dfan.day.unique():
            dfdy = dfan[dfan.day == dy]
            for celliid in dfdy.cellid.unique():
                dfcell = dfdy[dfdy.cellid == celliid]
                dfcell = dfcell.set_index('behavior').loc[order].reset_index()
                color=sns.color_palette('Dark2')[ii]
                yvals = dfcell['dff'].values
                if len(yvals) == 2:
                    ax.plot(xpos[ii], yvals, alpha=0.5, color=color, linewidth=1.5)
ax.set_xticklabels(ax.get_xticklabels(), rotation=10, ha="right")
# sns.barplot(x='behavior',y='latency (s)',data=df,fill=False)
ax.spines[['top','right']].set_visible(False)
ax.set_ylabel('')
ax.set_xlabel('')
pvals = []
comparisons = []

for ct in cell_types:
    d1 = dfagg[(dfagg.behavior == 'Movement Start') & (dfagg.cell_type == ct)]['dff']
    d2 = dfagg[(dfagg.behavior == 'Reward') & (dfagg.cell_type == ct)]['dff']
    if len(d1) > 0 and len(d2) > 0:
        stat, p = scipy.stats.wilcoxon(d1, d2)
        pvals.append(p)
        comparisons.append((ct, 'Reward', 'Movement Start'))
from statsmodels.stats.multitest import multipletests
# --- Correct p-values using FDR ---
reject, pvals_corrected, _, _ = multipletests(pvals, method='fdr_bh')
# --- Plot comparisons with corrected p-values ---
y_max = dfagg['dff'].max()
y_start = y_max + 0.05
y_step = 0.07
fs = 30
behavior_positions = {'Reward': 0, 'Movement Start': 1}
for i, ((ct, b1, b2), p_corr, sig) in enumerate(zip(comparisons, pvals_corrected, reject)):
    x1 = behavior_positions[b1] + (-0.2 if ct == 'pre' else 0.2)
    x2 = behavior_positions[b2] + (-0.2 if ct == 'pre' else 0.2)
    y = y_start + i * y_step
    h=0.05
    ax.plot([x1, x1, x2, x2], [y-h, y, y, y-h], lw=1.5, c='black')
    if p_corr<0.05:
        star='*'
    if p_corr<0.01:
        star='**'
    if p_corr<0.001:
        star='***'
    if p_corr>0.05: star=''
    ax.text((x1 + x2)/2, y + 0.01, f"{star}", 
            ha='center', va='bottom', fontsize=46)
    ax.text((x1 + x2)/2, y + 0.05, f"p={p_corr:.3g}", 
            ha='center', va='bottom', fontsize=12)
# Remove all current legends
ax.legend_.remove()

# Create a single legend for cell type
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))  # Remove duplicates
ax.legend(by_label.values(), by_label.keys(), title='Cell Type', fontsize=12, title_fontsize=14)

plt.savefig(os.path.join(savedst, 'postrew_dff_reward_v_movement.svg'),bbox_inches='tight')
# %%
# new fig 4
fig,ax=plt.subplots(figsize=(6,5))
colors = ['cornflowerblue', 'k']
# sns.stripplot(x='cell_type',y='dff',data=dfagg,order=hue_order,s=s,alpha=0.7,hue_order=order,ax=ax,dodge=True,hue='behavior',palette=colors)
sns.barplot(x='cell_type',y='dff',hue='behavior',order=hue_order,hue_order=order,ax=ax,
            data=dfagg,fill=False,palette=colors)
# ax.set_xticklabels(['Pre-reward','Post-reward'])
ax.set_xlabel('Cell type')
ax.set_ylabel('Mean $\Delta F/F$')
# sns.barplot(x='behavior',y='latency (s)',data=df,fill=False)
ax.spines[['top','right']].set_visible(False)
pvals = []
comparisons = []
for ct in cell_types:
    d1 = dfagg[(dfagg.behavior == 'Movement Start') & (dfagg.cell_type == ct)]['dff']
    d2 = dfagg[(dfagg.behavior == 'Reward') & (dfagg.cell_type == ct)]['dff']
    stat, p = scipy.stats.wilcoxon(d1, d2)
    pvals.append(p)
    comparisons.append((ct, 'Reward', 'Movement Start'))
        
from statsmodels.stats.multitest import multipletests
# --- Correct p-values using FDR ---
reject, pvals_corrected, _, _ = multipletests(pvals, method='fdr_bh')
# --- Plot comparisons with corrected p-values ---
y_max = dfagg['dff'].max()
y_start = y_max + 0.05
y_step = 0.07
fs = 30
behavior_positions = {'Reward': 0, 'Movement Start': .4}

for i, ((ct, b1, b2), p_corr, sig) in enumerate(zip(comparisons, pvals_corrected, reject)):
   if ct=='pre':
      ps = -0.2
   elif ct=='post':
      ps = .8
   else:
      ps = 2
   x1 = behavior_positions[b1] + ps
   x2 = behavior_positions[b2] + ps
   y = y_start + i * y_step
   h=0.05
   ax.plot([x1, x1, x2, x2], [y-h, y, y, y-h], lw=1.5, c='black')
   if p_corr<0.05:
      star='*'
   if p_corr<0.01:
      star='**'
   if p_corr<0.001:
      star='***'
   if p_corr>0.05: star=''
   ax.text((x1 + x2)/2, y + 0.01, f"{star}", 
         ha='center', va='bottom', fontsize=46)
   ax.text((x1 + x2)/2, y + 0.05, f"p={p_corr:.3g}", 
         ha='center', va='bottom', fontsize=12)
   
# Remove all current legends
ax.legend_.remove()
cell_type_order = hue_order  # same as used in plotting
behavior_order = ['Reward','Movement Start']
offset = 0.2  # half of the dodge width

for i, ct in enumerate(cell_type_order):
    df_ct = dfagg[dfagg.cell_type == ct]
    df_pivot = df_ct.pivot(index='animal', columns='behavior', values='dff')

    # Drop animals with missing values
    df_pivot = df_pivot.dropna(subset=behavior_order)

    for _, row in df_pivot.iterrows():
        x1 = i - offset  # Movement Start
        x2 = i + offset  # Reward
        y1 = row[behavior_order[0]]
        y2 = row[behavior_order[1]]
        ax.plot([x1, x2], [y1, y2], color='gray', linewidth=1.5, alpha=0.5, zorder=0)
ax.set_xticklabels(['Pre-reward','Post-reward','Post-place'])
# Create a single legend for cell type
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))  # Remove duplicates
ax.legend(by_label.values(), by_label.keys())

plt.savefig(os.path.join(savedst, 'postrew_dff_reward_v_movement.svg'),bbox_inches='tight')

# %%

#%%
# histogram of latencies
plt.rc('font', size=22) 
fig,ax=plt.subplots()
# per animal
sns.histplot(x='latency (s)',hue='animal',data=df[df.behavior=='Reward'],
            bins=40)
# ax.legend(bbox_to_anchor=(1.01, 1))  # Moves legend farther right
# sns.boxplot(x='behavior',y='latency (s)',data=df,fill=False,showfliers= False,whis=0)
# ax.axhline(0,color='k',linestyle='--')
# for an in df.animal.unique():
#     dfan = df[df.animal==an]
#     for dy in dfan.day.unique():
#         dfdy = dfan[dfan.day==dy]
#         for celliid in dfdy.cellid.unique():
#             sns.lineplot(x='behavior',y='latency (s)',data=dfdy[dfdy.cellid==celliid],
#                 alpha=0.1,color='gray')
ax.set_ylabel('# Post-reward cells')
ax.set_xlabel('Latency from reward (s)')
# sns.barplot(x='behavior',y='latency (s)',data=df,fill=False)
ax.spines[['top','right']].set_visible(False)
plt.savefig(os.path.join(savedst, 'latency_postrew_hist.svg'))
#%%
plt.close('all')
# per animal pair
ansq = int(np.sqrt(len(df.animal.unique())))
fig,axes=plt.subplots(nrows=ansq,ncols=ansq,figsize=(8,12),sharex=True, sharey=True)
axes = axes.flatten()
for ii,an in enumerate(df.animal.unique()):
    dfan = df[df.animal==an]
    ax=axes[ii]
    sns.stripplot(x='behavior',y='latency (s)',data=dfan,ax=ax,s=8,alpha=0.3,dodge=True)
    sns.boxplot(x='behavior',y='latency (s)',data=dfan,ax=ax,fill=False,showfliers= False,whis=0)
    for dy in dfan.day.unique():
        dfdy = dfan[dfan.day==dy]
        for celliid in dfdy.cellid.unique():
            sns.lineplot(x='behavior',y='latency (s)',data=dfdy[dfdy.cellid==celliid],
                alpha=0.1,color='gray',ax=ax)
    ax.set_title(an)
fig.tight_layout()

#%%

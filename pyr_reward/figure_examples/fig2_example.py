#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.opto.analysis.pyramdial.placecell import get_rew_cells_opto
from projects.pyr_reward.rewardcell import get_radian_position_first_lick_after_rew,intersect_arrays
from projects.pyr_reward.placecell import make_tuning_curves_by_trialtype_w_darktime,make_tuning_curves_by_trialtype_w_darktime_early,make_tuning_curves
from projects.opto.behavior.behavior import smooth_lick_rate
import warnings
warnings.filterwarnings("ignore")
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_performance_chrimson.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper\panels_main_figures'
savepth = os.path.join(savedst, 'vip_opto_reward_relative.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_reward_cell_bytrialtype_vipopto.p"
with open(saveddataset, "rb") as fp: #unpickle
        radian_alignment_saved = pickle.load(fp)
# initialize var
#%%
ii=65# for maps; and fig 2
ii=58
cm_window=20
day = int(conddf.days.values[ii])
animal = conddf.animals.values[ii]
if animal=='e145': pln=2  
else: pln=0
params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
print(params_pth)

fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
'timedFF', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
'stat', 'licks','putative_pcs'])
pcs=fall['putative_pcs'][0]
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
time = fall['timedFF'][0]
lick = fall['licks'][0]
if animal=='e145':
   ybinned=ybinned[:-1]
   forwardvel=forwardvel[:-1]
   changeRewLoc=changeRewLoc[:-1]
   trialnum=trialnum[:-1]
   rewards=rewards[:-1]
   time=time[:-1]
   lick=lick[:-1]
# set vars
eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
# only test opto vs. ctrl
eptest = conddf.optoep.values[ii]
if conddf.optoep.values[ii]<2: 
         eptest = random.randint(2,3)   
         if len(eps)<4: eptest = 2 # if no 3 epochs 
eptest=int(eptest)   
lasttr=8 # last trials
bins=90
rad = get_radian_position_first_lick_after_rew(eps, ybinned, lick, rewards, rewsize,rewlocs, trialnum, track_length) # get radian coordinates
track_length_rad = track_length*(2*np.pi/track_length)
bin_size=track_length_rad/bins
#%%
if True:  
# takes time
   fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
   Fc3 = fall_fc3['Fc3']
   dFF = fall_fc3['dFF']
   # for inhibi
   # Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool)&~(fall['bordercells'][0]).astype(bool))]
   # dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool)&~(fall['bordercells'][0]).astype(bool))]
   Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
   dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]

   skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
   # if animal!='z14' and animal!='e200' and animal!='e189':                
   Fc3 = Fc3[:, skew>2] # only keep cells with skew greater than 2
   # tc w/ dark time
   print('making tuning curves...\n')
   track_length_dt = 550 # cm estimate based on 99.9% of ypos
   track_length_rad_dt = track_length_dt*(2*np.pi/track_length_dt) # estimate bin for dark time
   bins_dt=150 
   bin_size_dt=track_length_rad_dt/bins_dt # typically 3 cm binswith ~ 475 track length
   tcs_correct_abs_no_si, coms_correct_abs_no_si,_,__ = make_tuning_curves(eps,rewlocs,ybinned,Fc3,trialnum,rewards,forwardvel,rewsize,3)

   tcs_correct, coms_correct, tcs_fail, coms_fail, ybinned_dt,rad = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,rewsize,ybinned,time,lick,Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
         bins=bins_dt,lasttr=8) 
   lick_rate=smooth_lick_rate(lick, 1/31.25)
   lick_tcs_correct, lick_coms_correct, lick_tcs_fail, lick_coms_fail, ybinned_dt,rad = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,rewsize,ybinned,time,lick,np.array([lick_rate]).T,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
      bins=bins_dt,lasttr=8) 
   vel_tcs_correct, vel_coms_correct, vel_tcs_fail, vel_coms_fail, ybinned_dt,rad = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,rewsize,ybinned,time,lick,np.array([forwardvel]).T,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
      bins=bins_dt,lasttr=8) 

   # early tc
   tcs_correct_early, coms_correct_early, tcs_fail_early, coms_fail_early, ybinned_dt = make_tuning_curves_by_trialtype_w_darktime_early(eps,rewlocs,rewsize,ybinned,time,lick,Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,bins=bins_dt,lasttr=8)  
   goal_window = cm_window*(2*np.pi/track_length) # cm converted to rad
   coms_rewrel = np.array([com-np.pi for com in coms_correct])
   perm = list(combinations(range(len(coms_correct)), 2)) 
   print(perm)
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
   com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap][:2]
   com_goal_all=np.unique(np.concatenate(com_goal))
   goal_cells = intersect_arrays(*com_goal) if len(com_goal) > 0 else []    
   Fc3 = fall_fc3['Fc3']
   dFF = fall_fc3['dFF']
   Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
   dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
   skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
   #if pc in all but 1
   # looser restrictions
   pc_bool = np.squeeze(np.sum(pcs,axis=0)>=1)
   Fc3 = Fc3[:,((skew>2)&pc_bool)] # only keep cells with skew greateer than 2
   # if no cells pass these crit
   if Fc3.shape[1]==0:
      Fc3 = fall_fc3['Fc3']
      Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
      pc_bool = np.sum(pcs,axis=0)>=1
      Fc3 = Fc3[:,((skew>2)&pc_bool)]
   bin_size=3 # cm
   # get abs dist tuning 
   tcs_correct_abs, coms_correct_abs,_,__ = make_tuning_curves(eps,rewlocs,ybinned,Fc3,trialnum,rewards,forwardvel,
            rewsize,bin_size)
   # get cells that maintain their coms across at least 2 epochs
   place_window = 20 # cm converted to rad                
   perm = list(combinations(range(len(coms_correct_abs)), 2))     
   com_per_ep = np.array([(coms_correct_abs[perm[jj][0]]-coms_correct_abs[perm[jj][1]]) for jj in range(len(perm))])        
   compc = [np.where((comr<place_window) & (comr>-place_window))[0] for comr in com_per_ep][:2] # just do 3 combinations for now
   # get cells across all epochs that meet crit
   pcs = np.unique(np.concatenate(compc))
   pcs_all = intersect_arrays(*compc)
   pcs_all=[xx for xx in pcs_all if xx not in goal_cells]
   # pcs_all = pcs

#%%#%% fig 2 eg
# add tuning curves
# Function for row-wise normalization
def normalize_rows(x):
    x_min = np.min(x, axis=1, keepdims=True)
    x_max = np.max(x, axis=1, keepdims=True)
    return (x - x_min) / (x_max - x_min + 1e-10)
# Moving average function
def moving_average(x, window_size=1):
    return np.convolve(x, np.ones(window_size)/window_size, mode='same')
plt.rc('font', size=16)
fig, axes = plt.subplots(ncols=4,nrows=3, figsize=(9,5), sharex=True, sharey='row',height_ratios=[3,1,1])
im_list = []  # to store imshow objects for colorbar
colors = ['k', 'slategray', 'darkcyan', 'darkgoldenrod', 'orchid']

for en,ep in enumerate([0,1,2,3]):
   ax = axes[0,en]
   # Apply moving average and stack
   pltt = [moving_average(tcs_correct[ep, cll, :]) for cll in goal_cells]
   pltt = np.array(pltt)
   # Sort by COM
   pltt = pltt[np.argsort(coms_correct[0][goal_cells])]
   # Normalize each row
   pltt = normalize_rows(pltt)
   ax.set_yticks([0,len(pltt)-1])
   ax.set_yticklabels([1,len(pltt)])
   # Plot
   im = ax.imshow(pltt, aspect='auto', cmap='viridis', vmin=0, vmax=1)
   im_list.append(im)
   thres1=75-75/4; thres2=75+(75/4)
   ax.axvline(thres1, color='y', linestyle='--',linewidth=2)
   ax.axvline(thres2, color='y', linestyle='--',linewidth=2)
   ax.axvline(bins_dt//2, color='w', linestyle='--',linewidth=3)
   if ep == 0:
      ax.set_ylabel('Reward cell #')
   ax.set_title(f'Epoch {en+1}\n Reward at {int(rewlocs[ep])} cm',fontsize=14)
   ax=axes[1,en]
   ax.plot(lick_tcs_correct[ep][0],color=colors[en])
   ax.spines[['top', 'right']].set_visible(False)
   ax.axvline(bins_dt//2, color='k', linestyle='--',linewidth=3)
   if ep == 0:ax.set_ylabel('Lick rate (licks/s)')
   ax=axes[2,en]
   ax.plot(vel_tcs_correct[ep][0],color=colors[en])
   ax.spines[['top', 'right']].set_visible(False)
   ax.axvline(bins_dt//2, color='k', linestyle='--',linewidth=3)
   if ep == 0:
      ax.set_ylabel('Velocity (cm/s)')
      ax.set_xlabel('Reward-centric distance ($\Theta$)')

ax.set_xticks([0, thres1, bins_dt/2, thres2, bins_dt])
ax.set_xticklabels(['-$\\pi$','-$\\pi/4$',0,'$\\pi/4$','$\\pi$'])
# Add colorbar
cbar = fig.colorbar(im_list[0], ax=axes[0,3], orientation='vertical', fraction=0.05, pad=0.05)
cbar.set_label('Norm. $\Delta F/F$')
plt.tight_layout()
plt.savefig(os.path.join(savedst, f'{animal}_{day}_fig2_rew_map.svg'), bbox_inches='tight')
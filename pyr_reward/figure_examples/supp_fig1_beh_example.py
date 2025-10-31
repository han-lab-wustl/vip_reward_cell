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
iis=conddf[conddf.animals=='z9'].index
iis=[133]
cm_window=20
for ii in iis:
   day = int(conddf.days.values[ii])
   animal = conddf.animals.values[ii]
   if animal=='e145': pln=2  
   else: pln=0
   params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
   print(params_pth)
   plt.rc('font', size=16)
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
   # behavior eg
   # example plot during learning
   eprng = np.arange(eps[0],eps[3])
   epoch_ids=np.zeros_like(eprng)
   for ep in range(3):
      epoch_ids[eps[ep]:eps[ep+1]]=ep+1

   mask = np.zeros_like(trialnum).astype(bool)
   mask[5850:14900]=1
   # mask[eps[0]+8500:eps[1]+2700]=True
   import matplotlib.patches as patches
   fig, ax = plt.subplots(figsize=(6,3))
   ypos=ybinned
   rew=rewards
   lick[ybinned<2]=0
   ybinned[ybinned<2]=np.nan
   # incorrect licks
   trials=trialnum[mask]
   incorrlick=np.zeros_like(lick[mask])
   for trial in np.unique(trials):
      tr = trials==trial
      if trial>2:
         if sum(rew[mask][tr])<1.5:
            incorrlick[tr]=lick[mask][tr]
   nolick = [tr for tr in np.unique(trials) if sum(lick[mask][trials==tr])==0]
   mask2 = np.ones_like(trials).astype(bool)
   mask2[[ii for ii,xx in enumerate(trials) if xx in nolick]]=0
   ax.plot(ypos[mask],zorder=1,label='Mouse Postion',color='slategray')
   s=9
   ax.scatter(np.where(lick[mask])[0], ypos[mask][np.where(lick[mask])[0]], color='k',zorder=2,s=s,label='Lick')
   ax.scatter(np.where(incorrlick)[0], ypos[mask][np.where(incorrlick)[0]], color='r',zorder=2,s=s, label='Lick in incorrect trial')
   ax.scatter(np.where(rew[mask])[0], ypos[mask][np.where(rew[mask])[0]], color='cyan',zorder=2,s=12, label='Reward')
   # ax.add_patch(
   # patches.Rectangle(
   #    xy=(probes[0],0),  # point of origin.
   #    width=probes[-1]-probes[0], height=270, linewidth=1, # width is s
   #    color='royalblue', alpha=0.2,
   #    label='Probe trials'))
   ax.add_patch(
   patches.Rectangle(
      xy=(0,rewlocs[1]-rewsize/2),  # point of origin.
      width=len(ypos[mask]), height=rewsize, linewidth=1, # width is s
      color='#42a5f5ff', alpha=0.3,label='Reward zone'))
   ax.set_ylim([0,270])
   ax.set_yticks([0,270])
   ax.set_yticklabels([0,270])
   ax.set_ylabel('Track position (cm)')
   ax.spines[['top','right']].set_visible(False)
   ax.set_xticks([0, len(ypos[mask])])
   ax.set_xticklabels([0, int(np.ceil((len(ypos[mask])/(31.25/2))/60))])
   ax.set_xlabel('Time (minutes)')
   probe_y = 267  # Y position above track
   from itertools import groupby
   from operator import itemgetter
   # Group contiguous indices
   probe_groups = []
   for k, g in groupby(enumerate(probes), lambda ix: ix[0] - ix[1]):
      group = list(map(itemgetter(1), g))
      probe_groups.append((group[0], group[-1]))
   ax.legend(fontsize=10)
   bar_height=3
   # Plot bars per contiguous span
   for i, (start, end) in enumerate(probe_groups):
      ax.add_patch(
         patches.Rectangle(
               (start, probe_y),
               end - start + 1,
               bar_height,
               color='royalblue',
               linewidth=2,
               label='Probes' if i == 0 else None,
               zorder=3
         )
      )
   # --- Epoch and probe bar params ---
   epoch_y = 267
   epoch_bar_height = 2

   # Define x-ranges for each epoch (in sample indices)
   epoch_spans = [
      (probe_groups[0][0], probe_groups[0][1], 'Probe', 'royalblue'),
      (probe_groups[0][1]+1, len(ypos[mask])-1, 'Epoch 2', '#42a5f5ff'),
      (probe_groups[1][0], probe_groups[1][1], 'Probe', 'royalblue'),
   ]

   # Draw bars
   for i, (start, end, label, color) in enumerate(epoch_spans):
      ax.add_patch(
         patches.Rectangle(
               (start, epoch_y),           # x = start index, y = above track
               end - start + 1,            # width = duration
               epoch_bar_height,
               color=color,
               edgecolor=None,
               linewidth=0,
               zorder=3,
         )
      )
      # Add text label above the bar
      ax.text(
         (start + end) / 2,
         epoch_y + 3.5,
         label,
         ha='center',
         va='bottom',
         fontsize=14,
         zorder=4
      )
   plt.savefig(os.path.join(savedst, f'supp_fig1_beh_{animal}_{day}.svg'),bbox_inches='tight')
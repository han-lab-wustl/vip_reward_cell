
"""
zahra
april 2025
get place cells and plot com histogram
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.pyr_reward.placecell import make_tuning_curves_by_trialtype_w_darktime, intersect_arrays
from projects.pyr_reward.rewardcell import get_radian_position, cosine_sim_ignore_nan
from projects.opto.behavior.behavior import get_success_failure_trials
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
savepth = os.path.join(savedst, 'true_pc.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
#%%
dataraster = {} # overwrite
coms_all = []
pc_ind = []
pc_prop = []
num_epochs = []
epoch_perm = []
pvals = []
total_cells = []
num_iterations=1000
place_cell_null=[]
# iterate through all animals
for ii in range(len(conddf)):
   day = conddf.days.values[ii]
   animal = conddf.animals.values[ii]
   if (animal!='e217') and (conddf.optoep.values[ii]<2):
      if animal=='e145' or animal=='e139': pln=2 
      else: pln=0
      params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
      
      fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
         'putative_pcs', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells','timedFF','licks',
         'stat'])
      pcs = np.vstack(np.array(fall['putative_pcs'][0]))
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
         lick=lick[:-1 ]
      # set vars
      eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf
      eps = np.append(eps, len(changeRewLoc))     
      diff =np.insert(np.diff(eps), 0, 1e15)
      eps=eps[diff>2000]
      if len(eps)>3:   
         print(params_pth)
         lasttr=8 # last trials
         bins=90
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

         tcs_correct, coms_correct, tcs_fail, coms_fail, ybinned_dt, rad = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,
            rewsize,ybinned,time,lick,
            Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
            bins=bins_dt)  
         goal_window = 20*(2*np.pi/track_length) # cm converted to rad
         # change to relative value 
         coms_rewrel = np.array([com-np.pi for com in coms_correct])
         perm = list(combinations(range(len(coms_correct)), 2)) 
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
         com_goal=[com for com in com_goal if len(com)>0]
         goal_cells = intersect_arrays(*com_goal) if len(com_goal)>0 else []
         # dropped cells
         ep12 = [(0,1)]
         arr=[]
         for kk, p in enumerate(perm):
            if p in ep12:
               arr.append(com_goal[kk])
         pcs_ep12 = intersect_arrays(*arr) if len(arr)>0 else []
         ep23 = [(1,2)]
         arr=[]
         for kk, p in enumerate(perm):
               if p in ep23:
                  arr.append(com_goal[kk])
         pcs_ep23 = intersect_arrays(*arr) if len(arr)>0 else []
         ep13 = [(0,2)]
         arr=[]
         for kk, p in enumerate(perm):
               if p in ep23:
                  arr.append(com_goal[kk])
         pcs_ep13 = intersect_arrays(*arr) if len(arr)>0 else []

         dropped3 = [xx for xx in pcs_ep12 if xx not in goal_cells]
         dropped1 = [xx for xx in pcs_ep23 if xx not in goal_cells]
         dropped2 = [xx for xx in pcs_ep13 if xx not in goal_cells]

         dataraster[f'{animal}_{day}']=[tcs_correct[:,goal_cells], tcs_correct[:,dropped3],tcs_correct[:,dropped1],tcs_correct[:,dropped2]]

#%%from sklearn.preprocessing import minmax_scale
import numpy as np
import matplotlib.pyplot as plt
def get_mean_and_cosine_similarity(mat1, mat2,results, label='', typ=''):
   mean1 = np.nanmean(mat1)
   mean2 = np.nanmean(mat2)

   cosines = np.array([cosine_sim_ignore_nan(mat1[i], mat2[i]) for i in range((mat1).shape[0])])
   mean_cosine = np.nanmean(cosines)

   # Append results
   results['animal'].append(an)
   results['label'].append(label)
   results['type'].append(typ)
   results['mean_init'].append(mean1)
   results['mean_dropped'].append(mean2)
   results['cos_sim'].append(mean_cosine)

   print(f"{label}:\n  Mean act. Epoch {init_ep[i]+1}: {mean1:.3f}, Epoch {dropped_ep[i]+1}: {mean2:.3f}\n  Cosine similarity: {mean_cosine:.3f}")

animals = ['e218', 'e216', 'e201', 'e186', 'e190',
       'z8', 'z9', 'z16']
plt.rc('font', size=16) 
comb_types = [1,2,3] # 1 = dropped in ep 3; 2 = dropped in ep 1; 3 = dropped in ep 2 but not in ep 1 or 3
lbls = ['Reward in epoch 1, 2', 'Reward in epoch 2, 3', 'Reward in epoch 1, 3']
init_ep = [0,1,0] # initial ep to plot
dropped_ep = [2,0,1]
# Store results across animals
results = {
    'animal': [],
    'label': [],
    'type': [],  # 'dedicated' or 'dropped'
    'mean_init': [],
    'mean_dropped': [],
    'cos_sim': []
}

for i in range(len(lbls)):
   for an in animals:
      # Stack and drop rows with NaNs
      # first epoch of both types
      dedicated_place = np.vstack([v[0][init_ep[i],:,:] for k, v in dataraster.items() if an in k])
      # dedicated_place = np.vstack([v[0][0,:,:] for k, v in dataraster.items() if an in k])
      dedicated_place_ep3 = np.vstack([v[0][dropped_ep[i],:,:] for k, v in dataraster.items() if an in k])
      dropped_place = np.vstack([v[comb_types[i]][init_ep[i],:,:] for k, v in dataraster.items() if an in k])
      dropped_place_ep = np.vstack([v[comb_types[i]][dropped_ep[i],:,:] for k, v in dataraster.items() if an in k])
      dedicated_place[np.isnan(dedicated_place).any(axis=1)]=0
      dropped_place[np.isnan(dropped_place).any(axis=1)]=0
      dropped_place_ep[np.isnan(dropped_place_ep).any(axis=1)]=0
      dedicated_place_ep3[np.isnan(dedicated_place_ep3).any(axis=1)]=0
      get_mean_and_cosine_similarity(dedicated_place, dedicated_place_ep3,results,
                           label=f"{lbls[i]}", typ='place in all epochs')

      get_mean_and_cosine_similarity(dropped_place, dropped_place_ep,results,
                                    label=f"{lbls[i]}", typ='place in some epochs')

      # or drop nans
      # dedicated_place = dedicated_place[~np.isnan(dedicated_place).any(axis=1)]
      # dedicated_place_ep3 = dedicated_place_ep3[~np.isnan(dedicated_place_ep3).any(axis=1)]
      # dropped_place = dropped_place[~np.isnan(dropped_place).any(axis=1)]
      # dropped_place_ep = dropped_place_ep[~np.isnan(dropped_place_ep).any(axis=1)]

      # Normalize each row to 0–1
      # dedicated_place = minmax_scale(dedicated_place, axis=1)
      # dropped_place = minmax_scale(dropped_place, axis=1)
      # Sort each by peak (argmax of each row)
      dedicated_sort_idx = np.argmax(dedicated_place, axis=1).argsort()
      dedicated_sort_idx_ep3 = np.argmax(dedicated_place, axis=1).argsort()
      dropped_sort_idx = np.argmax(dropped_place, axis=1).argsort()
      dropped_ep_sort_idx = np.argmax(dropped_place, axis=1).argsort()

      dedicated_place_ep3 = dedicated_place_ep3[dedicated_sort_idx_ep3]
      dedicated_place = dedicated_place[dedicated_sort_idx]
      dropped_place = dropped_place[dropped_sort_idx]
      dropped_place_ep = dropped_place_ep[dropped_ep_sort_idx]
      # Gamma normalization (e.g., gamma=0.5 for square root)
      gamma = .5
      import matplotlib.colors as mcolors
      norm = mcolors.PowerNorm(gamma=gamma)

      fig,axes=plt.subplots(ncols=4,nrows=2,sharex=True,figsize=(10,5),height_ratios=[2,1])
      # Top row: imshow with gamma + colorbar
      fs=12
      img0 = axes[0, 0].imshow(dedicated_place, aspect='auto', norm=norm, cmap='viridis')
      axes[0, 0].set_title(f'Reward cell in epochs 1,2,3\nEpoch {init_ep[i]+1}',fontsize=fs)
      fig.colorbar(img0, ax=axes[0, 0], orientation='vertical')
      axes[0,0].set_ylabel('Reward cell #')

      img0 = axes[0, 1].imshow(dedicated_place_ep3, aspect='auto', norm=norm, cmap='viridis')
      axes[0, 1].set_title(f"Reward cell in epochs 1,2,3\nEpoch {dropped_ep[i]+1}\nSorted by epoch {init_ep[i]+1}",fontsize=fs)
      fig.colorbar(img0, ax=axes[0, 1], orientation='vertical')

      img1 = axes[0, 2].imshow(dropped_place, aspect='auto', norm=norm, cmap='viridis')
      axes[0,2].set_title(f'{lbls[i]}\nEpoch {init_ep[i]+1}',fontsize=fs)
      fig.colorbar(img1, ax=axes[0, 2], orientation='vertical')

      img2 = axes[0, 3].imshow(dropped_place_ep, aspect='auto', norm=norm, cmap='viridis')
      axes[0, 3].set_title(f'{lbls[i]}\nEpoch {dropped_ep[i]+1}\nSorted by epoch {init_ep[i]+1}',fontsize=fs)
      fig.colorbar(img2, ax=axes[0, 3], orientation='vertical')
      vmin = 0
      vmax = 0.6
      if an == 'z16':
         vmax = 1

      # Calculate mean and SEM
      def plot_mean_with_sem(ax, data, color='k', label=''):
         mean = np.nanmean(data, axis=0)
         error = np.nanstd(data, axis=0) / np.sqrt(np.sum(~np.isnan(data), axis=0))
         x = np.arange(data.shape[1])
         ax.plot(x, mean, color=color, label=label)
         ax.fill_between(x, mean - error, mean + error, color=color, alpha=0.3)
         ax.set_ylim([vmin, vmax])

      # Dedicated
      col='cornflowerblue'
      ax = axes[1, 0]
      plot_mean_with_sem(ax, dedicated_place, color=col, label='Dedicated')
      ax.set_title("Mean $\pm$ SEM")
      ax.set_xticks([0,75,150])
      ax.set_xticklabels(['$-\\pi$',0,'$\\pi$'])
      ax.spines[['top', 'right']].set_visible(False)
      ax = axes[1, 1]
      plot_mean_with_sem(ax, dedicated_place_ep3, color=col, label='Dedicated')
      ax.set_xticks([0,75,150])
      # ax.set_xticklabels(['$-\\pi$',0,'$\\pi$'])
      ax.spines[['top', 'right']].set_visible(False)

      # Dropped
      ax = axes[1, 2]
      plot_mean_with_sem(ax, dropped_place, color=col, label='Dropped')
      ax.set_xticks([0,75,150])      
      ax.set_yticklabels([])
      ax.spines[['top', 'right']].set_visible(False)
      # Dropped EP
      ax = axes[1, 3]
      plot_mean_with_sem(ax, dropped_place_ep, color=col, label='Dropped EP')
      ax.set_xticks([0,75,150])      
      ax.set_yticklabels([])
      ax.spines[['top', 'right']].set_visible(False)
      ax.set_xlabel('Reward-centric distance ($\Theta$)')

      fig.suptitle(an)
#%%
import pandas as pd
import seaborn as sns

# Convert to DataFrame
dfres = pd.DataFrame(results)

# Plot cosine similarity summary
plt.figure(figsize=(8, 4))
sns.barplot(data=dfres, x='label', y='cos_sim', hue='type', palette='Set2', errorbar='se',fill=False)
sns.stripplot(data=dfres, x='label', y='cos_sim', hue='type', palette='Set2',dodge=True)
plt.title("Cosine similarity between initial and dropped epoch")
plt.ylabel("Cosine similarity")
plt.xlabel("Reward cell transition type")
plt.ylim(0, 1)
plt.legend(title="Cell type")
plt.tight_layout()

# Plot mean activity drop
plt.figure(figsize=(8, 4))
dfres['mean_drop'] = dfres['mean_init'] - dfres['mean_dropped']
sns.barplot(data=dfres, x='label', y='mean_drop', hue='type', palette='Set1', errorbar='se',legend=False)
sns.stripplot(data=dfres, x='label', y='mean_drop', hue='type', palette='Set1',dodge=True)
plt.title("Drop in mean activity between initial and dropped epoch")
plt.ylabel("Δ Mean Activity (initial-dropped ep)")
plt.xlabel("Reward cell transition type")
plt.tight_layout()

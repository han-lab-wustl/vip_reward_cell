
"""
zahra
pca on tuning curves of reward cells
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
from sklearn.decomposition import PCA

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
# plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.pyr_reward.placecell import make_tuning_curves_radians_by_trialtype, intersect_arrays
from projects.pyr_reward.rewardcell import get_radian_position,create_mask_from_coordinates,pairwise_distances,extract_data_rewcentric,\
    get_radian_position_first_lick_after_rew, get_rewzones
from projects.opto.behavior.behavior import get_success_failure_trials
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_rewardcentric_all.p"
with open(saveddataset, "rb") as fp: #unpickle
        radian_alignment_saved = pickle.load(fp)
#%%
# initialize var
# radian_alignment_saved = {} # overwrite
tcs_rew = []
goal_cells_all = []
bins = 90
goal_window_cm=20
epoch_perm=[]
# cm_window = [10,20,30,40,50,60,70,80] # cm
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
        if animal=='e145':
            ybinned=ybinned[:-1]
            forwardvel=forwardvel[:-1]
            changeRewLoc=changeRewLoc[:-1]
            trialnum=trialnum[:-1]
            rewards=rewards[:-1]
            licks=licks[:-1]
        # set vars
        eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
        rad = get_radian_position_first_lick_after_rew(eps, ybinned, licks, rewards, rewsize,rewlocs,
                        trialnum, track_length) # get radian coordinates
        track_length_rad = track_length*(2*np.pi/track_length)
        bin_size=track_length_rad/bins 
        rz = get_rewzones(rewlocs,1/scalingf)       
        # get average success rate
        rates = []
        for ep in range(len(eps)-1):
                eprng = range(eps[ep],eps[ep+1])
                success, fail, str_trials, ftr_trials, ttr, \
                total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
                rates.append(success/total_trials)
        rate=np.nanmean(np.array(rates))
        
        # added to get anatomical info
        # takes time
        fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
        Fc3 = fall_fc3['Fc3']
        dFF = fall_fc3['dFF']
        Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
        dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]
        skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
        Fc3 = Fc3[:, skew>2] # only keep cells with skew greateer than 2
        if f'{animal}_{day:03d}_index{ii:03d}' in radian_alignment_saved.keys():
            tcs_correct, coms_correct, tcs_fail, coms_fail, \
            com_goal, goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av = radian_alignment_saved[f'{animal}_{day:03d}_index{ii:03d}']            
        else:# remake tuning curves relative to reward        
            # 9/19/24
            # find correct trials within each epoch!!!!
            tcs_correct, coms_correct, tcs_fail, coms_fail = make_tuning_curves_radians_by_trialtype(eps,rewlocs,ybinned,rad,Fc3,trialnum,
            rewards,forwardvel,rewsize,bin_size)          
        # fall_stat = scipy.io.loadmat(params_pth, variable_names=['stat','ops'])
        # ops = fall_stat['ops']
        # stat = fall_stat['stat']
        # meanimg=np.squeeze(ops)[()]['meanImg']
        # s2p_iind = np.arange(stat.shape[1])
        # s2p_iind_filter = s2p_iind[(fall['iscell'][:,0]).astype(bool)]
        # s2p_iind_filter = s2p_iind_filter[skew>2]
        goal_window = goal_window_cm*(2*np.pi/track_length) # cm converted to rad
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

        print(f'Reward-centric cells total: {[len(xx) for xx in com_goal]}')
        epoch_perm.append([perm,rz_perm]) 
        # get goal cells across all epochs   
        if len(com_goal)>0:
            goal_cells = intersect_arrays(*com_goal); 
        else:
            goal_cells=[]
        
        # tcs of goal cells
        tcs = []
        for p,cg in enumerate(com_goal):
                pm = np.array([perm[p][0],perm[p][1]])
                tcs_ = np.array([tcs_correct[pmm,cg,:] for pmm in pm])
                tcs.append(tcs_)
        # cells that are considered rew cells across 2 epochs
        tcs_rew.append(np.hstack(tcs))
        # get indices
        goal_cells_all.append([com_goal, goal_cells])
        # mean of epochs
        # tcs = np.nanmean(tcs, axis=0)
        # pca = PCA(n_components=4)
        # tuning_pca = pca.fit_transform(tcs) 
        # from sklearn.cluster import KMeans
        # k = 4  # Try a few values!
        # kmeans = KMeans(n_clusters=k, random_state=0)
        # cluster_labels = kmeans.fit_predict(tuning_pca)
        # plt.figure(figsize=(6, 5))
        # for i in range(k):
        #     plt.scatter(tuning_pca[cluster_labels == i, 0], 
        #                 tuning_pca[cluster_labels == i, 1], 
        #                 label=f'Cluster {i}', alpha=0.6)
        # plt.xlabel("PC1")
        # plt.ylabel("PC2")
        # plt.title("Cell Clusters via Tuning Curve PCA")
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()


#%%
# collect tcs 
plt.rc('font', size=16) 
# just 1st epoch
ep=0
tcs_all = np.vstack([xx[ep] for xx in tcs_rew if len(xx)>ep])
tuning_curves_clean = tcs_all[~np.isnan(tcs_all).any(axis=1)]
# get num epochs per session
num_epochs = [np.nanmax(np.ravel(xx[0]))+1 for xx in epoch_perm]
# TODO
iind_2ep_per_session = [np.concatenate(xx[0]) for xx in goal_cells_all]
iind_2ep = np.hstack(iind_2ep_per_session)
iind_2ep = iind_2ep[~np.isnan(tcs_all).any(axis=1)]
iind_allep_per_session = [xx[1] for xx in goal_cells_all]
pca = PCA(n_components=7)
tuning_pca = pca.fit_transform(tuning_curves_clean) 

#%%
inertia = []
K = range(1, 50)
from sklearn.cluster import KMeans

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
    kmeans.fit(tuning_pca)
    inertia.append(kmeans.inertia_)

from kneed import KneeLocator

knee = KneeLocator(K, inertia, curve="convex", direction="decreasing")

# Plotting
plt.figure(figsize=(6,4))
plt.plot(K, inertia, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title(f'Elbow Method for Optimal k\n\
        Optimal number of clusters: {knee.knee}')
plt.tight_layout()
plt.show()

#%%
from sklearn.cluster import KMeans
k =13  # Try a few values!
kmeans = KMeans(n_clusters=k, random_state=0)
cluster_labels = kmeans.fit_predict(tuning_pca)
fig, ax = plt.subplots(figsize=(6, 5))
for i in range(k):
        ax.scatter(tuning_pca[cluster_labels == i, 0], 
                tuning_pca[cluster_labels == i, 1], 
                label=f'Cluster {i}', alpha=0.2)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
explained_var = pca.explained_variance_ratio_
print("Explained variance:", explained_var)
ax.set_title(f"Cell Clusters via Tuning Curve PCA\n\
        Explained variance in first 2 PCs = {round(sum(explained_var[:2]),1)*100}%")
ax.legend()
# ax.spines[['top', 'right']].set_visible(False)    
# Optional: how much variance is explained
# average tuning per cluster?
dim = int(np.ceil(np.sqrt(k)))
fig,axes = plt.subplots(ncols=dim, nrows=dim, sharex=True,
                        figsize=(dim*3, dim*3))
axes = axes.flatten()

for i in range(k):
        ax = axes[i]        
        m = np.nanmean(tuning_curves_clean[cluster_labels==i],axis=0)
        tc = tuning_curves_clean[cluster_labels==i]
        ax.plot(m)
        
        ax.fill_between(
        range(0, bins),
        m - scipy.stats.sem(tc, axis=0, nan_policy='omit'),
        m + scipy.stats.sem(tc, axis=0, nan_policy='omit'),
        alpha=0.5,
        )                    

        if i==k-1: ax.set_xlabel('Reward-relative distance ($\Theta$)')
        if i==0: ax.set_ylabel('Average $\Delta F/F$')
        ax.set_title(f'{tc.shape[0]} cells\ncluster {i}')
        ax.axvline(int(bins/2), color='k', linestyle='--')
        ax.spines[['top', 'right']].set_visible(False)    
fig.tight_layout()

# heatmap
fig,axes = plt.subplots(ncols=dim, nrows=dim, sharex=True,
                        figsize=(dim*3, dim*3))
axes = axes.flatten()
vmin = np.nanmin(tuning_curves_clean**0.7)
vmax = np.nanmax(tuning_curves_clean**0.7)
im = None  # We'll store one of the imshow handles for the colorbar

for i in range(k):
        ax = axes[i]        
        m = np.nanmean(tuning_curves_clean[cluster_labels==i],axis=0)
        tc = tuning_curves_clean[cluster_labels==i]
        im = ax.imshow(tc**0.7, vmin=vmin, vmax=vmax, aspect='auto', cmap='viridis')

        if i==k-1: ax.set_xlabel('Reward-relative distance ($\Theta$)')
        if i==0: ax.set_ylabel('Cells')
        ax.set_title(f'{tc.shape[0]} cells\ncluster {i}')
        ax.axvline(int(bins/2), color='w', linestyle='--')
        ax.spines[['top', 'right']].set_visible(False)    
fig.tight_layout()
        
fig.tight_layout(rect=[0, 0, 0.95, 1])  # leave space for colorbar
cbar_ax = fig.add_axes([0.96, 0.15, 0.015, 0.7])  # [left, bottom, width, height]
fig.colorbar(im, cax=cbar_ax, label='$(\Delta F/F)^{0.7}$')

#%%
# map back onto com ind
valid_mask = ~np.isnan(tcs_all).any(axis=1)
iind_2ep = np.hstack([np.array(x).flatten() for x in iind_2ep_per_session])
iind_2ep_clean = iind_2ep[valid_mask]
# get number of cells per session (after NaN filtering)
session_lengths = [len(np.array(xx).flatten()) for xx in iind_2ep_per_session]
session_lengths_clean = [np.sum(valid_mask[start:start+length])
                         for start, length in zip(np.cumsum([0]+session_lengths[:-1]), session_lengths)]

# split by session
cluster_labels_by_session = np.split(cluster_labels, np.cumsum(session_lengths_clean)[:-1])
iind_2ep_clean_by_session = np.split(iind_2ep_clean, np.cumsum(session_lengths_clean)[:-1])
allep_cluster = [cluster_labels_by_session[ii][np.where(np.isin(iind_2ep_clean_by_session[ii],gc))[0]] for ii, gc in enumerate(iind_allep_per_session)]
# count number of cells in each cluster that are dedicated rew cells
# Combine all values
all_values = np.concatenate(allep_cluster)
#%%
# Unique values and counts
counts = [np.unique(xx, return_counts=True) for xx in allep_cluster]
total_cells = [len(xx) for xx in allep_cluster]
percents = [xx[1]/total_cells[ii] for ii,xx in enumerate(counts)]
clusters = [xx[0] for ii,xx in enumerate(counts)]

df = pd.DataFrame()
df['percent_dedicated_cells'] = np.concatenate(percents)
df['percent_dedicated_cells']=df['percent_dedicated_cells']*100
df['cluster'] = np.concatenate(clusters)
df['num_epochs'] = np.concatenate([[xx]*len(percents[ii]) for ii,xx in enumerate(num_epochs)])
df=df[df.num_epochs<5]
# df=df[df.cluster>0]
fig,ax = plt.subplots(figsize=(15,4))
sns.stripplot(x='cluster',y='percent_dedicated_cells',hue='num_epochs',data=df,dodge=True,
        palette='colorblind')
sns.barplot(x='cluster',y='percent_dedicated_cells',hue='num_epochs',data=df,
        palette='colorblind',fill=False)
ax.set_ylim([0,30])
#%%
# Find clusters where max(num_epochs) == 2
eps = [2,3,4]
for ep in eps:
        cluster_max_epochs = df.groupby('cluster')['num_epochs'].max()
        clusters_max_2 = cluster_max_epochs[cluster_max_epochs == ep].index.tolist()

        # Set up plot grid
        n = len(clusters_max_2)
        ncols = int(np.ceil(np.sqrt(n)))
        fig, axes = plt.subplots(nrows=ncols, ncols=ncols, figsize=(ncols*3, ncols*3), squeeze=False)
        axes = axes.flatten()

        # Plot tuning curves
        for i, clust in enumerate(clusters_max_2):
                ax = axes[i]
                mask = cluster_labels == clust
                tc = tuning_curves_clean[mask]
                m = np.nanmean(tc, axis=0)
                sem = scipy.stats.sem(tc, axis=0, nan_policy='omit')
                
                ax.plot(m, label=f'Cluster {clust}')
                ax.fill_between(range(tc.shape[1]), m - sem, m + sem, alpha=0.3)
                ax.axvline(tc.shape[1] // 2, linestyle='--', color='k')
                ax.set_title(f'Cluster {clust}\n(n={tc.shape[0]})')
                ax.set_xlabel('Reward-relative bin')
                ax.set_ylabel('Avg ΔF/F')
                ax.spines[['top', 'right']].set_visible(False)

        # Hide unused subplots
        for j in range(i+1, len(axes)):
                axes[j].axis('off')
        fig.suptitle(f'Max epoch: {ep}')
        fig.tight_layout()
        plt.show()
        
        fig, axes = plt.subplots(nrows=ncols, ncols=ncols, figsize=(ncols*3, ncols*3), squeeze=False)
        axes = axes.flatten()

        vmin = np.nanmin(tuning_curves_clean**0.7)
        vmax = np.nanmax(tuning_curves_clean**0.7)

        for i, clust in enumerate(clusters_max_2):
                ax = axes[i]
                mask = cluster_labels == clust
                tc = tuning_curves_clean[mask]
                im = ax.imshow(tc**0.5, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
                ax.axvline(tc.shape[1] // 2, linestyle='--', color='w')
                ax.set_title(f'Cluster {clust}\n(n={tc.shape[0]})')
                ax.set_xlabel('Reward-relative bin')
                ax.set_ylabel('Cells')

        for j in range(i+1, len(axes)):
                axes[j].axis('off')

        fig.tight_layout()
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='$(\Delta F/F)^{0.7}$')
        plt.show()



#%%
# eg cluster
i = 0
fig,ax = plt.subplots(figsize=(6,20))
x1,x2=1000,3000
m = .3
tc = tuning_curves_clean[cluster_labels==i][x1:x2]
tc = tuning_curves_clean[cluster_labels==i]
peak_bins = np.argmax(tc, axis=1)
sort_idx = np.argsort(peak_bins)
ax.imshow(tc[sort_idx]**m,aspect='auto')       
# ax.imshow(tc**m,aspect='auto')       

if i==k-1: ax.set_xlabel('Reward-relative distance ($\Theta$)')
if i==0: ax.set_ylabel('Average $\Delta F/F$')
ax.set_title(f'{tc.shape[0]} cells\ncluster {i}')
ax.axvline(int(bins/2), color='w', linestyle='--')
ax.spines[['top', 'right']].set_visible(False)    

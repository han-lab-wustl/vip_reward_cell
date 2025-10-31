
"""
zahra
get goal cells
see if they maintain their coms in probe 1,2,3
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
# plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from placecell import make_tuning_curves_probes,make_tuning_curves_radians_by_trialtype, intersect_arrays
from rewardcell import get_radian_position,create_mask_from_coordinates,pairwise_distances
from projects.opto.behavior.behavior import get_success_failure_trials
from projects.memory.dopamine import get_rewzones
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
savepth = os.path.join(savedst, 'reward_relative_probe.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_reward_cell_bytrialtype_nopto_20241108.p"
with open(saveddataset, "rb") as fp: #unpickle
        radian_alignment_saved = pickle.load(fp)
#%%
# initialize var
cm_window = 20
data_dct= {}
# cm_window = [10,20,30,40,50,60,70,80] # cm
# iterate through all animals
for ii in range(0,len(conddf)):
    day = conddf.days.values[ii]
    animal = conddf.animals.values[ii]
    if (animal!='e217') & (conddf.optoep.values[ii]<2):
        if animal=='e145' or animal=='e139': pln=2 
        else: pln=0
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
        print(params_pth)
        fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
                'pyr_tc_s2p_cellind', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
                'stat'])
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
        if animal=='e145':
                ybinned=ybinned[:-1]
                forwardvel=forwardvel[:-1]
                changeRewLoc=changeRewLoc[:-1]
                trialnum=trialnum[:-1]
                rewards=rewards[:-1]
        # set vars
        eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
        lasttr=8 # last trials
        bins=90
        rad = get_radian_position(eps,ybinned,rewlocs,track_length,rewsize) # get radian coordinates
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
        # get each probe tc
        tcs_probe1, coms_probe1 = make_tuning_curves_probes(eps,rewlocs,ybinned,rad,Fc3,trialnum,
                rewards,forwardvel,rewsize,bin_size)  
        tcs_probe2, coms_probe2 = make_tuning_curves_probes(eps,rewlocs,ybinned,rad,Fc3,trialnum,
                rewards,forwardvel,rewsize,bin_size,probe=[1])          
        tcs_probe3, coms_probe3 = make_tuning_curves_probes(eps,rewlocs,ybinned,rad,Fc3,trialnum,
                rewards,forwardvel,rewsize,bin_size,probe=[2])          

        fall_stat = scipy.io.loadmat(params_pth, variable_names=['stat','ops'])
        ops = fall_stat['ops']
        stat = fall_stat['stat']
        meanimg=np.squeeze(ops)[()]['meanImg']
        s2p_iind = np.arange(stat.shape[1])
        s2p_iind_filter = s2p_iind[(fall['iscell'][:,0]).astype(bool)]
        s2p_iind_filter = s2p_iind_filter[skew>2]
        goal_window = cm_window*(2*np.pi/track_length) # cm converted to rad
        # change to relative value 
        coms_rewrel = np.array([com-np.pi for com in coms_correct])
        perm = list(combinations(range(len(coms_correct)), 2)) 
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
        # get goal cells across all epochs        
        goal_cells = intersect_arrays(*com_goal)   
        colors = ['k', 'slategray', 'darkcyan', 'darkgoldenrod', 'orchid']
        binsz = 3 # for 270 track
        if len(goal_cells)>0:
                rows = 4; cols=1
                for i,gc in enumerate(goal_cells):            
                        fig, axes = plt.subplots(cols, rows, figsize=(10,3),sharex=True)
                        axes = axes.flatten()
                        for ep in range(len(coms_correct)):
                                ax = axes[0]
                                ax.plot(tcs_correct[ep,gc,:], label=f'rewloc {rewlocs[ep]}', 
                                color=colors[ep],linewidth=2)
                                ax.axvline((coms_correct[ep,gc])/(2*np.pi/track_length)/binsz,
                                        color=colors[ep])
                                ax=axes[1]
                                try:
                                        ax.plot(tcs_probe1[ep+1,gc,:],
                                        color=colors[ep],linewidth=2)
                                        ax.axvline((coms_probe1[ep+1,gc])/(2*np.pi/track_length)/binsz,
                                        color=colors[ep])
                                        ax.set_title('Probe 1')
                                except Exception as e:
                                        print(e)
                                ax=axes[2]
                                try:
                                        ax.plot(tcs_probe2[ep+1,gc,:], 
                                        color=colors[ep], linewidth=2)
                                        ax.axvline((coms_probe2[ep+1,gc])/(2*np.pi/track_length)/binsz,
                                        color=colors[ep])
                                        ax.set_title('Probe 2')
                                except Exception as e:
                                        print(e)
                                ax=axes[3]
                                try:
                                        ax.plot(tcs_probe3[ep+1,gc,:], 
                                        color=colors[ep], linewidth=2)
                                        ax.axvline((coms_probe3[ep+1,gc])/(2*np.pi/track_length)/binsz,
                                        color=colors[ep])
                                        ax.set_title('Probe 3')
                                except Exception as e:
                                        print(e)                        
                        fig.suptitle(f'cell # {gc}')
                        ax.set_xticks(np.arange(0,bins+1,20))
                        ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi, np.pi/2.25),1))
                        ax.set_xlabel('Radian position (centered start rew loc)')
                        ax.set_ylabel('$\Delta F/F$')
                        pdf.savefig(fig)
                        plt.close('all')
        data_dct[f'{animal}_{day}'] = [coms_correct, coms_probe1, coms_probe2, coms_probe3, rates, rewlocs]

pdf.close()
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_probes.p"
# save pickle of dcts
with open(saveddataset, "wb") as fp:   #Pickling
        pickle.dump(data_dct, fp) 
#%%
plt.rc('font', size=16)          # controls default text sizes
# plot difference of com from ep to probe 1,2,3

diff = data_dct['e218_20'][0]-data_dct['e218_20'][1]
diff2 = data_dct['e218_20'][0]-data_dct['e218_20'][2]
diff3 = data_dct['e218_20'][0]-data_dct['e218_20'][3]

for ii,df in enumerate(diff):  
        plt.figure()
        plt.scatter(np.arange(len(data_dct['e218_20'][0][ii])), 
                data_dct['e218_20'][0][ii]-np.pi)      
        plt.scatter(np.arange(len(df)), df)
#%%
# convert back to cm
df=pd.DataFrame()
df['com_dff'] = np.concatenate([diff[1,:],diff2[1,:],diff3[1,:]])/(2*np.pi/track_length)
df['probe_num'] = np.concatenate([[1]*len(diff[1,:]),[2]*len(diff2[1,:]),
                        [3]*len(diff3[1,:])])
df['cellid'] = np.concatenate([np.arange(len(diff[1,:]))]*3)

fig,ax = plt.subplots(figsize=(3,5))
sns.barplot(x='probe_num',y='com_dff',data=df)
#%%
# count # nan
df=pd.DataFrame()
isnans = np.array([np.isnan(diff[1,:]),np.isnan(diff2[1,:]),np.isnan(diff3[1,:])])
df['com_nan'] = data_dct['e218_20'][0][np.sum(isnans,axis=1)>1]
df['probe_num'] = np.concatenate([[1]*len(diff[1,:]),[2]*len(diff2[1,:]),
                        [3]*len(diff3[1,:])])
df['cellid'] = np.concatenate([np.arange(len(diff[1,:]))]*3)

fig,ax = plt.subplots(figsize=(3,5))
sns.barplot(x='probe_num',y='com_dff',data=df)




"""
zahra
march 2025
find proportion of cells that are considered reward cells for 
multiple epochs and days
1) get day 1 reward cells
2) get the next 2 days of reward cells
3) get proportion of cells that are reward cells across all the epochs
3/10/25
split up day 1 into epochs
use or function (if a cell is a reward cell for 3/5 epochs for eg., keep it)
"""
#%%

import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd, matplotlib.backends.backend_pdf
import pickle, seaborn as sns, random, math, os, matplotlib as mpl
from collections import Counter
from itertools import combinations, chain
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
# plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.pyr_reward.placecell import make_tuning_curves_radians_by_trialtype, intersect_arrays
from projects.pyr_reward.rewardcell import get_days_from_cellreg_log_file, find_log_file, get_radian_position, \
    get_tracked_lut, get_tracking_vars, get_shuffled_goal_cell_indices, get_reward_cells_that_are_tracked
from projects.opto.behavior.behavior import get_success_failure_trials
# import condition df
animals = ['e218','e216','e217','e201','e186',
        'e190', 'e145', 'z8', 'z9']
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
radian_tuning_dct = r'Z:\\saved_datasets\\radian_tuning_curves_rewardcentric_all.p'
with open(radian_tuning_dct, "rb") as fp: #unpickle
    radian_alignment_saved = pickle.load(fp)
celltrackpth = r'Y:\analysis\celltrack'
# cell tracked days
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)

tracked_rew_cell_inds_all = {}
trackeddct = {}
#%%
per_day_goal_cells_all = []
# defined vars
maxep = 5
shuffles = 1000
# redo across days analysis but init array per animal
for animal in animals:
    # all rec days
    dys = conddf.loc[conddf.animals==animal, 'days'].values
    # index compared to org df
    dds = list(conddf[conddf.animals==animal].index)
    # init 
    iind_goal_cells_all_per_day=[]
    
    for ii, day in enumerate(dys[:4]): # iterate per day
        if animal!='e217' and conddf.optoep.values[dds[ii]]==-1:
            if animal=='e145': pln=2
            else: pln=0
            # get lut
            tracked_lut, days= get_tracked_lut(celltrackpth,animal,pln)
            if ii==0:
                # init with min 4 epochs
                # ep x cells x days
                # instead of filling w/ coms, fill w/ binary
                tracked = np.zeros((tracked_lut.shape[0]))
                tracked_shuf =np.zeros((shuffles, tracked_lut.shape[0]))
            # get vars
            params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
            dFF, suite2pind_remain, VR, scalingf, rewsize, ybinned, forwardvel, changeRewLoc,\
                rewards, eps, rewlocs, track_length = get_tracking_vars(params_pth)
            goal_window = 20*(2*np.pi/track_length) # cm converted to rad, consistent with quantified window sweep
            # find key
            k = [k for k,v in radian_alignment_saved.items() if f'{animal}_{day:03d}' in k][0]
            _, __, ___, ____,tcs_correct, coms_correct, tcs_fail_dt, coms_fail_dt, goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av = radian_alignment_saved[k]         
            perm = list(combinations(range(len(coms_correct)), 2))       
            goal_window = 20*(2*np.pi/track_length) # cm converted to rad
            coms_rewrel = np.array([com-np.pi for com in coms_correct])
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
            com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
            com_goal_all=np.unique(np.concatenate(com_goal))

            ########### shuffle
            shufs = [list(range(coms_correct[ii].shape[0])) for ii in range(1, len(coms_correct))]
            [random.shuffle(shuf) for shuf in shufs]
            # first com is as ep 1, others are shuffled cell identities
            com_shufs = np.zeros_like(coms_correct); com_shufs[0,:] = coms_correct[0]
            com_shufs[1:1+len(shufs),:] = [coms_correct[ii][np.array(shufs)[ii-1]] for ii in range(1, 1+len(shufs))]
            # OR shuffle cell identities
            # relative to reward
            coms_rewrel = np.array([com-np.pi for ii, com in enumerate(com_shufs)])             
            # account for cells that move to the end/front
            # Find COMs near pi and shift to -pi
            epsilon=.7
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
            # cont.
            coms_rewrel[:,com_loop_w_in_window]=abs(coms_rewrel[:,com_loop_w_in_window])
            com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
            # get goal cells across all epochs
            com_goal_shuf = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
            com_goal_shuf=[com for com in com_goal_shuf if len(com)>0]
            goal_cells_shuf_p_per_comparison = [len(xx)/len(coms_correct[0]) for xx in com_goal_shuf]
            # get goal cells across all epochs   
            if len(com_goal_shuf)>0:
                goal_cells_shuf = intersect_arrays(*com_goal_shuf); 
            else:
                goal_cells_shuf=[]
            ########### shuffl   
            assert suite2pind_remain.shape[0]==tcs_correct.shape[1]
            # indices per epo
            iind_goal_cells_all=[suite2pind_remain[xx] for xx in com_goal]
            iind_goal_cells_all_per_day.append(iind_goal_cells_all)
    # per day cells
    per_day_goal_cells = [intersect_arrays(*xx) for xx in iind_goal_cells_all_per_day]
    # split per ep
    # test per day + epoch 1 from next day, n so on...
    per_day_nextday_ep1=[]
    for ii,xx in enumerate(iind_goal_cells_all_per_day[:-1]):
        per_day_nextday_ep1.append(intersect_arrays(*list(chain.from_iterable([xx, np.array([iind_goal_cells_all_per_day[ii+1][0]])]))))
    per_day_nextday_ep2=[]; days_ep2=[]
    for ii,xx in enumerate(iind_goal_cells_all_per_day[:-1]):
        try:
            per_day_nextday_ep2.append(intersect_arrays(*list(chain.from_iterable([xx, np.array([iind_goal_cells_all_per_day[ii+1][0],
                iind_goal_cells_all_per_day[ii+1][1]])]))))
            days_ep2.append(ii)
        except Exception as e:
            print(e)
    per_day_nextday_ep3=[]; days_ep3=[]
    for ii,xx in enumerate(iind_goal_cells_all_per_day[:-1]):
        try:
            per_day_nextday_ep3.append(intersect_arrays(*list(chain.from_iterable([xx, np.array([iind_goal_cells_all_per_day[ii+1][0],
                iind_goal_cells_all_per_day[ii+1][1], iind_goal_cells_all_per_day[ii+1][2]])]))))
            print(len(list(chain.from_iterable([xx, np.array([iind_goal_cells_all_per_day[ii+1][0],
                iind_goal_cells_all_per_day[ii+1][1], iind_goal_cells_all_per_day[ii+1][2]])]))))
            days_ep3.append(ii)
        except Exception as e:
            print(e)
    per_day_next2day_ep1=[]; days_2day_ep1=[]
    for ii,xx in enumerate(iind_goal_cells_all_per_day[:-2]):
        try:
            if len(iind_goal_cells_all_per_day[ii+1])>2:
                per_day_next2day_ep1.append(intersect_arrays(*list(chain.from_iterable([xx, np.array([iind_goal_cells_all_per_day[ii+1][0],
                    iind_goal_cells_all_per_day[ii+1][1], iind_goal_cells_all_per_day[ii+1][2], iind_goal_cells_all_per_day[ii+2][0]])]))))            
            elif len(iind_goal_cells_all_per_day[ii+1])>1:
                per_day_next2day_ep1.append(intersect_arrays(*list(chain.from_iterable([xx, np.array([iind_goal_cells_all_per_day[ii+1][0],
                iind_goal_cells_all_per_day[ii+1][1], iind_goal_cells_all_per_day[ii+2][0]])]))))            
            else:
                per_day_next2day_ep1.append(intersect_arrays(*list(chain.from_iterable([xx, np.array([iind_goal_cells_all_per_day[ii+1][0],
                iind_goal_cells_all_per_day[ii+2][0]])]))))            

            days_2day_ep1.append(ii)
        except Exception as e:
            print(e)
    per_day_next2day_ep2=[]; days_2day_ep2=[]
    for ii,xx in enumerate(iind_goal_cells_all_per_day[:-2]):
        try:
            if len(iind_goal_cells_all_per_day[ii+1])>2:
                per_day_next2day_ep2.append(intersect_arrays(*list(chain.from_iterable([xx, np.array([iind_goal_cells_all_per_day[ii+1][0],
                    iind_goal_cells_all_per_day[ii+1][1], iind_goal_cells_all_per_day[ii+1][2], iind_goal_cells_all_per_day[ii+2][0],
                    iind_goal_cells_all_per_day[ii+2][1]])]))))            
            elif len(iind_goal_cells_all_per_day[ii+1])>1:
                per_day_next2day_ep2.append(intersect_arrays(*list(chain.from_iterable([xx, np.array([iind_goal_cells_all_per_day[ii+1][0],
                iind_goal_cells_all_per_day[ii+1][1], iind_goal_cells_all_per_day[ii+2][0],
                iind_goal_cells_all_per_day[ii+2][1]])]))))            
            else:
                per_day_next2day_ep2.append(intersect_arrays(*list(chain.from_iterable([xx, np.array([iind_goal_cells_all_per_day[ii+1][0],
                iind_goal_cells_all_per_day[ii+2][0],
                iind_goal_cells_all_per_day[ii+2][1]])]))))            

            days_2day_ep2.append(ii)
        except Exception as e:
            print(e)
            
    per_day_next2day_ep3=[]; days_2day_ep3=[]
    for ii,xx in enumerate(iind_goal_cells_all_per_day[:-2]):
        try:
            if len(iind_goal_cells_all_per_day[ii+1])>2:
                per_day_next2day_ep3.append(intersect_arrays(*list(chain.from_iterable([xx, np.array([iind_goal_cells_all_per_day[ii+1][0],
                    iind_goal_cells_all_per_day[ii+1][1], iind_goal_cells_all_per_day[ii+1][2], iind_goal_cells_all_per_day[ii+2][0],
                    iind_goal_cells_all_per_day[ii+2][1],
                    iind_goal_cells_all_per_day[ii+2][2]])]))))            
            elif len(iind_goal_cells_all_per_day[ii+1])>1:
                per_day_next2day_ep3.append(intersect_arrays(*list(chain.from_iterable([xx, np.array([iind_goal_cells_all_per_day[ii+1][0],
                iind_goal_cells_all_per_day[ii+1][1], iind_goal_cells_all_per_day[ii+2][0],
                iind_goal_cells_all_per_day[ii+2][1],
                iind_goal_cells_all_per_day[ii+2][2]])]))))            
            else:
                per_day_next2day_ep3.append(intersect_arrays(*list(chain.from_iterable([xx, np.array([iind_goal_cells_all_per_day[ii+1][0],
                iind_goal_cells_all_per_day[ii+2][0],
                iind_goal_cells_all_per_day[ii+2][1],
                iind_goal_cells_all_per_day[ii+2][2]])]))))            

            days_2day_ep3.append(ii)
        except Exception as e:
            print(e)

    # get number of cells in each comparison
    per_day_goal_cells_num = [[len(xx) for xx in yy] for yy in iind_goal_cells_all_per_day] # incl all epochs
    per_day_nextday_ep1_num = [len(xx) for xx in per_day_nextday_ep1]
    per_day_nextday_ep2_num = [len(xx) for xx in per_day_nextday_ep2]
    per_day_nextday_ep3_num = [len(xx) for xx in per_day_nextday_ep3]
    per_day_next2day_ep1_num = [len(xx) for xx in per_day_next2day_ep1]
    per_day_next2day_ep2_num = [len(xx) for xx in per_day_next2day_ep2]
    per_day_next2day_ep3_num = [len(xx) for xx in per_day_next2day_ep3]
    
    per_day_goal_cells_all.append([per_day_goal_cells_num,per_day_nextday_ep1_num,
                per_day_nextday_ep2_num,per_day_nextday_ep3_num,
                per_day_next2day_ep1_num,per_day_next2day_ep2_num,per_day_next2day_ep3_num])
    
#%%

df=pd.DataFrame()
lut = ['1_day', '1_day_1_epoch', '1_day_2_epochs',
    '1_day_3_epochs', '2_days_1_epoch', '2_days_2_epochs',
    '2_days_3_epochs']
biglst = []; bigann =[]; biganimal=[]
for ii in range(len(per_day_goal_cells_all[0])):
    biglst.append(np.concatenate([xx[ii] for xx in per_day_goal_cells_all]))
    bigann.append(np.concatenate([[lut[ii]]*len(xx[ii]) for xx in per_day_goal_cells_all]))
    biganimal.append(np.concatenate([[animals[jj]]*len(xx[ii]) for jj,xx in enumerate(per_day_goal_cells_all)]))

df['reward_cell_count']=np.concatenate(biglst)
df['epoch_type']=np.concatenate(bigann)
df['animal']=np.concatenate(biganimal)

#%%
plt.rc('font', size=16) 
fig, ax = plt.subplots(figsize=(12,9))
sns.stripplot(x='epoch_type',y='reward_cell_count',hue='animal',data=df, dodge=True)
sns.barplot(x='epoch_type',y='reward_cell_count',hue='animal',data=df)
ax.tick_params(axis='x', rotation=45)
#%%
s=12
sumdf = df.groupby(['animal','epoch_type']).mean(numeric_only=True)
sumdf = sumdf.sort_index(axis=1)
sumdf=sumdf.reset_index()
fig, ax = plt.subplots(figsize=(6,5))
sns.stripplot(x='epoch_type',y='reward_cell_count',data=sumdf, dodge=True,color='k',
        s=s)
sns.barplot(x='epoch_type',y='reward_cell_count',data=sumdf, fill=False,color='k')
ax.tick_params(axis='x', rotation=45)
# make lines
ans = sumdf.animal.unique()
for i in range(len(ans)):
    ax = sns.lineplot(x='epoch_type', y='reward_cell_count', 
    data=sumdf[sumdf.animal==ans[i]],
    errorbar=None, color='dimgray', linewidth=2)
ax.spines[['top','right']].set_visible(False)

# only show ectra epochs
#%%
s=12
sumdf=sumdf[sumdf.epoch_type!='1_day']
fig, ax = plt.subplots(figsize=(6,5))
sns.stripplot(x='epoch_type',y='reward_cell_count',data=sumdf, dodge=True,color='k',
        s=s)
sns.barplot(x='epoch_type',y='reward_cell_count',data=sumdf, fill=False,color='k')
ax.tick_params(axis='x', rotation=45)
# make lines
ans = sumdf.animal.unique()
for i in range(len(ans)):
    ax = sns.lineplot(x='epoch_type', y='reward_cell_count', 
    data=sumdf[sumdf.animal==ans[i]],
    errorbar=None, color='dimgray', linewidth=2)
ax.spines[['top','right']].set_visible(False)

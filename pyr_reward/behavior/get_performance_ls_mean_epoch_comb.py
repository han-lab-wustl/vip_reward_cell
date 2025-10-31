
"""
zahra
behavior metrics
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
from projects.opto.behavior.behavior import get_success_failure_trials, get_lick_selectivity
from projects.pyr_reward.rewardcell import get_radian_position_first_lick_after_rew, get_rewzones, make_tuning_curves_radians_by_trialtype
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper\panels_main_figures'
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_rewardcentric_all.p"
with open(saveddataset, "rb") as fp: #unpickle
        radian_alignment_saved = pickle.load(fp)
#%%
# initialize var

goal_window_cm = 20
lasttr=8 # last trials
bins=90
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_rewardcentric_all.p"
rates_perm_all=[]
ls_perm_all=[]
perms=[]
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
        lasttr=8 # last trials
        bins=90
        rad = get_radian_position_first_lick_after_rew(eps, ybinned, licks, rewards, rewsize,rewlocs,
                        trialnum, track_length) # get radian coordinates
        track_length_rad = track_length*(2*np.pi/track_length)
        bin_size=track_length_rad/bins 
        rz = get_rewzones(rewlocs,1/scalingf)       
        # get average success rate
        rates = []; ls = []
        for ep in range(len(eps)-1):
                eprng = range(eps[ep],eps[ep+1])
                success, fail, str_trials, ftr_trials, ttr, \
                total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
                lick_selectivity_per_trial = get_lick_selectivity(ybinned[eprng], trialnum[eprng], 
                licks[eprng], rewlocs[ep], rewsize,
                fails_only = False)
                rates.append(success/total_trials)
                # last 8 trials
                ls.append(np.nanmean(lick_selectivity_per_trial[:-8]))
        rate=np.nanmean(np.array(rates))
        if f'{animal}_{day:03d}_index{ii:03d}' in radian_alignment_saved.keys():
                tcs_correct, coms_correct, tcs_fail, coms_fail, \
                com_goal, goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av,_,__,___ = radian_alignment_saved[f'{animal}_{day:03d}_index{ii:03d}']            
        else:# remake tuning curves relative to reward        
        # 9/19/24
        # find correct trials within each epoch!!!!
                # takes time
                fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
                Fc3 = fall_fc3['Fc3']
                dFF = fall_fc3['dFF']
                Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
                dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]
                skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
                Fc3 = Fc3[:, skew>2] # only keep cells with skew greateer than 2

                tcs_correct, coms_correct, tcs_fail, coms_fail = make_tuning_curves_radians_by_trialtype(eps,rewlocs,ybinned,rad,Fc3,trialnum,rewards,forwardvel,rewsize,bin_size)          
        # change to relative value 
        coms_rewrel = np.array([com-np.pi for com in coms_correct])
        perm = list(combinations(range(len(coms_correct)), 2)) 
        # get average per combination
        rates_perm = [np.nanmean([rates[p[0]],rates[p[1]]]) for p in perm]
        ls_perm = [np.nanmean([ls[p[0]],ls[p[1]]]) for p in perm]
        rates_perm_all.append(rates_perm)
        ls_perm_all.append(ls_perm)
        perms.append(perm)
#%%
# plot goal cells across epochs
plt.rc('font', size=20)
df = conddf.copy()
df = df[((df.animals!='e217')) & (df.optoep<2)]
df['num_epochs'] = [max(max(xx))+1 for xx in perms]
df['rates'] = [np.nanmean(xx)*100 for xx in rates_perm_all]
df['lick_selectivity'] = [np.nanmean(xx) for xx in ls_perm_all]
# add epoch combinations
df2=df.copy()
df2['rates'] = [np.nanmean(xx)*100 for xx in rates_perm_all]
df['lick_selectivity'] = [np.nanmean(xx) for xx in ls_perm_all]
df2['num_epochs'] = [2]*len(df2)
df=pd.concat([df,df2])
# df=df[df.animals!='e189']
# number of epochs vs. rates    
fig,ax = plt.subplots(figsize=(3,4))
df_plt=df[df.num_epochs<5]
# av across mice
df_plt = df_plt.groupby(['animals','num_epochs']).mean(numeric_only=True)
sns.stripplot(x='num_epochs', y='rates',color='k',
        data=df_plt,alpha=0.7,
        s=10)
sns.barplot(x='num_epochs', y='rates',
        data=df_plt,
        fill=False,ax=ax, color='k', errorbar='se')
ax.spines[['top','right']].set_visible(False)
ax.set_ylabel('% Correct trials')
ax.set_xlabel('# of epochs')
# Group rates by number of epochs
grouped = df_plt.groupby('num_epochs')['rates'].apply(list)

# Filter out groups with <2 data points (optional for robustness)
valid_groups = [g for g in grouped if len(g) > 1]

# Run test if at least 2 valid groups
if len(valid_groups) > 1:
    stat, p = scipy.stats.kruskal(*valid_groups)
    print("Kruskal-Wallis Test: Effect of number of epochs on % correct trials")
    print(f"H-statistic = {stat:.3f}, p = {p:.4f}")
else:
    print("Not enough valid groups to run Kruskal-Wallis test.")
summary = df_plt.groupby('num_epochs')['rates'].agg(['mean', scipy.stats.sem]).reset_index()
summary.columns = ['num_epochs', 'mean_rate', 'sem_rate']

print("Mean ± SEM of % correct trials per number of epochs:")
print(summary)

plt.savefig(os.path.join(savedst, 'p_correct_trials.svg'))

#%%
fig,ax = plt.subplots(figsize=(3,5))
sns.stripplot(x='num_epochs', y='lick_selectivity',color='k',
        data=df_plt,alpha=0.7,
        s=10)
sns.barplot(x='num_epochs', y='lick_selectivity',
        data=df_plt,
        fill=False,ax=ax, color='k', errorbar='se')
ax.spines[['top','right']].set_visible(False)
ax.set_ylabel('Lick selectivity, last 8 trials')
ax.set_xlabel('# of reward loc. switches')
plt.savefig(os.path.join(savedst, 'lick_selectivity.svg'))

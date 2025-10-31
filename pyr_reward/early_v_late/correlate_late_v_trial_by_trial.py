
"""
zahra
late/stable tc  vs. trial by trial
TODO: bin trials by 3 trial tuning curves, get spatially tuned cells only
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
from projects.pyr_reward.placecell import get_tuning_curve, calc_COM_EH, make_tuning_curves_radians_by_trialtype,\
    make_tuning_curves_time_trial_by_trial
from projects.pyr_reward.rewardcell import get_radian_position,\
    get_radian_position_first_lick_after_rew, get_rewzones, get_goal_cells, goal_cell_shuffle,\
    trail_type_activity_quant, cosine_sim_ignore_nan

from projects.opto.behavior.behavior import get_success_failure_trials
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
saveddataset = r"Z:\saved_datasets\stable_v_trial_by_trial_tc.p"


#%%
####################################### RUN CODE #######################################
# initialize var
radian_alignment_saved = {} # overwrite
bins = 90
goal_window_cm=20
datadct = {}
# cm_window = [10,20,30,40,50,60,70,80] # cm
# iterate through all animals
#%%
for ii in range(len(conddf)):
    day = conddf.days.values[ii]
    animal = conddf.animals.values[ii]
    # check if its the last 3 days of animal behavior
    andf = conddf[(conddf.animals==animal) &( conddf.optoep<2)]
    lastdays = andf.days.values[-3:]
    if (animal!='e217') & (day in lastdays):
        if animal=='e145' or animal=='e139': pln=2 
        else: pln=0
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
        print(params_pth)
        fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 'timedFF',
                'putative_pcs', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
                'stat', 'licks'])
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
        rad = get_radian_position_first_lick_after_rew(eps, ybinned, 
            licks, rewards, rewsize,rewlocs,trialnum, track_length) # get radian coordinates
        # ONLY SPATIALLY TUNED CELLS
        fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
        Fc3 = fall_fc3['Fc3']
        dFF = fall_fc3['dFF']
        Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
        dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
        skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
        #if pc in all but 1
        # pc_bool = np.sum(pcs,axis=0)>=len(eps)-2
        # looser restrictions
        pc_bool = np.sum(pcs,axis=0)>=1
        Fc3 = Fc3[:,((skew>2)&pc_bool)] # only keep cells with skew greateer than 2
        # if no cells pass these crit
        if Fc3.shape[1]==0:
            Fc3 = fall_fc3['Fc3']
            Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
            # to avoid issues with e217 and z17?
            # pc_bool = np.sum(pcs,axis=0)>=1
            Fc3 = Fc3[:,((skew>1))]
        # normal tc
        tcs_correct, coms_correct, tcs_fail, coms_fail = make_tuning_curves_radians_by_trialtype(eps,rewlocs,ybinned,rad,Fc3,trialnum,
        rewards,forwardvel,rewsize,bin_size) 
        # get trial by trial tuning curves
        trialstates, licks_all, tcs, coms=make_tuning_curves_time_trial_by_trial(eps, rewlocs, licks, ybinned, time, Fc3, trialnum, rewards, forwardvel, rewsize, bin_size)
        # only corrects
        # per cell per epoch
        # max trials
        trialstates=trialstates[:len(tcs)] # correct for epoch discrepancy
        if len(tcs_correct)!=len(trialstates):
            if len(tcs_correct)>len(trialstates):
                tcs_correct=tcs_correct[:len(trialstates)]
            else:                
                trialstates=trialstates[:len(tcs_correct)]
        maxtr= [tr[:,trialstates[ep]==1].shape[1] for ep,tr in enumerate(tcs)]
        bin_size_trials = 3
        stable_v_tr_cosine_sim_corr = np.ones((len(tcs_correct), np.ceil(np.nanmax(maxtr) / bin_size_trials).astype(int), tcs_correct[0].shape[0])) * np.nan
        for ep, tc_c in enumerate(tcs_correct):
            # Get trial-by-trial data: trials x cells x bins
            trial_tc = np.transpose(tcs[ep][:, trialstates[ep] == 1, :], (1, 0, 2))
            n_trials = trial_tc.shape[0]
            n_bins = int(np.ceil(n_trials / bin_size_trials))

            for b in range(n_bins):
                start = b * bin_size_trials
                end = min((b + 1) * bin_size_trials, n_trials)
                if end - start < 1:
                    continue
                # Average across trials in bin
                bin_tc = np.nanmean(trial_tc[start:end, :, :], axis=0)  # shape: cell x bins
                stable_v_tr_cosine_sim_corr[ep, b, :] = [
                    cosine_sim_ignore_nan(bin_tc[cll], tc_c[cll, :]) for cll in range(tc_c.shape[0])
                ]

        datadct[f'{animal}_{day:03d}_index{ii:03d}'] = [tcs_correct, coms_correct, tcs_fail, coms_fail, trialstates, licks_all, tcs, coms, stable_v_tr_cosine_sim_corr, rates]

# save pickle of dcts
with open(saveddataset, "wb") as fp:   #Pickling
        pickle.dump(datadct, fp) 
####################################### RUN CODE #######################################
#%%
plt.rc('font', size=20)          # controls default text sizes

corr_succ = [v[8] for k,v in datadct.items()]
corr_all = [v[9] for k,v in datadct.items()]
# av across cells
corr_succ_o = [np.nanmean(c, axis=2) for c in corr_succ]
# just take epoch 1
fig, ax = plt.subplots(figsize=(6, 4))
colors=['k','slategray']
for ep in range(2):
    corr_succ = [c[ep] for c in corr_succ_o]
    # nan pad extra trials
    maxl = max([len(c) for c in corr_succ])
    corr_succ_arr = np.ones((len(corr_succ), maxl))*np.nan
    for k,c in enumerate(corr_succ):
        corr_succ_arr[k,:len(c)]=c

    # Compute mean and SEM
    mean_succ = np.nanmedian(corr_succ_arr, axis=0)
    sem_succ = scipy.stats.sem(corr_succ_arr,axis=0,nan_policy='omit')
    # Plot
    t = np.arange(mean_succ.shape[0])  # adjust to match your time axis
    ax.plot(mean_succ,color=colors[ep])
    ax.fill_between(t, mean_succ-sem_succ, mean_succ + sem_succ, color=colors[ep],alpha=0.2)
    # ax.set_xlim([0,21])
# t = np.arange(mean_all.shape[0])  # adjust to match your time axis
# ax.plot(mean_all)
# ax.fill_between(t, mean_all-sem_all, mean_all + sem_all, color='blue',alpha=0.2)

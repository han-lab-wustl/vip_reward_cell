
"""
zahra
get tuning curves with dark time
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
from projects.pyr_reward.placecell import get_tuning_curve, calc_COM_EH, make_tuning_curves_radians_by_trialtype
from projects.pyr_reward.rewardcell import get_radian_position,create_mask_from_coordinates,pairwise_distances,extract_data_rewcentric,\
    get_radian_position_first_lick_after_rew, get_rewzones, get_goal_cells
from projects.opto.behavior.behavior import get_success_failure_trials
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_rewardcentric_all.p"
with open(saveddataset, "rb") as fp: #unpickle
        radian_alignment_saved = pickle.load(fp)
savepth = os.path.join(savedst, 'dark_time_tuning.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)


def get_radian_position_first_lick_after_rew_w_dt(i, eps, ybinned, licks, reward, rewsize,rewlocs,
                    trialnum):
    """
    Computes radian position aligned to the first lick after reward.
    Parameters:
    - i = epoch
    - eps: List of trial start indices.
    - ybinned: 1D array of position values.
    - licks: 1D binary array (same length as ybinned) indicating lick events.
    - reward: 1D binary array (same length as ybinned) indicating reward delivery.
    - track_length: Total length of the circular track.

    Returns:
    - rad: 1D array of radian positions aligned to the first lick after reward.
    """
    rad = []  # Store radian coordinates
    # for i in range(len(eps) - 1):
    # Extract data for the current trial
    y_trial = ybinned#[eps[i]:eps[i+1]]
    licks_trial = licks#[eps[i]:eps[i+1]]
    reward_trial = reward#[eps[i]:eps[i+1]]
    trialnum_trial = trialnum#[eps[i]:eps[i+1]]
    unique_trials = np.unique(trialnum)  # Get unique trial numbers [eps[i]:eps[i+1]]
    for tr,trial in enumerate(unique_trials):
        # Extract data for the current trial
        trial_mask = trialnum_trial == trial  # Boolean mask for the current trial
        y = y_trial[trial_mask]
        licks_trial_ = licks_trial[trial_mask]
        reward_trial_ = reward_trial[trial_mask]
        # Find the reward location in this trial
        reward_indices = np.where(reward_trial_ > 0)[0]  # Indices where reward occurs
        if len(reward_indices) == 0:
            try:
                # 1.5 bc doesn't work otherwise?
                y_rew = np.where((y<(rewlocs[tr][i]+rewsize*.5)) & (y>(rewlocs[i][tr]-rewsize*.5)))[0][0]
                reward_idx=y_rew
            except Exception as e: # if trial is empty??
                reward_idx=int(len(y)/2) # put in random middle place of trials
        else:
            reward_idx = reward_indices[0]  # First occurrence of reward
        # Find the first lick after the reward
        lick_indices_after_reward = np.where((licks_trial_ > 0) & (np.arange(len(licks_trial_)) > reward_idx))[0]
        if len(lick_indices_after_reward) > 0:
            first_lick_idx = lick_indices_after_reward[0]  # First lick after reward
        else:
            # if animal did not lick after reward/no reward was given
            first_lick_idx=reward_idx
        # Convert positions to radians relative to the first lick
        first_lick_pos = y[first_lick_idx]
        track_length = np.max(y) # custom max for each trial w dark time
        rad.append((((((y - first_lick_pos) * 2 * np.pi) / track_length) + np.pi) % (2 * np.pi)) - np.pi)

    if len(rad) > 0:
        rad = np.concatenate(rad)
        return rad
    else:
        return np.array([])  # Return empty array if no valid trials

def make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,rewsize,ybinned,time,licks,
            Fc3,trialnum, rewards,forwardvel,scalingf,bin_size=3.5,
            lasttr=8,bins=90,
            velocity_filter=False):    
    rates = []; tcs_fail = []; tcs_correct = []; coms_correct = []; coms_fail = []        
    rewlocs_w_dt = []; ybinned_dt = []
    failed_trialnm = []; 
    # remake tuning curves relative to reward        
    for ep in range(len(eps)-1):
        eprng = np.arange(eps[ep],eps[ep+1])
        rewloc = rewlocs[ep]
        ypos_ep = ybinned[eprng]
        vel_ep = forwardvel[eprng]
        trial_ep = trialnum[eprng]        
        reward_ep = rewards[eprng]
        lick_ep = licks[eprng]
        # Get dark time frames and estimate distance
        ypos_dt = []
        rewloc_per_trial = []
        rewloc_bool = []
        # get the dark time and add it to the beginning of the trial
        for trial in np.unique(trial_ep):            
            trial_mask = trial_ep==trial
            # constant set to dt
            trial_ind = 5
            if not sum(trial_mask)>trial_ind and sum(trial_mask)>1:                
                trial_ind = 1
            elif sum(trial_mask)==1:
                trial_ind=0
            ypos_num = ypos_ep[trial_mask][trial_ind]
            ypos_trial = ypos_ep[trial_mask]
            # remove random end of track value            
            ypos_trial[:trial_ind] = ypos_num
            dark_mask = ypos_trial == ypos_num
            dark_vel = vel_ep[trial_mask][dark_mask]
            dark_frames = np.sum(dark_mask)
            
            dark_time = time[eprng][trial_mask][dark_mask]
            dark_dt_series = np.diff(dark_time, prepend=dark_time[0])  # time differences
            dark_distance = np.cumsum(dark_vel * dark_dt_series)  # distance traveled frame-by-frame
            dark_distance = dark_distance / scalingf  # convert to position
            # dark_dt = time[eprng][trial_mask][dark_mask] 
            # dark_distance = np.nanmean(dark_vel) * dark_dt
            # dark_distance = (dark_distance-dark_distance[0])/scalingf # scale to gain            
            from scipy.ndimage import gaussian_filter1d
            dt_ind = np.where(ypos_trial==ypos_num)[0]
            ypos_trial_new = ypos_trial.copy()
            ypos_trial_new[ypos_trial_new==ypos_num] = dark_distance
            ypos_trial_new[ypos_trial>ypos_num] = ypos_trial_new[ypos_trial>ypos_num]+dark_distance[-1]
            ypos_dt.append(ypos_trial_new)
            # find start of rew loc index
            rewloc = (ypos_trial >= (rewlocs[ep]-(rewsize/2)-5)) & (ypos_trial <= (rewlocs[ep]+(rewsize/2)+5))
            rewloc = consecutive_stretch(np.where(rewloc)[0])[0]
            if len(rewloc)>0:
                rewloc = min(rewloc)
            else: # set it to a random number in middle of the track...
                rewloc = int(len(ypos_trial_new)/2)
            rewloc_per_trial.append(ypos_trial_new[rewloc])
            rl_bool = np.zeros_like(ypos_trial_new)
            rl_bool[rewloc]=1
            rewloc_bool.append(rl_bool)
        
        # nan pad position
        ypos_w_dt = np.concatenate(ypos_dt)
        ybinned_dt.append(ypos_w_dt)
        # realign to reward????        
        rewloc_bool = np.concatenate(rewloc_bool)
        # test
        # plt.plot(ypos_w_dt)
        # plt.plot(rewloc_bool*400)        
        relpos = get_radian_position_first_lick_after_rew_w_dt(ep, eps, ypos_w_dt, lick_ep, 
                reward_ep, rewsize, rewloc_per_trial,
                trial_ep)
        success, fail, strials, ftrials, ttr, total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
        rates.append(success/total_trials)
        # in between failed trials only!!!!! 4/2025
        if len(strials)>0:
            failed_inbtw = np.array([int(xx)-strials[0] for xx in ftrials])
            failed_inbtw=np.array(ftrials)[failed_inbtw>0]
        else: # for cases where an epoch was started but not enough trials
            failed_inbtw=np.array(ftrials)
        failed_trialnm.append(failed_inbtw)
        # trials going into tuning curve
        F_all = Fc3[eprng,:]            
        # simpler metric to get moving time
        if velocity_filter==True:
            moving_middle = forwardvel[eprng]>5 # velocity > 5 cm/s
        else:
            moving_middle = np.ones_like(forwardvel[eprng]).astype(bool)
        F_all = F_all[moving_middle,:]
        relpos_all = np.array(relpos)[moving_middle]
        if len(ttr)>lasttr: # only if ep has more than x trials
            # last 8 correct trials
            if len(strials)>0:
                mask = [True if xx in strials[-lasttr:] else False for xx in trialnum[eprng][moving_middle]]
                F = F_all[mask,:]
                relpos = relpos_all[mask]                
                tc = np.array([get_tuning_curve(relpos, f, bins=bins) for f in F.T])
                com = calc_COM_EH(tc,bin_size)
                tcs_correct.append(tc)
                coms_correct.append(com)
            # failed trials                        
            # UPDATE 4/16/25
            # only take last 8 failed trials?
            if len(failed_inbtw)>0:
                mask = [True if xx in failed_inbtw[-lasttr:] else False for xx in trialnum[eprng][moving_middle]]
                F = F_all[mask,:]
                # print(f'Fluorescence array size:\n{F.shape}\n')
                relpos = relpos_all[mask]                
                tc = np.array([get_tuning_curve(relpos, f, bins=bins) for f in F.T])
                com = calc_COM_EH(tc,bin_size)
                tcs_fail.append(tc)
                coms_fail.append(com)
        rewlocs_w_dt.append(rewloc_per_trial)
    tcs_correct = np.array(tcs_correct); coms_correct = np.array(coms_correct)  
    tcs_fail = np.array(tcs_fail); coms_fail = np.array(coms_fail)  
    return tcs_correct, coms_correct, tcs_fail, coms_fail, rewlocs_w_dt, ybinned_dt

#%%
# initialize var
# radian_alignment_saved = {} # overwrite
data_dct={}
bins = 90
goal_window_cm=20
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
        rad = get_radian_position_first_lick_after_rew(eps, ybinned, licks, rewards, rewsize,rewlocs,
                    trialnum, track_length) # get radian coordinates

        # added to get anatomical info
        # takes time
        fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
        Fc3 = fall_fc3['Fc3']
        dFF = fall_fc3['dFF']
        Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
        dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]
        skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
        Fc3 = Fc3[:, skew>2] # only keep cells with skew greateer than 2
        # tc w/ dark time
        track_length_rad_dt = 475*(2*np.pi/track_length) # estimate bin for dark time
        bins_dt=150 
        bin_size_dt=track_length_rad/bins_dt # typically 3 cm binswith ~ 475 track length
        tcs_correct_dt, coms_correct_dt, tcs_fail, coms_fail, rewloc_dt, ybinned_dt = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,rewsize,ybinned,time,licks,
            Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
            bins=bins_dt)
        # normal tc
        tcs_correct, coms_correct, tcs_fail, coms_fail = make_tuning_curves_radians_by_trialtype(eps,rewlocs,ybinned,rad,Fc3,trialnum,
        rewards,forwardvel,rewsize,bin_size)          

        #test
        # fig,axes = plt.subplots(ncols=len(tcs_correct_dt))
        # for ep in range(len(tcs_correct_dt)):
        #     ax=axes[ep]
        #     ax.imshow(tcs_correct_dt[ep][np.argsort(coms_correct_dt[0])]**.3)
        #     ax.axvline(bins_dt/2, color='w',linestyle='--')
            
        # fig,axes = plt.subplots(ncols=len(tcs_correct))
        # for ep in range(len(tcs_correct)):
        #     ax=axes[ep]
        #     ax.imshow(tcs_correct[ep][np.argsort(coms_correct[0])]**.3)
        #     ax.axvline(bins/2, color='w',linestyle='--')
        
        goal_window = goal_window_cm*(2*np.pi/track_length) # cm converted to rad
        goal_cells, com_goal_postrew, perm, rz_perm = get_goal_cells(rz, goal_window, coms_correct, cell_type = 'all')
        goal_cells_dt, com_goal_postrew_dt, perm_dt, rz_perm_dt = get_goal_cells(rz, goal_window, coms_correct_dt, cell_type = 'all')
        #only get perms with non zero cells
        
        p_goal_cells = len(goal_cells)/len(coms_correct[0])
        p_goal_cells_dt = len(goal_cells_dt)/len(coms_correct_dt[0])
        goal_cells_iind = [goal_cells, goal_cells_dt]
        # save!!!
        data_dct[f'{animal}_{day:03d}_index{ii:03d}'] = [p_goal_cells,p_goal_cells_dt,
                                goal_cells_iind]
        print(f'Goal cells w/o dt: {goal_cells}\n\
            Goal cells w/ dt: {goal_cells_dt}')
        # plot separately
        colors = ['k', 'slategray', 'darkcyan', 'darkgoldenrod', 'orchid']
        if len(goal_cells)>0:
            rows = int(np.ceil(np.sqrt(len(goal_cells))))
            cols = int(np.ceil(len(goal_cells) / rows))
            # scale fig based on num cells
            fig, axes = plt.subplots(rows, cols, figsize=(rows*5,cols*5),sharex=True)
            if len(goal_cells) > 1:
                axes = axes.flatten()
            else:
                axes = [axes]
            for i,gc in enumerate(goal_cells):            
                for ep in range(len(coms_correct)):
                    ax = axes[i]
                    ax.plot(tcs_correct[ep,gc,:], label=f'rewloc {rewlocs[ep]}', color=colors[ep])
                    # if len(tcs_fail)>0:
                    #         ax.plot(tcs_fail[ep,gc,:], label=f'fail rewloc {rewlocs[ep]}', color=colors[ep], linestyle = '--')
                    ax.axvline((bins/2), color='k')
                    ax.set_title(f'cell # {gc}')
                    ax.spines[['top','right']].set_visible(False)
            ax.set_xticks(np.arange(0,bins+1,20))
            ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi, np.pi/2.25),3))
            ax.set_xlabel('Radian position (centered start rew loc)')
            ax.set_ylabel('Fc3')
            # fig.tight_layout()
            fig.suptitle(f'{animal}, {day}, goal cells w/o dt')
            pdf.savefig(fig)
            plt.show()
            # plt.close(fig)
        if len(goal_cells_dt)>0:
            rows = int(np.ceil(np.sqrt(len(goal_cells_dt))))
            cols = int(np.ceil(len(goal_cells_dt) / rows))
            # scale fig based on num cells
            fig, axes = plt.subplots(rows, cols, figsize=(rows*5,cols*5),sharex=True)
            if len(goal_cells_dt) > 1:
                axes = axes.flatten()
            else:
                axes = [axes]
            for i,gc in enumerate(goal_cells_dt):            
                for ep in range(len(coms_correct_dt)):
                    ax = axes[i]
                    ax.plot(tcs_correct_dt[ep,gc,:], label=f'rewloc {rewlocs[ep]}', color=colors[ep])
                    # if len(tcs_fail)>0:
                    #         ax.plot(tcs_fail[ep,gc,:], label=f'fail rewloc {rewlocs[ep]}', color=colors[ep], linestyle = '--')
                    ax.axvline((bins_dt/2), color='k')
                    ax.set_title(f'cell # {gc}')
                    ax.spines[['top','right']].set_visible(False)
            ax.set_xticks(np.arange(0,bins_dt+1,30))
            ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi, np.pi/3),2))
            ax.set_xlabel('Radian position (centered start rew loc)')
            ax.set_ylabel('Fc3')
            # fig.tight_layout()
            fig.suptitle(f'{animal}, {day}, goal cells w/ dt')
            pdf.savefig(fig)
            plt.show()
            # plt.close(fig)

pdf.close()
#%%
# distribution of % goal cells
df = conddf.copy()
df = df[((df.animals!='e217')) & (df.optoep<2)]
a=0.3
plt.hist(p_goal_cells,alpha=a,label = 'w/o dark time',color='darkcyan',bins=20)
plt.hist(p_goal_cells_dt,alpha=a,label = 'w/ dark time',color='k',bins=20)
plt.legend()
plt.xlabel('Reward cell %')
plt.ylabel('Sessions')
#%%
fig, ax = plt.subplots()
x_ = np.array(p_goal_cells)
x = x_[x_>0]
y = np.array(p_goal_cells_dt)
y = y[x_>0]
x = x[y>0]
y = y[y>0]
ax.scatter(x,y,s=50)
ax.set_xlabel('Reward cell % without delay period')
ax.set_ylabel('Reward cell % with delay period')
ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c="k")

slope, intercept = np.polyfit(x,y, 1)
x_fit = np.array(ax.get_xlim())
y_fit = slope * x_fit + intercept
ax.plot(x_fit, y_fit, color='red', label=f'Fit: y = {slope:.2f}x + {intercept:.2f}')


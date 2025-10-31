
"""
zahra
field width transitions
may 2025
dark time updates
"""
#%%
import numpy as np, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
from scipy.ndimage import gaussian_filter1d
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
# plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.pyr_reward.placecell import make_tuning_curves_radians_by_trialtype, intersect_arrays,\
    make_tuning_curves_by_trialtype_w_darktime
from projects.pyr_reward.rewardcell import get_radian_position,create_mask_from_coordinates,pairwise_distances,extract_data_rewcentric,\
    get_radian_position_first_lick_after_rew, get_rewzones
from projects.pyr_reward.placecell import get_tuning_curve, calc_COM_EH, make_tuning_curves_by_trialtype_w_darktime, make_tuning_curves_time_trial_by_trial_w_darktime, intersect_arrays
from projects.opto.behavior.behavior import get_success_failure_trials
from width import smooth_lick_rate,detect_lick_bouts, check_reward_in_bouts
from projects.pyr_reward.rewardcell import perireward_binned_activity_early_late, perireward_binned_activity

# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
#%%
# initialize var
# radian_alignment_saved = {} # overwrite
bins = 150
goal_window_cm=20
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
        lick=fall['licks'][0]
        time=fall['timedFF'][0]
        if animal=='e145':
            ybinned=ybinned[:-1]
            forwardvel=forwardvel[:-1]
            changeRewLoc=changeRewLoc[:-1]
            trialnum=trialnum[:-1]
            rewards=rewards[:-1]
            lick=licks[:-1]
            time=time[:-1]
        # set vars
        eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
        rz = get_rewzones(rewlocs,1/scalingf)       
        # get average success rate
        rates = []
        for ep in range(len(eps)-1):
                eprng = range(eps[ep],eps[ep+1])
                success, fail, str_trials, ftr_trials, ttr, \
                total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
                rates.append(success/total_trials)
        rate=np.nanmean(np.array(rates))
        # dark time params
        track_length_dt = 550 # cm estimate based on 99.9% of ypos
        track_length_rad_dt = track_length_dt*(2*np.pi/track_length_dt) # estimate bin for dark time
        bins_dt=150 
        bin_size_dt=track_length_rad_dt/bins_dt # typically 3 cm binswith ~ 475 track length
        # added to get anatomical info
        # takes time
        fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
        Fc3 = fall_fc3['Fc3']
        dFF = fall_fc3['dFF']
        Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
        dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]
        skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
        Fc3 = Fc3[:, skew>2] # only keep cells with skew greateer than 2
        # tc w/ dark time added to the end of track
        tcs_correct, coms_correct, tcs_fail, coms_fail, ybinned_dt, rad = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,
            rewsize,ybinned,time,lick,
            Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
            bins=bins_dt)  
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
        # all cells before 0
        com_goal_postrew = [[xx for xx in com if ((np.nanmedian(coms_rewrel[:,xx], axis=0)<0)&(np.nanmedian(coms_rewrel[:,xx], axis=0)>-np.pi/4))] if len(com)>0 else [] for com in com_goal]
        # get goal cells across all epochs        
        if len(com_goal_postrew)>0:
            goal_cells = intersect_arrays(*com_goal_postrew); 
        else:
            goal_cells=[]
        goal_all = np.unique(np.concatenate(com_goal_postrew)).astype(int)        
        # lick rate
        dt = np.median(np.diff(time))  # make sure time is 1D and regularly sampled
        lickrate = smooth_lick_rate(lick,dt)
        bout_starts,bout_ends  = detect_lick_bouts(lickrate, time, threshold=.2, min_duration=0.01)
        # Now you can use this to filter:
        lick_bout_starts = np.array(bout_starts)
        # peri align
        allprerew_r = []
        allprerew_nr = []
        # corresponding licks
        allprerew_lickr = []
        allprerew_licknr = []
        range_val,binsize=12,.1
        for gc in goal_all:
            _, meanrflick, __, rewrflick = perireward_binned_activity(Fc3[:,gc], rew_bout_bin.astype(bool), 
            time, trialnum, range_val,binsize)
            _, meanflick, __, rewflick = perireward_binned_activity(Fc3[:,gc], nonrew_bout_bin.astype(bool), 
            time, trialnum, range_val,binsize)
            _, mean_rlick, __, all_rlick = perireward_binned_activity(lick, rew_bout_bin.astype(bool), 
            time, trialnum, range_val,binsize)
            _, mean_lick, __, all_lick = perireward_binned_activity(lick, nonrew_bout_bin.astype(bool), time, trialnum, range_val,binsize)
            _, mean_rrew, __, all_rrew = perireward_binned_activity(rewards==1, rew_bout_bin.astype(bool), 
            time, trialnum, range_val,binsize)
            _, mean_rew, __, all_rew = perireward_binned_activity(rewards==1, nonrew_bout_bin.astype(bool), time, trialnum, range_val,binsize)

            allprerew_r.append(rewrflick)
            allprerew_nr.append(rewflick)
            allprerew_lickr.append(all_rlick)
            allprerew_licknr.append(all_lick)
        # correlation b/wn licks and fc3
        ep=0
        eprng = np.arange(eps[ep],eps[ep+1])
        trial_ids = np.unique(trialnum[eprng])[-8:]
        cell_lick_corrs = []
        for cell_idx in goal_all:
            corrs = []
            for tr in trial_ids:
                tr_mask = trialnum[eprng] == tr
                if np.sum(tr_mask) < 5:  # skip too-short trials
                    continue
                lick_rate_tr = lickrate[eprng][tr_mask]
                fc3_tr = Fc3[eprng][tr_mask, cell_idx]
                if np.any(np.isnan(lick_rate_tr)) or np.any(np.isnan(fc3_tr)):
                    continue
                corr, _ = scipy.stats.pearsonr(fc3_tr, lick_rate_tr)
                corrs.append(corr)
            cell_lick_corrs.append(np.nanmean(corrs))
        cell_lick_corrs = np.array(cell_lick_corrs)
        plt.hist(cell_lick_corrs)
        
        # Calculate distance from last lick to reward for each trial
        for ep in range(len(eps)-1):
            eprng = np.arange(eps[ep],eps[ep+1])
            trialnumep = trialnum[eprng]
            trial_ids = np.unique(trialnumep)
            trial_lick_dists = []
            trial_com_widths = []
            for tr in trial_ids:
                tr_mask = trialnumep == tr
                if np.sum(tr_mask) < 5:
                    continue
                tr_licks = lick[eprng][tr_mask]
                tr_y = ybinned[eprng][tr_mask]
                tr_rewards = rewards[eprng][tr_mask]
                first_lick_idx = np.where(tr_licks)[0][0]                
                reward_idx = np.where(tr_rewards)[0]
                # before cs technically
                if len(reward_idx)>0:
                    last_lick_before_rew_idx = np.where(tr_licks)[0][np.where(tr_licks)[0]<reward_idx[0]][-1]
                else:
                    last_lick_before_rew_idx = np.where(tr_licks)[0][-1]
                dist = np.abs(tr_y[last_lick_before_rew_idx] - tr_y[first_lick_idx])
                trial_lick_dists.append(dist)
                # Compute width from tuning curve
                widths, tcs = compute_field_width(Fc3[eprng,:][tr_mask,:][:,goal_all], ybinned[eprng][tr_mask], bins=150, smooth_sigma=.2,threshold_ratio=0.2)  # You'll need to implement `compute_field_width`
                trial_com_widths.append(widths)

            trial_lick_dists = np.array(trial_lick_dists)
            trial_com_widths = np.vstack(trial_com_widths)  # trials × cells

            # Classify trials
            threshold = np.nanmean(trial_lick_dists)
            short_idx = trial_lick_dists < threshold-10
            long_idx = trial_lick_dists >= threshold+10
            # Compare mean field width across neurons
            short_widths = np.nanmean(trial_com_widths[short_idx], axis=0)
            long_widths = np.nanmean(trial_com_widths[long_idx], axis=0)
            # Keep only neurons with non-NaN values in both conditions
            valid_mask = ~np.isnan(short_widths) & ~np.isnan(long_widths)
            short_clean = short_widths[valid_mask]
            long_clean = long_widths[valid_mask]
            # Paired test on cleaned arrays
            tstat, pval = scipy.stats.wilcoxon(short_clean, long_clean)
            print(f"T-test: t={tstat:.2f}, p={pval:.3g}")

        # test
        # fig, axes = plt.subplots(ncols=3, figsize=(12, 5))
        # # Convert to arrays and normalize by row
        # allprerew_r = np.array([xx for xx in allprerew_r if np.nanmax(xx)>0.2])
        # allprerew_nr = np.array([xx for xx in allprerew_nr if np.nanmax(xx)>0.2])
        # # Row-wise normalization
        # allprerew_r_norm = (allprerew_r - np.nanmin(allprerew_r, axis=1, keepdims=True)) / \
        #                 (np.nanmax(allprerew_r, axis=1, keepdims=True) - np.nanmin(allprerew_r, axis=1, keepdims=True))
        # allprerew_nr_norm = (allprerew_nr - np.nanmin(allprerew_nr, axis=1, keepdims=True)) / \
        #                     (np.nanmax(allprerew_nr, axis=1, keepdims=True) - np.nanmin(allprerew_nr, axis=1, keepdims=True))
        # # Sort by max
        # sort_idx = np.argsort(np.nanmax(allprerew_r,axis=1))
        # sort_idx_nr = np.argsort(np.nanmax(allprerew_nr,axis=1))
        # # Plot prereward cells
        # ax = axes[0]
        # ax.set_title('Pre-reward cell')
        # ax.imshow(allprerew_r_norm[sort_idx], cmap='Greens', aspect='auto')
        # ax.axvline(int(range_val/binsize), color='k')
        # # Plot non-reward cells
        # ax = axes[1]
        # ax.imshow(allprerew_nr_norm[sort_idx], cmap='Reds', aspect='auto')
        # ax.axvline(int(range_val/binsize), color='k')
        # # Lick rate
        # ax = axes[2]
        # ax.set_title('Lick rate')
        # ax.plot(mean_rlick, color='seagreen')
        # ax.plot(mean_lick, color='firebrick')
        # ax.axvline(int(range_val/binsize), color='k')




        
#%%
# get all cells width cm 
bigdf = pd.concat(dfs)
# bigdf = bigdf[bigdf['75_quantile']>0]
# exclude some animals 
# bigdf = bigdf[(bigdf.animal!='e139') & (bigdf.animal!='e145') & (bigdf.animal!='e200')]

# transition from 1 to 3 only
plt.rc('font', size=20)
rzdf = bigdf[(bigdf.epoch.str.contains('epoch1_rz1') | bigdf.epoch.str.contains('epoch2_rz3'))]
rzdf2 = bigdf[(bigdf.epoch.str.contains('epoch2_rz1') | bigdf.epoch.str.contains('epoch3_rz3'))]
rzdf3 = bigdf[(bigdf.epoch.str.contains('epoch3_rz1') | bigdf.epoch.str.contains('epoch4_rz3'))]
rzdf4 = bigdf[(bigdf.epoch.str.contains('epoch4_rz1') | bigdf.epoch.str.contains('epoch5_rz3'))]
rzdf = pd.concat([rzdf,rzdf2,rzdf3,rzdf4])

rzdf=rzdf.reset_index()
rzdf['epoch'] = [xx[-3:] for xx in rzdf.epoch.values]
# fig, ax = plt.subplots(figsize=(4,5))
# sns.stripplot(x='epoch',y='width_cm',data=rzdf,hue='cell_type',alpha=0.05,dodge=True)
# sns.boxplot(x='epoch',y='width_cm',data=rzdf,fill=False,hue='cell_type')
# ax.spines[['top','right']].set_visible(False)

#%%
# per animal
s=10
hue_order = ['Pre','Post']
fig, ax = plt.subplots(figsize=(4,5))
# only get super close rz1 rewlocs
rzdf['rewloc_cm'] = [float(xx[-5:]) for xx in rzdf.rewloc.values]
rzdf = rzdf[((rzdf.epoch=='rz3')&(rzdf.rewloc_cm.values>210))|((rzdf.epoch=='rz1')&(rzdf.rewloc_cm.values<120))]
anrzdf = rzdf.groupby(['animal', 'epoch', 'cell_type']).mean(numeric_only=True)
anrzdf=anrzdf.reset_index()
anrzdf = anrzdf.sort_values(by="cell_type", ascending=False)
anrzdf = anrzdf.sort_values(by="epoch", ascending=True)
# anrzdf=anrzdf[(anrzdf.animal!='e189') & (anrzdf.animal!='e190')]
sns.stripplot(x='epoch',y='width_cm',data=anrzdf,hue='cell_type',alpha=0.7,dodge=True,s=s,
    palette='Dark2',hue_order=hue_order)
h_strip, l_strip = ax.get_legend_handles_labels()
sns.boxplot(x='epoch',y='width_cm',hue='cell_type',data=anrzdf,
        fill=False,palette='Dark2',hue_order=hue_order,
           showfliers=False)
ax.spines[['top','right']].set_visible(False)

ans = anrzdf.animal.unique()
ind = ['Pre','Post']
for j,ct in enumerate(ind):
    for i in range(len(ans)):    
        df_ = anrzdf[(anrzdf.animal==ans[i]) & (anrzdf.cell_type==ct)]
        df_ = df_.sort_values(by="epoch", ascending=True)
        color=sns.color_palette('Dark2')[j]
        ax = sns.lineplot(x=np.arange(len(df_.epoch.values))+(j*.2)-.1,y='width_cm',
        data=df_,
        errorbar=None, color=color, linewidth=2, alpha=0.3,ax=ax)

# 3) remove whatever legend was just created
ax.legend_.remove()
# 4) re-add only the stripplot legend, placing it outside
ax.legend(
    h_strip, l_strip,
    title='Cell Type',
    loc='upper left',
    bbox_to_anchor=(1.02, 1),
    borderaxespad=0.
)
ax.spines[['top','right']].set_visible(False)
ax.set_ylabel('Field width (cm)')
ax.set_xlabel('')
ax.set_xticklabels(['Near', 'Far'],rotation=45)
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM
palette=sns.color_palette('Dark2')
# anrzdf has columns: ['animal','epoch','cell_type','width_cm']
# make sure epoch & cell_type are categorical
anrzdf['epoch'] = anrzdf['epoch'].astype('category')
anrzdf['cell_type'] = anrzdf['cell_type'].astype('category')

aov = AnovaRM(
    data=anrzdf,
    depvar='width_cm',
    subject='animal',
    within=['epoch','cell_type']
).fit()
print(aov)

results = []
for ct in ['Pre','Post']:
    sub = anrzdf[anrzdf['cell_type']==ct]
    near = sub[sub['epoch']=='rz1']['width_cm']
    far  = sub[sub['epoch']=='rz3']['width_cm']
    t, p = scipy.stats.ttest_rel(near, far)
    results.append({'cell_type':ct, 't_stat':t, 'p_uncorrected':p})

posthoc = pd.DataFrame(results)
posthoc['p_bonferroni'] = np.minimum(posthoc['p_uncorrected']*len(posthoc), 1.0)
print(posthoc)
# mapping epoch→x-position (as you offset earlier)
x_map = {'Pre':0, 'Post':1}
offset = {'Pre':-0.1, 'Post':+0.1}

for _, row in posthoc.iterrows():
    ct = row['cell_type']
    p = row['p_bonferroni']
    stars = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
    x = x_map[ct]+offset[ct]  # test is Pre vs Post, annotate at Pre position
    # y-height: just above max for that group
    ymax = anrzdf[(anrzdf.cell_type==ct)]['width_cm'].max()
    ax.text(x, ymax, stars, ha='center', va='bottom',fontsize=42)
    ax.text(x, ymax+5, f'{ct} near v far\np={p:.2g}', ha='center', va='bottom',fontsize=12,
            rotation=45)
plt.savefig(os.path.join(os.path.join(savedst, 'near_to_far_dark_time_field_width.svg')))

#%%
# transition from 3 to 1 only
plt.rc('font', size=20)
rzdf = bigdf[(bigdf.epoch.str.contains('epoch1_rz3') | bigdf.epoch.str.contains('epoch2_rz1'))]
rzdf2 = bigdf[(bigdf.epoch.str.contains('epoch2_rz3') | bigdf.epoch.str.contains('epoch3_rz1'))]
rzdf3 = bigdf[(bigdf.epoch.str.contains('epoch3_rz3') | bigdf.epoch.str.contains('epoch4_rz1'))]
rzdf4 = bigdf[(bigdf.epoch.str.contains('epoch4_rz3') | bigdf.epoch.str.contains('epoch5_rz1'))]
rzdf = pd.concat([rzdf,rzdf2,rzdf3,rzdf4])
rzdf=rzdf.reset_index()
rzdf['epoch'] = [xx[-3:] for xx in rzdf.epoch.values]
# only get super close rz1 rewlocs
rzdf['rewloc_cm'] = [float(xx[-5:]) for xx in rzdf.rewloc.values]
rzdf = rzdf[((rzdf.epoch=='rz3')&(rzdf.rewloc_cm.values>210))|((rzdf.epoch=='rz1')&(rzdf.rewloc_cm.values<120))]
fig, ax = plt.subplots(figsize=(4,5))
sns.stripplot(x='epoch',y='width_cm',data=rzdf,hue='cell_type',alpha=0.05,dodge=True)
sns.boxplot(x='epoch',y='width_cm',data=rzdf,fill=False,hue='cell_type')
ax.spines[['top','right']].set_visible(False)
# per animal
s=10
hue_order = ['Pre','Post']
fig, ax = plt.subplots(figsize=(4,5))
anrzdf = rzdf.groupby(['animal', 'epoch', 'cell_type']).mean(numeric_only=True)
anrzdf=anrzdf.reset_index()
anrzdf = anrzdf.sort_values(by="cell_type", ascending=False)
anrzdf = anrzdf.sort_values(by="epoch", ascending=True)
# anrzdf=anrzdf[(anrzdf.animal!='e189') & (anrzdf.animal!='e190') & (anrzdf.animal!='e139')
#              & (anrzdf.animal!='e216')]
# anrzdf=anrzdf[(anrzdf.animal!='z16')]
order = ['rz3', 'rz1']
sns.stripplot(x='epoch',y='width_cm',data=anrzdf,hue='cell_type',alpha=0.7,dodge=True,s=s,
    palette='Dark2',hue_order=hue_order,order=order)
h_strip, l_strip = ax.get_legend_handles_labels()
sns.boxplot(x='epoch',y='width_cm',hue='cell_type',data=anrzdf,
        fill=False,palette='Dark2',hue_order=hue_order,order=order,
           showfliers=False)
ax.spines[['top','right']].set_visible(False)

ans = anrzdf.animal.unique()
ind = ['Pre','Post']
for j,ct in enumerate(ind):
    for i in range(len(ans)):    
        df_ = anrzdf[(anrzdf.animal==ans[i]) & (anrzdf.cell_type==ct)]
        df_ = df_.sort_values(by="epoch", ascending=False)
        color=sns.color_palette('Dark2')[j]
        ax = sns.lineplot(x=np.arange(len(df_.epoch.values))+(j*.2)-.1,y='width_cm',
        data=df_,
        errorbar=None, color=color, linewidth=2, alpha=0.3,ax=ax)

# 3) remove whatever legend was just created
ax.legend_.remove()
# 4) re-add only the stripplot legend, placing it outside
ax.legend(
    h_strip, l_strip,
    title='Cell Type',
    loc='upper left',
    bbox_to_anchor=(1.02, 1),
    borderaxespad=0.
)
ax.spines[['top','right']].set_visible(False)
ax.set_ylabel('Field width (cm)')
ax.set_xlabel('')
ax.set_xticklabels(['Far', 'Near'],rotation=45)
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM
palette=sns.color_palette('Dark2')
# anrzdf has columns: ['animal','epoch','cell_type','width_cm']
# make sure epoch & cell_type are categorical
anrzdf['epoch'] = anrzdf['epoch'].astype('category')
anrzdf['cell_type'] = anrzdf['cell_type'].astype('category')
aov = AnovaRM(
    data=anrzdf,
    depvar='width_cm',
    subject='animal',
    within=['epoch','cell_type']
).fit()
print(aov)

results = []
for ct in ['Pre','Post']:
    sub = anrzdf[anrzdf['cell_type']==ct]
    near = sub[sub['epoch']=='rz3']['width_cm']
    far  = sub[sub['epoch']=='rz1']['width_cm']
    t, p = scipy.stats.ttest_rel(near[~np.isnan(near)], far[~np.isnan(far)])
    results.append({'cell_type':ct, 't_stat':t, 'p_uncorrected':p})

posthoc = pd.DataFrame(results)
posthoc['p_bonferroni'] = np.minimum(posthoc['p_uncorrected']*len(posthoc), 1.0)
print(posthoc)
# mapping epoch→x-position (as you offset earlier)
x_map = {'Pre':0, 'Post':1}
offset = {'Pre':-0.1, 'Post':+0.1}

for _, row in posthoc.iterrows():
    ct = row['cell_type']
    p = row['p_bonferroni']
    stars = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
    x = x_map[ct]+offset[ct]  # test is Pre vs Post, annotate at Pre position
    # y-height: just above max for that group
    ymax = anrzdf[(anrzdf.cell_type==ct)]['width_cm'].max()
    ax.text(x, ymax+1, stars, ha='center', va='bottom',fontsize=42)
    ax.text(x, ymax+4, f'{ct} far v near\np={p:.2g}', ha='center', va='bottom',fontsize=12,
            rotation=45)
plt.savefig(os.path.join(os.path.join(savedst, 'far_to_near_dark_time_field_width.svg')))

#%%
# get corresponding licking behavior 
# scatterplot of lick distance vs. field width
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats

# --- Preprocess and merge bigdf (tuning) ---
def extract_epochs(df):
    epochs = [
        ('epoch1_rz1', 'epoch2_rz3'), ('epoch2_rz1', 'epoch3_rz3'),
        ('epoch3_rz1', 'epoch4_rz3'), ('epoch4_rz1', 'epoch5_rz3'),
        ('epoch1_rz3', 'epoch2_rz1'), ('epoch2_rz3', 'epoch3_rz1'),
        ('epoch3_rz3', 'epoch4_rz1'), ('epoch4_rz3', 'epoch5_rz1')
    ]
    return pd.concat([
        df[df.epoch.str.contains(e1) | df.epoch.str.contains(e2)]
        for e1, e2 in epochs
    ])

# Pre-process tuning data
rzdf = extract_epochs(bigdf)
rzdf = rzdf.groupby(['animal', 'day', 'epoch', 'cell_type']).median(numeric_only=True).reset_index()
rzdf = rzdf[rzdf.cell_type == 'Pre'].drop(columns=['cell_type'])
rzdf['day'] = rzdf['day'].astype(int)

# Pre-process licking data
lrzdf = extract_epochs(lickbigdf)
lrzdf = lrzdf.groupby(['animal', 'day', 'epoch']).median(numeric_only=True).reset_index()
lrzdf['day'] = lrzdf['day'].astype(int)

# Merge licking + tuning data
alldf = pd.merge(lrzdf, rzdf, on=['animal', 'day', 'epoch'], how='inner')

# Compute lick distance and clean
alldf['lick_dist'] = alldf['last_lick_loc_cm'] - alldf['first_lick_loc_cm']
alldf['width_cm'] = alldf['width_cm'].astype(float)
alldf = alldf.dropna(subset=['width_cm', 'lick_dist'])
alldf = alldf[alldf['width_cm'] > 0]

# --- Plot original + shuffled ---
fig, axes = plt.subplots(ncols = 2, figsize=(10,5))
ax=axes[0]
# Original regression and scatter
sns.regplot(x='lick_dist', y='width_cm', data=alldf,
            scatter=True, color='k', line_kws={'color': 'dodgerblue'},
            scatter_kws={'alpha': 0.5, 's': 50}, ax=ax,label='Real')

# --- Compute original r and p ---
r_obs, p_obs = scipy.stats.pearsonr(alldf['lick_dist'], alldf['width_cm'])

# --- Compute r and p for one shuffle ---
shuffled = alldf.copy()
shuffled['lick_dist'] = np.random.permutation(shuffled['lick_dist'].values)
r_shuff, p_shuff = scipy.stats.pearsonr(shuffled['lick_dist'], shuffled['width_cm'])

# Shuffled data overlay
sns.scatterplot(x='lick_dist', y='width_cm', data=shuffled,
                color='gray', alpha=0.4, s=50, ax=ax, label='Shuffle')

# Display r and p values
ax.text(0.05, 0.95,
        f'Original:\n r = {r_obs:.2g}, p = {p_obs:.3g}\n'
        f'Shuffled:\n r = {r_shuff:.2g}, p = {p_shuff:.3g}',
        transform=ax.transAxes, fontsize=14, verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.6))

# --- Null distribution ---
n_shuffles = 1000
r_null = []
for _ in range(n_shuffles):
    shuffled = alldf.copy()
    shuffled['lick_dist'] = np.random.permutation(shuffled['lick_dist'].values)
    r_shuff, _ = scipy.stats.pearsonr(shuffled['lick_dist'], shuffled['width_cm'])
    r_null.append(r_shuff)

# --- Empirical two-sided p-value ---
r_null = np.array(r_null)
p_empirical = np.mean(np.abs(r_null) >= np.abs(r_obs))

# Labels and cleanup
ax.set_xlabel('Lick Distance (cm)')
ax.set_ylabel('Field Width (cm)')
ax.spines[['top', 'right']].set_visible(False)
ax.legend()
# --- Plot null distribution ---
ax=axes[1]
sns.histplot(r_null, kde=True, color='gray', bins=30, ax=ax)
ax.axvline(r_obs, color='dodgerblue', linewidth = 2, label=f'Observed r = {r_obs:.2f}')
ax.set_title(f'Empirical p = {p_empirical:.4f}')
ax.set_xlabel('Shuffled Correlation (r)')
ax.set_ylabel('Frequency')
ax.legend()
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()

plt.savefig(os.path.join(os.path.join(savedst, 'lick_field_width.svg')))

#%%
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests

def get_transition_df(lickbigdf, rz_pairs):
    rzdf = pd.concat([
        lickbigdf[lickbigdf['epoch'].str.contains(pair[0]) | lickbigdf['epoch'].str.contains(pair[1])]
        for pair in rz_pairs
    ])
    rzdf = rzdf.reset_index(drop=True)
    rzdf['epoch'] = [e[-3:] for e in rzdf['epoch']]
    return rzdf

def plot_transition_panel(rzdf, axes, label_order, color='mediumvioletred'):
    # Mean per animal x epoch
    anrzdf = rzdf.groupby(['animal', 'epoch']).mean(numeric_only=True).reset_index()
    anrzdf = anrzdf.sort_values(by="epoch", ascending=(label_order == ['Near', 'Far']))

    metrics = [
        ('lick_cm_diff', '$\Delta$ Distance (cm) (first-last lick)'),
        ('lick_time_diff', '$\Delta$ Time (s) (first-last lick)'),
        ('lick_rate_hz', 'Lick rate (Hz)'),
        ('avg_velocity_cm_s', 'Velocity (cm/s)'),
    ]

    pvals = []

    for i, (metric, ylabel) in enumerate(metrics):
        ax = axes[i]
        sns.stripplot(x='epoch', y=metric, data=anrzdf, alpha=0.7, dodge=True, s=10, ax=ax, color=color)
        sns.boxplot(x='epoch', y=metric, data=anrzdf, fill=False, ax=ax, color=color, showfliers=False)
        ax.spines[['top','right']].set_visible(False)

        for animal in anrzdf['animal'].unique():
            df_ = anrzdf[anrzdf.animal == animal].sort_values('epoch', ascending=(label_order == ['Near', 'Far']))
            x_vals = np.arange(len(df_)) + (i * 0.1) - 0.1
            sns.lineplot(x=x_vals, y=df_[metric].values, ax=ax, color='dimgrey', linewidth=2, alpha=0.3)

        ax.set_ylabel(ylabel)
        ax.set_xticklabels(label_order)
        ax.set_xlabel('')

        # Stats
        df1 = anrzdf[anrzdf['epoch'] == 'rz1'].sort_values('animal')
        df2 = anrzdf[anrzdf['epoch'] == 'rz3'].sort_values('animal')
        assert all(df1['animal'].values == df2['animal'].values)
        p = stats.wilcoxon(df1[metric], df2[metric]).pvalue
        pvals.append(p)

    # Bonferroni correction
    reject, pvals_corr, _, _ = multipletests(pvals, alpha=0.05, method='bonferroni')

    def get_asterisks(p):
        return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'

    for i, ax in enumerate(axes):
        ymax = anrzdf[metrics[i][0]].max()
        y_text = ymax + 0.01 * abs(ymax)
        y_p = ymax - 0.005 * abs(ymax)
        ax.annotate(get_asterisks(pvals_corr[i]), xy=(.5, y_text), ha='center', fontsize=46, fontweight='bold')
        ax.annotate(f'{pvals_corr[i]:.2g}', xy=(.5, y_p), ha='center', fontsize=11)

    return anrzdf

# === USAGE ===

# Preprocessing
lickbigdf = pd.concat(lick_dfs)
lickbigdf['lick_cm_diff'] = lickbigdf['last_lick_loc_cm'] - lickbigdf['first_lick_loc_cm']
lickbigdf['lick_time_diff'] = lickbigdf['last_lick_time'] - lickbigdf['first_lick_time']

# Plot setup
fig, axes_all = plt.subplots(nrows=2, ncols=4, figsize=(11, 10))
plt.tight_layout()

# Plot 1→3
rz_pairs_13 = [('epoch1_rz1', 'epoch2_rz3'), ('epoch2_rz1', 'epoch3_rz3'),
               ('epoch3_rz1', 'epoch4_rz3'), ('epoch4_rz1', 'epoch5_rz3')]
rzdf13 = get_transition_df(lickbigdf, rz_pairs_13)
plot_transition_panel(rzdf13, axes_all[0], label_order=['Near', 'Far'])

# Plot 3→1
rz_pairs_31 = [('epoch1_rz3', 'epoch2_rz1'), ('epoch2_rz3', 'epoch3_rz1'),
               ('epoch3_rz3', 'epoch4_rz1'), ('epoch4_rz3', 'epoch5_rz1')]
rzdf31 = get_transition_df(lickbigdf, rz_pairs_31)
plot_transition_panel(rzdf31, axes_all[1], label_order=['Far', 'Near'],color='mediumslateblue')
fig.suptitle('Last 8 correct trials')
plt.savefig(os.path.join(os.path.join(savedst, 'transition_licks_velocity.svg')))


#%%
ii=152 # near to far
ii=132 # far to near
# behavior
day = conddf.days.values[ii]
animal = conddf.animals.values[ii]
pln=0
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
ypos = fall['ybinned'][0]/scalingf
track_length=180/scalingf    
forwardvel = fall['forwardvel'][0]    
changeRewLoc = np.hstack(fall['changeRewLoc'])
trialnum=fall['trialnum'][0]
rewards = fall['rewards'][0]
lick=fall['licks'][0]
lick[ypos<3]=0
# set vars
eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
rz = get_rewzones(rewlocs,1/scalingf)       

eps = np.where(changeRewLoc)[0]
rew = (rewards==1).astype(int)
mask = np.array([True if xx>10 and xx<28 else False for xx in trialnum])
mask = np.zeros_like(trialnum).astype(bool)
# mask[eps[0]+3000:eps[1]+12000]=True # near to far
mask[eps[0]+7000:eps[1]+18000]=True # far to near
import matplotlib.patches as patches
fig, ax = plt.subplots(figsize=(20,5))
ax.plot(ypos[mask],zorder=1)
ax.scatter(np.where(lick[mask])[0], ypos[mask][np.where(lick[mask])[0]], color='k',
        zorder=2)
ax.scatter(np.where(rew[mask])[0], ypos[mask][np.where(rew[mask])[0]], color='cyan',
    zorder=2)
# ax.add_patch(
# patches.Rectangle(
#     xy=(0,newrewloc-10),  # point of origin.
#     width=len(ypos[mask]), height=20, linewidth=1, # width is s
#     color='slategray', alpha=0.3))
ax.add_patch(
patches.Rectangle(
    xy=(0,(changeRewLoc[eps][0]/scalingf)-10),  # point of origin.
    width=len(ypos[mask]), height=20, linewidth=1, # width is s
    color='slategray', alpha=0.3))

ax.set_ylim([0,270])
ax.spines[['top','right']].set_visible(False)
ax.set_title(f'{day}')
# plt.savefig(os.path.join(savedst, f'{animal}_day{day:03d}_behavior.svg'),bbox_inches='tight')

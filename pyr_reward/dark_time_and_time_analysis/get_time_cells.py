
"""
zahra
get tuning curves with dark time
oct 2025- for figure!!!
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
# plt.rc('font', size=20)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.memory.behavior import consecutive_stretch
from projects.pyr_reward.placecell import get_tuning_curve, calc_COM_EH, make_tuning_curves_radians_by_trialtype,\
        make_time_tuning_curves, make_tuning_curves_radians_trial_by_trial, \
        make_tuning_curves_time_trial_by_trial
from projects.pyr_reward.rewardcell import get_radian_position,\
    get_radian_position_first_lick_after_rew, get_rewzones, get_goal_cells, goal_cell_shuffle
from projects.opto.behavior.behavior import get_success_failure_trials
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_rewardcentric_all.p"
with open(saveddataset, "rb") as fp: #unpickle
        radian_alignment_saved = pickle.load(fp)
savepth = os.path.join(savedst, 'time_tuning.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)

#%%
####################################### RUN CODE #######################################
# initialize var
# radian_alignment_saved = {} # overwrite
p_goal_cells=[]
p_goal_cells_dt = []
goal_cells_iind=[]
pvals = []
bins = 90
goal_window_cm=20
datadct = {}
goal_cell_null= []
perms = []
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
        animals_w_2_planes = ['z9', 'e190']
        animals_w_3_planes = ['e139', 'e145']
        # framerate
        fr = 31.25
        if animal in animals_w_2_planes: fr/2
        if animal in animals_w_3_planes: fr/3
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
        rz = get_rewzones(rewlocs,1/scalingf)       
        # get average success rate
        rates = []
        for ep in range(len(eps)-1):
            eprng = range(eps[ep],eps[ep+1])
            success, fail, str_trials, ftr_trials, ttr, \
            total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
            rates.append(success/total_trials)
        rate=np.nanmean(np.array(rates))
        rad = get_radian_position_first_lick_after_rew(eps, ybinned, licks, rewards, rewsize,rewlocs,trialnum, track_length) # get radian coordinates

        # added to get anatomical info
        # takes time
        fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
        Fc3 = fall_fc3['Fc3']
        dFF = fall_fc3['dFF']
        Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
        dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]
        skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
        Fc3 = Fc3[:, skew>2] # only keep cells with skew greateer than 2
        # tc w/ time
        tcs_correct_time, coms_correct_time, tcs_fail_time, coms_fail_time, trial_times = make_time_tuning_curves(eps, 
                time, Fc3, trialnum, rewards, licks, ybinned, rewlocs, rewsize,
                lasttr=8, bins=bins, velocity_filter=False)
        # time bin is roughly (16/90) or 170 ms
        # test
        # max time trials
        # max_trial_times = [tr.shape[1]/31.25 for tr in trial_times]
        # fig, axes = plt.subplots(ncols = len(tcs_correct_time),figsize=(12,10))
        # for ep in range(len(coms_correct_time)):
        #     ax=axes[ep]
        #     ax.imshow(tcs_correct_time[ep][np.argsort(coms_correct_time[0])]**.3,aspect='auto')
        #     ax.set_title(f'Rew. Loc. {rewlocs[ep]} cm\n Max trial time {max_trial_times[ep]:.1f}s',fontsize=12)
        #     ax.axvline(int(bins/2),color='w',linestyle='--')
        # fig.tight_layout()
        # normal tc
        track_length_rad = track_length*(2*np.pi/track_length)
        bin_size=track_length_rad/bins 
        tcs_correct, coms_correct, tcs_fail, coms_fail = make_tuning_curves_radians_by_trialtype(eps,rewlocs,ybinned,rad,Fc3,trialnum,rewards,forwardvel,rewsize,bin_size)  
        goal_window = goal_window_cm*(2*np.pi/track_length) # cm converted to rad
        goal_cells, com_goal_postrew, perm, rz_perm = get_goal_cells(rz, goal_window, coms_correct, cell_type = 'all')
        goal_cells_p_per_comparison = [len(xx)/len(coms_correct[0]) for xx in com_goal_postrew]           
        # get goal cells aligned to time
        # expand window to 7.5% of trial time?
        # roughly equivalent to 20 cm of 270 cm
        max_trial_times = np.nanmax(np.array([tr.shape[1]/fr for tr in trial_times]))
        time_window = max_trial_times*.074 #s
        time_window = time_window*(2*np.pi/max_trial_times) # s converted to rad
        print(f'Window for time cells: {time_window:.2f}')
        goal_cells_time, com_goal_postrew_time, perm_time, rz_perm_time = get_goal_cells(rz, goal_window, coms_correct_time, cell_type = 'all')
        goal_cells_p_per_comparison_time = [len(xx)/len(coms_correct[0]) for xx in com_goal_postrew_time]            
        # eg cell         
        # for cellid in goal_cells:
        #     fig, axes = plt.subplots(nrows = 2,figsize=(3,5),sharex=True,sharey=True)
        #     for ep in range(len(tcs_correct)-1):            
        #         axes[0].plot(tcs_correct[ep,cellid,:])
        #         axes[0].set_title('Track aligned')
        #         axes[0].axvline(int(bins/2),color='k',linestyle='--')
        #         axes[1].plot(tcs_correct_time[ep,cellid,:])
        #         axes[1].set_title('Time aligned')
        #         axes[1].axvline(int(bins/2),color='k',linestyle='--')
        #     fig.suptitle(f'{animal}, {day}, Reward cells')
        # pdf.savefig(fig)        
        # for cellid in goal_cells_time:
        #     fig, axes = plt.subplots(nrows = 2,figsize=(3,5),sharex=True,sharey=True)
        #     for ep in range(len(tcs_correct)-1):            
        #         axes[0].plot(tcs_correct[ep,cellid,:])
        #         axes[0].set_title('Track aligned')
        #         axes[0].axvline(int(bins/2),color='k',linestyle='--')
        #         axes[1].plot(tcs_correct_time[ep,cellid,:])
        #         axes[1].set_title('Time aligned')
        #         axes[1].axvline(int(bins/2),color='k',linestyle='--')
        #     fig.suptitle(f'{animal}, {day}, Time cells')
        # pdf.savefig(fig)
        # test time cells raster
        # lick=licks
        # trialstates, licks_all, tcs, coms=make_tuning_curves_time_trial_by_trial(eps, 
        #         rewlocs, lick, ybinned, time, Fc3[:,goal_cells_time], trialnum, rewards, forwardvel, rewsize, bin_size, lasttr=8, bins=90)
        # trialstates, licks_all_dist, tcs_dist, coms_dist=make_tuning_curves_radians_trial_by_trial(eps,rewlocs,lick,ybinned,rad,Fc3[:,goal_cells_time],trialnum,rewards,forwardvel,rewsize,bin_size)
#         from mpl_toolkits.axes_grid1 import make_axes_locatable
#         from matplotlib.colors import ListedColormap, BoundaryNorm
#         cmap = ListedColormap(['gray', 'red', 'lime'])  # -1 = gray, 0 = red, 1 = green
#         bounds = [-1.5, -0.5, 0.5, 1.5]  # boundaries between categories
#         norm = BoundaryNorm(bounds, cmap.N)

#         plt.close('all')
#         for cll in range(len(goal_cells_time)):
#                 fig, axes_all = plt.subplots(nrows=3, ncols=len(tcs_correct),figsize=(10, 6), height_ratios=[3, 1, 1],sharex=True)
#                 # Normalize activity by row
#                 # per epoch
#                 for ep in range(len(tcs_correct)):
#                         axes=axes_all[:,ep]
#                         data = tcs[ep][cll]
#                         norm_data = (data - np.nanmin(data, axis=1, keepdims=True)) / \
#                                         (np.nanmax(data, axis=1, keepdims=True) - np.nanmin(data, axis=1, keepdims=True) + 1e-10)

#                         # Add divider to move colorbar outside of axes[0]
#                         divider = make_axes_locatable(axes[0])
#                         cax = divider.append_axes("right", size="5%", pad=0.05)
                        
#                         im = axes[0].imshow(tcs[ep][cll], aspect='auto')
#                         fig.colorbar(im, cax=cax, orientation='vertical', label='Activity')
#                         trial_mask = np.array(trialstates[ep])  # shape (n_trials,)
#                         trial_mask_2d = trial_mask[:, np.newaxis] * np.ones((1, data.shape[1]))  # shape (n_trials, n_timebins)
#                         axes[0].imshow(trial_mask_2d, cmap=cmap, norm=norm, aspect='auto', alpha=0.3)
#                         axes[2].plot(tcs_correct_time[:, goal_cells_time[cll]].T)                
#                         axes[1].plot(np.nanmean(licks_all[ep], axis=0), color='dimgrey')                
#                         center_bin = 45
#                         for ax in axes:
#                                 ax.axvline(center_bin, color='k')
#                         axes[0].axvline(center_bin, color='w')  # for contrast on heatmap
#                         axes[2].set_title('Time tuning curve')
#                         axes[0].set_title(f'{animal}, {day}, cell: {goal_cells_time[cll]}')        
#                         min_tr = np.nanmin(trial_times[ep])
#                         max_tr = np.nanmax(trial_times[ep])
#                         ticks = [0, center_bin, bins - 1]
#                         tick_labels = [f"{min_tr:.1f}", "0", f"{max_tr:.1f}"]
#                         # Set x-ticks on both the heatmap and the licks plot
#                         for ax in [axes[0], axes[1]]:
#                                 ax.set_xticks(ticks)
#                                 ax.set_xticklabels(tick_labels)
#                                 ax.set_xlabel('Time (s)')
#                         axes[1].set_ylabel('Licks')
#                 plt.tight_layout()
# #%%                
#         # plot the distance tuning
#         for cll in range(len(goal_cells_time)):
#                 fig, axes_all = plt.subplots(nrows=3, ncols=len(tcs_correct),figsize=(10, 6), height_ratios=[3, 1, 1],sharex=True)
#                 # Normalize activity by row
#                 # per epoch
#                 for ep in range(len(tcs_correct)):
#                         axes=axes_all[:,ep]
#                         data = tcs_dist[ep][cll]
#                         norm_data = (data - np.nanmin(data, axis=1, keepdims=True)) / \
#                                         (np.nanmax(data, axis=1, keepdims=True) - np.nanmin(data, axis=1, keepdims=True) + 1e-10)

#                         # Add divider to move colorbar outside of axes[0]
#                         divider = make_axes_locatable(axes[0])
#                         cax = divider.append_axes("right", size="5%", pad=0.05)
                        
#                         im = axes[0].imshow(tcs_dist[ep][cll], aspect='auto')
#                         fig.colorbar(im, cax=cax, orientation='vertical', label='Activity')
#                         trial_mask = np.array(trialstates[ep])  # shape (n_trials,)
#                         trial_mask_2d = trial_mask[:, np.newaxis] * np.ones((1, data.shape[1]))  # shape (n_trials, n_timebins)
#                         axes[0].imshow(trial_mask_2d, cmap=cmap, norm=norm, aspect='auto', alpha=0.3)

#                         axes[2].plot(tcs_correct[:, goal_cells_time[cll]].T)                
#                         axes[1].plot(np.nanmean(licks_all[ep], axis=0), color='dimgrey')                
#                         center_bin = 45
#                         for ax in axes:
#                                 ax.axvline(center_bin, color='k')
#                         axes[0].axvline(center_bin, color='w')  # for contrast on heatmap
#                         axes[2].set_title('Distance tuning curve')
#                         axes[0].set_title(f'{animal}, {day}, cell: {goal_cells_time[cll]}')        
#                         min_tr = np.nanmin(trial_times[ep])
#                         max_tr = np.nanmax(trial_times[ep])
#                         ticks = [0, center_bin, bins - 1]
#                         tick_labels = [f"{min_tr:.1f}", "0", f"{max_tr:.1f}"]
#                         # Set x-ticks on both the heatmap and the licks plot
#                         axes[2].set_xticks([0, center_bin, bins - 1])
#                         axes[2].set_xticklabels(["$-\\pi$", "0", "$\\pi$"])
#                         axes[2].set_xlabel('Reward-relative distance')
#                         axes[1].set_ylabel('Licks')
#                 plt.tight_layout()

        #only get perms with non zero cells
        # get per comparison and also across epochs
        p_goal_cells.append([len(goal_cells)/len(coms_correct[0]),goal_cells_p_per_comparison])
        p_goal_cells_dt.append([len(goal_cells_time)/len(coms_correct_time[0]), goal_cells_p_per_comparison_time])
        goal_cells_iind.append([goal_cells, goal_cells_time])
        # save perm
        perms.append([[perm, rz_perm],
            [perm_time, rz_perm_time]])
        print(f'Goal cells w/o dt: {goal_cells}\n\
            Time cells: {goal_cells_time}')
        # shuffle
        num_iterations=1000
        goal_cell_shuf_ps_per_comp, goal_cell_shuf_ps, shuffled_dist=goal_cell_shuffle(coms_correct, goal_window,\
                    perm,num_iterations = num_iterations)
        goal_cell_shuf_ps_per_comp_time, goal_cell_shuf_ps_time, shuffled_dist_time=goal_cell_shuffle(coms_correct_time, \
                time_window, perm_time, num_iterations = num_iterations)
        goal_cell_shuf_ps_per_comp_av = np.nanmedian(goal_cell_shuf_ps_per_comp,axis=0)        
        goal_cell_shuf_ps_av = np.nanmedian(np.array(goal_cell_shuf_ps))
        goal_cell_p=len(goal_cells)/len(coms_correct[0]) 
        p_value = sum(shuffled_dist>goal_cell_p)/num_iterations
        # dark time
        goal_cell_shuf_ps_per_comp_av_time = np.nanmedian(goal_cell_shuf_ps_per_comp_time,axis=0)        
        goal_cell_shuf_ps_av_time = np.nanmedian(np.array(goal_cell_shuf_ps_time))
        goal_cell_p_time=len(goal_cells_time)/len(coms_correct[0]) 
        p_value_time = sum(shuffled_dist_time>goal_cell_p_time)/num_iterations
        print(f'{animal}, day {day}: significant goal cells proportion p-value: {p_value} v time aligned {p_value_time}')
        goal_cell_null.append([[goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av],
                        [goal_cell_shuf_ps_per_comp_av_time,goal_cell_shuf_ps_av_time]])
        pvals.append([p_value,p_value_time]); 
        datadct[f'{animal}_{day:03d}_index{ii:03d}'] = [tcs_correct, coms_correct, tcs_fail, coms_fail,
                tcs_correct_time, coms_correct_time, tcs_fail_time, coms_fail_time]

pdf.close()
####################################### RUN CODE #######################################
#%%
plt.rc('font', size=20)          # controls default text sizes
# plot goal cells across epochs
inds = [int(xx[-3:]) for xx in datadct.keys()]
df = conddf.copy()
df = df[((df.animals!='e217')) & (df.optoep<2) & (df.index.isin(inds))]
df['num_epochs'] = [len(xx[1]) for k,xx in datadct.items()]
df['goal_cell_prop'] =  [xx[0] for xx in p_goal_cells]
df['opto'] = df.optoep.values>1
df['day'] = df.days
df['session_num_opto'] = np.concatenate([[xx-df[df.animals==an].days.values[0] for xx in df[df.animals==an].days.values] for an in np.unique(df.animals.values)])
df['session_num'] = np.concatenate([[ii for ii,xx in enumerate(df[df.animals==an].days.values)] for an in np.unique(df.animals.values)])
df['condition'] = ['vip' if xx=='vip' else 'ctrl' for xx in df.in_type.values]
df['p_value'] = [xx[0] for xx in pvals]
df['goal_cell_prop_shuffle'] = [xx[0][1] for xx in goal_cell_null]

fig,ax = plt.subplots(figsize=(5,5))
ax = sns.histplot(data = df.loc[df.opto==False], x='p_value', 
                hue='animals', bins=40)
ax.spines[['top','right']].set_visible(False)
ax.axvline(x=0.05, color='k', linestyle='--')
sessions_sig = sum(df.loc[df.opto==False,'p_value'].values<0.05)/len(df.loc[df.opto==False])
ax.set_title(f'{(sessions_sig*100):.2f}% of sessions are significant\n Reward cells w/o dark time')
ax.set_xlabel('P-value')
ax.set_ylabel('Sessions')
#%%
# w/ dark time
df_dt = conddf.copy()
df_dt = df_dt[((df_dt.animals!='e217')) & (df_dt.optoep<2) & (df_dt.index.isin(inds))]
df_dt['num_epochs'] = [len(xx[1]) for k,xx in datadct.items()]
df_dt['goal_cell_prop'] = [xx[0] for xx in p_goal_cells_dt]
df_dt['opto'] = df_dt.optoep.values>1
df_dt['day'] = df_dt.days
df_dt['session_num_opto'] = np.concatenate([[xx-df_dt[df_dt.animals==an].days.values[0] for xx in df_dt[df_dt.animals==an].days.values] for an in np.unique(df_dt.animals.values)])
df_dt['session_num'] = np.concatenate([[ii for ii,xx in enumerate(df_dt[df_dt.animals==an].days.values)] for an in np.unique(df_dt.animals.values)])
df_dt['condition'] = ['vip' if xx=='vip' else 'ctrl' for xx in df_dt.in_type.values]
df_dt['p_value'] = [xx[1] for xx in pvals]
df_dt['goal_cell_prop_shuffle'] = [xx[1][1] for xx in goal_cell_null]

fig,ax = plt.subplots(figsize=(5,5))
ax = sns.histplot(data = df_dt.loc[df_dt.opto==False], x='p_value', 
                hue='animals', bins=40)
ax.spines[['top','right']].set_visible(False)
ax.axvline(x=0.05, color='k', linestyle='--')
sessions_sig = sum(df_dt.loc[df_dt.opto==False,'p_value'].values<0.05)/len(df_dt.loc[df_dt.opto==False])
ax.set_title(f'{(sessions_sig*100):.2f}% of sessions are significant\n Time cells')
ax.set_xlabel('P-value')
ax.set_ylabel('Sessions')
#%%
# number of epochs vs. reward cell prop    
fig,axes = plt.subplots(ncols = 2, figsize=(6,5),sharex=True,sharey=True)
ax = axes[0]
df_plt = df[df.num_epochs<5]
df_plt['goal_cell_prop']=df_plt['goal_cell_prop']*100
df_plt['goal_cell_prop_shuffle']=df_plt['goal_cell_prop_shuffle']*100
# av across mice
df_plt = df_plt.groupby(['animals','num_epochs']).mean(numeric_only=True)
sns.stripplot(x='num_epochs', y='goal_cell_prop',
        data=df_plt,color='k',alpha=0.7,
        s=10,ax=ax)
sns.barplot(x='num_epochs', y='goal_cell_prop',
        data=df_plt,
        fill=False,ax=ax, color='k', errorbar='se')
# bar plot of shuffle instead
ax = sns.barplot(data=df_plt, # correct shift
        x='num_epochs', y='goal_cell_prop_shuffle',color='grey', 
        label='shuffle', alpha=0.5, err_kws={'color': 'grey'},errorbar=None,ax=ax)
ax.spines[['top','right']].set_visible(False)
ax.get_legend().remove()
eps = [2,3,4]
y = 40
pshift=2
fs=36
ax.set_title('Reward-distance cells\n')
ax.set_ylabel('% Cells') 
ax.set_xlabel('') 

# time
ax = axes[1]
df_plt_dt = df_dt[df_dt.num_epochs<5]
# av across mice
df_plt_dt['goal_cell_prop']=df_plt_dt['goal_cell_prop']*100
df_plt_dt['goal_cell_prop_shuffle']=df_plt_dt['goal_cell_prop_shuffle']*100

df_plt_dt = df_plt_dt.groupby(['animals','num_epochs']).mean(numeric_only=True)
sns.stripplot(x='num_epochs', y='goal_cell_prop',
       data=df_plt_dt,color='k',alpha=0.7,
        s=10,ax=ax)
sns.barplot(x='num_epochs', y='goal_cell_prop',
        data=df_plt_dt,
        fill=False,ax=ax, color='k', errorbar='se')
# bar plot of shuffle instead
ax = sns.barplot(data=df_plt_dt, # correct shift
        x='num_epochs', y='goal_cell_prop_shuffle',color='grey', 
        label='shuffle', alpha=0.5, err_kws={'color': 'grey'},errorbar=None,ax=ax)
ax.spines[['top','right']].set_visible(False)
ax.get_legend().remove()
ax.set_title('Time cells\n')
ax.set_xlabel('# of rew. loc. switches') 
# eps = [2,3,4]
# for ii,ep in enumerate(eps):
#         # rewprop = df_plt.loc[(df_plt.num_epochs==ep), 'goal_cell_prop']
#         rewprop = df_plt.loc[(df_plt.index.get_level_values('num_epochs')==ep), 'goal_cell_prop']
#         shufprop = df_plt.loc[(df_plt.index.get_level_values('num_epochs')==ep), 'goal_cell_prop_shuffle']
#         t,pval = scipy.stats.wilcoxon(rewprop, shufprop)
#         print(f'{ep} epochs, pval: {pval}')
#         # statistical annotation        
#         if pval < 0.001:
#                 ax.text(ii, y, "***", ha='center', fontsize=fs)
#         elif pval < 0.01:
#                 ax.text(ii, y, "**", ha='center', fontsize=fs)
#         elif pval < 0.05:
#                 ax.text(ii, y, "*", ha='center', fontsize=fs)


#%%
# include all comparisons 
df_perms = pd.DataFrame()
goal_cell_perm = [xx[1] for xx in p_goal_cells]
goal_cell_perm_shuf = [xx[0][0][~np.isnan(xx[0][0])] for xx in goal_cell_null]
df_perms['goal_cell_prop'] = np.concatenate(goal_cell_perm)
df_perms['goal_cell_prop'] =df_perms['goal_cell_prop'] *100
df_perms['goal_cell_prop_shuffle'] = np.concatenate(goal_cell_perm_shuf)
df_perms['goal_cell_prop_shuffle']=df_perms['goal_cell_prop_shuffle']*100
df_perm_animals = [[xx]*len(goal_cell_perm[ii]) for ii,xx in enumerate(df.animals.values)]
df_perms['animals'] = np.concatenate(df_perm_animals)
df_perm_days = [[xx]*len(goal_cell_perm[ii]) for ii,xx in enumerate(df.session_num.values)]
df_perms['session_num'] = np.concatenate(df_perm_days)

# take a mean of all epoch comparisons
df_perms['num_epochs'] = [2]*len(df_perms)
df_permsav2 = df_perms.groupby(['animals', 'num_epochs']).mean(numeric_only=True)

# compare to shuffle
df_plt2 = pd.concat([df_permsav2,df_plt])
# df_plt2 = df_plt2[df_plt2.index.get_level_values('animals')!='e189']
df_plt2 = df_plt2[df_plt2.index.get_level_values('num_epochs')<5]
df_plt2 = df_plt2.groupby(['animals', 'num_epochs']).mean(numeric_only=True)
df_plt2=df_plt2.reset_index()

# v dark time
df_perms = pd.DataFrame()
goal_cell_perm = [xx[1] for xx in p_goal_cells_dt]
goal_cell_perm_shuf = [xx[1][0][~np.isnan(xx[1][0])] for xx in goal_cell_null]
df_perms['goal_cell_prop'] = np.concatenate(goal_cell_perm)
df_perms['goal_cell_prop'] =df_perms['goal_cell_prop'] *100
df_perms['goal_cell_prop_shuffle'] = np.concatenate(goal_cell_perm_shuf)
df_perms['goal_cell_prop_shuffle']=df_perms['goal_cell_prop_shuffle']*100
df_perm_animals = [[xx]*len(goal_cell_perm[ii]) for ii,xx in enumerate(df.animals.values)]
df_perms['animals'] = np.concatenate(df_perm_animals)
df_perm_days = [[xx]*len(goal_cell_perm[ii]) for ii,xx in enumerate(df.session_num.values)]
df_perms['session_num'] = np.concatenate(df_perm_days)

# take a mean of all epoch comparisons
df_perms['num_epochs'] = [2]*len(df_perms)
df_permsav2 = df_perms.groupby(['animals', 'num_epochs']).mean(numeric_only=True)

# compare to shuffle
df_plt2_dt = pd.concat([df_permsav2,df_plt_dt])
# df_plt2_dt = df_plt2_dt[df_plt2_dt.index.get_level_values('animals')!='e189']
df_plt2_dt = df_plt2_dt[df_plt2_dt.index.get_level_values('num_epochs')<5]
df_plt2_dt = df_plt2_dt.groupby(['animals', 'num_epochs']).mean(numeric_only=True)
df_plt2_dt=df_plt2_dt.reset_index()
#%%
# number of epochs vs. reward cell prop incl combinations    
fig,axes = plt.subplots(figsize=(7,5),ncols=2,sharex=True)
ax=axes[0]
# av across mice
sns.stripplot(x='num_epochs', y='goal_cell_prop',color='k',
        data=df_plt2,s=10,alpha=0.7,ax=ax)
sns.barplot(x='num_epochs', y='goal_cell_prop',
        data=df_plt2,
        fill=False,ax=ax, color='k', errorbar='se')
# bar plot of shuffle instead
ax = sns.barplot(data=df_plt2, # correct shift
        x='num_epochs', y='goal_cell_prop_shuffle',color='grey', 
        label='shuffle', alpha=0.5, err_kws={'color': 'grey'},errorbar=None,ax=ax)

ax.spines[['top','right']].set_visible(False)
ax.legend()#.set_visible(False)
# make lines
alpha=0.5
ans = df_plt2.animals.unique()
for i in range(len(ans)):
    sns.lineplot(x=df_plt2.num_epochs-2, y='goal_cell_prop', 
    data=df_plt2[df_plt2.animals==ans[i]],
    errorbar=None, color='dimgray', linewidth=2, alpha=alpha,ax=ax)
ax.set_xlabel('')
ax.set_ylabel('Reward cell % ')
ax.set_ylim([0,40])
eps = [2,3,4]
y = 37
pshift = 4
fs=36
pvals=[]
for ii,ep in enumerate(eps):
        rewprop = df_plt2.loc[(df_plt2.num_epochs==ep), 'goal_cell_prop']
        shufprop = df_plt2.loc[(df_plt2.num_epochs==ep), 'goal_cell_prop_shuffle']
        t,pval = scipy.stats.wilcoxon(rewprop, shufprop)
        pvals.append(pval)

ax=axes[1]
# av across mice
color='cadetblue'
sns.stripplot(x='num_epochs', y='goal_cell_prop',color=color,
        data=df_plt2_dt,s=10,alpha=0.7,ax=ax)
sns.barplot(x='num_epochs', y='goal_cell_prop',
        data=df_plt2_dt,
        fill=False,ax=ax, color=color, errorbar='se')
# bar plot of shuffle instead
ax = sns.barplot(data=df_plt2_dt, # correct shift
        x='num_epochs', y='goal_cell_prop_shuffle',color='grey', 
        label='shuffle', alpha=0.5, err_kws={'color': 'grey'},errorbar=None,ax=ax)

ax.spines[['top','right']].set_visible(False)
ax.legend()#.set_visible(False)
# make lines
alpha=0.5
ans = df_plt2.animals.unique()
for i in range(len(ans)):
    sns.lineplot(x=df_plt2_dt.num_epochs-2, y='goal_cell_prop', 
    data=df_plt2_dt[df_plt2_dt.animals==ans[i]],
    errorbar=None, color='dimgray', linewidth=2, alpha=alpha,ax=ax)
ax.set_xlabel('# of rew. loc. switches')
ax.set_ylabel('Time cell % ')

eps = [2,3,4]
y = 37
pshift = 4
fs=36
for ii,ep in enumerate(eps):
        rewprop = df_plt2_dt.loc[(df_plt2_dt.num_epochs==ep), 'goal_cell_prop']
        shufprop = df_plt2_dt.loc[(df_plt2_dt.num_epochs==ep), 'goal_cell_prop_shuffle']
        t,pval = scipy.stats.wilcoxon(rewprop, shufprop)
        pvals.append(pval)
from statsmodels.stats.multitest import multipletests
corrected, pvals_corrected, _, _ = multipletests(pvals, method='bonferroni')

for ii,pval in enumerate(pvals_corrected):                
        if ii<3:
                ax=axes[0]
        else:
                ax=axes[1]
                ii=ii-3
        # statistical annotation        
        if pval < 0.001:
                ax.text(ii, y, "***", ha='center', fontsize=fs)
        elif pval < 0.01:
                ax.text(ii, y, "**", ha='center', fontsize=fs)
        elif pval < 0.05:
                ax.text(ii, y, "*", ha='center', fontsize=fs)
        ax.text(ii-0.5, y+pshift, f'p={pval:.3g}',fontsize=10,rotation=45)

        # statistical annotation        
        if pval < 0.001:
                ax.text(ii, y, "***", ha='center', fontsize=fs)
        elif pval < 0.01:
                ax.text(ii, y, "**", ha='center', fontsize=fs)
        elif pval < 0.05:
                ax.text(ii, y, "*", ha='center', fontsize=fs)
        ax.text(ii-0.5, y+pshift, f'p={pval:.3g}',fontsize=10,rotation=45)
ax.set_ylim([0,40])
fig.tight_layout()
plt.savefig(os.path.join(savedst, 'time_cell_prop_per_an.svg'), 
        bbox_inches='tight')
#%%
# compare reward v. time
# --- Merge DataFrames ---
df_compare = pd.merge(
    df_plt2[['animals', 'num_epochs', 'goal_cell_prop']],
    df_plt2_dt[['animals', 'num_epochs', 'goal_cell_prop']],
    on=['animals', 'num_epochs'],
    suffixes=('_reward', '_time')
)

# --- Plot comparison ---
fig, ax = plt.subplots(figsize=(4, 5))
sns.pointplot(
    data=pd.melt(df_compare, id_vars=['animals', 'num_epochs'], 
                 value_vars=['goal_cell_prop_reward', 'goal_cell_prop_time'], 
                 var_name='cell_type', value_name='cell_prop'),
    x='num_epochs', y='cell_prop', hue='cell_type', 
    dodge=0.3, err_kws={'linewidth': 2}, capsize=0.1, palette=['black', 'cadetblue'],
    ax=ax
)
ax.set_ylabel('% of cells')
ax.set_xlabel('# of rew. location switches')
ax.spines[['top','right']].set_visible(False)

# --- Stats per epoch ---
epochs = sorted(df_compare['num_epochs'].unique())
ymax = df_compare[['goal_cell_prop_reward','goal_cell_prop_time']].max().max()
y = 30
offset=3
pvals = []

for i, ep in enumerate(epochs):
    sub = df_compare[df_compare.num_epochs == ep]
    stat, pval = scipy.stats.ttest_rel(sub['goal_cell_prop_reward'], sub['goal_cell_prop_time'])
    pvals.append(pval)

reject, pvals_corrected, _, _ = multipletests(pvals, method='bonferroni')
for i, ep in enumerate(epochs):
        pval = pvals_corrected[i]
        # Annotate plot
        if pval < 0.001:
                stars = '***'
        elif pval < 0.01:
                stars = '**'
        elif pval < 0.05:
                stars = '*'
        else:
                stars = ''
        ax.text(i, y, stars, ha='center', fontsize=46)
        ax.text(i, y+offset, f'{pval:.3g}', ha='center', rotation=45,fontsize=12)
ax.set_title('Reward vs Time cell % by epoch\n\n')
ax.legend(
    loc='center left',
    bbox_to_anchor=(1.0, 0.5),
    title='Cell Type'
)
# plt.tight_layout()
plt.show()

#%%
# overlap of reward and time cells

goal_cell_ind = [xx[0] for xx in goal_cells_iind]
time_cell_ind = [xx[1] for xx in goal_cells_iind]

# find % of time cells that are goal cells
time_reward = [len([xx for xx in tc if xx in goal_cell_ind[ii]])/len(tc) if len(tc)>0 else np.nan for ii,tc in enumerate(time_cell_ind)]
df = conddf.copy()
df = df[((df.animals!='e217')) & (df.optoep<2) & (df.index.isin(inds))]
df['num_epochs'] = [len(xx[1]) for k,xx in datadct.items()]
df['goal_cell_prop'] =  [xx[0] for xx in p_goal_cells]
df['opto'] = df.optoep.values>1
df['day'] = df.days
df['session_num_opto'] = np.concatenate([[xx-df[df.animals==an].days.values[0] for xx in df[df.animals==an].days.values] for an in np.unique(df.animals.values)])
df['session_num'] = np.concatenate([[ii for ii,xx in enumerate(df[df.animals==an].days.values)] for an in np.unique(df.animals.values)])
df['condition'] = ['vip' if xx=='vip' else 'ctrl' for xx in df.in_type.values]
df['goal_cell_prop_shuffle'] = [xx[0][1] for xx in goal_cell_null]
df['p_time_reward'] = time_reward
df['p_time_reward'] = df['p_time_reward']*100
fig,ax=plt.subplots()
sns.histplot(x='p_time_reward',hue='animals',data=df,bins=20)
ax.set_xlabel('% of time cells that are reward cells')
ax.set_ylabel('Sessions across animals')
ax.axvline(df.p_time_reward.mean(), color='k', linestyle='--',linewidth=3,label='Mean')
# ax.legend()
ax.spines[['top','right']].set_visible(False)
ax.set_title('Time Reward overlap')
plt.savefig(os.path.join(savedst, 'time_reward_overlap.svg'), 
        bbox_inches='tight')

#%%
# time cells and a function of total cells?
coms_correct = [v[1] for k,v in datadct.items()]
total_cells = [len(xx[0]) for xx in coms_correct]
s=70
fig,ax=plt.subplots()
sns.scatterplot(x=total_cells, y='p_time_reward',hue='animals',data=df,s=s,alpha=0.8)
x = np.array(df[total_cells] if isinstance(total_cells, str) else total_cells)
y = df['p_time_reward'].values
x = x[~np.isnan(y)]
y = y[~np.isnan(y)]
# Linear regression
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
# Plot regression line
x_vals = np.linspace(min(x), max(x), 100)
y_vals = intercept + slope * x_vals
ax.plot(x_vals, y_vals, color='black', label='Linear fit')

# Annotate R² and p-value
r2 = r_value**2
textstr = f'$R^2 = {r2:.2f}$\n$p = {p_value:.3g}$'
ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
        verticalalignment='top', fontsize=12, bbox=dict(boxstyle="round", facecolor='white', alpha=0.6))
ax.set_ylabel('% of time cells that are reward cells')
ax.set_xlabel('Total cells in session')
ax.spines[['top','right']].set_visible(False)

plt.savefig(os.path.join(savedst, 'p_time_cells_v_total_cells.svg'), 
        bbox_inches='tight')

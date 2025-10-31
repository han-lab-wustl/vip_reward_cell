
"""
zahra
pure time cells (not rew or place)
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
from projects.pyr_reward.placecell import get_tuning_curve, calc_COM_EH, make_tuning_curves_by_trialtype_w_darktime,\
    make_time_tuning_curves_radians, make_tuning_curves, intersect_arrays
from projects.pyr_reward.rewardcell import get_radian_position,\
    get_radian_position_first_lick_after_rew, get_rewzones, get_goal_cells, goal_cell_shuffle,\
        goal_cell_shuffle_time, get_goal_cells_time
from projects.opto.behavior.behavior import get_success_failure_trials
from projects.pyr_reward.dark_time_and_time_analysis.time import filter_cells_by_field_selectivity
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
saveddataset = r"Z:\saved_datasets\time_tuning_circ.p"
with open(saveddataset, "rb") as fp: #unpickle
    datadct = pickle.load(fp)
#%%
savepth = os.path.join(savedst, 'time_tuning_wo_rew_cells_circ_moving.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
####################################### RUN CODE #######################################
# initialize var
p_goal_cells=[]
p_goal_cells_pure_time = []
p_goal_cells_all_time=[]
p_goal_cells_pc=[]
goal_cells_iind=[]
pvals = []
bins = 90
goal_window_cm=20
goal_cell_null= []
perms = []

# iterate through all animals
for ii in range(len(conddf)):
    day = conddf.days.values[ii]
    animal = conddf.animals.values[ii]
    # check if its the last 4 days of animal behavior
    andf = conddf[(conddf.animals==animal) &( conddf.optoep<2)]
    lastdays = andf.days.values[-4:]
    if (animal!='e217') and (day in lastdays):
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
        rad = get_radian_position_first_lick_after_rew(eps, ybinned, licks, rewards, rewsize,rewlocs,
                    trialnum, track_length) # get radian coordinates

        fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
        Fc3 = fall_fc3['Fc3']
        dFF = fall_fc3['dFF']
        Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
        dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]
        skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
        Fc3 = Fc3[:, skew>2] # only keep cells with skew greateer than 2
        # get tcs from saved pickle
        tcs_correct, coms_correct, tcs_fail, coms_fail, tcs_correct_time, coms_correct_time, tcs_fail_time, coms_fail_time=datadct[f'{animal}_{day:03d}_index{ii:03d}']
        #################### remake tc w/ time but w/o stopped periods ###################
        tcs_correct_time, coms_correct_time, tcs_fail_time, coms_fail_time, trial_times = make_time_tuning_curves_radians(eps, 
                time, Fc3, trialnum, rewards, licks, ybinned, rewlocs, rewsize,forwardvel,
                lasttr=8, bins=bins, velocity_filter=True)
        # filter by in/out field firing
        # selected_cells, in_field_fraction = filter_cells_by_field_selectivity(tcs_correct_time, 
        #                 threshold=0.5, fraction_cutoff=0.6)
        # time bin is roughly (16/90) or 170 ms
        # test
        # max_trial_times =np.array([tr.shape[1]/fr for tr in trial_times])
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
        ################### get rew cells ###################
        # dark time params
        track_length_dt = 550 # cm estimate based on 99.9% of ypos
        track_length_rad_dt = track_length_dt*(2*np.pi/track_length_dt) # estimate bin for dark time
        bins_dt=150 
        bin_size_dt=track_length_rad_dt/bins_dt # typically 3 cm binswith ~ 475 track length
        # tc w/ dark time added to the end of track
        # tcs_correct, coms_correct, tcs_fail, coms_fail, ybinned_dt = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,
        # rewsize,ybinned,time,licks,
        # Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
        # bins=bins_dt)  
        # get abs dist tuning for rew cells
        # binsize = 3 for place
        ################### get place cells ###################
        tcs_correct_abs, coms_correct_abs = make_tuning_curves(eps,rewlocs,ybinned,
                Fc3,trialnum,rewards,forwardvel,
                rewsize,3)
        goal_window = goal_window_cm*(2*np.pi/track_length) # cm converted to rad
        goal_cells, com_goal_postrew, perm, rz_perm = get_goal_cells(rz, goal_window, coms_correct, cell_type = 'all')
        goal_cells_p_per_comparison = [len(xx)/len(coms_correct[0]) for xx in com_goal_postrew]           
        # get goal cells aligned to time
        # same % as goal window
        max_trial_times = np.nanmax(np.array([tr.shape[1]/fr for tr in trial_times]))
        time_window = max_trial_times*.074 #s/rad
        # 5/19/25: halved window of time compared to distance?
        # 5/20/25:back to original window
        time_window = time_window*(2*np.pi/max_trial_times) # s converted to rad        
        goal_cells_time, com_goal_postrew_time, perm_time, rz_perm_time = get_goal_cells_time(rz, time_window, coms_correct_time)
        goal_cells_p_per_comparison_time = [len(xx)/len(coms_correct[0]) for xx in com_goal_postrew_time]            
        ################### get place cells ###################
        # get cells that maintain their coms across at least 2 epochs
        place_window = 20 # cm converted to rad                
        perm = list(combinations(range(len(coms_correct_abs)), 2))     
        com_per_ep = np.array([(coms_correct_abs[perm[jj][0]]-coms_correct_abs[perm[jj][1]]) for jj in range(len(perm))])        
        compc = [np.where((comr<place_window) & (comr>-place_window))[0] for comr in com_per_ep]
        # get cells across all epochs that meet crit
        pcs = np.unique(np.concatenate(compc))
        pcs_all = intersect_arrays(*compc)
        place_cells_p_per_comparison = [len(xx)/len(coms_correct_abs[0]) for xx in compc]           
        # get indices
        goal_cells_iind.append([goal_cells, goal_cells_time, pcs_all])
        # remove rew and place cells
        pure_time = [xx for xx in goal_cells_time if (xx not in goal_cells) and (xx not in pcs_all)]
        # also for per comparison
        pure_time_cells_p_per_comparison = [[xx for xx in com_time if xx not in compc[ll] and xx not in com_goal_postrew[ll]] for ll,com_time in enumerate(com_goal_postrew_time)]
        # eg cell         
        # For goal_cells (Reward cells)
        xticks_time = [r'$-\pi$', '0', r'$\pi$']
        xticks_labels_pi = [r'$-\pi$', '0', r'$\pi$']
        xticks_rew = [0,75,150]
        xticks_poslbl = np.arange(0,track_length+1,90).astype(int)
        if track_length>180:
                xticks_pos = np.arange(0,90+1,30)
        else:
                xticks_pos = np.arange(0,90+1,45)
        n_cells = len(goal_cells)
        if n_cells>0:
                figlength = n_cells if n_cells<30 else n_cells/2
                fig, axes = plt.subplots(nrows=n_cells, ncols=2,figsize=(6,figlength))
                if n_cells == 1:
                        axes = [axes]  # Ensure it's iterable
                for i, cellid in enumerate(goal_cells):
                        axes[i][0].set_title('Reward-relative distance aligned') if i == 0 else None
                        axes[i][1].set_title('Time aligned') if i == 0 else None
                        axes[i][0].plot(tcs_correct[:, cellid, :].T)
                        axes[i][0].axvline(int(bins_dt / 2), color='k', linestyle='--')
                        axes[i][1].plot(tcs_correct_time[:, cellid, :].T)
                        axes[i][1].axvline(int(bins / 2), color='k', linestyle='--')
                        # Set xticks for both plots
                        axes[i][0].set_xticks(xticks_rew)
                        axes[i][0].set_xticklabels(xticks_labels_pi)
                        axes[i][1].set_xticks([0,45,90])
                        axes[i][1].set_xticklabels(xticks_time)
                fig.suptitle(f'{animal}, day {day}\nReward cells')
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
        # For pcs_all (Place cells)
        n_cells = len(pcs_all)
        if n_cells>0:
                figlength = n_cells if n_cells<30 else n_cells/2
                fig, axes = plt.subplots(nrows=n_cells, ncols=2,figsize=(6,figlength))
                if n_cells == 1:
                        axes = [axes]
                for i, cellid in enumerate(pcs_all):
                        axes[i][0].set_title('Track aligned') if i == 0 else None
                        axes[i][1].set_title('Time aligned') if i == 0 else None
                        axes[i][0].plot(tcs_correct_abs[:, cellid, :].T)
                        axes[i][1].plot(tcs_correct_time[:, cellid, :].T)
                        axes[i][1].axvline(int(bins / 2), color='k', linestyle='--')
                        # Set xticks for both plots
                        axes[i][0].set_xticks(xticks_pos)
                        axes[i][0].set_xticklabels(xticks_poslbl)
                        axes[i][1].set_xticks([0,45,90])
                        axes[i][1].set_xticklabels(xticks_time)
                fig.suptitle(f'{animal}, day {day}\nPlace cells')
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
        # For goal_cells_time (Time cells)
        n_cells = len(pure_time)
        if n_cells>0:
                figlength = n_cells if n_cells<30 else n_cells/2
                fig, axes = plt.subplots(nrows=n_cells, ncols=3,figsize=(6,figlength))
                if n_cells == 1:
                        axes = [axes]
                for i, cellid in enumerate(pure_time):
                        axes[i][0].set_title('Track aligned') if i == 0 else None
                        axes[i][1].set_title('Reward-relative distance aligned') if i == 0 else None
                        axes[i][2].set_title('Time aligned') if i == 0 else None
                        axes[i][0].plot(tcs_correct_abs[:, cellid, :].T)
                        axes[i][1].plot(tcs_correct[:, cellid, :].T)
                        axes[i][1].axvline(int(bins_dt / 2), color='k', linestyle='--')
                        axes[i][2].plot(tcs_correct_time[:, cellid, :].T)
                        axes[i][2].axvline(int(bins / 2), color='k', linestyle='--')
                        # Set xticks for both plots
                axes[i][0].set_xticks(xticks_pos)
                axes[i][0].set_xticklabels(xticks_poslbl)
                axes[i][0].set_xlabel('Distance (cm)')
                axes[i][1].set_xticks(xticks_rew)
                axes[i][1].set_xticklabels(xticks_labels_pi)
                axes[i][1].set_xlabel('Reward-relative distance (rad)')
                axes[i][2].set_xticks([0,45,90])
                axes[i][2].set_xticklabels(xticks_time)
                axes[i][2].set_xlabel('Reward-relative time (rad)')
                fig.suptitle(f'{animal}, day {day}\nTime cells\n rewlocs: {rewlocs}')
                plt.tight_layout()
                pdf.savefig(fig)
                plt.show()
                # plt.close(fig)        #only get perms with non zero cells
        # get per comparison and also across epochs
        ####### rew #######
        p_goal_cells.append([len(goal_cells)/len(coms_correct[0]),goal_cells_p_per_comparison])
        ####### pure time #######
        p_goal_cells_pure_time.append([len(pure_time)/len(coms_correct_time[0]), pure_time_cells_p_per_comparison])
        ####### all time #######
        p_goal_cells_all_time.append([len(goal_cells_time)/len(coms_correct_time[0]), goal_cells_p_per_comparison_time])
        ####### place #######
        p_goal_cells_pc.append([len(pcs_all)/len(coms_correct_abs[0]), place_cells_p_per_comparison])
        # save perm
        perms.append([[perm, rz_perm],
            [perm_time, rz_perm_time]])
        print(f'Goal cells w/ dt: {len(goal_cells)} cells\n\
            Time cells: {len(pure_time)} cells')
        ################ rew shuffle ################
        num_iterations=1000
        goal_cell_shuf_ps_per_comp, goal_cell_shuf_ps, shuffled_dist=goal_cell_shuffle(coms_correct, goal_window,perm)
        ################ all time rew shuffle ################
        time_cell_shuf_ps_per_comp, time_cell_shuf_ps, shuffled_time_all=goal_cell_shuffle(coms_correct_time, goal_window,perm_time)
        # gets shuffle w/o rew and place cells
        ################ for pure time cells ################
        # normalize by time epochs
        shuf_ps_per_comp_pure_time, shuf_ps_pure_time, shuffled_dist_time=goal_cell_shuffle_time(coms_correct[:len(coms_correct_time),:],coms_correct_abs[:len(coms_correct_time),:],coms_correct_time, time_window, perm_time)
        ################## for rew cells
        goal_cell_shuf_ps_per_comp_av = np.nanmedian(goal_cell_shuf_ps_per_comp,axis=0)        
        goal_cell_shuf_ps_av = np.nanmedian(np.array(goal_cell_shuf_ps))
        goal_cell_p=len(goal_cells)/len(coms_correct[0]) 
        p_value = sum(shuffled_dist>goal_cell_p)/num_iterations
        ######## for pure time cells only 
        shuf_ps_per_comp_av_pure_time = np.nanmedian(shuf_ps_per_comp_pure_time,axis=0)        
        shuf_ps_av_pure_time = np.nanmedian(np.array(shuf_ps_pure_time))
        p_pure_time=len(pure_time)/len(coms_correct[0]) 
        p_value_time = sum(shuffled_dist_time>p_pure_time)/num_iterations
        ########### for all time cells
        shuf_ps_per_comp_av_all_time = np.nanmedian(time_cell_shuf_ps_per_comp,axis=0)        
        shuf_ps_av_all_time = np.nanmedian(np.array(time_cell_shuf_ps))
        p_all_time=len(goal_cells_time)/len(coms_correct[0]) 
        p_value_all_time = sum(shuffled_time_all>p_all_time)/num_iterations

        print(f'{animal}, day {day}: significant goal cells proportion p-value: {p_value} v time aligned {p_value_time}')
        goal_cell_null.append([[goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av],
                        [shuf_ps_per_comp_av_all_time,shuf_ps_av_all_time],
                        [shuf_ps_per_comp_av_pure_time,shuf_ps_av_pure_time]])
        pvals.append([p_value,p_value_time,p_value_all_time]); 
        datadct[f'{animal}_{day:03d}_index{ii:03d}'] = [tcs_correct, coms_correct, tcs_fail, coms_fail,
                tcs_correct_time, coms_correct_time, tcs_fail_time, coms_fail_time]

pdf.close()
# save pickle of dcts
with open(saveddataset, "wb") as fp:   #Pickling
        pickle.dump(datadct, fp) 
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
ax.set_title(f'{(sessions_sig*100):.2f}% of sessions are significant\n Reward cells')
ax.set_xlabel('P-value')
ax.set_ylabel('Sessions')
# all time cells (not pure)
df_dt = conddf.copy()
df_dt = df_dt[((df_dt.animals!='e217')) & (df_dt.optoep<2) & (df_dt.index.isin(inds))]
df_dt['num_epochs'] = [len(xx[1]) for k,xx in datadct.items()]
df_dt['goal_cell_prop'] = [xx[0] for xx in p_goal_cells_all_time]
df_dt['opto'] = df_dt.optoep.values>1
df_dt['day'] = df_dt.days
df_dt['session_num_opto'] = np.concatenate([[xx-df_dt[df_dt.animals==an].days.values[0] for xx in df_dt[df_dt.animals==an].days.values] for an in np.unique(df_dt.animals.values)])
df_dt['session_num'] = np.concatenate([[ii for ii,xx in enumerate(df_dt[df_dt.animals==an].days.values)] for an in np.unique(df_dt.animals.values)])
df_dt['condition'] = ['vip' if xx=='vip' else 'ctrl' for xx in df_dt.in_type.values]
df_dt['p_value'] = [xx[2] for xx in pvals]
df_dt['goal_cell_prop_shuffle'] = [xx[1][1] for xx in goal_cell_null]

fig,ax = plt.subplots(figsize=(5,5))
ax = sns.histplot(data = df_dt.loc[df_dt.opto==False], x='p_value', 
                hue='animals', bins=40)
ax.spines[['top','right']].set_visible(False)
ax.axvline(x=0.05, color='k', linestyle='--')
sessions_sig = sum(df_dt.loc[df_dt.opto==False,'p_value'].values<0.05)/len(df_dt.loc[df_dt.opto==False])
ax.set_title(f'{(sessions_sig*100):.2f}% of sessions are significant\n All time cells')
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
eps = [2,3,4]
for ii,ep in enumerate(eps):
        # rewprop = df_plt.loc[(df_plt.num_epochs==ep), 'goal_cell_prop']
        rewprop = df_plt.loc[(df_plt.index.get_level_values('num_epochs')==ep), 'goal_cell_prop']
        shufprop = df_plt.loc[(df_plt.index.get_level_values('num_epochs')==ep), 'goal_cell_prop_shuffle']
        t,pval = scipy.stats.wilcoxon(rewprop, shufprop)
        print(f'{ep} epochs, pval: {pval}')
        # statistical annotation        
        if pval < 0.001:
                ax.text(ii, y, "***", ha='center', fontsize=fs)
        elif pval < 0.01:
                ax.text(ii, y, "**", ha='center', fontsize=fs)
        elif pval < 0.05:
                ax.text(ii, y, "*", ha='center', fontsize=fs)

#%%
# include all comparisons 
df_perms = pd.DataFrame()
goal_cell_perm = [xx[1] for xx in p_goal_cells]
goal_cell_perm_shuf = [xx[0][0][~np.isnan(xx[0][0])] for xx in goal_cell_null]
df_perms['goal_cell_prop'] = np.concatenate(goal_cell_perm)
df_perms['goal_cell_prop'] =df_perms['goal_cell_prop'] *100
df_perms['goal_cell_prop_shuffle'] = np.concatenate(goal_cell_perm_shuf)[:len(df_perms)]
df_perms['goal_cell_prop_shuffle']=df_perms['goal_cell_prop_shuffle']*100
df_perm_animals = [[xx]*len(goal_cell_perm[ii]) for ii,xx in enumerate(df.animals.values)]
df_perms['animals'] = np.concatenate(df_perm_animals)
df_perm_days = [[xx]*len(goal_cell_perm[ii]) for ii,xx in enumerate(df.session_num.values)]
df_perms['session_num'] = np.concatenate(df_perm_days)
# take a mean of all epoch comparisons
df_perms['num_epochs'] = [2]*len(df_perms)
df_permsav2 = df_perms.groupby(['animals', 'num_epochs']).mean(numeric_only=True)
# compare to shuffl
df_plt2 = pd.concat([df_permsav2,df_plt])
# df_plt2 = df_plt2[df_plt2.index.get_level_values('animals')!='e189']
df_plt2 = df_plt2[df_plt2.index.get_level_values('num_epochs')<5]
df_plt2 = df_plt2.groupby(['animals', 'num_epochs']).mean(numeric_only=True)
df_plt2=df_plt2.reset_index()

# v time
# HACK TO FIX UNEQUAL 2 COMPARISON SHUFFLES??? IT SHOULDNT MATTER
df_perms = pd.DataFrame()
goal_cell_perm = [xx[1] for xx in p_goal_cells_all_time]
goal_cell_perm_shuf = [xx[1][0][~np.isnan(xx[1][0])] for xx in goal_cell_null]
df_perms['goal_cell_prop_shuffle'] = np.concatenate(goal_cell_perm_shuf)
df_perms['goal_cell_prop_shuffle']=df_perms['goal_cell_prop_shuffle']*100
df_perms['goal_cell_prop'] = np.concatenate(goal_cell_perm)[:len(df_perms)]
df_perms['goal_cell_prop'] =df_perms['goal_cell_prop'] *100
df_perm_animals = [[xx]*len(goal_cell_perm_shuf[ii]) for ii,xx in enumerate(df.animals.values)]
df_perms['animals'] = np.concatenate(df_perm_animals)
df_perm_days = [[xx]*len(goal_cell_perm_shuf[ii]) for ii,xx in enumerate(df.session_num.values)]
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
# combine the dataframes 
df_plt2['type'] = ['Distance']*len(df_plt2)
df_plt2_dt['type'] = ['Time']*len(df_plt2_dt)
df_new = pd.concat([df_plt2_dt,df_plt2])
df_new['prop_diff'] = df_new['goal_cell_prop'] - df_new['goal_cell_prop_shuffle']
df_av = df_new.groupby(['animals', 'type']).median(numeric_only=True)
df_av = df_av.reset_index()
distance_diff = df_av[df_av['type'] == 'Distance']['prop_diff'].reset_index(drop=True)
time_diff = df_av[df_av['type'] == 'Time']['prop_diff'].reset_index(drop=True)
# Make sure they're aligned properly — this assumes same number and order
t_stat, p_val = scipy.stats.wilcoxon(distance_diff, time_diff)
df_new = df_new.reset_index()
df_new = df_new[df_new.animals!='e139']
fig,axes = plt.subplots(figsize=(14,5),ncols=3,width_ratios=[2.5,2,1])
ax=axes[0]
custom_palette = ['black', 'navy']
sns.set_palette(custom_palette)
hue_order = ['Distance', 'Time']
df_new = df_new[df_new.animals!='e200']
# av across mice
sns.stripplot(x='num_epochs', y='goal_cell_prop',hue='type',
        data=df_new,s=10,alpha=0.7,ax=ax,dodge=True,hue_order=hue_order)
sns.barplot(x='num_epochs', y='goal_cell_prop',hue='type',
        data=df_new,hue_order=hue_order,
        fill=False,ax=ax, errorbar='se')
# bar plot of shuffle instead
ax = sns.barplot(data=df_new, # correct shift
        x='num_epochs', y='goal_cell_prop_shuffle',hue='type',color='grey', hue_order=hue_order,
        alpha=0.5, err_kws={'color': 'grey'},errorbar=None,ax=ax)
ax.spines[['top','right']].set_visible(False)
ax.set_ylabel('Cell %')
ax.set_xlabel('')
ans = df_new.animals.unique()
# lines
alpha=0.5
# One value per animal per epoch per type
df_lines = df_new.groupby(['animals', 'num_epochs', 'type'])['goal_cell_prop'].median().reset_index()
for epoch in df_lines['num_epochs'].unique():
    df_ep = df_lines[df_lines['num_epochs'] == epoch]
    for animal in df_ep['animals'].unique():
        sub = df_ep[df_ep['animals'] == animal]
        if len(sub) == 2:  # Ensure both Distance and Time are present
            x = ['Distance', 'Time']
            y = sub.sort_values('type')['goal_cell_prop'].values
            ax.plot([epoch-2 - 0.2, epoch-2 + 0.2], y, color='dimgray', alpha=0.5, linewidth=2)
ax.legend().set_visible(False)
ax.set_xlabel('# of rew. loc. switches')

ax=axes[1]
sns.stripplot(x='num_epochs', y='prop_diff',hue='type',
        data=df_new,s=10,alpha=0.7,ax=ax,dodge=True,hue_order=hue_order)
sns.barplot(x='num_epochs', y='prop_diff',hue='type',
        data=df_new,hue_order=hue_order,
        fill=False,ax=ax, errorbar='se')
ax.spines[['top','right']].set_visible(False)
ax.legend().set_visible(False)
ax.set_ylabel('Cell %-shuffle')
ax.set_xlabel('# of rew. loc. switches')
# average of epochs
ax=axes[2]
df_new = df_new.groupby(['animals','type']).mean(numeric_only=True)
sns.stripplot(x='type', y='prop_diff',hue='type',
        data=df_new,s=10,alpha=0.7,ax=ax,hue_order=hue_order)
sns.barplot(x='type', y='prop_diff',hue='type',
        data=df_new,hue_order=hue_order,
        fill=False,ax=ax, errorbar='se')
ax.spines[['top','right']].set_visible(False)
ax.set_title('Mean of epochs')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_ylabel('Cell %-shuffle')
ax.set_xlabel('')
# One value per animal per epoch per type
df_new=df_new.reset_index()
for ai, animal in enumerate(df_new['animals'].unique()):
    sub = df_new[df_new['animals'] == animal]
    # if len(sub) == 2:  # Ensure both Distance and Time are present
    x = ['Distance', 'Time']
    y = sub.sort_values('type')['prop_diff'].values
    ax.plot([0.1, 0.9], y, color='dimgray', alpha=0.5, linewidth=2)
ax.legend().set_visible(False)

ax.text(.5, 15, f'p={p_val:.3g}', fontsize=12)
fig.suptitle('Time v. Distance Reward cells during movement')
plt.savefig(os.path.join(savedst, 'time_v_reward_sidebyside_wo_stopped.svg'), 
        bbox_inches='tight')
#%%
# side by side
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
alpha=0.5
# make lines
ans = df_plt2.animals.unique()
for i in range(len(ans)):
    sns.lineplot(x=df_plt2.num_epochs-2, y='goal_cell_prop', 
    data=df_plt2[df_plt2.animals==ans[i]],
    errorbar=None, color='dimgray', linewidth=2, alpha=alpha,ax=ax)
ax.set_xlabel('')
ax.set_ylabel('Reward cell % ')
ax.set_ylim([0,40])
eps = [2,3,4]
y = 40
pshift = 4
fs=36
from statsmodels.stats.multitest import multipletests
pvalues = []
# First, compute all raw p-values
for ep in eps:
    rewprop = df_plt2.loc[(df_plt2.num_epochs == ep), 'goal_cell_prop']
    shufprop = df_plt2.loc[(df_plt2.num_epochs == ep), 'goal_cell_prop_shuffle']
    t, pval = scipy.stats.wilcoxon(rewprop, shufprop, alternative='greater')
    pvalues.append(pval)
# Apply FDR correction
rejected, pvals_corr, _, _ = multipletests(pvalues, alpha=0.05, method='fdr_bh')
# Plot and annotate
for ii, (ep, pval_corr, reject) in enumerate(zip(eps, pvals_corr, rejected)):
    if pval_corr < 0.001:
        ax.text(ii, y, "***", ha='center', fontsize=fs)
    elif pval_corr < 0.01:
        ax.text(ii, y, "**", ha='center', fontsize=fs)
    elif pval_corr < 0.05:
        ax.text(ii, y, "*", ha='center', fontsize=fs)
    ax.text(ii - 0.5, y + pshift, f'p={pval_corr:.3g}', fontsize=10, rotation=45)

# time
ax=axes[1]
# av across mice
color='navy'
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
pvalues = []
# First, compute all raw p-values
for ep in eps:
    rewprop = df_plt2_dt.loc[(df_plt2_dt.num_epochs == ep), 'goal_cell_prop']
    shufprop = df_plt2_dt.loc[(df_plt2_dt.num_epochs == ep), 'goal_cell_prop_shuffle']
    t, pval = scipy.stats.wilcoxon(rewprop, shufprop, alternative='greater')
    pvalues.append(pval)
# Apply FDR correction
rejected, pvals_corr, _, _ = multipletests(pvalues, alpha=0.05, method='fdr_bh')
# Plot and annotate
for ii, (ep, pval_corr, reject) in enumerate(zip(eps, pvals_corr, rejected)):
    if pval_corr < 0.001:
        ax.text(ii, y, "***", ha='center', fontsize=fs)
    elif pval_corr < 0.01:
        ax.text(ii, y, "**", ha='center', fontsize=fs)
    elif pval_corr < 0.05:
        ax.text(ii, y, "*", ha='center', fontsize=fs)
    ax.text(ii - 0.5, y + pshift, f'p={pval_corr:.3g}', fontsize=10, rotation=45)

# make lines
alpha=0.5
ans = df_plt2_dt.animals.unique()
for i in range(len(ans)):
    sns.lineplot(x=df_plt2_dt.num_epochs-2, y='goal_cell_prop', 
    data=df_plt2_dt[df_plt2_dt.animals==ans[i]],
    errorbar=None, color='dimgray', linewidth=2, alpha=alpha,ax=ax)
ax.set_xlabel('# of rew. loc. switches')
ax.set_ylabel('Time cell % \n(incl. reward & place cell)')
ax.set_ylim([0,40])
fig.tight_layout()
plt.savefig(os.path.join(savedst, 'time_cell_prop_wo_stopped_time.svg'), 
        bbox_inches='tight')

#%%
# overlap of reward and time cells

goal_cell_ind = [xx[0] for xx in goal_cells_iind]
time_cell_ind = [xx[1] for xx in goal_cells_iind]
place_cell_ind = [xx[2] for xx in goal_cells_iind]

# find % of time cells that are goal cells
time_reward = [len([xx for xx in tc if xx in goal_cell_ind[ii]])/len(tc) if len(tc)>0 else 0 for ii,tc in enumerate(time_cell_ind)]
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
# df=df.iloc[:-1]
df['p_time_reward'] = time_reward
df['p_time_reward'] = df['p_time_reward']*100
fig,ax=plt.subplots()
sns.histplot(x='p_time_reward',hue='animals',data=df,bins=20)
ax.set_xlabel('% of time cells that are reward cells')
ax.set_ylabel('Sessions across animals')
mean_val = df.p_time_reward.mean()
ax.axvline(mean_val, color='k', linestyle='--', linewidth=3, label='Mean')
ax.text(mean_val, ax.get_ylim()[1], f'Mean={mean_val:.2g}%', color='k', ha='center', va='bottom', fontsize=14)
# ax.legend()
ax.spines[['top','right']].set_visible(False)
ax.set_title('Time Reward overlap',pad=30)
plt.savefig(os.path.join(savedst, 'time_reward_overlap_wo_stopped.svg'), 
        bbox_inches='tight')

# find % of time cells that are place cells
time_reward = [len([xx for xx in tc if xx in place_cell_ind[ii]])/len(tc) if len(tc)>0 else 0 for ii,tc in enumerate(time_cell_ind)]
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
df['p_time_reward'] = time_reward
df['p_time_reward'] = df['p_time_reward']*100
fig,ax=plt.subplots()
sns.histplot(x='p_time_reward',hue='animals',data=df,bins=20)
ax.set_xlabel('% of time cells that are place cells')
ax.set_ylabel('Sessions across animals')
mean_val = df.p_time_reward.mean()
ax.axvline(mean_val, color='k', linestyle='--',linewidth=3,label='Mean')
ax.text(mean_val, ax.get_ylim()[1], f'Mean={mean_val:.2g}%', color='k', ha='center', va='bottom', fontsize=14)
# ax.legend()
ax.spines[['top','right']].set_visible(False)
ax.set_title('Time Place overlap',pad=20)
plt.savefig(os.path.join(savedst, 'time_place_overlap_wo_stopped.svg'), 
        bbox_inches='tight')


#%%
# time cells and a function of total cells?
coms_correct = [v[1] for k,v in datadct.items()]
total_cells = [len(xx[0]) for xx in coms_correct]
s=70
fig,ax=plt.subplots()
sns.scatterplot(x=total_cells, y='p_time_reward',hue='animals',data=df,s=s,alpha=0.8)
ax.set_ylabel('% of time cells that are reward cells')
ax.set_xlabel('Total cells in session')
ax.spines[['top','right']].set_visible(False)

plt.savefig(os.path.join(savedst, 'p_time_cells_v_total_cells_wo_stopped.svg'), 
        bbox_inches='tight')

#%% 
# com histogram of time cells
time_iind = [v[1] for v in goal_cells_iind]
com_time = [v[5]-np.pi for k,v in datadct.items()]
com_time = [np.nanmedian(com[:,time_iind[iind]],axis=0) for iind,com in enumerate(com_time)]
com_rew = [v[1]-np.pi for k,v in datadct.items()]
rew_iind = [v[0] for v in goal_cells_iind]
com_rew = [np.nanmedian(com[:,rew_iind[iind]],axis=0) for iind,com in enumerate(com_rew)]
# compare to pure time cells?
goal_cell_ind = [xx[0] for xx in goal_cells_iind]
time_cell_ind = [xx[1] for xx in goal_cells_iind]
place_cell_ind = [xx[2] for xx in goal_cells_iind]
pure_time = [[xx for xx in time_cell if xx not in goal_cell_ind[ind] and xx not in place_cell_ind[ind]]
              for ind,time_cell in enumerate(time_cell_ind) ]
com_time_tc = [v[5]-np.pi for k,v in datadct.items()]
com_pure_time = [np.nanmedian(com[:,pure_time[iind]],axis=0) for iind,com in enumerate(com_time_tc)]

fig,ax = plt.subplots()
ncells = len(np.concatenate(com_time))
a=0.4
ax.hist(np.concatenate(com_time),color=color,density=True,alpha=a, label=f'Time, {ncells} cells',bins=20)
ncells = len(np.concatenate(com_pure_time))
ax.hist(np.concatenate(com_pure_time),color='orangered',density=True,alpha=a, label=f'Pure Time, {ncells} cells',bins=20)
ncells = len(np.concatenate(com_rew))
ax.hist(np.concatenate(com_rew),color='k',density=True,alpha=a, label=f'Distance, {ncells} cells',bins=20)

ax.axvline(0, color='k',linewidth=3,linestyle='--')
ax.spines[['top','right']].set_visible(False)
ax.set_ylabel('Density of cells')
ax.set_xticks([-np.pi, -np.pi/4,0, np.pi/4,np.pi])
ax.set_xticklabels(["$-\\pi$", '$-\\pi/4$', "0",  '$\\pi/4$', "$\\pi$"])
ax.set_xlabel('Reward-relative time or distance ($\Theta$)')
ax.legend(loc='center left', bbox_to_anchor=(0.6, 0.5))
ax.set_title('During movement')
plt.savefig(os.path.join(savedst, 'time_v_distance_com_wo_stopped.svg'), 
        bbox_inches='tight')

# datadct[f'{animal}_{day:03d}_index{ii:03d}'] = [tcs_correct, coms_correct, tcs_fail, coms_fail,
#                 tcs_correct_time, coms_correct_time, tcs_fail_time, coms_fail_time]
# goal_cells_iind.append([goal_cells, goal_cells_time, pcs_all])

#%%
# eg
colors = ['k', 'slategray', 'darkcyan', 'darkgoldenrod', 'orchid']
sns.set_palette(colors)
plt.rc('font', size=12)
n_cells = len(goal_cells[:4])
if n_cells>0:
        figlength = n_cells if n_cells<30 else n_cells/2
        fig, axes = plt.subplots(nrows=n_cells, ncols=2,figsize=(5,7))
        if n_cells == 1:
            axes = [axes]  # Ensure it's iterable
        for i, cellid in enumerate(goal_cells[:4]):
            axes[i][0].set_title('Distance aligned') if i == 0 else None
            axes[i][1].set_title('Time aligned') if i == 0 else None
            axes[i][0].plot(tcs_correct[:, cellid, :].T)
            axes[i][0].axvline(int(bins_dt / 2), color='k', linestyle='--')
            for ep in range(len(tcs_correct_time)):
                axes[i][1].plot(tcs_correct_time[ep, cellid, :].T, label=f'{int(rewlocs[ep])} cm')
            axes[i][1].axvline(int(bins / 2), color='k', linestyle='--')
            # Set xticks for both plots
            axes[i][0].set_xticks(xticks_rew)
            axes[i][0].set_xticklabels(xticks_labels_pi)
            axes[i][1].set_xticks([0,45,90])
            axes[i][1].set_xticklabels(xticks_time)
            axes[i][0].spines[['top','right']].set_visible(False)
            axes[i][1].spines[['top','right']].set_visible(False)
            if i==0:
                axes[i][0].set_ylabel('$\Delta$ F/F')
        axes[i][0].set_xlabel('Reward-relative distance ($\Theta$)')
        axes[i][1].set_xlabel('Reward-relative time ($\Theta$)')
        axes[i][1].legend(title='Rew. zone center')
        fig.suptitle(f'{animal}, day {day}\nReward-distance cells')
        plt.tight_layout()
plt.savefig(os.path.join(savedst, 'time_aligned_rew_cells.svg'), 
        bbox_inches='tight')
#%%
# For pcs_all (Place cells)
cellid=[0,1,4,6]
n_cells = len(pcs_all[cellid])
if n_cells>0:
        figlength = n_cells if n_cells<30 else n_cells/2
        fig, axes = plt.subplots(nrows=n_cells, ncols=2,figsize=(5,7))
        if n_cells == 1:
                axes = [axes]
        for i, cellid in enumerate(pcs_all[cellid]):
            axes[i][0].set_title('Track aligned') if i == 0 else None
            axes[i][1].set_title('Time aligned') if i == 0 else None
            axes[i][0].plot(tcs_correct_abs[:, cellid, :].T)
            axes[i][1].plot(tcs_correct_time[:, cellid, :].T)
            axes[i][1].axvline(int(bins / 2), color='k', linestyle='--')
            # Set xticks for both plots
            axes[i][0].set_xticks(xticks_pos)
            axes[i][0].set_xticklabels(xticks_poslbl)
            axes[i][1].set_xticks([0,45,90])
            axes[i][1].set_xticklabels(xticks_time)
            axes[i][0].spines[['top','right']].set_visible(False)
            axes[i][1].spines[['top','right']].set_visible(False)
            if i==0:
                axes[i][0].set_ylabel('$\Delta$ F/F')
        axes[i][0].set_xlabel('Track position (cm)')
        axes[i][1].set_xlabel('Reward-relative time ($\Theta$)')
        fig.suptitle(f'{animal}, day {day}\nPlace cells')
        plt.tight_layout()
plt.savefig(os.path.join(savedst, 'time_aligned_place_cells.svg'), 
        bbox_inches='tight')
#%%
# For goal_cells_time (Time cells)
pure_time=np.array(pure_time)
cellid = [1,3,4,5]
n_cells = len(pure_time[cellid])
if n_cells>0:
        figlength = n_cells if n_cells<30 else n_cells/2
        fig, axes = plt.subplots(nrows=n_cells, ncols=3,figsize=(7,7))
        if n_cells == 1:
            axes = [axes]
        for i, cellid in enumerate(pure_time[cellid]):
            axes[i][0].set_title('Track aligned') if i == 0 else None
            axes[i][1].set_title('Distance aligned') if i == 0 else None
            axes[i][2].set_title('Time aligned') if i == 0 else None
            axes[i][0].plot(tcs_correct_abs[:, cellid, :].T)
            axes[i][1].plot(tcs_correct[:, cellid, :].T)
            axes[i][1].axvline(int(bins_dt / 2), color='k', linestyle='--')
            axes[i][2].plot(tcs_correct_time[:, cellid, :].T)
            axes[i][2].axvline(int(bins / 2), color='k', linestyle='--')
            if i==0:
                axes[i][0].set_ylabel('$\Delta$ F/F')
                # Set xticks for both plots
            axes[i][0].set_xticks(xticks_pos)
            axes[i][0].set_xticklabels(xticks_poslbl)        
            axes[i][1].set_xticks(xticks_rew)
            axes[i][1].set_xticklabels(xticks_labels_pi)        
            axes[i][2].set_xticks([0,45,90])
            axes[i][2].set_xticklabels(xticks_time)
            axes[i][0].spines[['top','right']].set_visible(False)
            axes[i][1].spines[['top','right']].set_visible(False)
            axes[i][2].spines[['top','right']].set_visible(False)
        axes[i][0].set_xlabel('Track position (cm)')
        axes[i][1].set_xlabel('Reward-relative distance ($\Theta$)')
        axes[i][2].set_xlabel('Reward-relative time ($\Theta$)')
        fig.suptitle(f'{animal}, day {day}\nPure time cells')
        plt.tight_layout()
plt.savefig(os.path.join(savedst, 'time_aligned_time_cells.svg'), 
        bbox_inches='tight')
#%%
# venn diagram
from matplotlib_venn import venn3
colors = ['dimgrey', 'navy', sns.color_palette('Dark2')[2]]
# Example sets
goal_cell_ind = np.concatenate([[f'{yy}_{ii}' for yy in xx[0]] for ii,xx in enumerate(goal_cells_iind)])
time_cell_ind = np.concatenate([[f'{yy}_{ii}' for yy in xx[1]] for ii,xx in enumerate(goal_cells_iind)])
place_cell_ind = np.concatenate([[f'{yy}_{ii}' for yy in xx[2]] for ii,xx in enumerate(goal_cells_iind)])
goal_cell_ind=set(goal_cell_ind)
time_cell_ind=set(time_cell_ind)
place_cell_ind=set(place_cell_ind)
venn3([goal_cell_ind,time_cell_ind,place_cell_ind], set_labels=('Reward-distance', 'Reward-time', 'Place'),set_colors=colors)
plt.title("All cells (sessions across mice)")
plt.savefig(os.path.join(savedst, 'time_venn_diagram.svg'), 
        bbox_inches='tight')

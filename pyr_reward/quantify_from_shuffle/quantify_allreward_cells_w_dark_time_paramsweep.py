
"""
zahra
get tuning curves with dark time
get cells in 2, 3, or 4 epochs
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
    make_tuning_curves_by_trialtype_w_darktime,get_radian_position_first_lick_after_rew_w_dt
from projects.pyr_reward.rewardcell import get_radian_position,\
    get_radian_position_first_lick_after_rew, get_rewzones, get_goal_cells, goal_cell_shuffle
from projects.opto.behavior.behavior import get_success_failure_trials
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_rewardcentric_all.p"
with open(saveddataset, "rb") as fp: #unpickle
        radian_alignment_saved = pickle.load(fp)
savepth = os.path.join(savedst, 'dark_time_tuning.pdf')
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
datadct = {}
goal_cell_null= []
perms = []
goal_window_cm = np.arange(10,270,10) # cm
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
        track_length_dt = 550 # cm estimate based on 99.9% of ypos
        track_length_rad_dt = track_length_dt*(2*np.pi/track_length_dt) # estimate bin for dark time
        bins_dt=150 
        bin_size_dt=track_length_rad_dt/bins_dt # typically 3 cm binswith ~ 475 track length
        tcs_correct_dt, coms_correct_dt, tcs_fail_dt, coms_fail_dt, ybinned_dt = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,rewsize,ybinned,time,licks,
            Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
            bins=bins_dt)
        # normal tc
        tcs_correct, coms_correct, tcs_fail, coms_fail = make_tuning_curves_radians_by_trialtype(eps,rewlocs,ybinned,rad,Fc3,trialnum,
        rewards,forwardvel,rewsize,bin_size) 
        param_sweep = {}    
        for cm_window in goal_window_cm: 
            goal_window = cm_window*(2*np.pi/track_length) # cm converted to rad
            goal_cells, com_goal_postrew, perm, rz_perm = get_goal_cells(rz, goal_window, coms_correct)
            goal_cells_p_per_comparison = [len(xx)/len(coms_correct[0]) for xx in com_goal_postrew]
            goal_cells_dt, com_goal_postrew_dt, perm_dt, rz_perm_dt = get_goal_cells(rz, goal_window, coms_correct_dt)
            goal_cells_p_per_comparison_dt = [len(xx)/len(coms_correct[0]) for xx in com_goal_postrew_dt]
            #only get perms with non zero cells
            # get per comparison and also across epochs
            p_goal_cells.append([len(goal_cells)/len(coms_correct[0]),goal_cells_p_per_comparison])
            p_goal_cells_dt.append([len(goal_cells_dt)/len(coms_correct_dt[0]), goal_cells_p_per_comparison_dt])
            goal_cells_iind.append([goal_cells, goal_cells_dt])
            # save perm
            perms.append([[perm, rz_perm],
                [perm_dt, rz_perm_dt]])
            print(f'Goal window: {goal_window} rad\n\
                Goal cells w/o dt: {len(goal_cells)}\n\
                Goal cells w/ dt: {len(goal_cells_dt)}')
            # shuffle
            num_iterations=1000
            goal_cell_shuf_ps_per_comp, goal_cell_shuf_ps, shuffled_dist=goal_cell_shuffle(coms_correct, goal_window,perm,num_iterations = num_iterations)
            goal_cell_shuf_ps_per_comp_dt, goal_cell_shuf_ps_dt, shuffled_dist_dt=goal_cell_shuffle(coms_correct_dt,goal_window, perm_dt, num_iterations = num_iterations)
            goal_cell_shuf_ps_per_comp_av = np.nanmedian(goal_cell_shuf_ps_per_comp,axis=0)        
            goal_cell_shuf_ps_av = np.nanmedian(np.array(goal_cell_shuf_ps))
            goal_cell_p=len(goal_cells)/len(coms_correct[0]) 
            p_value = sum(shuffled_dist>goal_cell_p)/num_iterations
            # dark time
            goal_cell_shuf_ps_per_comp_av_dt = np.nanmedian(goal_cell_shuf_ps_per_comp_dt,axis=0)        
            goal_cell_shuf_ps_av_dt = np.nanmedian(np.array(goal_cell_shuf_ps_dt))
            goal_cell_p_dt=len(goal_cells_dt)/len(coms_correct[0]) 
            p_value_dt = sum(shuffled_dist_dt>goal_cell_p_dt)/num_iterations
            print(f'{animal}, day {day}: goal cells w/ dark time and window {cm_window} cm; p-value: {p_value_dt}')
            goal_cell_null.append([[goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av],
                            [goal_cell_shuf_ps_per_comp_av_dt,goal_cell_shuf_ps_av_dt]])
            pvals.append([p_value,p_value_dt]); 
            # save
            param_sweep[cm_window] = [goal_cell_p_dt,com_goal_postrew_dt,goal_cells_dt,goal_cell_shuf_ps_per_comp_av_dt,goal_cell_shuf_ps_av_dt,p_value_dt]
        # save~
        datadct[f'{animal}_{day:03d}_index{ii:03d}'] = [tcs_correct_dt, coms_correct_dt,param_sweep]

pdf.close()
####################################### RUN CODE #######################################
#%%
plt.rc('font', size=20)          # controls default text sizes
color = 'cornflowerblue' # reward cell color
sweep = list(datadct['e218_020_index000'][2].keys())
# get values for each sweep per session
sessions = list(datadct.keys())
pvals_per_an = []
sweep_labels = []

for session in sessions:
    pvals = [datadct[session][2][sw][5] for sw in sweep]
    pvals_per_an.extend(pvals)
    sweep_labels.extend(sweep)

# Repeat sweep labels for each session
# sweep_column = sweep_labels * len(sessions)

# Create DataFrame
df = pd.DataFrame({
    'p_value': pvals_per_an,
    'sweep': sweep_labels
})

df = df[df.sweep<180]
fig,axes=plt.subplots(ncols = 3,figsize=(15,5))
ax=axes[0]
sns.lineplot(x='sweep',y = 'p_value',data=df, color=color,ax=ax)
ax.axhline(0.05, color='grey', linestyle='--',linewidth=3)
ax.axvline(20, color='grey', linestyle='-.',linewidth=3)
ax.set_ylabel('P-value \n(Reward cell % > shuffle)')
ax.set_xlabel('')
ax.text(
    40,                           # x-position (same as vertical line)
    ax.get_ylim()[1] * 1.02,      # y-position slightly above the top of y-axis
    'Threshold = 20 cm',                      # text to display
    ha='center', va='bottom',     # horizontal and vertical alignment
)

ax.spines[['top', 'right']].set_visible(False)
# # of cells per threshold
# 2 - rew cell indices
goal_cells_per_an = []
sweep_labels = []
for session in sessions:
    gcs = [len(datadct[session][2][sw][2]) for sw in sweep]
    goal_cells_per_an.extend(gcs)
    sweep_labels.extend(sweep)
# Create DataFrame
df = pd.DataFrame({
    'numgc': goal_cells_per_an,
    'sweep': sweep_labels
})

df = df[df.sweep<180]
ax=axes[1]
sns.lineplot(x='sweep',y = 'numgc',data=df, color=color,ax=ax)
ax.axvline(20, color='grey', linestyle='-.',linewidth=3)
ax.set_ylabel('# of Reward cells')
ax.set_xlabel('')
ax.text(
    40,                           # x-position (same as vertical line)
    ax.get_ylim()[1] * 1.02,      # y-position slightly above the top of y-axis
    'Threshold = 20 cm',                      # text to display
    ha='center', va='bottom',     # horizontal and vertical alignment
)

ax.spines[['top', 'right']].set_visible(False)
# % of cells per threshold
# 0 - rew cell prop
# 4 - shuffle prop
goal_cells_per_an = []
shuffle_per_an = []
sweep_labels = []
for session in sessions:
    gcs = [datadct[session][2][sw][0] for sw in sweep]
    shf = [datadct[session][2][sw][4] for sw in sweep]
    goal_cells_per_an.extend(gcs)
    shuffle_per_an.extend(shf)
    sweep_labels.extend(sweep)
# Create DataFrame
df = pd.DataFrame({
    'numgc': goal_cells_per_an,
    'shufprop': shuffle_per_an,
    'sweep': sweep_labels
})

df = df[df.sweep<180]
ax=axes[2]
sns.lineplot(x='sweep',y = 'numgc',data=df,ax=ax,color=color,label='Real')
sns.lineplot(x='sweep',y = 'shufprop',data=df, color='k',ax=ax,label='Shuffle')
ax.axvline(20, color='grey', linestyle='-.',linewidth=3)
ax.set_ylabel('Reward cell %')
ax.set_xlabel('Distance between COM threshold (cm)')
ax.text(
    40,                           # x-position (same as vertical line)
    ax.get_ylim()[1] * 1.02,      # y-position slightly above the top of y-axis
    'Threshold = 20 cm',                      # text to display
    ha='center', va='bottom',     # horizontal and vertical alignment
)

ax.spines[['top', 'right']].set_visible(False)
ax.legend()
panel_labels = ['A', 'B', 'C']
for i, ax in enumerate(axes):
    ax.text(
        -0.3, 1.1,                      # x, y in axes coords (left-top outside corner)
        panel_labels[i],               # label text
        transform=ax.transAxes,
        fontsize=26, 
        va='top', ha='right'
    )

plt.tight_layout()
plt.savefig(os.path.join(savedst, 'pvalue_per_threshold.svg'), bbox_inches='tight')
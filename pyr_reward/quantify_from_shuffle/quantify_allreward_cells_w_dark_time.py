
"""
zahra
get tuning curves with dark time
get cells in 2, 3, or 4 epochs
only use spatially tuned!!!
july 2025
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
    get_radian_position_first_lick_after_rew, get_rewzones, get_goal_cells, goal_cell_shuffle,wilcoxon_r
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
radian_alignment_saved = {} # overwrite
p_goal_cells=[]
p_goal_cells_dt = []
goal_cells_iind=[]
pvals = []
bins = 90
goal_window_cm=20
datadct = {}
goal_cell_null= []
perms = []
# goal_window_cm = np.arange(5,135,5) # cm
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
        rad = get_radian_position_first_lick_after_rew(eps, ybinned, licks, rewards, rewsize,rewlocs,trialnum, track_length) # get radian coordinates
        # added to get anatomical info
        # takes time
        fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
        Fc3 = fall_fc3['Fc3']
        dFF = fall_fc3['dFF']
        Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & ~(fall['bordercells'][0].astype(bool)))]
        dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool) & ~(fall['bordercells'][0].astype(bool)))]
        skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
        #if pc in all but 1
        # looser restrictions
        pc_bool = np.sum(pcs,axis=0)>0
        Fc3 = Fc3[:,((skew>2)&pc_bool)] # only keep cells with skew greateer than 2
        # tc w/ dark time
        track_length_dt = 550 # cm estimate based on 99.9% of ypos
        track_length_rad_dt = track_length_dt*(2*np.pi/track_length_dt) # estimate bin for dark time
        bins_dt=150 
        bin_size_dt=track_length_rad_dt/bins_dt # typically 3 cm binswith ~ 475 track length
        tcs_correct_dt, coms_correct_dt, tcs_fail_dt, coms_fail_dt, ybinned_dt,raddt = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,rewsize,ybinned,time,licks,
            Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
            bins=bins_dt)
        # normal tc
        tcs_correct, coms_correct, tcs_fail, coms_fail = make_tuning_curves_radians_by_trialtype(eps,rewlocs,ybinned,rad,Fc3,trialnum,
        rewards,forwardvel,rewsize,bin_size)      
        goal_window = goal_window_cm*(2*np.pi/track_length) # cm converted to rad
        goal_cells, com_goal_postrew, perm, rz_perm = get_goal_cells(rz, goal_window, coms_correct, cell_type = 'all')
        goal_cells_p_per_comparison = [len(xx)/len(coms_correct[0]) for xx in com_goal_postrew]
        goal_cells_dt, com_goal_postrew_dt, perm_dt, rz_perm_dt = get_goal_cells(rz, goal_window, coms_correct_dt, cell_type = 'all')
        goal_cells_p_per_comparison_dt = [len(xx)/len(coms_correct[0]) for xx in com_goal_postrew_dt]
        #only get perms with non zero cells
        # get per comparison and also across epochs
        p_goal_cells.append([len(goal_cells)/len(coms_correct[0]),goal_cells_p_per_comparison])
        p_goal_cells_dt.append([len(goal_cells_dt)/len(coms_correct_dt[0]), goal_cells_p_per_comparison_dt])
        goal_cells_iind.append([goal_cells, goal_cells_dt])
        # save perm
        perms.append([[perm, rz_perm],
            [perm_dt, rz_perm_dt]])
        print(f'Goal cells w/o dt: {goal_cells}\n\
            Goal cells w/ dt: {goal_cells_dt}')
        # shuffle
        num_iterations=1000
        goal_cell_shuf_ps_per_comp, goal_cell_shuf_ps, shuffled_dist=goal_cell_shuffle(coms_correct, goal_window,\
                            perm,num_iterations = num_iterations)
        goal_cell_shuf_ps_per_comp_dt, goal_cell_shuf_ps_dt, shuffled_dist_dt=goal_cell_shuffle(coms_correct_dt, \
                        goal_window, perm_dt, num_iterations = num_iterations)
        goal_cell_shuf_ps_per_comp_av = np.nanmedian(goal_cell_shuf_ps_per_comp,axis=0)        
        goal_cell_shuf_ps_av = np.nanmedian(np.array(goal_cell_shuf_ps))
        goal_cell_p=len(goal_cells)/len(coms_correct[0]) 
        p_value = sum(shuffled_dist>goal_cell_p)/num_iterations
        # dark time
        goal_cell_shuf_ps_per_comp_av_dt = np.nanmedian(goal_cell_shuf_ps_per_comp_dt,axis=0)        
        goal_cell_shuf_ps_av_dt = np.nanmedian(np.array(goal_cell_shuf_ps_dt))
        goal_cell_p_dt=len(goal_cells_dt)/len(coms_correct[0]) 
        p_value_dt = sum(shuffled_dist_dt>goal_cell_p_dt)/num_iterations
        print(f'{animal}, day {day}: significant goal cells proportion p-value: {p_value} v w/ dark ttime {p_value_dt}')
        goal_cell_null.append([[goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av],
                        [goal_cell_shuf_ps_per_comp_av_dt,goal_cell_shuf_ps_av_dt]])
        pvals.append([p_value,p_value_dt]); 
        datadct[f'{animal}_{day:03d}_index{ii:03d}'] = [tcs_correct, coms_correct, tcs_fail, coms_fail,tcs_correct_dt, coms_correct_dt, tcs_fail_dt, coms_fail_dt, goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av]

pdf.close()
with open(saveddataset, "wb") as fp:   #Pickling
        pickle.dump(datadct, fp) 

####################################### RUN CODE #######################################
#%%
plt.rc('font', size=16)          # controls default text sizes
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
ax.set_title(f'{(sessions_sig*100):.2f}% of sessions are significant\n Reward cells w/ dark time')
ax.set_xlabel('P-value')
ax.set_ylabel('Sessions')
#%%
# number of epochs vs. reward cell prop    
fig,axes = plt.subplots(ncols = 2, figsize=(6,5),sharex=True,sharey=True)
ax = axes[0]
df_plt = df[df.num_epochs<5]
# av across mice
df_plt = df_plt.groupby(['animals','num_epochs']).mean(numeric_only=True)
sns.stripplot(x='num_epochs', y='goal_cell_prop',
        hue='animals',data=df_plt,
        s=10,ax=ax)
sns.barplot(x='num_epochs', y='goal_cell_prop',
        data=df_plt,
        fill=False,ax=ax, color='k', errorbar='se')
sns.lineplot(data=df_plt, # correct shift
        x=df_plt.index.get_level_values('num_epochs')-2, y='goal_cell_prop_shuffle',color='grey', 
        label='shuffle',ax=ax)
ax.spines[['top','right']].set_visible(False)
ax.get_legend().remove()

eps = [2,3,4]
for ep in eps:
    # rewprop = df_plt.loc[(df_plt.num_epochs==ep), 'goal_cell_prop']
    rewprop = df_plt.loc[(df_plt.index.get_level_values('num_epochs')==ep), 'goal_cell_prop']
    shufprop = df_plt.loc[(df_plt.index.get_level_values('num_epochs')==ep), 'goal_cell_prop_shuffle']
    t,pval = scipy.stats.wilcoxon(rewprop, shufprop)
    print(f'{ep} epochs, pval: {pval}')

ax = axes[1]
df_plt = df_dt[df_dt.num_epochs<5]
# av across mice
df_plt = df_plt.groupby(['animals','num_epochs']).mean(numeric_only=True)
sns.stripplot(x='num_epochs', y='goal_cell_prop',
        hue='animals',data=df_plt,
        s=10,ax=ax)
sns.barplot(x='num_epochs', y='goal_cell_prop',
        data=df_plt,
        fill=False,ax=ax, color='k', errorbar='se')
sns.lineplot(data=df_plt, # correct shift
        x=df_plt.index.get_level_values('num_epochs')-2, y='goal_cell_prop_shuffle',color='grey', 
        label='shuffle',ax=ax)
ax.spines[['top','right']].set_visible(False)
ax.get_legend().remove()

eps = [2,3,4]
for ep in eps:
    # rewprop = df_plt.loc[(df_plt.num_epochs==ep), 'goal_cell_prop']
    rewprop = df_plt.loc[(df_plt.index.get_level_values('num_epochs')==ep), 'goal_cell_prop']
    shufprop = df_plt.loc[(df_plt.index.get_level_values('num_epochs')==ep), 'goal_cell_prop_shuffle']
    t,pval = scipy.stats.wilcoxon(rewprop, shufprop)
    print(f'{ep} epochs, pval: {pval}')
df_plt['goal_cell_prop']=df_plt['goal_cell_prop']*100
df_plt['goal_cell_prop_shuffle']=df_plt['goal_cell_prop_shuffle']*100


#%%    
# include all comparisons 
df_perms = pd.DataFrame()
goal_cell_perm = [xx[1] for xx in p_goal_cells]
goal_cell_perm_shuf = [xx[1][0][~np.isnan(xx[1][0])] for xx in goal_cell_null]
epoch_perm = [xx[1][0] for xx in perms]
df_perms['goal_cell_prop'] = np.concatenate(goal_cell_perm)
df_perms['epoch_comparison'] = [str(tuple(xx)) for xx in np.concatenate(epoch_perm)][:len(df_perms)] # HACK
df_perms['goal_cell_prop_shuffle'] = np.concatenate(goal_cell_perm_shuf)[:len(df_perms)] # HACK
df_perm_animals = [[xx]*len(goal_cell_perm[ii]) for ii,xx in enumerate(df.animals.values)]
df_perms['animals'] = np.concatenate(df_perm_animals)
df_perm_days = [[xx]*len(goal_cell_perm[ii]) for ii,xx in enumerate(df.session_num.values)]
df_perms['session_num'] = np.concatenate(df_perm_days)

# take a mean of all epoch comparisons
df_perms['num_epochs'] = [2]*len(df_perms)
df_permsav = df_perms.groupby(['animals', 'epoch_comparison']).mean(numeric_only=True)
df_permsav=df_permsav.reset_index()
epcomp = [f'Epoch {int(xx[1])+1}, Epoch {int(xx[4])+1}' for xx in df_permsav.epoch_comparison.values]
df_permsav['epoch_comparison']=epcomp
df_permsav['goal_cell_prop']=df_permsav['goal_cell_prop']*100
df_permsav['goal_cell_prop_shuffle']=df_permsav['goal_cell_prop_shuffle']*100

#%% 
# for figure; place cells in any epoch
df_permsav=df_permsav[(df_permsav.epoch_comparison!='Epoch 1, Epoch 5') & (df_permsav.epoch_comparison!='Epoch 2, Epoch 5') & (df_permsav.epoch_comparison!='Epoch 3, Epoch 5') & (df_permsav.epoch_comparison!='Epoch 4, Epoch 5')]

df_permsav=df_permsav[(df_permsav.epoch_comparison=='Epoch 1, Epoch 2') | (df_permsav.epoch_comparison=='Epoch 2, Epoch 3') | (df_permsav.epoch_comparison=='Epoch 3, Epoch 4')]

fig,ax = plt.subplots(figsize=(3,4))
sns.barplot(x='epoch_comparison', y='goal_cell_prop',
        data=df_permsav,
        fill=False,ax=ax, color='cornflowerblue', errorbar='se')
sns.barplot(data=df_permsav, # correct shift
        x='epoch_comparison', y='goal_cell_prop_shuffle',color='grey', 
        label='shuffle', alpha=0.5, err_kws={'color': 'grey'},errorbar=None,ax=ax)

ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.01, 1.05))
ax.set_xticklabels(ax.get_xticklabels(),rotation=20)
eps = df_permsav.epoch_comparison.unique()
pvalues=[]
for ep in eps:
        # rewprop = df_plt.loc[(df_plt.num_epochs==ep), 'goal_cell_prop']
        rewprop = df_permsav.loc[(df_permsav.epoch_comparison==ep), 
        'goal_cell_prop'].values
        shufprop = df_permsav.loc[(df_permsav.epoch_comparison==ep), 
                'goal_cell_prop_shuffle'].values
        t,pval = wilcoxon_r(rewprop, shufprop)
        print(f'{ep} epochs, pval: {pval},w = {t},n={len(rewprop)}')
        pvalues.append(pval)

from statsmodels.stats.multitest import multipletests
reject, pvals_corrected, _, _ = multipletests(pvalues, method='fdr_bh')

y=45
fs=38
for ii,ep in enumerate(eps):
        pval=pvals_corrected[ii]
        # statistical annotation        
        if pval < 0.001:
                ax.text(ii, y, "***", ha='center', fontsize=fs)
        elif pval < 0.01:
                ax.text(ii, y, "**", ha='center', fontsize=fs)
        elif pval < 0.05:
                ax.text(ii, y, "*", ha='center', fontsize=fs)

ax.set_xlabel('')
ans = df_permsav.animals.unique()
for i in range(len(ans)):
    ax = sns.lineplot(x=np.arange(len(df_permsav[df_permsav.animals==ans[i]])), y='goal_cell_prop', 
    data=df_permsav[df_permsav.animals==ans[i]],
    errorbar=None, color='dimgray', linewidth=1.5, alpha=0.5,ax=ax)

groups = [df_permsav.loc[df_permsav.epoch_comparison==ep, 'goal_cell_prop'].values
          for ep in df_permsav.epoch_comparison.unique()]

H, p_kw = scipy.stats.kruskal(*groups)
print(f"Kruskal–Wallis: H={H:.3f}, p={p_kw:.3g}")

# =========================
# Post-hoc pairwise Dunn test if KW significant
# =========================
import scikit_posthocs as sp_post
if p_kw < 0.05:
        dunn = sp_post.posthoc_dunn(df_permsav, val_col='goal_cell_prop', 
                                group_col='epoch_comparison', p_adjust='fdr_bh')
        print(dunn)
        
import itertools

# Set label
ax.set_ylabel('Reward cell %')

# Add subject-level lines
ans = df_permsav.animals.unique()
for i in range(len(ans)):
    ax = sns.lineplot(x=np.arange(len(df_permsav[df_permsav.animals==ans[i]])),
                      y='goal_cell_prop',
                      data=df_permsav[df_permsav.animals==ans[i]],
                      errorbar=None, color='dimgray',
                      linewidth=1.5, alpha=0.2, ax=ax)

# Kruskal–Wallis
groups = [df_permsav.loc[df_permsav.epoch_comparison==ep, 'goal_cell_prop'].values
          for ep in df_permsav.epoch_comparison.unique()]
H, p_kw = scipy.stats.kruskal(*groups)
print(f"Kruskal–Wallis: H={H:.3f}, p={p_kw:.3g}")
h=1.5
# Post-hoc Dunn test
import scikit_posthocs as sp_post
if p_kw < 0.05:
    dunn = sp_post.posthoc_dunn(df_permsav,
                                val_col='goal_cell_prop',
                                group_col='epoch_comparison',
                                p_adjust='fdr_bh')
    print(dunn)

    # =============================
    # Draw comparison bars manually
    # =============================
    xticks = df_permsav.epoch_comparison.unique()
    y_max = df_permsav['goal_cell_prop'].max()
    bar_height = y_max * 0.1   # height increment for bars
    current_y = y_max + bar_height

    for (i, j) in itertools.combinations(range(len(xticks)), 2):
        pval = dunn.iloc[i, j]
        if pval < 0.05:
            # draw line
            x1, x2 = i, j
            ax.plot([x1, x1, x2, x2],
                    [current_y, current_y+h, current_y+h, current_y],
                    lw=1.5, c='k')
            # significance stars
            if pval < 0.001:
                stars = '***'
            elif pval < 0.01:
                stars = '**'
            else:
                stars = '*'
            ax.text((x1+x2)/2, current_y+0.015, stars,
                    ha='center', va='bottom', fontsize=fs)
            current_y += bar_height  # increment for next bar
fig.suptitle('Reward cells between two epochs')
plt.savefig(os.path.join(savedst, 'rew_cell_prop_btwn_two_ep.svg'), 
        bbox_inches='tight')
df_permsav.to_csv(r"C:\Users\Han\Documents\MATLAB\vip_reward_cell\raw_data\fig2c.csv")

#%%
# compare to shuffle
plt.rc('font', size=20)          # controls default text sizes
df_plt2 = pd.concat([df_permsav,df_plt.reset_index()])
# df_plt2 = df_plt2[df_plt2.animals!='e200']
df_plt2 = df_plt2[df_plt2.num_epochs<5]
df_plt2 = df_plt2.groupby(['animals', 'num_epochs']).mean(numeric_only=True)
df_plt2=df_plt2.reset_index()
# df_plt2=df_plt2[df_plt2.animals!='e189']
# number of epochs vs. reward cell prop incl combinations    
fig,axes = plt.subplots(figsize=(6.5,4),ncols=2,sharex=True)
ax=axes[0]
# av across mice
# sns.stripplot(x='num_epochs', y='goal_cell_prop',color='k',
        # data=df_plt2,s=10,alpha=0.7,ax=ax)
sns.barplot(x='num_epochs', y='goal_cell_prop',
        data=df_plt2,
        fill=False,ax=ax, color='cornflowerblue', errorbar='se')
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
    errorbar=None, color='gray', alpha=0.5, linewidth=1.5,ax=ax)
ax.set_xlabel('')
ax.set_ylabel('Reward cell % ')

eps = [2,3,4]
y = 37
pshift = 4
fs=38
pvalues = []
ts=[]
# Step 1: Compute p-values
for ep in eps:
    rewprop = df_plt2.loc[(df_plt2.num_epochs == ep), 'goal_cell_prop']
    shufprop = df_plt2.loc[(df_plt2.num_epochs == ep), 'goal_cell_prop_shuffle']
    t, pval = wilcoxon_r(rewprop, shufprop)
    print(f'{ep} epochs, pval: {pval}, r ={t}, n={len(rewprop)}')    
    pvalues.append(pval)
    ts.append(t)
    
from statsmodels.stats.multitest import fdrcorrection
# Step 2: FDR correction
reject, pvals_fdr = fdrcorrection(pvalues, alpha=0.05)

# Step 3: Annotate plot
for ii, (ep, pval_corr, sig) in enumerate(zip(eps, pvals_fdr, reject)):
    if pval_corr < 0.001:
        stars = "***"
    elif pval_corr < 0.01:
        stars = "**"
    elif pval_corr < 0.05:
        stars = "*"
    else:
        stars = "n.s."
    if sig:
        ax.text(ii, y, stars, ha='center', fontsize=fs)
    else:
        ax.text(ii, y, stars, ha='center', fontsize=fs, color='gray')  # Optional: fade non-sig
    ax.text(ii, y-10, f'p={pval_corr:.3g}\nt={ts[ii]:.3g}', ha='center', fontsize=10, color='k')  # Optional: fade non-sig

ax.set_title('Reward cells')

# subtract from shuffle
# df_plt2=df_plt2.reset_index()
df_plt3 = df_plt2.groupby(['num_epochs']).mean(numeric_only=True)
df_plt3=df_plt3.reset_index()
# subtract by average across animals?
sub = []
for ii,xx in df_plt2.iterrows():
        ep = xx.num_epochs
        sub.append(xx.goal_cell_prop-df_plt3.loc[df_plt3.num_epochs==ep, 'goal_cell_prop_shuffle'].values[0])
        
df_plt2['goal_cell_prop_sub_shuffle']=sub
# vs. average within animal
df_plt2['goal_cell_prop_sub_shuffle']=df_plt2['goal_cell_prop']-df_plt2['goal_cell_prop_shuffle']
ax=axes[1]# av across mice
# sns.stripplot(x='num_epochs', y='goal_cell_prop_sub_shuffle',color='cornflowerblue',
#         data=df_plt2,s=10,alpha=0.7,ax=ax)
sns.barplot(x='num_epochs', y='goal_cell_prop_sub_shuffle',
        data=df_plt2,
        fill=False,ax=ax, color='cornflowerblue', errorbar='se')
ax.spines[['top','right']].set_visible(False)
ax.set_title('Reward cell %-shuffle')
ax.set_ylim([0, 22])
# make lines
ans = df_plt2.animals.unique()
for i in range(len(ans)):
    ax = sns.lineplot(x=df_plt2.num_epochs-2, y='goal_cell_prop_sub_shuffle', 
    data=df_plt2[df_plt2.animals==ans[i]],
    errorbar=None, color='dimgray', alpha=0.5, linewidth=1.5)
    
ax.set_xlabel('# of epochs')
ax.set_ylabel('')
y=18
# Step 3: Annotate plot
for ii, (ep, pval_corr, sig) in enumerate(zip(eps, pvals_fdr, reject)):
    if pval_corr < 0.001:
        stars = "***"
    elif pval_corr < 0.01:
        stars = "**"
    elif pval_corr < 0.05:
        stars = "*"
    else:
        stars = "n.s."
    if sig:
        ax.text(ii, y, stars, ha='center', fontsize=fs)
    else:
        ax.text(ii, y, stars, ha='center', fontsize=fs, color='gray')  # Optional: fade non-sig

fig.suptitle('Dedicated reward cells')
plt.savefig(os.path.join(savedst, 'allreward_w_darktime_cell_prop-shuffle_per_an.svg'), 
        bbox_inches='tight')
df_plt2.to_csv(r"C:\Users\Han\Documents\MATLAB\vip_reward_cell\raw_data\fig2d.csv")

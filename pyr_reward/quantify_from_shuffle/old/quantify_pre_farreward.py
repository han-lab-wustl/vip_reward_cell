
"""
zahra
2025
far reward
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
from placecell import make_tuning_curves_radians_by_trialtype, intersect_arrays
from rewardcell import get_radian_position,create_mask_from_coordinates,pairwise_distances,extract_data_pre_farrew
from projects.opto.behavior.behavior import get_success_failure_trials
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
savepth = os.path.join(savedst, 'pre_far_rew.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_rewardcentric_all.p"
with open(saveddataset, "rb") as fp: #unpickle
        radian_alignment_saved = pickle.load(fp)
#%%
# initialize var
# radian_alignment_saved = {} # overwrite
goal_cell_iind = []
goal_cell_prop = []
goal_cell_null = []
dist_to_rew = [] # per epoch
num_epochs = []
pvals = []
rates_all = []
total_cells = []
epoch_perm = []
radian_alignment = {}
goal_window_cm = 20
lasttr=8 # last trials
bins=90
dists = []
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_farreward_all.p"
# iterate through all animals
for ii in range(len(conddf)):
        day = conddf.days.values[ii]
        animal = conddf.animals.values[ii]
        if (animal!='e217') & (conddf.optoep.values[ii]<2):
                if animal=='e145' or animal=='e139': pln=2 
                else: pln=0
                params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
                radian_alignment,rate,p_value,total_cells,goal_cell_iind,goal_cell_prop,num_epochs,\
                        goal_cell_null,epoch_perm,pvals=extract_data_pre_farrew(ii,params_pth,\
                        animal,day,bins,radian_alignment,radian_alignment_saved,goal_window_cm,
                        pdf,epoch_perm,goal_cell_iind,goal_cell_prop,num_epochs,goal_cell_null,pvals,
                        total_cells)
pdf.close()
# # save pickle of dcts
# with open(saveddataset, "wb") as fp:   #Pickling
#         pickle.dump(radian_alignment, fp) 
#%%
plt.rc('font', size=16)          # controls default text sizes
# plot goal cells across epochs
inds = [int(xx[-3:]) for xx in radian_alignment.keys()]
df = conddf.copy()
df = df[((df.animals!='e217')) & (df.optoep<2) & (df.index.isin(inds))]
df['num_epochs'] = num_epochs
df['goal_cell_prop'] = [xx[1] for xx in goal_cell_prop]
df['opto'] = df.optoep.values>1
df['day'] = df.days
df['session_num_opto'] = np.concatenate([[xx-df[df.animals==an].days.values[0] for xx in df[df.animals==an].days.values] for an in np.unique(df.animals.values)])
df['session_num'] = np.concatenate([[ii for ii,xx in enumerate(df[df.animals==an].days.values)] for an in np.unique(df.animals.values)])
df['condition'] = ['vip' if xx=='vip' else 'ctrl' for xx in df.in_type.values]
df['p_value'] = pvals
df['goal_cell_prop_shuffle'] = [xx[1] for xx in goal_cell_null]

fig,ax = plt.subplots(figsize=(5,5))
ax = sns.histplot(data = df.loc[df.opto==False], x='p_value', 
                hue='animals', bins=40)
ax.spines[['top','right']].set_visible(False)
ax.axvline(x=0.05, color='k', linestyle='--')
sessions_sig = sum(df.loc[df.opto==False,'p_value'].values<0.05)/len(df.loc[df.opto==False])
ax.set_title(f'{(sessions_sig*100):.2f}% of sessions are significant')
ax.set_xlabel('P-value')
ax.set_ylabel('Sessions')

# number of epochs vs. reward cell prop    
fig,ax = plt.subplots(figsize=(5,5))
df_plt = df
# av across mice
df_plt = df_plt.groupby(['animals','num_epochs']).mean(numeric_only=True)
df_plt=df_plt.reset_index()
df_plt=df_plt[df_plt.num_epochs<5]
sns.stripplot(x='num_epochs', y='goal_cell_prop',
        hue='animals',data=df_plt,
        s=10)
sns.barplot(x='num_epochs', y='goal_cell_prop',
        data=df_plt,
        fill=False,ax=ax, color='k', errorbar='se')
ax = sns.lineplot(data=df_plt, # correct shift
        x=df_plt.num_epochs-2, y='goal_cell_prop_shuffle',color='grey', 
        label='shuffle')
ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.01, 1.05))

eps = [2,3,4]
for ep in eps:
    # rewprop = df_plt.loc[(df_plt.num_epochs==ep), 'goal_cell_prop']
    rewprop = df_plt.loc[(df_plt.num_epochs==ep), 'goal_cell_prop']
    shufprop = df_plt.loc[(df_plt.num_epochs==ep), 'goal_cell_prop_shuffle']
    t,pval = scipy.stats.wilcoxon(rewprop, shufprop)
    print(f'{ep} epochs, pval: {pval}')
    
# include all comparisons 
df_perms = pd.DataFrame()
goal_cell_perm = [xx[0] for xx in goal_cell_prop]
goal_cell_perm_shuf = [xx[0][~np.isnan(xx[0])] for xx in goal_cell_null]
df_perms['goal_cell_prop'] = np.concatenate(goal_cell_perm)
df_perms['goal_cell_prop_shuffle'] = np.concatenate(goal_cell_perm_shuf)
df_perm_animals = [[xx]*len(goal_cell_perm[ii]) for ii,xx in enumerate(df.animals.values)]
df_perms['animals'] = np.concatenate(df_perm_animals)
df_perm_days = [[xx]*len(goal_cell_perm[ii]) for ii,xx in enumerate(df.session_num.values)]
df_perms['session_num'] = np.concatenate(df_perm_days)
# take a mean of all epoch comparisons
df_perms['num_epochs'] = [2]*len(df_perms)
df_permsav2 = df_perms.groupby(['animals', 'num_epochs']).mean(numeric_only=True)
df_permsav2=df_permsav2.reset_index()

# compare to shuffle
# df_permsav2=df_permsav2.reset_index()
df_plt2 = pd.concat([df_permsav2,df_plt])
df_plt2 = df_plt2[(df_plt2.animals!='e200') & (df_plt2.animals!='e189')]
df_plt2 = df_plt2[df_plt2.num_epochs<5]
df_plt2 = df_plt2.groupby(['animals', 'num_epochs']).mean(numeric_only=True)
df_plt2['goal_cell_prop']=df_plt2['goal_cell_prop']*100
df_plt2['goal_cell_prop_shuffle']=df_plt2['goal_cell_prop_shuffle']*100
df_plt2=df_plt2.reset_index()
# df_plt2 = df_plt2[df_plt2.animals!='z9']

# number of epochs vs. reward cell prop incl combinations    
fig,axes = plt.subplots(ncols=2,figsize=(7,5))
ax=axes[0]
# av across mice
sns.stripplot(x='num_epochs', y='goal_cell_prop',color='k',
        data=df_plt2,s=10,alpha=0.7,ax=ax)
sns.barplot(x='num_epochs', y='goal_cell_prop',
        data=df_plt2,
        fill=False,ax=ax, color='k', errorbar='se')
# ax = sns.lineplot(data=df_plt2, # correct shift
#         x=df_plt2.index.get_level_values('num_epochs').astype(int)-2, 
#         y='goal_cell_prop_shuffle',color='grey', 
#         label='shuffle')
# bar plot of shuffle instead
sns.barplot(data=df_plt2, # correct shift
        x='num_epochs', y='goal_cell_prop_shuffle',color='grey', 
        label='shuffle', alpha=0.5, err_kws={'color': 'grey'},errorbar=None,ax=ax)

ax.spines[['top','right']].set_visible(False)
ax.legend()#.set_visible(False)
ax.set_ylabel('Far reward cell %')
eps = [2,3,4]
y = 28
pshift = 1
fs=36
for ii,ep in enumerate(eps):
    rewprop = df_plt2.loc[(df_plt2.num_epochs==ep), 'goal_cell_prop']
    shufprop = df_plt2.loc[(df_plt2.num_epochs==ep), 'goal_cell_prop_shuffle']
    t,pval = scipy.stats.wilcoxon(rewprop, shufprop)
    print(f'{ep} epochs, pval: {pval}')
    # statistical annotation        
    if pval < 0.001:
            ax.text(ii, y, "***", ha='center', fontsize=fs)
    elif pval < 0.01:
            ax.text(ii, y, "**", ha='center', fontsize=fs)
    elif pval < 0.05:
            ax.text(ii, y, "*", ha='center', fontsize=fs)
        # ax.text(ii-0.5, y+pshift, f'p={pval:.3g}',fontsize=10,rotation=45)
# make lines
ans = df_plt2.animals.unique()
for i in range(len(ans)):
    ax = sns.lineplot(x=df_plt2.num_epochs-2, y='goal_cell_prop', 
    data=df_plt2[df_plt2.animals==ans[i]],
    errorbar=None, color='dimgray', linewidth=2, alpha=0.7,ax=ax)
ax.set_title('Far pre-reward cells',pad=30)
ax.set_xlabel('')
ax.set_ylim([0,30])
ax=axes[1]
# subtract from shuffle
# df_plt2=df_plt2.reset_index()
df_plt2['goal_cell_prop_sub_shuffle'] = df_plt2['goal_cell_prop']-df_plt2['goal_cell_prop_shuffle']
# av across mice
sns.stripplot(x='num_epochs', y='goal_cell_prop_sub_shuffle',color='cornflowerblue',
        data=df_plt2,s=10,alpha=0.7,ax=ax)
sns.barplot(x='num_epochs', y='goal_cell_prop_sub_shuffle',
        data=df_plt2,
        fill=False,ax=ax, color='cornflowerblue', errorbar='se')
# make lines
df_plt2=df_plt2.reset_index()
ans = df_plt2.animals.unique()
for i in range(len(ans)):
    ax = sns.lineplot(x=df_plt2.num_epochs-2, y='goal_cell_prop_sub_shuffle', 
    data=df_plt2[df_plt2.animals==ans[i]],
    errorbar=None, color='dimgray', linewidth=2, alpha=0.7,ax=ax)

ax.spines[['top','right']].set_visible(False)
ax.set_xlabel('# of reward loc. switches')
ax.set_ylabel('')
ax.set_title('Far pre-reward cell %-shuffle',pad=30)
ax.set_ylim([-1,8])

plt.savefig(os.path.join(savedst, 'pre_farreward_cell_prop-shuffle_per_an.svg'), 
        bbox_inches='tight')

#%% 
# find tau/decay

from scipy.optimize import curve_fit

# Define the exponential decay function
def exponential_decay(t, A, tau):
    return A * np.exp(-t / tau)
tau_all = []; y_fit_all = []
# df_plt2=df_plt2.reset_index()
for an in df_plt2.animals.unique():
        # Initial guesses for the optimization
        initial_guess = [4, 2]  # Amplitude guess and tau guess
        y = df_plt2[df_plt2.animals==an]
        t=np.array([2,3,4])
        # Fit the model to the data using curve_fit
        params, params_covariance = curve_fit(exponential_decay, t, y.goal_cell_prop.values, p0=initial_guess)
        # Extract the fitted parameters
        A_fit, tau_fit = params
        tau_all.append(tau_fit)
        # Generate the fitted curve using the optimized parameters
        y_fit = exponential_decay(t, A_fit, tau_fit)
        y_fit_all.append(y_fit)
        
# plot fit
fig,ax = plt.subplots(figsize=(3,5))
sns.stripplot(x='num_epochs', y='goal_cell_prop',color='k',
        data=df_plt2,s=10,alpha=0.7)
plt.plot(np.array(y_fit_all).T,color='grey')

ax.spines[['top','right']].set_visible(False)
ax.legend()#.set_visible(False)
ax.set_xlabel('# of reward loc. switches')
ax.set_ylabel('Reward-centric cell proportion')
# plt.savefig(os.path.join(savedst, 'expo_fit_reward_centric.png'), 
#         bbox_inches='tight')

#%%
# compare to persistent cells
tau_all_postrew = [2.5081012042756297,
 1.3564559842969026,
 3.067144722017443,
 4.017594663159857,
 2.307820130958938,
 1.814554708027948,
 1.9844914154882163,
 1.7613987163758171,
 2.403541822072123]

tau_all_prerew =[1.6056888447006052,
 1.7426458455678147,
 5.050873750828834,
 2.2320785654220607,
 1.8576189935003193,
 1.386477229747204,
 1.6253067659011684,
 1.4999119163136394,
 2.5827747398002434]

tau_far_pre = tau_all
tau_far_post = [
    
]
df = pd.DataFrame()
df['tau'] = np.concatenate([tau_all,tau_all_postrew,tau_all_prerew])
df['cell_type'] =np.concatenate([['Far-reward']*len(tau_all),
                                ['Post-reward']*len(tau_all_postrew),
                                ['Pre-reward']*len(tau_all_prerew)])
# number of epochs vs. reward cell prop incl combinations    
fig,ax = plt.subplots(figsize=(3,5))
# av across mice
sns.stripplot(x='cell_type', y='tau',color='k',
        data=df,s=10,alpha=0.7)
sns.barplot(x='cell_type', y='tau',
        data=df, fill=False,ax=ax, color='k', errorbar='se')
ax.spines[['top','right']].set_visible(False)
ax.legend().set_visible(False)
ax.set_ylabel(f'Decay over epochs ($\\tau$)')
ax.set_xlabel('')
ax.tick_params(axis='x', rotation=45)

import scipy.stats as stats
from scikit_posthocs import posthoc_dunn
from statsmodels.stats.multitest import multipletests

df = df.reset_index()

# Perform Kruskal-Wallis test
grouped = [df[df['cell_type'] == group]['tau'] for group in df['cell_type'].unique()]
kruskal_result = stats.kruskal(*grouped)
print(f"Kruskal-Wallis H-statistic: {kruskal_result.statistic}, p-value: {kruskal_result.pvalue}")

# Perform Dunn's test for pairwise comparisons and apply Bonferroni correction
dunn_result = posthoc_dunn(df, val_col='tau', group_col='cell_type',p_adjust='bonferroni')
# Annotate the plot
# Annotate the plot
def add_stat_annotation(ax, x1, x2, y, adjusted_p):
        h = 0.2  # height offset
        line_offset = 0.1  # offset of the horizontal line
        # Draw horizontal line
        ax.plot([x1, x1, x2, x2], [y, y + line_offset, y + line_offset, y], lw=1.5, c='k')
        # Determine significance level
        if adjusted_p < 0.001:
                significance = '***'
        elif adjusted_p < 0.01:
                significance = '**'
        elif adjusted_p < 0.05:
                significance = '*'
        else:
                significance = 'ns'  # Not significant
        # Combine significance level and p-value in the annotation
        text = f'{significance}\n(p={adjusted_p:.3g})'
        line_offset2=.5
        # Add p-value text
        ax.text((x1 + x2) * 0.5, y + line_offset2, text, ha='center', va='bottom', color='k',
                fontsize=14)

# Example usage of add_stat_annotation with group annotations and asterisks
y_max = df['tau'].max()-5  # maximum y value of the plot
y_range = df['tau'].max() - df['tau'].min()  # range of y values

cell_types = df['cell_type'].unique()
for i, group1 in enumerate(cell_types):
        for j, group2 in enumerate(cell_types):
                if i < j:
                        p_value = dunn_result.loc[group1, group2]
                        adjusted_p = p_value  # Already Bonferroni-adjusted
                        x1 = list(cell_types).index(group1)
                        x2 = list(cell_types).index(group2)
                        y = y_max + (i + j+3) * y_range * .1  # adjust y position
                        print(i,y)
                        add_stat_annotation(ax, x1, x2, y+i+j, adjusted_p)

plt.savefig(os.path.join(savedst, 'decay_rewardcell.svg'), 
        bbox_inches='tight')

#%%
# as a function of session/day
df_plt = df.groupby(['animals','session_num','num_epochs']).mean(numeric_only=True)
df_permsav2 = df_perms.groupby(['animals', 'session_num','num_epochs']).mean(numeric_only=True)
# compare to shuffle
df_plt2 = pd.concat([df_permsav2,df_plt])
# df_plt2 = df_plt2[df_plt2.index.get_level_values('animals')!='e189']
df_plt2 = df_plt2[df_plt2.index.get_level_values('num_epochs')==2]
df_plt2 = df_plt2.groupby(['animals', 'session_num','num_epochs']).mean(numeric_only=True)
# number of epochs vs. reward cell prop incl combinations    
fig,ax = plt.subplots(figsize=(7,5))
# av across mice
sns.stripplot(x='session_num', y='goal_cell_prop',hue='animals',
        data=df_plt2,s=10,alpha=0.7)
sns.barplot(x='session_num', y='goal_cell_prop',color='darkslateblue',
        data=df_plt2,fill=False,ax=ax, errorbar='se')
ax.set_xlim([-.5,9.5])
# ax = sns.lineplot(data=df_plt2, # correct shift
#         x=df_plt2.index.get_level_values('num_epochs').astype(int)-2, y='goal_cell_prop_shuffle',color='grey', 
#         label='shuffle')
ax.spines[['top','right']].set_visible(False)
# ax.legend().set_visible(False)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax.set_xlabel('# of sessions')
ax.set_ylabel('Reward-distance cell proportion')
df_reset = df_plt2.reset_index()
sns.regplot(x='session_num', y='goal_cell_prop',
        data=df_reset, scatter=False, color='k')
r, p = scipy.stats.pearsonr(df_reset['session_num'], 
        df_reset['goal_cell_prop'])
ax = plt.gca()
ax.text(.5, .8, 'r={:.3f}, p={:.3g}'.format(r, p),
        transform=ax.transAxes)
ax.set_title('2 epoch combinations')

#%%
df['recorded_neurons_per_session'] = total_cells
df_plt_ = df[(df.opto==False)&(df.p_value<0.05)]
df_plt_= df_plt_[(df_plt_.animals!='e200')&(df_plt_.animals!='e189')]
df_plt_ = df_plt_.groupby(['animals']).mean(numeric_only=True)

fig,ax = plt.subplots(figsize=(7,5))
sns.scatterplot(x='recorded_neurons_per_session', y='goal_cell_prop',hue='animals',
        data=df_plt_,
        s=150, ax=ax)
sns.regplot(x='recorded_neurons_per_session', y='goal_cell_prop',
        data=df_plt_,
        ax=ax, scatter=False, color='k'
)
r, p = scipy.stats.pearsonr(df_plt_['recorded_neurons_per_session'], 
        df_plt_['goal_cell_prop'])
ax = plt.gca()
ax.text(.5, .8, 'r={:.2f}, p={:.2g}'.format(r, p),
        transform=ax.transAxes)

ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.01, 1.05))
ax.set_xlabel('Av. # of neurons per session')
ax.set_ylabel('Reward cell proportion')

plt.savefig(os.path.join(savedst, 'rec_cell_rew_prop_per_an.svg'), 
        bbox_inches='tight')

#%%
df['success_rate'] = rates_all

# all animals
fig,ax = plt.subplots(figsize=(7,5))
sns.scatterplot(x='success_rate', y='goal_cell_prop',hue='animals',
        data=df,
        s=150, ax=ax)
sns.regplot(x='success_rate', y='goal_cell_prop',
        data=df,
        ax=ax, scatter=False, color='k'
)
r, p = scipy.stats.pearsonr(df['success_rate'], 
        df['goal_cell_prop'])
ax = plt.gca()
ax.text(.5, .8, 'r={:.2f}, p={:.2g}'.format(r, p),
        transform=ax.transAxes)

ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.01, 1.05))
ax.set_xlabel('Success rate')
ax.set_ylabel('Reward cell proportion')
#%%
an_nms = df.animals.unique()
rows = int(np.ceil(np.sqrt(len(an_nms))))
cols = int(np.ceil(np.sqrt(len(an_nms))))
fig,axes = plt.subplots(nrows=rows, ncols=cols,
            figsize=(10,10))
rr=0;cc=0
for an in an_nms:        
    ax = axes[rr,cc]
    sns.scatterplot(x='success_rate', y='goal_cell_prop',
            data=df[(df.animals==an)&(df.opto==False)&(df.p_value<0.05)],
            s=200, ax=ax)
    ax.spines[['top','right']].set_visible(False)
    ax.set_title(an)
    rr+=1
    if rr>=rows: rr=0; cc+=1    
fig.tight_layout()
#%%
# # #examples
fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
Fc3 = fall_fc3['Fc3']
dFF = fall_fc3['dFF']
Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
Fc3 = Fc3[:,(skew>2)] # only keep cells with skew greateer than 2
bin_size=3 # cm
# get abs dist tuning 
tcs_correct_abs, coms_correct_abs, tcs_fail, coms_fail = make_tuning_curves_by_trialtype(eps,rewlocs,ybinned,
Fc3,trialnum,rewards,forwardvel,rewsize,bin_size)

# # #plot example tuning curve
plt.rc('font', size=30)  
fig,axes = plt.subplots(1,3,figsize=(20,20), sharex = True)
for ep in range(3):
        axes[ep].imshow(tcs_correct_abs[ep,com_goal[0]][np.argsort(coms_correct_abs[0,com_goal[0]])[:60],:]**.3)
        axes[ep].set_title(f'Epoch {ep+1}')
        axes[ep].axvline((rewlocs[ep]-rewsize/2)/bin_size, color='w', linestyle='--', linewidth=4)
        axes[ep].set_xticks(np.arange(0,(track_length/bin_size)+bin_size,30))
        axes[ep].set_xticklabels(np.arange(0,track_length+bin_size*30,bin_size*30).astype(int))
axes[0].set_ylabel('Reward-distance cells')
axes[2].set_xlabel('Absolute distance (cm)')
plt.savefig(os.path.join(savedst, 'abs_dist_tuning_curves_3_ep.svg'), bbox_inches='tight')

# fig,axes = plt.subplots(1,4,figsize=(15,20), sharey=True, sharex = True)
# axes[0].imshow(tcs_correct[0,com_goal[0]][np.argsort(coms_correct[0,com_goal[0]])[:60],:]**.5)
# axes[0].set_title('Epoch 1')
# im = axes[1].imshow(tcs_correct[1,com_goal[0]][np.argsort(coms_correct[0,com_goal[0]])[:60],:]**.5)
# axes[1].set_title('Epoch 2')
# im = axes[2].imshow(tcs_correct[2,com_goal[0]][np.argsort(coms_correct[0,com_goal[0]])[:60],:]**.5)
# axes[2].set_title('Epoch 3')
# im = axes[3].imshow(tcs_correct[3,com_goal[0]][np.argsort(coms_correct[0,com_goal[0]])[:60],:]**.5)
# axes[3].set_title('Epoch 4')
# ax = axes[1]
# ax.set_xticks(np.arange(0,bins+1,10))
# ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi+np.pi/4.5, np.pi/4.5),1))
# ax.axvline((bins/2), color='w', linestyle='--')
# axes[0].axvline((bins/2), color='w', linestyle='--')
# axes[2].axvline((bins/2), color='w', linestyle='--')
# axes[3].axvline((bins/2), color='w', linestyle='--')
# axes[0].set_ylabel('Reward distance cells')
# axes[3].set_xlabel('Reward-relative distance (rad)')
# fig.tight_layout()
# plt.savefig(os.path.join(savedst, 'tuning_curves_4_ep.png'), bbox_inches='tight')
# for gc in goal_cells:
#%%
gc = 51
plt.rc('font', size=24)  
fig2,ax2 = plt.subplots(figsize=(5,5))

for ep in range(3):        
        ax2.plot(tcs_correct_abs[ep,gc,:], label=f'rewloc {rewlocs[ep]}', color=colors[ep],linewidth=3)
        ax2.axvline(rewlocs[ep]/bin_size, color=colors[ep], linestyle='--',linewidth=3)
        
        ax2.spines[['top','right']].set_visible(False)
ax2.set_title(f'animal: {animal}, day: {day}\ncell # {gc}')
ax2.set_xticks(np.arange(0,(track_length/bin_size)+bin_size,30))
ax2.set_xticklabels(np.arange(0,track_length+bin_size*30,bin_size*30).astype(int))
ax2.set_xlabel('Absolute position (cm)')
ax2.set_ylabel('$\Delta$ F/F')
        
plt.savefig(os.path.join(savedst, f'rewardd_cell_{gc}_tuning_per_ep.svg'), bbox_inches='tight')

fig2,ax2 = plt.subplots(figsize=(5,5))
for ep in range(3):        
        ax2.plot(tcs_correct[ep,gc,:], label=f'rewloc {rewlocs[ep]}', color=colors[ep],linewidth=3)
        ax2.axvline(bins/2, color="k", linestyle='--',linewidth=3)
        
        ax2.spines[['top','right']].set_visible(False)
ax2.set_title(f'animal: {animal}, day: {day}\ncell # {gc}')
ax2.set_xticks(np.arange(0,bins+1,30))
ax2.set_xticklabels(np.round(np.arange(-np.pi, 
        np.pi+np.pi/1.5, np.pi/1.5),1))
ax2.set_xlabel('Radian position ($\Theta$)')
ax2.set_ylabel('$\Delta$ F/F')
plt.savefig(os.path.join(savedst, f'rewardd_cell_{gc}_aligned_tuning_per_ep.svg'), bbox_inches='tight')
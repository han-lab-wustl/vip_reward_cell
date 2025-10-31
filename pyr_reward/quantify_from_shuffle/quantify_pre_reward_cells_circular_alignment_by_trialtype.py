
#%%
"""
zahra
july 2024
quantify reward-relative cells near reward
"""

import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd, os
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"]=8
mpl.rcParams["ytick.major.size"]=8
# plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from placecell import make_tuning_curves_radians_by_trialtype, intersect_arrays
from projects.opto.behavior.behavior import get_success_failure_trials
from rewardcell import get_radian_position,extract_data_prerew
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
savepth = os.path.join(savedst, 'pre_rew.pdf')
#%%
goal_cm_window=20 # to search for rew cells
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
saveddataset = rf'Z:\saved_datasets\radian_tuning_curves_prereward_cell_bytrialtype_nopto_{goal_cm_window}cm_window.p'
with open(saveddataset, "rb") as fp: #unpickle
    radian_alignment_saved = pickle.load(fp)
radian_alignment_saved = {} # overwrite
goal_cell_iind = []
goal_cell_prop = []
goal_cell_null = []
num_epochs = []
pvals = []
rates_all = []
total_cells = []
epoch_perm = []
radian_alignment = {}
lasttr=8 #  last trials
bins=90
saveto = rf'Z:\saved_datasets\radian_tuning_curves_prereward_cell_bytrialtype_nopto_{goal_cm_window}cm_window.p'
# iterate through all animals
for ii in range(len(conddf)):
        day = conddf.days.values[ii]
        animal = conddf.animals.values[ii]
        if (animal!='e217') & (conddf.optoep.values[ii]<2):
                if animal=='e145' or animal=='e139': pln=2 
                else: pln=0
                params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
                radian_alignment,rate,p_value,total_cells,goal_cell_iind,\
                goal_cell_prop,num_epochs,goal_cell_null,epoch_perm,pvals=extract_data_prerew(ii,params_pth,
                animal,day,bins,radian_alignment,radian_alignment_saved,goal_cm_window,
                pdf,epoch_perm,goal_cell_iind,goal_cell_prop,num_epochs,goal_cell_null,pvals,
                total_cells)
pdf.close()
# save pickle of dcts
with open(saveto, "wb") as fp:   #Pickling
        pickle.dump(radian_alignment, fp)
#%%
plt.rc('font', size=16)          # controls default text sizes
# plot goal cells across epochs
inds = [int(xx[-3:]) for xx in radian_alignment.keys()]
df = conddf.copy()
df = df[((df.animals!='e217')) & (df.optoep<2)]
df['num_epochs'] = num_epochs
df['goal_cell_prop'] = [xx[1] for xx in goal_cell_prop]
df['opto'] = df.optoep.values>1
df['condition'] = ['vip' if xx=='vip' else 'ctrl' for xx in df.in_type.values]
df['p_value'] = pvals
df['goal_cell_prop_shuffle'] = [xx[1] for xx in goal_cell_null]
df['session_num_opto'] = np.concatenate([[xx-df[df.animals==an].days.values[0] for xx in df[df.animals==an].days.values] for an in np.unique(df.animals.values)])
df['session_num'] = np.concatenate([[ii for ii,xx in enumerate(df[df.animals==an].days.values)] for an in np.unique(df.animals.values)])

fig,ax = plt.subplots(figsize=(5,5))
ax = sns.histplot(data = df.loc[df.opto==False], x='p_value', hue='animals', bins=40)
ax.spines[['top','right']].set_visible(False)
ax.axvline(x=0.05, color='k', linestyle='--')
sessions_sig = sum(df.loc[df.opto==False ,'p_value'].values<0.05)/len(df.loc[df.opto==False])
ax.set_title(f'{(sessions_sig*100):.2f}% of sessions are significant')
ax.set_xlabel('P-value')
ax.set_ylabel('Sessions')
#%%
# number of epochs vs. reward cell prop    
fig,ax = plt.subplots(figsize=(3,5))
df_plt = df[(df.opto==False)]
# av across mice
df_plt = df_plt.groupby(['animals','num_epochs']).mean(numeric_only=True)
sns.stripplot(x='num_epochs', y='goal_cell_prop',
        hue='animals',data=df_plt,
        s=10)
sns.barplot(x='num_epochs', y='goal_cell_prop',
        data=df_plt,
        fill=False,ax=ax, color='k', errorbar='se')
ax = sns.lineplot(data=df_plt, # correct shift
        x=df_plt.index.get_level_values('num_epochs')-2, y='goal_cell_prop_shuffle',color='grey', 
        label='shuffle')
ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.01, 1.05))

eps = [2,3,4]
for ep in eps:
    # rewprop = df_plt.loc[(df_plt.num_epochs==ep), 'goal_cell_prop']
    rewprop = df_plt.loc[(df_plt.index.get_level_values('num_epochs')==ep), 'goal_cell_prop']
    shufprop = df_plt.loc[(df_plt.index.get_level_values('num_epochs')==ep), 'goal_cell_prop_shuffle']
    t,pval = scipy.stats.ranksums(rewprop, shufprop)
    print(f'{ep} epochs, pval: {pval}')
    
# include all comparisons 
df_perms = pd.DataFrame()
# epcomp= [str(tuple(xx)) for xx in np.concatenate(epoch_perm)]
goal_cell_perm = [xx[0] for xx in goal_cell_prop]
goal_cell_perm_shuf = [xx[0][~np.isnan(xx[0])] for xx in goal_cell_null]
# df_perms['epoch_comparison']=
df_perms['goal_cell_prop'] = np.concatenate(goal_cell_perm)
# df_perms['goal_cell_prop_shuffle'] = np.concatenate(goal_cell_perm_shuf)
df_perm_animals = [[xx]*len(goal_cell_perm[ii]) for ii,xx in enumerate(df.animals.values)]
df_perms['animals'] = np.concatenate(df_perm_animals)
df_perm_days = [[xx]*len(goal_cell_perm[ii]) for ii,xx in enumerate(df.session_num.values)]
df_perms['session_num'] = np.concatenate(df_perm_days)

df_perms = df_perms[df_perms.animals!='e189']
# skipped fro now because it wasn't working
# df_permsav = df_perms.groupby(['animals','epoch_comparison']).mean(numeric_only=True)

# fig,ax = plt.subplots(figsize=(7,5))
# sns.stripplot(x='epoch_comparison', y='goal_cell_prop',
#         hue='animals',data=df_permsav,
#         s=8,ax=ax)
# sns.barplot(x='epoch_comparison', y='goal_cell_prop',
#         data=df_permsav,
#         fill=False,ax=ax, color='k', errorbar='se')
# ax = sns.lineplot(data=df_permsav, # correct shift
#         x='epoch_comparison', y='goal_cell_prop_shuffle',
#         color='grey', label='shuffle')

# ax.spines[['top','right']].set_visible(False)
# ax.legend(bbox_to_anchor=(1.01, 1.05))
# #%%
# eps = df_permsav.index.get_level_values("epoch_comparison").unique()
# for ep in eps:
#     # rewprop = df_plt.loc[(df_plt.num_epochs==ep), 'goal_cell_prop']
#     rewprop = df_permsav.loc[(df_permsav.index.get_level_values('epoch_comparison')==ep), 'goal_cell_prop'].values
#     shufprop = df_permsav.loc[(df_permsav.index.get_level_values('epoch_comparison')==ep), 'goal_cell_prop_shuffle'].values
#     t,pval = scipy.stats.ranksums(rewprop, shufprop)
#     print(f'{ep} epochs, pval: {pval}')

# take a mean of all epoch comparisons
df_perms['num_epochs'] = [2]*len(df_perms)
df_permsav2 = df_perms.groupby(['animals', 'num_epochs']).mean(numeric_only=True)
s=10
df_plt2 = pd.concat([df_permsav2,df_plt])
df_plt2 = df_plt2[df_plt2.index.get_level_values('num_epochs')<5]
df_plt2 = df_plt2.groupby(['animals', 'num_epochs']).mean(numeric_only=True)
df_plt2=df_plt2.reset_index()
df_plt2 = df_plt2[(df_plt2.animals!='e189') & (df_plt2.animals!='e200')]
df_plt2['goal_cell_prop']=df_plt2['goal_cell_prop']*100
df_plt2['goal_cell_prop_shuffle']=df_plt2['goal_cell_prop_shuffle']*100
# number of epochs vs. reward cell prop incl combinations    
fig,axes = plt.subplots(ncols=2,figsize=(7,5))
ax=axes[0]
# av across mice
sns.stripplot(x='num_epochs', y='goal_cell_prop',color='k',
        data=df_plt2,
        s=s,alpha=0.7,ax=ax)
sns.barplot(x='num_epochs', y='goal_cell_prop',
        data=df_plt2,
        fill=False,ax=ax, color='k', errorbar='se')
# bar plot of shuffle instead
sns.barplot(data=df_plt2, # correct shift
        x='num_epochs', y='goal_cell_prop_shuffle',color='grey', 
        label='shuffle', alpha=0.5, err_kws={'color': 'grey'},errorbar=None,ax=ax)
# make lines
ans = df_plt2.animals.unique()
for i in range(len(ans)):
    ax = sns.lineplot(x=df_plt2.num_epochs-2, y='goal_cell_prop', 
    data=df_plt2[df_plt2.animals==ans[i]],
    errorbar=None, color='dimgray', linewidth=2, alpha=0.7,ax=ax)
ax.spines[['top','right']].set_visible(False)
ax.legend()
ax.set_ylabel('Pre-reward cell %')
ax.set_xlabel('')
eps = [2,3,4]
y = 28
pshift=3
fs=36
for ii,ep in enumerate(eps):
        rewprop = df_plt2.loc[(df_plt2.num_epochs==ep), 'goal_cell_prop']
        shufprop = df_plt2.loc[(df_plt2.num_epochs==ep), 'goal_cell_prop_shuffle']
        t,pval = scipy.stats.ttest_rel(rewprop[~np.isnan(shufprop.values)], shufprop.values[~np.isnan(shufprop.values)])
        print(f'{ep} epochs, pval: {pval}')
        # statistical annotation        
        if pval < 0.001:
                ax.text(ii, y, "***", ha='center', fontsize=fs)
        elif pval < 0.01:
                ax.text(ii, y, "**", ha='center', fontsize=fs)
        elif pval < 0.05:
                ax.text(ii, y, "*", ha='center', fontsize=fs)
        # ax.text(ii, y+pshift, f'p={pval:.2g}',rotation=45,fontsize=12)
ax.set_title('Pre-reward cells',pad=30)
ax.set_ylim([0, 30])
# subtract from shuffle
# df_plt2=df_plt2.reset_index()
df_plt2['goal_cell_prop_sub_shuffle'] = df_plt2['goal_cell_prop']-df_plt2['goal_cell_prop_shuffle']
ax=axes[1]
# av across mice
sns.stripplot(x='num_epochs', y='goal_cell_prop_sub_shuffle',color='cornflowerblue',
        data=df_plt2,s=s,alpha=0.7,ax=ax)
sns.barplot(x='num_epochs', y='goal_cell_prop_sub_shuffle',
        data=df_plt2,
        fill=False,ax=ax, color='cornflowerblue', errorbar='se')
# make lines
ans = df_plt2.animals.unique()
for i in range(len(ans)):
    ax = sns.lineplot(x=df_plt2.num_epochs-2, y='goal_cell_prop_sub_shuffle', 
    data=df_plt2[df_plt2.animals==ans[i]],
    errorbar=None, color='dimgray', linewidth=2, alpha=0.7,ax=ax)

ax.spines[['top','right']].set_visible(False)
ax.set_ylabel('')
ax.set_xlabel('# of reward loc. switches')
ax.set_ylim([-1, 8])
ax.set_title('Pre-reward cell %-shuffle',pad=30)

plt.savefig(os.path.join(savedst, 'prereward_cell_prop-shuffle_per_an.svg'), 
        bbox_inches='tight')

#%%
# find tau/decay
from scipy.optimize import curve_fit
# Define the exponential decay function
def exponential_decay(t, A, tau):
    return A * np.exp(-t / tau)
tau_all = []
for an in df_plt2.animals.unique():
        try:
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
        except:
                print(an)


#%%
# as a function of session/day
df_plt = df.groupby(['animals','session_num','num_epochs']).mean(numeric_only=True)
df_permsav2 = df_perms.groupby(['animals', 'session_num','num_epochs']).mean(numeric_only=True)
# compare to shuffle
df_plt2 = pd.concat([df_permsav2,df_plt])
# df_plt2 = df_plt2[df_plt2.index.get_level_values('animals')!='e189']
df_plt2 = df_plt2[(df_plt2.index.get_level_values('num_epochs')==2) & (df_plt2.index.get_level_values('animals')!='e200')]
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
# per session
df_plt2 = pd.concat([df_perms,df])
# df_plt2 = df_plt2[df_plt2.index.get_level_values('animals')!='e189']
df_plt2 = df_plt2[(df_plt2.num_epochs<5) & (df_plt2.num_epochs>1)]
# df_plt2 = df_plt2.groupby(['animals', 'num_epochs']).mean(numeric_only=True)
# number of epochs vs. reward cell prop incl combinations    
fig,ax = plt.subplots(figsize=(6,5))
# av across mice
sns.stripplot(x='num_epochs', y='goal_cell_prop',hue='animals',s=10,alpha=0.4,
        data=df_plt2,dodge=True)
sns.barplot(x='num_epochs', y='goal_cell_prop',hue='animals',
        data=df_plt2,
        fill=False,ax=ax, errorbar='se')

ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.01, 1.1))
ax.set_ylabel('Post reward cell proportion')
ax.set_title('Post-reward cells')
plt.savefig(os.path.join(savedst, 'postrew_cell_prop_per_session.svg'), 
        bbox_inches='tight')
#%%
df['success_rate'] = rates_all
dffil = df[df.goal_cell_prop>0]
dffil=dffil[dffil.success_rate>.6]
# all animals
fig,ax = plt.subplots(figsize=(7,5))
sns.scatterplot(x='success_rate', y='goal_cell_prop',hue='animals',
        data=dffil,
        s=100, ax=ax)
sns.regplot(x='success_rate', y='goal_cell_prop',
        data=dffil,
        ax=ax, scatter=False, color='k'
)
r, p = scipy.stats.pearsonr(dffil['success_rate'], 
        dffil['goal_cell_prop'])
ax = plt.gca()
ax.text(.5, .8, 'r={:.2f}, p={:.2g}'.format(r, p),
        transform=ax.transAxes)

ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.01, 1.05))
ax.set_xlabel('Success rate')
ax.set_ylabel('Post-reward cell proportion')
plt.savefig(os.path.join(savedst, 'postrew_v_correctrate.svg'), 
        bbox_inches='tight')
#%%

an_nms = dffil[dffil.animals!='e189'].animals.unique()
num_plots = len(an_nms)
rows = int(np.ceil(np.sqrt(num_plots)))
cols = int(np.ceil(num_plots / rows))

fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 10))
axes = axes.flatten()  # Flatten the axes array for easier plotting

for i, an in enumerate(an_nms):
        ax = axes[i]
        sns.scatterplot(x='success_rate', y='goal_cell_prop', data=dffil[(dffil.animals == an)], s=100, ax=ax)
        sns.regplot(x='success_rate', y='goal_cell_prop', data=dffil[(dffil.animals == an)], ax=ax, scatter=False, color='k')
        ax.spines[['top','right']].set_visible(False)
        ax.set_title(an)
        try:
                r, p = scipy.stats.pearsonr(dffil[(dffil.animals == an)]['success_rate'], 
                dffil[(dffil.animals == an)]['goal_cell_prop'])
                ax.text(.2, .5, 'r={:.2f}, p={:.2g}'.format(r, p),
                        transform=ax.transAxes)
        except Exception as e:
                print(e)

# Hide any remaining unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

fig.tight_layout()
plt.show()

#%%


"""
zahra
april 2025
get place cells and plot com histogram
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.pyr_reward.placecell import make_tuning_curves, intersect_arrays
from projects.pyr_reward.rewardcell import get_radian_position, wilcoxon_r
from projects.opto.behavior.behavior import get_success_failure_trials
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
savepth = os.path.join(savedst, 'true_pc.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
saveddataset = r"Z:\saved_datasets\tuning_curves_pcs_nopto.p"
# with open(saveddataset, "rb") as fp: #unpickle
#         radian_alignment_saved = pickle.load(fp)
# initialize var
#%%
datadct = {} # overwrite
coms_all = []
pc_ind = []
pc_prop = []
num_epochs = []
epoch_perm = []
pvals = []
total_cells = []
num_iterations=1000
place_cell_null=[]
# iterate through all animals
for ii in range(len(conddf)):
    day = conddf.days.values[ii]
    animal = conddf.animals.values[ii]
    if (animal!='e217') and (conddf.optoep.values[ii]<2):
        if animal=='e145' or animal=='e139': pln=2 
        else: pln=0
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
        print(params_pth)
        fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
            'putative_pcs', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
            'stat'])
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
        if animal=='e145':
                ybinned=ybinned[:-1]
                forwardvel=forwardvel[:-1]
                changeRewLoc=changeRewLoc[:-1]
                trialnum=trialnum[:-1]
                rewards=rewards[:-1]
        # set vars
        eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf
        eps = np.append(eps, len(changeRewLoc)) 
        # check if done full epoch for last epoch
        trials=np.unique(trialnum[eps[len(eps)-2]:eps[len(eps)-1]])
        if len(trials)<15:
                eps = eps[:-1]
        epochs = len(eps)-1
        # if epochs>3:   
        lasttr=8 # last trials
        bins=90
        fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
        Fc3 = fall_fc3['Fc3']
        dFF = fall_fc3['dFF']
        Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
        dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
        skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
        #if pc in all but 1
        pc_bool = np.sum(pcs,axis=0)>=len(eps)-2
        # looser restrictions
        pc_bool = np.sum(pcs,axis=0)>=1
        Fc3 = Fc3[:,((skew>2)&pc_bool)] # only keep cells with skew greateer than 2
        # if no cells pass these crit
        if Fc3.shape[1]==0:
                Fc3 = fall_fc3['Fc3']
                Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
                pc_bool = np.sum(pcs,axis=0)>=1
                Fc3 = Fc3[:,((skew>2)&pc_bool)]
        bin_size=3 # cm
        # get abs dist tuning 
        tcs_correct_abs, coms_correct_abs,_,__ = make_tuning_curves(eps,rewlocs,ybinned,
        Fc3,trialnum,rewards,forwardvel,
        rewsize,bin_size)
        # get cells that maintain their coms across at least 2 epochs
        place_window = 20 # cm converted to rad                
        perm = list(combinations(range(len(coms_correct_abs)), 2))     
        com_per_ep = np.array([(coms_correct_abs[perm[jj][0]]-coms_correct_abs[perm[jj][1]]) for jj in range(len(perm))])        
        compc = [np.where((comr<place_window) & (comr>-place_window))[0] for comr in com_per_ep]
        # get cells across all epochs that meet crit
        pcs = np.unique(np.concatenate(compc))
        pcs_all = intersect_arrays(*compc)
        # get per comparison
        pcs_p_per_comparison = [len(xx)/len(coms_correct_abs[0]) for xx in compc]
        pc_ind.append(pcs_all);pc_p=len(pcs_all)/len(coms_correct_abs[0])
        epoch_perm.append(perm)
        pc_prop.append([pcs_p_per_comparison,pc_p])
        num_epochs.append(len(coms_correct_abs))

        colors = ['k', 'slategray', 'darkcyan', 'darkgoldenrod', 'steelblue']
        coms_all.append(coms_correct_abs)
        for gc in pcs:
                fig, ax = plt.subplots()
                for ep in range(len(coms_correct_abs)):
                        ax.plot(tcs_correct_abs[ep,gc,:], label=f'rewloc {rewlocs[ep]}', color=colors[ep],linewidth=3)
                        ax.axvline(rewlocs[ep]/bin_size, color=colors[ep], linestyle='--',linewidth=3)
                        ax.set_title(f'animal: {animal}, day: {day}\ncell # {gc}')
                        ax.set_xticks(np.arange(0,(track_length/bin_size)+bin_size,10))
                        ax.set_xticklabels(np.arange(0,track_length+bin_size*10,bin_size*10).astype(int))
                        ax.set_xlabel('Absolute position (cm)')
                        ax.set_ylabel('Fc3')
                        ax.spines[['top','right']].set_visible(False)
                        ax.legend()
        #     plt.savefig(os.path.join(savedst, 'true_place_cell.png'), bbox_inches='tight', dpi=500)
                plt.close('all')
                pdf.savefig(fig)
        # get shuffled iterations
        shuffled_dist = np.zeros((num_iterations))
        # max of 5 epochs = 10 perms
        place_cell_shuf_ps_per_comp = np.ones((num_iterations,10))*np.nan
        place_cell_shuf_ps = []
        for i in range(num_iterations):
                # shuffle locations
                shufs = [list(range(coms_correct_abs[ii].shape[0])) for ii in range(1, len(coms_correct_abs))]
                [random.shuffle(shuf) for shuf in shufs]
                # first com is as ep 1, others are shuffled cell identities
                com_shufs = np.zeros_like(coms_correct_abs); com_shufs[0,:] = coms_correct_abs[0]
                com_shufs[1:1+len(shufs),:] = [coms_correct_abs[ii][np.array(shufs)[ii-1]] for ii in range(1, 1+len(shufs))]
                # get cells that maintain their coms across at least 2 epochs
                perm = list(combinations(range(len(com_shufs)), 2))     
                com_per_ep = np.array([(com_shufs[perm[jj][0]]-com_shufs[perm[jj][1]]) for jj in range(len(perm))])        
                compc = [np.where((comr<place_window) & (comr>-place_window))[0] for comr in com_per_ep]
                # get cells across all epochs that meet crit
                pcs = np.unique(np.concatenate(compc))
                pcs_all = intersect_arrays(*compc)
                # get per comparison
                pcs_p_per_comparison = [len(xx)/len(coms_correct_abs[0]) for xx in compc]
                shuffled_dist[i] = len(pcs_all)/len(coms_correct_abs[0])
                place_cell_shuf_p=len(pcs_all)/len(com_shufs[0])
                place_cell_shuf_ps.append(place_cell_shuf_p)
                place_cell_shuf_ps_per_comp[i, :len(pcs_p_per_comparison)] = pcs_p_per_comparison
        # save median of goal cell shuffle
        place_cell_shuf_ps_per_comp_av = np.nanmedian(place_cell_shuf_ps_per_comp,axis=0)        
        place_cell_shuf_ps_av = np.nanmedian(np.array(place_cell_shuf_ps)[1])
        place_cell_null.append([place_cell_shuf_ps_per_comp_av,place_cell_shuf_ps_av])
        p_value = sum(shuffled_dist>pc_p)/num_iterations
        print(f'{animal}, day {day}: significant place cells proportion p-value: {p_value}')
        pvals.append(p_value);     
        total_cells.append(len(coms_correct_abs[0]))
        datadct[f'{animal}_{day:03d}_index{ii:03d}'] = [tcs_correct_abs, coms_correct_abs, 
                pcs_all, place_cell_shuf_ps_per_comp_av, place_cell_shuf_ps_av]

pdf.close()
# save pickle of dcts
# with open(saveddataset, "wb") as fp:   #Pickling
#         pickle.dump(datadct, fp) 

#%%
radian_alignment=datadct
plt.rc('font', size=20)          # controls default text sizes
# plot goal cells across epochs
inds = [int(xx[-3:]) for xx in radian_alignment.keys()]
df = conddf.copy()
df = df[((df.animals!='e217')) & (df.index.isin(inds))]
df['num_epochs'] = num_epochs
df['place_cell_prop'] = [xx[1] for xx in pc_prop]
df['opto'] = df.optoep.values>1
df['p_value'] = pvals
df['place_cell_prop_shuffle'] = [xx[1] for xx in place_cell_null]
df['session_num'] = np.concatenate([[ii for ii,xx in enumerate(df[df.animals==an].days.values)] for an in np.unique(df.animals.values)])

fig,ax = plt.subplots(figsize=(5,5))
ax = sns.histplot(data = df.loc[df.opto==False], x='p_value', hue='animals', bins=40)
ax.spines[['top','right']].set_visible(False)
ax.axvline(x=0.05, color='k', linestyle='--')
sessions_sig = sum(df.loc[df.opto==False,'p_value'].values<0.05)/len(df.loc[df.opto==False])
ax.set_title(f'{(sessions_sig*100):.2f}% of sessions are significant')
ax.set_xlabel('P-value')
ax.set_ylabel('Sessions')
#%%
# number of epochs vs. reward cell prop    
fig,ax = plt.subplots(figsize=(5,5))
df_plt = df[(df.opto==False)]
# av across mice
df_plt = df_plt.groupby(['animals','num_epochs']).mean(numeric_only=True)
sns.stripplot(x='num_epochs', y='place_cell_prop',
        hue='animals',data=df_plt,
        s=10)
sns.barplot(x='num_epochs', y='place_cell_prop',
        data=df_plt,
        fill=False,ax=ax, color='k', errorbar='se')
ax = sns.lineplot(data=df_plt, # correct shift
        x=df_plt.index.get_level_values('num_epochs')-2, y='place_cell_prop_shuffle',color='grey', 
        label='shuffle')
ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.01, 1.05))

eps = [2,3,4]
for ep in eps:
        # rewprop = df_plt.loc[(df_plt.num_epochs==ep), 'goal_cell_prop']
        rewprop = df_plt.loc[(df_plt.index.get_level_values('num_epochs')==ep), 
                'place_cell_prop']
        shufprop = df_plt.loc[(df_plt.index.get_level_values('num_epochs')==ep),
        'place_cell_prop_shuffle']
        t,pval = scipy.stats.ranksums(rewprop, shufprop)
        print(f'{ep} epochs, pval: {pval}')
#%%
# include all comparisons 
df_perms = pd.DataFrame()
df_perms['epoch_comparison'] = [str(tuple(xx)) for xx in np.concatenate(epoch_perm)]
goal_cell_perm = [xx[0] for xx in pc_prop]
goal_cell_perm_shuf = [xx[0][~np.isnan(xx[0])] for xx in place_cell_null]
df_perms['place_cell_prop'] = np.concatenate(goal_cell_perm)
df_perms['place_cell_prop_shuffle'] = np.concatenate(goal_cell_perm_shuf)
df_perm_animals = [[xx]*len(goal_cell_perm[ii]) for ii,xx in enumerate(df.animals.values)]
df_perms['animals'] = np.concatenate(df_perm_animals)
# df_perms = df_perms[df_perms.animals!='e189']
df_permsav = df_perms.groupby(['animals','epoch_comparison']).mean(numeric_only=True)
# df_perm_days = [[xx]*len(goal_cell_perm[ii]) for ii,xx in enumerate(df.session_num.values)]
# df_perms['session_num'] = np.concatenate(df_perm_days)


fig,ax = plt.subplots(figsize=(7,5))
sns.stripplot(x='epoch_comparison', y='place_cell_prop',
        hue='animals',data=df_permsav,
        s=8,ax=ax)
sns.barplot(x='epoch_comparison', y='place_cell_prop',
        data=df_permsav,
        fill=False,ax=ax, color='k', errorbar='se')
ax = sns.lineplot(data=df_permsav, # correct shift
        x='epoch_comparison', y='place_cell_prop_shuffle',
        color='grey', label='shuffle')

ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.01, 1.05))

eps = df_permsav.index.get_level_values("epoch_comparison").unique()
for ep in eps:
        # rewprop = df_plt.loc[(df_plt.num_epochs==ep), 'goal_cell_prop']
        rewprop = df_permsav.loc[(df_permsav.index.get_level_values('epoch_comparison')==ep), 
        'place_cell_prop'].values
        shufprop = df_permsav.loc[(df_permsav.index.get_level_values('epoch_comparison')==ep), 
                'place_cell_prop_shuffle'].values
        t,pval = scipy.stats.ranksums(rewprop, shufprop)
        print(f'{ep} epochs, pval: {pval}')

# take a mean of all epoch comparisons
df_perms['num_epochs'] = [2]*len(df_perms)
df_permsav2 = df_perms.groupby(['animals', 'num_epochs']).mean(numeric_only=True)
df_permsav=df_permsav.reset_index()
epcomp = [f'Epoch {int(xx[1])+1}, Epoch {int(xx[4])+1}' for xx in df_permsav.epoch_comparison.values]
df_permsav['epoch_comparison']=epcomp
df_permsav['place_cell_prop']=df_permsav['place_cell_prop']*100
df_permsav['place_cell_prop_shuffle']=df_permsav['place_cell_prop_shuffle']*100

#%% 
# for figure; place cells in any epoch
# consecutive epochs
df_permsav=df_permsav[(df_permsav.epoch_comparison!='Epoch 1, Epoch 5') & (df_permsav.epoch_comparison!='Epoch 2, Epoch 5') & (df_permsav.epoch_comparison!='Epoch 3, Epoch 5') & (df_permsav.epoch_comparison!='Epoch 4, Epoch 5')]

df_permsav=df_permsav[(df_permsav.epoch_comparison=='Epoch 1, Epoch 2') | (df_permsav.epoch_comparison=='Epoch 2, Epoch 3') | (df_permsav.epoch_comparison=='Epoch 3, Epoch 4')]

fig,ax = plt.subplots(figsize=(3,4))
sns.barplot(x='epoch_comparison', y='place_cell_prop',
        data=df_permsav,
        fill=False,ax=ax, color='indigo', errorbar='se')
sns.barplot(data=df_permsav, # correct shift
        x='epoch_comparison', y='place_cell_prop_shuffle',color='grey', 
        label='shuffle', alpha=0.5, err_kws={'color': 'grey'},errorbar=None,ax=ax)

ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.01, 1.05))
ax.set_xticklabels(ax.get_xticklabels(),rotation=20)
eps = df_permsav.epoch_comparison.unique()
pvalues=[]
for ep in eps:
        # rewprop = df_plt.loc[(df_plt.num_epochs==ep), 'goal_cell_prop']
        rewprop = df_permsav.loc[(df_permsav.epoch_comparison==ep), 
        'place_cell_prop'].values
        shufprop = df_permsav.loc[(df_permsav.epoch_comparison==ep), 
                'place_cell_prop_shuffle'].values
        t,pval = wilcoxon_r(rewprop, shufprop)
        print(f'{ep} epochs, pval: {pval},w = {t},n={len(rewprop)}')
        pvalues.append(pval)

from statsmodels.stats.multitest import multipletests
reject, pvals_corrected, _, _ = multipletests(pvalues, method='fdr_bh')

y=32
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
ax.set_ylabel('Place cell %')
ans = df_permsav.animals.unique()
for i in range(len(ans)):
    ax = sns.lineplot(x=np.arange(len(df_permsav[df_permsav.animals==ans[i]])), y='place_cell_prop', 
    data=df_permsav[df_permsav.animals==ans[i]],
    errorbar=None, color='dimgray', linewidth=1.5, alpha=0.5,ax=ax)

groups = [df_permsav.loc[df_permsav.epoch_comparison==ep, 'place_cell_prop'].values
          for ep in df_permsav.epoch_comparison.unique()]

H, p_kw = scipy.stats.kruskal(*groups)
print(f"Kruskal–Wallis: H={H:.3f}, p={p_kw:.3g}")

# =========================
# Post-hoc pairwise Dunn test if KW significant
# =========================
import scikit_posthocs as sp_post
if p_kw < 0.05:
        dunn = sp_post.posthoc_dunn(df_permsav, val_col='place_cell_prop', 
                                group_col='epoch_comparison', p_adjust='fdr_bh')
        print(dunn)
        
import itertools

# Set label
ax.set_ylabel('Place cell %')

# Add subject-level lines
ans = df_permsav.animals.unique()
for i in range(len(ans)):
    ax = sns.lineplot(x=np.arange(len(df_permsav[df_permsav.animals==ans[i]])),
                      y='place_cell_prop',
                      data=df_permsav[df_permsav.animals==ans[i]],
                      errorbar=None, color='dimgray',
                      linewidth=1.5, alpha=0.2, ax=ax)

# Kruskal–Wallis
groups = [df_permsav.loc[df_permsav.epoch_comparison==ep, 'place_cell_prop'].values
          for ep in df_permsav.epoch_comparison.unique()]
H, p_kw = scipy.stats.kruskal(*groups)
print(f"Kruskal–Wallis: H={H:.3f}, p={p_kw:.3g}")
h=1.5
# Post-hoc Dunn test
import scikit_posthocs as sp_post
if p_kw < 0.05:
    dunn = sp_post.posthoc_dunn(df_permsav,
                                val_col='place_cell_prop',
                                group_col='epoch_comparison',
                                p_adjust='fdr_bh')
    print(dunn)

    # =============================
    # Draw comparison bars manually
    # =============================
    xticks = df_permsav.epoch_comparison.unique()
    y_max = df_permsav['place_cell_prop'].max()
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
fig.suptitle('Place cells between two epochs')
plt.savefig(os.path.join(savedst, 'place_cell_prop_btwn_two_ep.svg'), 
        bbox_inches='tight')
df_permsav.to_csv(r"C:\Users\Han\Documents\MATLAB\vip_reward_cell\raw_data\fig2a.csv")
#%%
df_plt2 = pd.concat([df_permsav2,df_plt])
df_plt2 = df_plt2[df_plt2.index.get_level_values('num_epochs')<5]
df_plt2 = df_plt2.groupby(['animals', 'num_epochs']).mean(numeric_only=True)
df_plt2['place_cell_prop']=df_plt2['place_cell_prop']*100
df_plt2['place_cell_prop_shuffle']=df_plt2['place_cell_prop_shuffle']*100
df_plt2=df_plt2.reset_index()
# df_plt2=df_plt2[(df_plt2.animals!='e189') & (df_plt2.animals!='e139')]
# number of epochs vs. reward cell prop incl combinations    
# number of epochs vs. reward cell prop incl combinations    
fig,axes = plt.subplots(ncols=2,figsize=(6.5,4))
ax=axes[0]
# av across mice
# sns.stripplot(x='num_epochs', y='place_cell_prop',color='k',
        # data=df_plt2,s=10,alpha=0.7,ax=ax)
sns.barplot(x='num_epochs', y='place_cell_prop',
        data=df_plt2,
        fill=False,ax=ax, color='indigo', errorbar='se')
# ax = sns.lineplot(data=df_plt2, # correct shift
#         x=df_plt2.index.get_level_values('num_epochs').astype(int)-2, 
#         y='goal_cell_prop_shuffle',color='grey', 
#         label='shuffle')
# bar plot of shuffle instead
sns.barplot(data=df_plt2, # correct shift
        x='num_epochs', y='place_cell_prop_shuffle',color='grey', 
        label='shuffle', alpha=0.5, err_kws={'color': 'grey'},errorbar=None,ax=ax)

ax.spines[['top','right']].set_visible(False)
ax.legend()#.set_visible(False)
ax.set_ylabel('Place cell %')
eps = [2,3,4]
y = 28
pshift = 1
fs=46
pvalues=[]
ts=[]
for ii,ep in enumerate(eps):
        rewprop = df_plt2.loc[(df_plt2.num_epochs==ep), 'place_cell_prop']
        shufprop = df_plt2.loc[(df_plt2.num_epochs==ep), 'place_cell_prop_shuffle']
        tstat,pval = wilcoxon_r(rewprop, shufprop)
        pvalues.append(pval)
        ts.append(tstat)
        print(f'{ep} epochs, pval: {pval}, r ={tstat}, n={len(rewprop)}')
# correct pvalues
from statsmodels.stats.multitest import multipletests
reject, pvals_corrected, _, _ = multipletests(pvalues, method='fdr_bh')

for ii,ep in enumerate(eps):
        pval=pvals_corrected[ii]
        # statistical annotation        
        if pval < 0.001:
                ax.text(ii, y, "***", ha='center', fontsize=fs)
        elif pval < 0.01:
                ax.text(ii, y, "**", ha='center', fontsize=fs)
        elif pval < 0.05:
                ax.text(ii, y, "*", ha='center', fontsize=fs)
        # ax.text(ii-0.5, y-pshift*2, f't={ts[ii]:.3g}\np={pval:.3g}',fontsize=10,rotation=45)
# make lines
ans = df_plt2.animals.unique()
for i in range(len(ans)):
    ax = sns.lineplot(x=df_plt2.num_epochs-2, y='place_cell_prop', 
    data=df_plt2[df_plt2.animals==ans[i]],
    errorbar=None, color='dimgray', linewidth=1.5, alpha=0.5,ax=ax)
ax.set_title('Place cells',pad=30)
ax.set_xlabel('')
ax.set_ylim([0,35])
ax=axes[1]
# subtract from shuffle
# df_plt2=df_plt2.reset_index()
df_plt2['place_cell_prop_sub_shuffle'] = df_plt2['place_cell_prop']-df_plt2['place_cell_prop_shuffle']
# av across mice
# sns.stripplot(x='num_epochs', y='place_cell_prop_sub_shuffle',color='cornflowerblue',data=df_plt2,s=10,alpha=0.7,ax=ax)
sns.barplot(x='num_epochs', y='place_cell_prop_sub_shuffle',
        data=df_plt2,
        fill=False,ax=ax, color='indigo', errorbar='se')
# make lines
df_plt2=df_plt2.reset_index()
ans = df_plt2.animals.unique()
for i in range(len(ans)):
    ax = sns.lineplot(x=df_plt2.num_epochs-2, y='place_cell_prop_sub_shuffle', 
    data=df_plt2[df_plt2.animals==ans[i]],
    errorbar=None, color='dimgray', linewidth=1.5, alpha=0.5,ax=ax)
y=15
for ii,ep in enumerate(eps):
        pval=pvals_corrected[ii]
        # statistical annotation        
        if pval < 0.001:
                ax.text(ii, y, "***", ha='center', fontsize=fs)
        elif pval < 0.01:
                ax.text(ii, y, "**", ha='center', fontsize=fs)
        elif pval < 0.05:
                ax.text(ii, y, "*", ha='center', fontsize=fs)
        # ax.text(ii-0.5, y-pshift*2, f't={ts[ii]:.3g}\np={pval:.3g}',fontsize=10,rotation=45)
        
ax.spines[['top','right']].set_visible(False)
ax.set_xlabel('# of epochs')
ax.set_ylabel('')
ax.set_title('Place cell %-shuffle',pad=30)
ax.set_ylim([-1,20
             ])
fig.suptitle('Dedicated place cells')

plt.savefig(os.path.join(savedst, 'place_cell_prop_per_an.svg'), 
        bbox_inches='tight')
df_plt2.to_csv(r"C:\Users\Han\Documents\MATLAB\vip_reward_cell\raw_data\fig2b.csv")
#%%

df['recorded_neurons_per_session'] = total_cells
df_plt_ = df[(df.opto==False)&(df.p_value<0.05)]
df_plt_= df_plt_[(df_plt_.animals!='e200')&(df_plt_.animals!='e189')]
df_plt_ = df_plt_.groupby(['animals']).mean(numeric_only=True)

fig,ax = plt.subplots(figsize=(7,5))
sns.scatterplot(x='recorded_neurons_per_session', y='place_cell_prop',hue='animals',
        data=df_plt_,
        s=150, ax=ax)
sns.regplot(x='recorded_neurons_per_session', y='place_cell_prop',
        data=df_plt_,
        ax=ax, scatter=False, color='k'
)
r, p = scipy.stats.pearsonr(df_plt_['recorded_neurons_per_session'], 
        df_plt_['place_cell_prop'])
ax = plt.gca()
ax.text(.5, .8, 'r={:.2f}, p={:.2g}'.format(r, p),
        transform=ax.transAxes)

ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.01, 1.05))
plt.savefig(os.path.join(savedst, 'place_cell_nearrew_prop_per_an.svg'), 
        bbox_inches='tight')
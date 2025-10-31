
"""
zahra
pca on tuning curves of reward cells
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
import matplotlib.backends.backend_pdf, matplotlib as mpl
from itertools import combinations
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
# plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from ensemble import get_all_ensemble_data
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
savepth = os.path.join(savedst, 'rew_dt_assemblies.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)

#%%
# initialize var
# gets place, pre, and post rew cells
p_rewcells_in_assemblies=[]
bins = 90
goal_window_cm=20
assembly_cells_all_an=[]
cell_type = 'pre'
# iterate through all animals
for ii in range(len(conddf)):
    day = conddf.days.values[ii]
    animal = conddf.animals.values[ii]
    if (animal!='e217') & (conddf.optoep.values[ii]<2):
        if animal=='e145' or animal=='e139': pln=2 
        else: pln=0
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
        print(params_pth)
        datadct = get_all_ensemble_data(params_pth, animal, day, pdf, bins=90, goal_window_cm=20)
        assembly_cells_all_an.append(datadct)

# Assembly activity is a time series that measures how active a particular 
# neuronal ensemble (identified via PCA) is at each time point. It reflects 
# coordinated activity, not just individual spikes.
pdf.close()
#%%
# look through all the assemblies
df = pd.DataFrame()#
dfc = conddf.copy()
dfc = dfc[(dfc.animals!='e217') & (dfc.optoep<2)]
# df = df[(df.animals!='e217') & (df.optoep.values<2)]
pre_fraction=[assembly_cells_all_an[i]['pre']['fraction'] for i in range(len(assembly_cells_all_an))]
post_fraction=[assembly_cells_all_an[i]['post']['fraction'] for i in range(len(assembly_cells_all_an))]
place_fraction=[assembly_cells_all_an[i]['place']['fraction'] for i in range(len(assembly_cells_all_an))]
df['p_cells_in_assemblies'] = np.concatenate([pre_fraction,post_fraction,place_fraction])
df['cell_type'] = np.concatenate([['pre']*len(pre_fraction),['post']*len(post_fraction),
                    ['place']*len(place_fraction)])
df['animal'] = np.concatenate([dfc.animals.values]*3)
df['days'] = np.concatenate([dfc.days.values]*3)
df['p_cells_in_assemblies'] = df['p_cells_in_assemblies'] *100
ax = sns.histplot(x = 'p_cells_in_assemblies',data=df, hue='cell_type')

from projects.pyr_reward.rewardcell import cosine_sim_ignore_nan
from matplotlib import colors
bins_dt=150
# look through all the assemblies
# df = conddf.copy()
# df = df[(df.animals!='e217') & (df.optoep.values<2)]
an_plt = 'z16' # 1 eg animal
an_day = 9
plt.close('all')
cs_all_celltype = []; num_epochs_celltype = []
cell_types = ['pre', 'post', 'place']
for cell_type in cell_types:
    cs_all = []; num_epochs = []
    plot=True
    # plot = False
    assembly=[assembly_cells_all_an[i][cell_type]['assemblies'] for i in range(len(assembly_cells_all_an))]
    for ii,ass in enumerate(assembly):
        if df.iloc[ii].animal==an_plt and df.iloc[ii].days==an_day:
            print(f'{df.iloc[ii].animal}, {df.iloc[ii].days}')
            ass_all = [xx[0] for xx in list(ass.values())] # all assemblies
            # split into correct v fail
            ass_all_incorr = [xx[1] for xx in list(ass.values())] # all assemblies
            cs_per_ep = []; ne = []
            for jj,asm in enumerate(ass_all):
                # only pick cells > 10
                cell_nm = asm.shape[1]
                if cell_nm<5:
                    continue
                perm = list(combinations(range(len(asm)), 2)) 
                # consecutive ep only
                perm = [p for p in perm if p[0]-p[1]==-1]
                cs = [cosine_sim_ignore_nan(asm[p[0]], asm[p[1]]) for p in perm]
                cs_per_ep.append(cs)
                if plot:
                    fig,axes = plt.subplots(ncols = len(asm), figsize=(14,5),sharex=True,sharey=True)
                    gamma=1
                    for kk,tcs in enumerate(asm):
                        ax = axes[kk]
                        time_bins = np.arange(tcs.shape[1])
                        if kk==0: com_per_cell = [np.sum(tc * time_bins) / np.sum(tc) if np.sum(tc) > 0 else np.nan for tc in tcs]            
                        # Normalize per row
                        tcs_norm = tcs / np.nanmax(tcs, axis=1, keepdims=True)
                        tcs_norm[np.isnan(tcs_norm)] = 0
                        vmin = np.min(tcs_norm)
                        vmax = np.max(tcs_norm)
                        norm = colors.Normalize(vmin=vmin, vmax=vmax)
                        im = ax.imshow(tcs_norm[np.argsort(com_per_cell)]**gamma, aspect='auto', norm=norm)
                        ax.set_title(f'Epoch {kk+1}')
                        ax.axvline(time_bins[-1]/2, color='w', linestyle='--')
                    ax.set_xticks([0, tcs.shape[1]/2,tcs.shape[1]])
                    if cell_type!='place':
                        ax.set_xticklabels(['-$\\pi$',0,'$\\pi$'])
                        ax.set_xlabel('Reward-relative distance ($\Theta$)')
                    else:
                        ax.set_xticklabels([0,135,270])
                        ax.set_xlabel('Track position (cm)')
                    fig.suptitle(f'Correct \n {cell_type} ensemble \n {df.iloc[ii].animal}, {df.iloc[ii].days} \n\
                        Assembly: {jj}, Cosine sim b/wn epochs average: {np.round(np.nanmean(cs),2)}\n\n')
                    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
                    fig.colorbar(im, cax=cbar_ax, label=f'Norm. $\Delta$ F/F')
                    ax.set_ylabel('Cell ID #')
                    if jj==0:
                        plt.savefig(os.path.join(savedst,f'{an_plt}_{an_day}_darktime_{cell_type}_ensemble_eg.svg'),bbox_inches='tight')
                    ############## incorrects #############
                    # asm = ass_all_incorr[jj]
                    # fig,axes = plt.subplots(ncols = len(asm), figsize=(14,5),sharex=True,sharey=True)
                    # for kk,tcs in enumerate(asm):
                    #     ax = axes[kk]
                    #     vmin = np.min(tcs)
                    #     vmax = np.max(tcs)
                    #     norm = colors.Normalize(vmin=vmin, vmax=vmax)
                    #     # same order as corrects
                    #     # if kk==0: com_per_cell = [np.sum(tc * time_bins) / np.sum(tc) if np.sum(tc) > 0 else np.nan for tc in tcs]            
                    #     im=ax.imshow(tcs[np.argsort(com_per_cell)]**gamma,aspect='auto',norm=norm)
                    #     ax.set_title(f'Epoch {kk+1}')
                    #     ax.axvline(bins_dt/2, color='w', linestyle='--')
                    # # ax.set_xticks(np.arange(0,bins_dt+10,30))
                    # # ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi+.6, np.pi/2.5),2))
                    # fig.suptitle(f'Incorrect \n Pre-reward ensemble \n {df.iloc[ii].animal}, {df.iloc[ii].days} \n\
                    #     Assembly: {jj}, Cosine similarity b/wn epochs average: {np.round(np.nanmean(cs),2)}\n\n')
                    # cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
                    # fig.colorbar(im, cax=cbar_ax, label=f'$\Delta$ F/F ^ {gamma}')
                    # if jj==0:
                    #     plt.savefig(os.path.join(savedst,f'{an_plt}_{an_day}_darktime_prerew_ensemble_eg.svg'),bbox_inches='tight')

            num_epochs.append(len(asm))
            cs_all.append(cs_per_ep)
    cs_all_celltype.append(cs_all)
    num_epochs_celltype.append(num_epochs)
            # plt.figure()
            # plt.plot(tcs[np.argsort(com_per_cell)].T)
#%%

######################## shuffle ########################
# look through all the assemblies
df = conddf.copy()
df = df[(df.animals!='e217') & (df.optoep.values<2)]
an_plt = 'z9' # 1 eg animal
an_day = 19
cs_all_shuffle = []; num_epochs_shuffle = []
plt.close('all')
plot = False
assembly=[assembly_cells_all_an[i]['pre']['assemblies'] for i in range(len(assembly_cells_all_an))]
for ii,ass in enumerate(assembly):
    # if df.iloc[ii].animals==an_plt and df.iloc[ii].days==an_day:
        print(f'{df.iloc[ii].animals}, {df.iloc[ii].days}')
        ass_all = [xx[0] for xx in list(ass.values())] # all assemblies
        # split into correct v fail
        ass_all_incorr = [xx[1] for xx in list(ass.values())] # all assemblies
        cs_per_ep = []; ne = []
        for jj,asm in enumerate(ass_all):
            # only pick cells > 10
            cell_nm = asm.shape[1]
            if cell_nm<5:
                continue
            perm = list(combinations(range(len(asm)), 2)) 
            # consecutive ep only
            perm = [p for p in perm if p[0]-p[1]==-1]
            ############ SHUFFLE ############
            shufs = np.arange(len(asm[perm[0][0]]))
            random.shuffle(shufs)
            ############ SHUFFLE ############
            # shuffle 2nd epoch
            cs = [cosine_sim_ignore_nan(asm[p[0]], asm[p[1]][shufs,:]) for p in perm]
            cs_per_ep.append(cs)
            if plot:
                fig,axes = plt.subplots(ncols = len(asm), figsize=(14,5),sharex=True,sharey=True)
                gamma=.5
                for kk,tcs in enumerate(asm):
                    ax = axes[kk]
                    vmin = np.min(tcs)
                    vmax = np.max(tcs)
                    norm = colors.Normalize(vmin=vmin, vmax=vmax)
                    if kk==0: com_per_cell = [np.sum(tc * time_bins) / np.sum(tc) if np.sum(tc) > 0 else np.nan for tc in tcs]            
                    im=ax.imshow(tcs[np.argsort(com_per_cell)]**gamma,aspect='auto',norm=norm)
                    ax.set_title(f'Epoch {kk+1}')
                    ax.axvline(bins/2, color='w', linestyle='--')
                ax.set_xticks(np.arange(0,bins,30))
                ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi+.6, np.pi),2))
                fig.suptitle(f'Pre-reward ensemble \n {df.iloc[ii].animals}, {df.iloc[ii].days} \n\
                    Assembly: {jj}, Cosine similarity b/wn epochs average: {np.round(np.nanmean(cs),2)}')
                cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
                fig.colorbar(im, cax=cbar_ax, label=f'$\Delta$ F/F ^ {gamma}')
                if jj==0:
                    plt.savefig(os.path.join(savedst,f'{an_plt}_{an_day}_prerew_ensemble_eg.svg'),bbox_inches='tight')
        num_epochs_shuffle.append(len(asm))
        cs_all_shuffle.append(cs_per_ep)
            
# add 2 ep combinaitions as 2 ep
df2 = pd.DataFrame()
df2['cosine_sim_across_ep'] = np.hstack([np.concatenate(xx) if len(xx)>0 else np.nan for xx in cs_all_shuffle])
df2['animals'] = np.concatenate([[df.iloc[ii].animals]*len(np.concatenate(xx)) if len(xx)>0 else [df.iloc[ii].animals] for ii,xx in enumerate(cs_all_shuffle)])
df2['num_epochs'] =[2]*len(df2)

df['num_epochs'] = num_epochs_shuffle
# df['cosine_sim_across_ep'] = [np.quantile(xx,.75) if len(xx)>0 else np.nan for xx in cs_all]
df['cosine_sim_across_ep'] = [np.nanmean(xx) if len(xx)>0 else np.nan for xx in cs_all_shuffle]

df = pd.concat([df,df2])
df = df.dropna(subset=['cosine_sim_across_ep', 'num_epochs'])
dfan = df.groupby(['animals', 'num_epochs']).mean(numeric_only=True)
dfan = dfan.reset_index()
dfan = dfan[dfan.num_epochs<5]
df_clean = dfan
# temp
# df_clean = df_clean[(df_clean.animals!='e139') & (df_clean.animals!='e200') & (df_clean.animals!='e190') & (df_clean.animals!='e189')]
# Plot
s=10
plt.figure(figsize=(3,5))
ax = sns.barplot(x='num_epochs', y='cosine_sim_across_ep', data=df_clean, errorbar='se',
            fill=False, color='k')
sns.stripplot(x='num_epochs', y='cosine_sim_across_ep', data=df_clean, color='k', jitter=True,
            s=s,alpha=0.7)
ax.set_title('Shuffle ensembles', pad=50)
plt.tight_layout()
plt.show()
df_clean.to_csv(r'Z:\condition_df\shuffle_ensemble_w_dt.csv', index=None)
######################## shuffle ########################

#%% pre-reward cells
# add 2 ep combinaitions as 2 ep
cell_types = ['pre', 'post', 'place']
df2 = pd.DataFrame()
df2['cosine_sim_across_ep'] = np.concatenate([np.concatenate([np.concatenate(xx) if len(xx)>0 else [np.nan ]for xx in cs_all]) for cs_all in cs_all_celltype])
df2['cell_type'] = np.concatenate([np.concatenate([[cell_types[ctype]]*len(np.concatenate(xx)) if len(xx)>0 else [cell_types[ctype]] for xx in cs_all]) for ctype,cs_all in enumerate(cs_all_celltype)])
# FIX
dfc = conddf.copy()
dfc = dfc[(dfc.animals!='e217') & (dfc.optoep.values<2)]
df2['animal'] = np.concatenate([np.concatenate([[dfc.animals.values[ii]]*len(np.concatenate(xx)) if len(xx)>0 else [np.nan]for ii,xx in enumerate(cs_all)]) for cs_all in cs_all_celltype])
df2['days'] = np.concatenate([np.concatenate([[dfc.days.values[ii]]*len(np.concatenate(xx)) if len(xx)>0 else [np.nan]for ii,xx in enumerate(cs_all)]) for cs_all in cs_all_celltype])
df2['num_epochs'] =[2]*len(df2)
df['num_epochs'] = np.concatenate(num_epochs_celltype)
# df['cosine_sim_across_ep'] = [np.quantile(xx,.75) if len(xx)>0 else np.nan for xx in cs_all]
df['cosine_sim_across_ep'] = np.concatenate([[np.nanmean(xx) if len(xx)>0 else np.nan for xx in cs_all] for cs_all in cs_all_celltype])
df_all = pd.concat([df,df2])
# df_all = df.dropna(subset=['cosine_sim_across_ep'])
#%%
plt.rc('font', size=20) 
dfan = df_all.groupby(['animal', 'num_epochs', 'cell_type']).mean(numeric_only=True)
dfan = dfan.reset_index()
dfan = dfan[dfan.num_epochs<5]
df_clean = dfan[(dfan.animal!='e139')]
# palette = seaborn Dark2
s=10
df_all = df_clean
cell_types = ['pre', 'post', 'place']
plt.figure(figsize=(6,4))
ax = sns.barplot(x='num_epochs', y='cosine_sim_across_ep', hue='cell_type',data=df_all, errorbar='se',
            fill=False, palette = 'Dark2',hue_order=cell_types)
# sns.barplot(x='num_epochs', y='cosine_sim_across_ep',data=df_shuffle, errorbar='se',
#             color='dimgrey',alpha=0.3,
#             label='shuffle', err_kws={'color': 'grey'},ax=ax)
ax = sns.stripplot(x='num_epochs', y='cosine_sim_across_ep', hue='cell_type',data=df_all, dodge=True,
            s=s,alpha=0.7,palette = 'Dark2',hue_order=cell_types)

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.0, 0.5), title='Cell Type')
ax.spines[['top','right']].set_visible(False)
# Get unique epochs
epochs = sorted(df_all.num_epochs.unique())
ymax = .6
y_offsets = [ymax + (i * 0.03) for i in range(len(epochs))]

fs = 40  # font size for stars
pshift = 0.08  # p-value label offset

# non rew vs. pre reward
# for i, epoch in enumerate(epochs):
#     data_epoch = df_all[df_all.num_epochs == epoch]
#     pre_vals = data_epoch[data_epoch.cell_type == 'Pre-reward']['cosine_sim_across_ep'].dropna()
#     post_vals = data_epoch[data_epoch.cell_type == 'Place']['cosine_sim_across_ep'].dropna()

#     # t-test
#     stat, pval = scipy.stats.ranksums(pre_vals, post_vals)
#     # Plot annotation
#     x = i
#     y = y_offsets[i]
#     # Show p-value (optional)
#     ax.text(x, y + pshift, f'place vs. pre p={pval:.2g}', ha='center', fontsize=12, rotation=45)
groups = [
    df_all[df_all.cell_type == ct]['cosine_sim_across_ep'].dropna()
    for ct in ['place', 'pre', 'post']
]
f_stat, p_anova = scipy.stats.f_oneway(*groups)
print(f"ANOVA F={f_stat:.3f}, p={p_anova:.3g}")
comparisons = [('pre', 'post')]
pvals = []
epochs = sorted(df_all.num_epochs.unique())
y_offsets = [0.6 + 0.03 * i for i in range(len(comparisons))]  # adjust if needed
from statsmodels.stats.multitest import multipletests

for i, epoch in enumerate(epochs):
    data_epoch = df_all[df_all.num_epochs == epoch]
    pairwise_pvals = []
    for (a, b) in comparisons:
        vals_a = data_epoch[data_epoch.cell_type == a]['cosine_sim_across_ep']
        vals_b = data_epoch[data_epoch.cell_type == b]['cosine_sim_across_ep']
        _, p = scipy.stats.ranksums(vals_a[~np.isnan(vals_a)], vals_b[~np.isnan(vals_b)])
        pairwise_pvals.append(p)
    # FDR correction
    reject, pvals_corr, _, _ = multipletests(pairwise_pvals, method='fdr_bh')
    
    # Plot annotations
    for j, ((a, b), pval, sig) in enumerate(zip(comparisons, pvals_corr, reject)):
        x = i-j/3+.5
        y = 0.62 + j * 0.05  # stagger vertical position
        if sig:
            if pval < 0.001:
                star = '***'
            elif pval < 0.01:
                star = '**'
            elif pval < 0.05:
                star = '*'
            else:
                star = ''
            ax.text(x, y, star, ha='center', fontsize=16)
        ax.text(x, y + 0.02, f'{a} vs {b}\np={pval:.2g}', ha='center', fontsize=10, rotation=45)

ax.set_ylabel('Mean ensemble cosine similarity')
ax.set_xlabel('# of reward loc. switches')

plt.savefig(os.path.join(savedst, 'dark_time_ensemble_cosine_sim_pre_v_post.svg'))
#%%
# histogram of cell % in assemblies
# df_all=df_all.reset_index()
fig, axes = plt.subplots(ncols=2,figsize=(10,5))
ax=axes[0]
sns.histplot(
    x='p_cells_in_assemblies',
    hue='cell_type',
    data=df_all,hue_order=cell_types,
    bins=5,
    palette='Dark2',
    multiple='dodge',  # This avoids overlapping
ax=ax)
ax.set_xlabel('Cell % in ensemble')
ax.set_ylabel('Sessions')
ax.spines[['top','right']].set_visible(False)
ax=axes[1]
sns.boxplot(
    x='cell_type',
    y='p_cells_in_assemblies',
    hue='cell_type',
    data=df_all,order=cell_types,hue_order=cell_types,
    palette='Dark2',    
ax=ax)
ax.set_ylabel('% of cells in ensemble')
ax.set_xlabel('')
ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
plt.savefig(os.path.join(savedst, 'dark_time_pcells_in_ensembles.svg'))

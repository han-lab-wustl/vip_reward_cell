
"""
zahra
pca on tuning curves of reward cells
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
from itertools import combinations
import matplotlib.backends.backend_pdf, matplotlib as mpl
from sklearn.cluster import KMeans
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
# plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from ensemble import get_ensemble_data
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_rewardcentric_all.p"
with open(saveddataset, "rb") as fp: #unpickle
        radian_alignment_saved = pickle.load(fp)
savepth = os.path.join(savedst, 'post_rew_assemblies.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
#%%
# initialize var
# radian_alignment_saved = {} # overwrite
p_rewcells_in_assemblies=[]
bins = 90
goal_window_cm=20
assembly_cells_all_an=[]
cell_type = 'post'
# iterate through all animals
for ii in range(len(conddf)):
    day = conddf.days.values[ii]
    animal = conddf.animals.values[ii]
    if (animal!='e217') & (conddf.optoep.values[ii]<2):
        if animal=='e145' or animal=='e139': pln=2 
        else: pln=0
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
        print(params_pth)
        assembly_cells_all, pcells=get_ensemble_data(params_pth, animal, day, pdf,cell_type=cell_type)
        assembly_cells_all_an.append(assembly_cells_all)
        p_rewcells_in_assemblies.append(pcells)

# Assembly activity is a time series that measures how active a particular 
# neuronal ensemble (identified via PCA) is at each time point. It reflects 
# coordinated activity, not just individual spikes.
pdf.close()

#%%
# look through all the assemblies
df = conddf.copy()
df = df[(df.animals!='e217') & (df.optoep.values<2)]
df['p_cells_in_assemblies'] = p_rewcells_in_assemblies
df['p_cells_in_assemblies'] = df['p_cells_in_assemblies'] *100
ax = sns.histplot(x = 'p_cells_in_assemblies',data=df)
# ax.set_xlim([0,100])
#%%
from projects.pyr_reward.rewardcell import cosine_sim_ignore_nan
from matplotlib import colors
plt.rc('font', size=12)
an_plt = 'e201' # 1 eg animal
an_day = 50
cs_all = []; num_epochs = []
plt.close('all')
# plot = True
bins_dt=150
plot=False
for ii,ass in enumerate(assembly_cells_all_an):
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
            cs = [cosine_sim_ignore_nan(asm[p[0]], asm[p[1]]) for p in perm]
            cs_per_ep.append(cs)
            if plot:
                fig,axes = plt.subplots(ncols = len(asm), figsize=(14,5),sharex=True,sharey=True)
                gamma=.8
                for kk,tcs in enumerate(asm):
                    ax = axes[kk]
                    vmin = np.min(tcs)
                    vmax = np.max(tcs)
                    print(tcs.shape[0])
                    norm = colors.Normalize(vmin=vmin, vmax=vmax)
                    time_bins = np.arange(bins_dt)
                    if kk==0: com_per_cell = [np.sum(tc * time_bins) / np.sum(tc) if np.sum(tc) > 0 else np.nan for tc in tcs]            
                    im=ax.imshow(tcs[np.argsort(com_per_cell)]**gamma,aspect='auto',norm=norm)
                    ax.set_title(f'Epoch {kk+1}')
                    ax.axvline(bins_dt/2, color='w', linestyle='--')
                ax.set_xticks(np.arange(0,bins_dt+10,30))
                ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi+.6, np.pi/2.5),2))
                fig.suptitle(f'Correct \nPost-reward ensemble \n {df.iloc[ii].animals}, {df.iloc[ii].days} \n\
                    Assembly: {jj}, Cosine similarity b/wn epochs average: {np.round(np.nanmean(cs),2)}\n\n')
                cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
                fig.colorbar(im, cax=cbar_ax, label=f'$\Delta$ F/F ^ {gamma}')
                # if jj==0:
                #     plt.savefig(os.path.join(savedst,f'{an_plt}_{an_day}_dark_time_postrew_ensemble_eg.svg'),bbox_inches='tight')
                # incorrects
                asm = ass_all_incorr[jj]
                fig,axes = plt.subplots(ncols = len(asm), figsize=(14,5),sharex=True,sharey=True)
                for kk,tcs in enumerate(asm):
                    ax = axes[kk]
                    vmin = np.min(tcs)
                    vmax = np.max(tcs)
                    norm = colors.Normalize(vmin=vmin, vmax=vmax)
                    # use same order as correct trials
                    # if kk==0: com_per_cell = [np.sum(tc * time_bins) / np.sum(tc) if np.sum(tc) > 0 else np.nan for tc in tcs]            
                    im=ax.imshow(tcs[np.argsort(com_per_cell)]**gamma,aspect='auto',norm=norm)
                    ax.set_title(f'Epoch {kk+1}')
                    ax.axvline(bins_dt/2, color='w', linestyle='--')
                ax.set_xticks(np.arange(0,bins_dt+10,30))
                ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi+.6, np.pi/2.5),2))
                fig.suptitle(f'Incorrect \n Post-reward ensemble \n {df.iloc[ii].animals}, {df.iloc[ii].days} \n\
                    Assembly: {jj}, Cosine similarity b/wn epochs average: {np.round(np.nanmean(cs),2)}\n\n')
                cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
                fig.colorbar(im, cax=cbar_ax, label=f'$\Delta$ F/F ^ {gamma}')

        cs_all.append(cs_per_ep)
        num_epochs.append(len(asm))
            # plt.figure()
            # plt.plot(tcs[np.argsort(com_per_cell)].T)
# %%
# add 2 ep combinaitions as 2 ep
df2 = pd.DataFrame()
df2['cosine_sim_across_ep'] = np.hstack([np.concatenate(xx) if len(xx)>0 else np.nan for xx in cs_all])
ans = np.concatenate([[df.iloc[ii].animals]*len(np.concatenate(xx)) if len(xx)>0 else [df.iloc[ii].animals] for ii,xx in enumerate(cs_all)])
df2['animals'] = ans
df2['num_epochs'] =[2]*len(df2)

# df = conddf.copy()
# df = df[(df.animals!='e217') & (df.optoep.values<2)]
df['num_epochs'] = num_epochs
df['cosine_sim_across_ep'] = [np.nanmean(xx) if len(xx)>0 else np.nan for xx in cs_all]
df = pd.concat([df,df2])
df = df.dropna(subset=['cosine_sim_across_ep', 'num_epochs'])
dfan = df.groupby(['animals', 'num_epochs']).mean(numeric_only=True)
dfan = dfan.reset_index()
dfan = dfan[dfan.num_epochs<5]
df_clean = dfan
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

# temp
df_clean = df_clean[(df_clean.animals!='e139') & (df_clean.animals!='e200') & (df_clean.animals!='e190')]
# Pairwise comparisons (Bonferroni)
unique_groups = sorted(df_clean['num_epochs'].unique())
group_data = {group: df_clean[df_clean['num_epochs'] == group]['cosine_sim_across_ep'] for group in unique_groups}

comparisons = list(combinations(unique_groups, 2))
raw_pvals = []
for g1, g2 in comparisons:
    _, pval = scipy.stats.ranksums(group_data[g1], group_data[g2])
    raw_pvals.append(pval)

# Bonferroni correction
reject, corrected_pvals, _, _ = multipletests(raw_pvals, method='bonferroni')

# Plot
s=10
plt.figure(figsize=(3,5))
ax = sns.barplot(x='num_epochs', y='cosine_sim_across_ep', data=df_clean, errorbar='se',
            fill=False, color='k')
sns.stripplot(x='num_epochs', y='cosine_sim_across_ep', data=df_clean, color='k', jitter=True,
            s=s,alpha=0.7)

# Annotate
fs = 30
pshift = 0.05
max_y = df_clean['cosine_sim_across_ep'].max()

for i, ((g1, g2), pval, rej) in enumerate(zip(comparisons, corrected_pvals, reject)):
    x1, x2 = int(g1)-2, int(g2)-2
    y = max_y + 0.05 * (i + 1)
    ax.plot([x1, x1, x2, x2], [y, y+0.01, y+0.01, y], lw=1.5, c='k')

    if pval < 0.001:
        star = '***'
    elif pval < 0.01:
        star = '**'
    elif pval < 0.05:
        star = '*'
    else:
        star = 'ns'

    ax.text((x1 + x2) / 2, y + 0.015, star, ha='center', fontsize=fs)
    ax.text((x1 + x2) / 2, y + 0.015 + pshift, f'p={pval:.2g}', ha='center', rotation=45, fontsize=12)

ax.set_title('Post-reward ensembles', pad=50)
plt.tight_layout()
plt.show()
ax.spines[['top','right']].set_visible(False)
#%%
df_clean.to_csv(r'Z:\condition_df\postrew_ensemble_w_dt.csv', index=None)
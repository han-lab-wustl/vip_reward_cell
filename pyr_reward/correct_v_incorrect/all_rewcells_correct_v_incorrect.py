
#%%
"""
zahra
2025
dff by trial type
added all cell subtype function
also gets cosine sim
"""
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd, os
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
from projects.pyr_reward.rewardcell import trail_type_activity_quant, cosine_sim_ignore_nan, wilcoxon_r
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper\panels_main_figures'
savepth = os.path.join(savedst, 'all_rew_corr_v_incorr.pdf')
#%%
goal_cm_window=20 # to search for rew cells
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
radian_alignment_saved = {} # overwrite
radian_alignment = {}
lasttr=8 #  last trials
bins=90
tcs_correct_all=[]
tcs_fail_all=[]
# not used
epoch_perm=[]
goal_cell_iind=[]
goal_cell_prop=[]
num_epochs=[]
goal_cell_null=[]
pvals=[]
total_cells=[]
vel_corr_all=[]
vel_fail_all=[]
# iterate through all animals
dfs = []
for ii in range(len(conddf)):
    day = conddf.days.values[ii]
    animal = conddf.animals.values[ii]
    if (animal!='e217') & (conddf.optoep.values[ii]<2):
        if animal=='e145' or animal=='e139': pln=2 
        else: pln=0
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
        df,tcs_correct,tcs_fail,vel_correct,vel_fail=trail_type_activity_quant(ii,params_pth,animal,day,bins,radian_alignment,radian_alignment_saved,goal_cm_window,\
                pdf,epoch_perm,goal_cell_iind,goal_cell_prop,num_epochs,goal_cell_null,
                pvals,
                total_cells)
        dfs.append(df)
        tcs_correct_all.append(tcs_correct)
        tcs_fail_all.append(tcs_fail)
        vel_corr_all.append(vel_correct)
        vel_fail_all.append(vel_fail)
pdf.close()

#%%
# get examples of correct vs. fail
# take the first epoch and first cell?
# v take all cells
# per day per animal
import scipy.stats
# Normalize each row to [0, 1]
def normalize_rows_0_to_1(arr):
    row_max = np.nanmax(arr, axis=1, keepdims=True)
    normed = arr / row_max
    return normed
plt.rc('font', size=26)
# --- Settings ---
animals = [xx for ii, xx in enumerate(conddf.animals.values) if (xx != 'e217') and (conddf.optoep.values[ii] < 2)]
animals_test = np.unique(animals)
animals_test = [ 'e145', 'e186', 'e189', 'e190', 'e200', 'e201', 'e216',
    'e218', 'z8', 'z9','z16']
# animals_test=['e186','z8','z9','e201']
cell_types = ['pre', 'post', 'far_pre', 'far_post']
bins = 150
# recalc tc
dff_correct_per_type = []
dff_fail_per_type = []
cs_per_type = []
# --- Loop through cell types ---
for cll, cell_type in enumerate(cell_types):
    dff_correct_per_an = []
    dff_fail_per_an = []
    cs_per_an=[]

    for animal in animals_test:
        dff_correct, dff_fail = [], []
        tcs_correct, tcs_fail = [], []
        if 'pre' in cell_type:
            activity_window='pre'
        else:
            activity_window='post'
        # print(cell_type, activity_window)
        # Get relevant traces for correct trials
        tcs_correct_ct = [xx[cll] for xx in tcs_correct_all]
        for ii, tcs_corr in enumerate(tcs_correct_ct):
            if animals[ii] == animal and tcs_corr.shape[1] > 0:
                tc = np.nanmean(tcs_corr, axis=0)
                # all epochs
                tc=np.vstack(tcs_corr)
                if tc.ndim == 1: tc = np.expand_dims(tc, 0)
                tcs_correct.append(tc)
                # Quantile per trial
                # switched to mean
                if activity_window == 'pre':
                    dff_correct.append(np.nanmean(tc[:, bins//3:bins//2], axis=1))
                else: # post
                    dff_correct.append(np.nanmean(tc[:, bins//2:], axis=1))

        # Get relevant traces for failed trials
        tcs_fail_ct = [xx[cll] for xx in tcs_fail_all]
        for ii, tcs_f in enumerate(tcs_fail_ct):
            if animals[ii] == animal and tcs_f.shape[1] > 0:
                tc = np.nanmean(tcs_f, axis=0)
                tc=np.vstack(tcs_f)
                if tc.ndim == 1: tc = np.expand_dims(tc, 0)
                tcs_fail.append(tc)
                if np.sum(np.isnan(tc)) == 0:
                    if activity_window == 'pre':
                        dff_fail.append(np.nanmean(tc[:, bins//3:bins//2], axis=1))
                    else: # post
                        dff_fail.append(np.nanmean(tc[:, bins//2:], axis=1))

        dff_correct_per_an.append(dff_correct)
        dff_fail_per_an.append(dff_fail)
        # get cosine similarity per cell
        cs = [cosine_sim_ignore_nan(np.vstack(tcs_correct)[i],np.vstack(tcs_fail)[i])
            for i in range(np.vstack(tcs_correct).shape[0])]
        cs=np.array(cs); cs_per_an.append(cs)
        # --- Plotting ---
        fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(10, 10),
            gridspec_kw={'height_ratios': [2, 1], 'width_ratios': [1, 1, 0.05]},
            constrained_layout=True)
        axes = axes.flatten()

        # Panel: Correct Trials Heatmap
        ax = axes[0]
        tc = np.vstack(tcs_correct)
        tc_z = normalize_rows_0_to_1(tc)
        valid_rows = ~(np.sum(np.isnan(tc_z),axis=1)>0)
        tc_z=tc_z[valid_rows,:]
        peak_bins = np.argmax(tc_z, axis=1)
        sort_idx = np.argsort(peak_bins)
        vmin, vmax = 0,1
        im = ax.imshow(tc_z[sort_idx], vmin=vmin, vmax=vmax, aspect='auto', cmap='viridis')
        ax.axvline(bins // 2, color='w', linestyle='--',linewidth=3)
        ax.set_xticks([0, tc.shape[1] // 2, tc.shape[1]])
        ax.set_yticks([0,tc_z.shape[0]])
        ax.set_yticklabels([0,tc_z.shape[0]])
        ax.set_xticklabels(['-$\\pi$', 0, '$\\pi$'])
        ax.set_ylabel('Reward cell # per epoch (sorted)')
        ax.set_xlabel('Reward-centric distance ($\\Theta$)')
        ax.set_title(f'mouse {animal}\nCorrect Trials')

        # Panel: Failed Trials Heatmap
        ax = axes[1]
        tc = np.vstack(tcs_fail)[sort_idx]
        # Remove rows with all NaNs before sorting
        # only for figures, keep for calc
        # tc_fail_valid = tc
        tc = normalize_rows_0_to_1(tc)
        valid_rows = ~(np.sum(np.isnan(tc),axis=1)>0)
        tc_fail_valid = tc[valid_rows,:]
        peak_bins = np.argmax(tc_fail_valid, axis=1)
        sort_idx = np.argsort(peak_bins)
        if len(tc_fail_valid)>0:
            im2 = ax.imshow(tc_fail_valid[sort_idx], aspect='auto', cmap='viridis')
            ax.axvline(bins // 2, color='w', linestyle='--',linewidth=3)
            ax.set_xticks([0, tc.shape[1] // 2, tc.shape[1]])
            ax.set_xticklabels(['-$\\pi$', 0, '$\\pi$'])
            ax.set_yticks([0,tc_fail_valid.shape[0]])
            ax.set_yticklabels([0,tc_fail_valid.shape[0]])
            ax.set_title('Incorrect Trials')

        # Colorbar
        cbar_ax = axes[2]
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar_ax.set_ylabel('Norm. $\Delta$F/F', rotation=270)
        cbar_ax.yaxis.set_label_position('left')
        cbar_ax.yaxis.tick_left()
        cbar.set_ticks([vmin, vmax])
        cbar.set_ticklabels([f"{vmin:.1f}", f"{vmax:.1f}"])

        # Panel: Correct Mean
        ax = axes[3]
        tc_all = np.vstack(tcs_correct)
        m = np.nanmean(tc_all, axis=0)
        sem = scipy.stats.sem(tc_all, axis=0, nan_policy='omit')
        ax.plot(m, color='seagreen')
        ax.fill_between(np.arange(len(m)), m - sem, m + sem, color='seagreen', alpha=0.5)
        ax.axvline(bins // 2, color='k', linestyle='--',linewidth=3)
        ax.set_xticks([0, len(m) // 2, len(m)])
        ax.set_xticklabels(['-$\\pi$', 0, '$\\pi$'])
        ax.set_xlabel('Reward-centric distance ($\\Theta$)')

        ax.spines[['top', 'right']].set_visible(False)
        ax.set_ylabel('$\Delta$F/F')
        height = m.max() + m.max()/2
        ax.set_ylim([0, height])
        ax.set_title('Correct Mean')

        # Panel: Failed Mean
        ax = axes[4]
        tc_all = np.vstack(tcs_fail)
        m = np.nanmean(tc_all, axis=0)
        sem = scipy.stats.sem(tc_all, axis=0, nan_policy='omit')
        ax.plot(m, color='firebrick')
        ax.fill_between(np.arange(len(m)), m - sem, m + sem, color='firebrick', alpha=0.5)
        ax.axvline(bins // 2, color='k', linestyle='--',linewidth=3)
        ax.set_xticks([0, len(m) // 2, len(m)])
        ax.set_xticklabels(['-$\\pi$', 0, '$\\pi$'])
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_ylabel('')
        ax.set_ylim([0, height])
        ax.set_title('Incorrect Mean')
        ax.set_xlabel('Reward-centric distance ($\\Theta$)')
        # Hide empty axis
        axes[5].axis('off')
        fig.suptitle(cell_type)
        # plt.close(fig)        
        plt.savefig(os.path.join(savedst, f'{animal}_{cell_type}_correctvfail.svg'),bbox_inches='tight')
    dff_correct_per_type.append(dff_correct_per_an)
    dff_fail_per_type.append(dff_fail_per_an)
    cs_per_type.append(cs_per_an)

#%%

# recalculate tc
dfsall = []
for cll,celltype in enumerate(cell_types):
    animals_unique = animals_test
    df=pd.DataFrame()
    correct = np.concatenate([np.concatenate(xx) for xx in dff_correct_per_type[cll]])
    incorrect = np.concatenate([np.concatenate(xx) if len(xx)>0 else [] for xx in dff_fail_per_type[cll]])
    df['mean_dff'] = np.concatenate([correct,incorrect])
    df['trial_type']=np.concatenate([['correct']*len(correct),['incorrect']*len(incorrect)])
    ancorr = np.concatenate([[animals_unique[ii]]*len(np.concatenate(xx)) for ii,xx in enumerate(dff_correct_per_type[cll])])
    anincorr = np.concatenate([[animals_unique[ii]]*len(np.concatenate(xx)) if len(xx)>0 else [] for ii,xx in enumerate(dff_fail_per_type[cll])])
    df['animal'] = np.concatenate([ancorr, anincorr])
    df['cell_type'] = [celltype]*len(df)
    dfsall.append(df)
    
# average
bigdf = pd.concat(dfsall)
bigdf=bigdf.groupby(['animal', 'trial_type', 'cell_type']).mean(numeric_only=True)
bigdf=bigdf.reset_index()
bigdf2=pd.read_csv(r'Z:\saved_datasets\performance_place.csv')
bigdf=bigdf[(bigdf.animal!='e189') & (bigdf.animal!='z16') & (bigdf.animal!='e145')]
bigdf=pd.concat([bigdf,bigdf2])
bigdf=bigdf.drop(columns=['Unnamed: 0'])
s=10
bigdf['trial_type'] = bigdf['trial_type'].str.capitalize()

cell_order = ['pre', 'post', 'far_pre', 'far_post', 'place']
fig,ax = plt.subplots(figsize=(7.5,4))
# sns.stripplot(x='cell_type', y='mean_dff', data=bigdf,hue='trial_type',
#         dodge=True,palette={'Correct':'seagreen', 'Incorrect': 'firebrick'},
#         s=s,alpha=0.7,order=cell_order)
sns.barplot(x='cell_type', y='mean_dff', data=bigdf,hue='trial_type',
        fill=False,palette={'Correct':'seagreen', 'Incorrect': 'firebrick'},
            order=cell_order,errorbar='se')

ax.spines[['top','right']].set_visible(False)
ax.set_ylabel('Mean $\Delta F/F$ (pre or post-reward)')
ax.set_xlabel('Cell type')

ax.set_xticklabels(['Pre', 'Post', 'Far pre', 'Far post', 'Place'])
# Use the last axis to get handles/labels
handles, labels = ax.get_legend_handles_labels()
# Create a single shared legend with title "Trial type" 
ax.legend(
    handles, labels,
    loc='upper left',
    bbox_to_anchor=(.7, 1),
    borderaxespad=0.,
    title='Trial type'
)
xpos = {ct: i for i, ct in enumerate(cell_order)}

# Draw dim gray connecting lines between paired trial types
for animal in bigdf['animal'].unique():
    for ct in cell_order:
        sub = bigdf[(bigdf['animal'] == animal) & (bigdf['cell_type'] == ct)]
        if len(sub) == 2:  # both trial types present
            # Get x locations for dodge-separated points
            x_base = xpos[ct]
            offsets = [-0.2, 0.2]  # match sns stripplot dodge
            y_vals = sub.sort_values('trial_type')['mean_dff'].values
            x_vals = [x_base + offset for offset in offsets]
            ax.plot(x_vals, y_vals, color='dimgray', alpha=0.5, linewidth=1)

# ans = bigdf.animal.unique()
# for i in range(len(ans)):
#     for j,tr in enumerate(np.unique(bigdf.cell_type.values)):
#         testdf= bigdf[(bigdf.animal==ans[i]) & (bigdf.cell_type==tr)]
#         ax = sns.lineplot(x='trial_type', y='mean_dff', 
#         data=testdf,
#         errorbar=None, color='dimgray', linewidth=2, alpha=0.7,ax=ax)
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM

# 1) Two-way repeated measures ANOVA
aov = AnovaRM(
    data=bigdf,
    depvar='mean_dff',
    subject='animal',
    within=['trial_type','cell_type']
).fit()
print(aov)    # F-stats and p-values for main effects and interaction
aov_table = aov.anova_table

# Print full p-values
with pd.option_context('display.precision', 10):
    print(aov_table)
# 2) Post-hoc paired comparisons: correct vs incorrect within each cell_type
posthoc = []
for ct in cell_order:
    sub = bigdf[bigdf['cell_type']==ct]
    cor = sub[sub['trial_type']=='Correct']['mean_dff'].values
    inc = sub[sub['trial_type']=='Incorrect']['mean_dff'].values
    t, p_unc = wilcoxon_r(cor, inc)
    posthoc.append({
        'cell_type': ct,
        't_stat':    t,
        'p_uncorrected': p_unc
    })

posthoc = pd.DataFrame(posthoc)
# Bonferroni
import statsmodels
reject, pvals_corrected, _, _ = statsmodels.stats.multitest.multipletests(posthoc['p_uncorrected'], method='fdr_bh')
posthoc['p_fdr_bh'] = pvals_corrected
posthoc['significant'] = reject
# map cell_type → x-position
xpos = {ct: i for i, ct in enumerate(cell_order)}
for _, row in posthoc.iterrows():
    x = xpos[row['cell_type']]
    y = bigdf[
        (bigdf['cell_type']==row['cell_type'])
    ]['mean_dff'].quantile(.7) + 0.1  # just above the tallest bar
    p = row['p_fdr_bh']
    stars = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
    ax.text(x, y, stars, ha='center', va='bottom', fontsize=42)    
    r=row['t_stat']
    ax.text(x, y, f'r={r:.3g}\np={p:.3g}', ha='center', va='bottom', fontsize=12)
# Assuming `axes` is a list of subplots and `ax` is the one with the legend (e.g., the last one)

# Example interpretation (fill in with your numbers)
# trial_type (Num DF = 1, Den DF = 9, F = 12.3, p = 0.006)
# -- There is a significant main effect of trial type: across all cell types, mean ΔF/F is different on correct vs. incorrect trials.

# cell_type (Num DF = 3, Den DF = 27, F = 8.7, p < 0.001)
# -- There is a significant main effect of cell type: some cell types have higher overall ΔF/F than others, regardless of trial outcome.

# trial_type × cell_type (Num DF = 3, Den DF = 27, F = 4.2, p = 0.014)
# -- The interaction is significant: the difference between correct vs. incorrect ΔF/F depends on which cell type you look at.

# Because the interaction is significant, you should then examine post-hoc tests (e.g., the paired comparisons you ran) to see for each cell type whether correct vs. incorrect is significant.
plt.savefig(os.path.join(savedst, 'allcelltype_trialtype.svg'),bbox_inches='tight')
#%%
# quantify cosine sim
# TODO: get COM per cell
dfsall = []
for cll,celltype in enumerate(cell_types):
    animals_unique = animals_test
    df=pd.DataFrame()
    cs = cs_per_type[cll]
    df['cosine_sim'] = np.concatenate(cs)
    # df['com'] = np.nanmax(tc) for tc
    ancorr = np.concatenate([[animals_unique[ii]]*len(xx) for ii,xx in enumerate(cs)])
    df['animal'] = ancorr
    df['cell_type'] = [celltype]*len(df)
    dfsall.append(df)

# average
bigdf = pd.concat(dfsall)
bigdf=bigdf[(bigdf.animal!='e189') & (bigdf.animal!='z16') & (bigdf.animal!='e145')]

bigdf_avg = bigdf.groupby(['animal', 'cell_type'])['cosine_sim'].mean().reset_index()
bigdf = bigdf.groupby(['animal', 'cell_type']).mean().reset_index()
# Step 2: Check if data is balanced
pivoted = bigdf_avg.pivot(index='animal', columns='cell_type', values='cosine_sim')
if pivoted.isnull().any().any():
    print("⚠️ Warning: Data is unbalanced — some animals are missing data for some cell types.")
else:
    print("✅ Data is balanced.")

# Step 2: Split cosine similarity by cell_type into a list of arrays
groups = [
    group['cosine_sim'].values
    for name, group in bigdf_avg.groupby('cell_type')
]
# Step 3: Kruskal–Wallis test
stat, p = scipy.stats.kruskal(*groups)
# Print results
print(f"Kruskal–Wallis H-statistic: {stat:.4f}")
print(f"p-value: {p:.6g}")

# Step 4: Post-hoc paired comparisons between all cell types
posthoc = []
pvals = []

comb = [ ('post', 'far_post'),
 ('pre', 'far_pre')]
for ct1, ct2 in comb:
    sub1 = pivoted[ct1]
    sub2 = pivoted[ct2]
    t, p_unc = scipy.stats.wilcoxon(sub1, sub2)
    posthoc.append({
        'comparison': f"{ct1} vs {ct2}",
        'W-statistic': t,
        'p_uncorrected': p_unc
    })
    pvals.append(p_unc)

from statsmodels.stats.multitest import fdrcorrection
# Step 5: FDR correction
rej, p_fdr = fdrcorrection(pvals, alpha=0.05)
for i, fdr_p in enumerate(p_fdr):
    posthoc[i]['p_fdr'] = fdr_p
    posthoc[i]['significant'] = rej[i]
import matplotlib.pyplot as plt
import seaborn as sns

# Re-map cell type names to x-axis positions
cell_order = ['pre', 'post', 'far_pre', 'far_post']  # or whatever order you're using
xpos = {ct: i for i, ct in enumerate(cell_order)}
color='saddlebrown'
# Plot
fig,ax = plt.subplots(figsize=(5,6))
sns.stripplot(data=bigdf, x='cell_type', y='cosine_sim',
    order=cell_order,color=color,
    alpha=0.7, size=10,)
sns.barplot(data=bigdf, x='cell_type', y='cosine_sim',color=color,fill=False,
            order=cell_order, errorbar='se')
ax.set_xticklabels(['Pre', 'Post', 'Far pre', 'Far post'],rotation=30)
# Annotate pairwise comparisons
height = bigdf['cosine_sim'].max() + 0.02
step = 0.02
# Add lines connecting same-animal points across cell types
for animal, subdf in bigdf_avg.groupby('animal'):
    subdf = subdf.set_index('cell_type').reindex(cell_order)
    xs = [xpos[ct] for ct in subdf.index]
    ys = subdf['cosine_sim'].values
    ax.plot(xs, ys, marker='o', color='gray', linewidth=1, alpha=0.5, zorder=0)

posthoc=pd.DataFrame(posthoc)
for i, row in posthoc.iterrows():
    if row['significant']:
        group1, group2 = row['comparison'].split(' vs ')
        x1, x2 = xpos[group1], xpos[group2]
        y = height + i * step* 4

        # Connecting line
        ax.plot([x1, x1, x2, x2], [y, y + step / 2, y + step / 2, y], lw=1.2, color='black')

        # Stars
        p = row['p_fdr']
        stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        ax.text((x1 + x2) / 2, y + step / 2 + 0.002, stars, ha='center', va='bottom', fontsize=46)

# Axis formatting
ax.set_ylabel("Cosine similarity\n Correct vs. Incorrect tuning curve")
ax.set_xlabel("Reward cell type")
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(savedst, 'allcelltype_corr_v_incorr_cs.svg'),bbox_inches='tight')

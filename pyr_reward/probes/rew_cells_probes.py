
#%%
"""
zahra
2025
dff by trial type
probes 1,2,3
added all cell subtype function
also gets cosine sim
"""
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd, os
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
# plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.pyr_reward.rewardcell import trail_type_probe_activity_quant, cosine_sim_ignore_nan
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper\panels_main_figures'
savepth = os.path.join(savedst, 'all_rew_corr_v_incorr_v_probe.pdf')
#%%
goal_cm_window=20 # to search for rew cells
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
radian_alignment_saved = {} # overwrite
radian_alignment = {}
lasttr=8 #  last trials
bins=90
tcs_correct_all=[]
tcs_fail_all=[]
tcs_probes_all=[]
# not used
epoch_perm=[]
goal_cell_iind=[]
goal_cell_prop=[]
num_epochs=[]
goal_cell_null=[]
pvals=[]
total_cells=[]
# iterate through all animals
dfs = []
for ii in range(len(conddf)):
    day = conddf.days.values[ii]
    animal = conddf.animals.values[ii]
    if (animal!='e217') & (conddf.optoep.values[ii]<2):
        if animal=='e145' or animal=='e139': pln=2 
        else: pln=0
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
        df,tcs_correct,tcs_fail,tcs_probes=trail_type_probe_activity_quant(ii,params_pth,\
                animal,day,bins,radian_alignment,radian_alignment_saved,goal_cm_window,
                pdf,epoch_perm,goal_cell_iind,goal_cell_prop,num_epochs,goal_cell_null,
                pvals,
                total_cells)
        dfs.append(df)
        tcs_correct_all.append(tcs_correct)
        tcs_fail_all.append(tcs_fail)
        tcs_probes_all.append(tcs_probes)
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
    # Identify rows where max is NaN (i.e., all values were NaN)
    bad_rows = np.isnan(row_max).flatten()
    row_max[bad_rows] = 1  # avoid division by zero
    normed = arr / row_max
    normed[bad_rows] = 0.001  # set all-NaN rows to 0
    return normed

plt.rc('font', size=20)

# --- Settings ---
animals = [xx for ii, xx in enumerate(conddf.animals.values) if (xx != 'e217') and (conddf.optoep.values[ii] < 2)]
animals_test = np.unique(animals)
animals_test = [ 'e145', 'e186', 'e189', 'e190', 'e200', 'e201', 'e216',
       'e218', 'z8', 'z9']
cell_types = ['pre', 'post', 'far_pre', 'far_post']
bins = 150
# recalc tc
dff_correct_per_type = []
dff_fail_per_type = []
cs_per_type = []
dff_probe_per_type=[]
cs_probe_per_type=[]
cs_probe_correct_per_type=[]
# --- Loop through cell types ---
for cll, cell_type in enumerate(cell_types):
        dff_correct_per_an = []
        dff_fail_per_an = []
        dff_probe_per_an = []
        cs_per_an = []
        cs_probe_per_an = []
        cs_probe_correct_per_an = []

        for animal in animals_test:
            # --- Initialize containers ---
            tcs_correct, tcs_fail = [], []
            tcs_probe =[]  # probe 0, 1, 2 traces

            if 'pre' in cell_type:
                activity_window = 'pre'
                win = slice(bins // 3, bins // 2)
            else:
                activity_window = 'post'
                win = slice(bins // 2, bins)

            # --- Get correct trial data ---
            for ii, xx in enumerate(tcs_correct_all):
                if animals[ii] == animal:
                    tc = xx[cll]
                    if tc.shape[1] > 0:
                        tc_avg = np.nanmean(tc,axis=0) #do not average across epochs??!
                        if tc_avg.ndim == 1: tc_avg = np.expand_dims(tc_avg, 0)
                        tcs_correct.append(tc_avg)

            # --- Get fail trial data ---
            for ii, xx in enumerate(tcs_fail_all):
                if animals[ii] == animal:
                    tc = xx[cll]
                    if tc.shape[1] > 0:
                        tc_avg = np.nanmean(tc,axis=0)
                        if tc_avg.ndim == 1: tc_avg = np.expand_dims(tc_avg, 0)
                        tcs_fail.append(tc_avg)
            #probe
            for ii, xx in enumerate(tcs_probes_all):
               if animals[ii] == animal:
                  tc = xx[cll]
                  if tc.shape[2] > 0:
                     # average across ep
                     tc_avg = [np.nanmean(tc[:,k,:,:],axis=0) for k in range(3)]
                     tcs_probe.append(tc_avg)
            # --- Stack and sort ---
            tc_corr = np.vstack(tcs_correct)
            tc_fail = np.vstack(tcs_fail)
            tc_prob1=np.vstack([xx[0] for xx in tcs_probe])
            tc_prob2=np.vstack([xx[1] for xx in tcs_probe])
            tc_prob3=np.vstack([xx[2] for xx in tcs_probe])
            # Remove rows where all bins are NaN in correct trials (sets reference for sorting)
            # valid_rows = ~np.all(np.isnan(tc_fail), axis=1)
            # tc_fail = tc_fail[valid_rows]
            # tc_prob1 = tc_prob1[valid_rows]
            # tc_prob2 = tc_prob2[valid_rows]
            # tc_prob3 = tc_prob3[valid_rows]
            # Normalize
            tc_corr_norm = normalize_rows_0_to_1(tc_corr)
            valid_rows = ~np.all(np.isnan(tc_corr_norm), axis=1)
            tc_corr_norm = tc_corr_norm[valid_rows]
            sort_idx = np.argsort(np.nanargmax(tc_corr_norm, axis=1))
            # Sort all trial types using the same cell order
            tc_corr_sorted = tc_corr_norm[sort_idx]
            tc_fail_sorted = normalize_rows_0_to_1(tc_fail)[valid_rows][sort_idx]
            tc_prob1_sorted = normalize_rows_0_to_1(tc_prob1)[valid_rows][sort_idx]
            tc_prob2_sorted = normalize_rows_0_to_1(tc_prob2)[valid_rows][sort_idx]
            tc_prob3_sorted = normalize_rows_0_to_1(tc_prob3)[valid_rows][sort_idx]
            # --- dF/F from activity window ---
            dff_correct = np.nanmean(tc_corr[:, win], axis=1)
            dff_fail = np.nanmean(tc_fail[:, win], axis=1)
            dff_correct_per_an.append(dff_correct)
            dff_fail_per_an.append(dff_fail)

            # --- Cosine similarity (correct vs fail) ---
            cs = [cosine_sim_ignore_nan(tc_corr[i], tc_fail[i]) for i in range(tc_corr.shape[0])]
            cs_per_an.append(np.array(cs))

            # --- Probes ---

            # --- Cosine similarity (between probes) ---
            # cs_probe = []
            # cs_probe_correct = []
            # if len(probe_avg) == 3:
            #     probe_avg = np.array(probe_avg)  # shape: 3 x cells x bins
            #     for i in range(probe_avg.shape[1]):  # cells
            #         cs_vals = [
            #             cosine_sim_ignore_nan(probe_avg[0, i], probe_avg[1, i]),
            #             cosine_sim_ignore_nan(probe_avg[0, i], probe_avg[2, i]),
            #             cosine_sim_ignore_nan(probe_avg[1, i], probe_avg[2, i]),
            #         ]
            #         cs_probe.append(cs_vals)
            #         # probe-correct similarity
            #         cs_vals_corr = [
            #             cosine_sim_ignore_nan(probe_avg[0, i], tc_corr[i]),
            #             cosine_sim_ignore_nan(probe_avg[1, i], tc_corr[i]),
            #             cosine_sim_ignore_nan(probe_avg[2, i], tc_corr[i]),
            #         ]
            #         cs_probe_correct.append(cs_vals_corr)
            # else:
            #     cs_probe = np.full((tc_corr.shape[0], 3), np.nan)
            #     cs_probe_correct = np.full((tc_corr.shape[0], 3), np.nan)

            # cs_probe_per_an.append(np.array(cs_probe))
            # cs_probe_correct_per_an.append(np.array(cs_probe_correct))

            # --- Plot ---
            fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 15),
                                     constrained_layout=True,
                                     gridspec_kw={'height_ratios': [2, 1]})
            axes = axes.flatten()

            titles = ['Correct Trials', 'Failed Trials', 'Probe 1', 'Probe 2', 'Probe 3']
            data_to_plot = [tc_corr_sorted, tc_fail_sorted,
                tc_prob1_sorted,tc_prob2_sorted,tc_prob3_sorted
            ]

            for i, ax in enumerate(axes[:5]):
               im = ax.imshow(data_to_plot[i], aspect='auto', vmin=0, vmax=1, cmap='viridis')
               ax.axvline(bins // 2, color='w', linestyle='--')
               ax.set_title(titles[i])
               ax.set_xticks([0, bins // 2, bins])
               ax.set_xticklabels(['-$\\pi$', 0, '$\\pi$'])
               if i % 3 == 0:
                  ax.set_ylabel('Cells (sorted)')

            # Mean traces
            trace_data = [tc_corr, tc_fail, tc_prob1,tc_prob2,tc_prob3]
            colors = ['seagreen', 'firebrick', 'royalblue', 'goldenrod', 'purple']

            for i in range(5):
               ax = axes[5 + (i // 3) * 3 + i % 3]
               if not i==0:
                  ax.sharey(axes[5])
               tc = trace_data[i]
               if len(tc) == 0: continue
               m = np.nanmean(tc, axis=0)
               sem = scipy.stats.sem(tc, axis=0, nan_policy='omit')
               ax.plot(m, color=colors[i])
               ax.fill_between(np.arange(len(m)), m - sem, m + sem, color=colors[i], alpha=0.5)
               ax.axvline(bins // 2, color='k', linestyle='--')
               ax.set_xticks([0, bins // 2, bins])
               ax.set_xticklabels(['-$\\pi$', 0, '$\\pi$'])
               ax.set_title(f'{titles[i]} Mean')
               ax.set_ylim(bottom=0)

            fig.suptitle(f'{animal}, {cell_type}')
            # plt.close(fig)

        # Store results
        dff_correct_per_type.append(dff_correct_per_an)
        dff_fail_per_type.append(dff_fail_per_an)
      #   dff_probe_per_type.append(dff_probe_per_an)
        cs_per_type.append(cs_per_an)
      #   cs_probe_per_type.append(cs_probe_per_an)
      #   cs_probe_correct_per_type.append(cs_probe_correct_per_an)

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
bigdf=bigdf[bigdf.animal!='e189']
s=12
cell_order = cell_types
fig,ax = plt.subplots(figsize=(6,4))
sns.stripplot(x='cell_type', y='mean_dff', data=bigdf,hue='trial_type',
        dodge=True,palette={'correct':'seagreen', 'incorrect': 'firebrick'},
        s=s,alpha=0.7,    order=cell_order)
sns.barplot(x='cell_type', y='mean_dff', data=bigdf,hue='trial_type',
        fill=False,palette={'correct':'seagreen', 'incorrect': 'firebrick'},
            order=cell_order)

ax.spines[['top','right']].set_visible(False)
ax.set_ylabel('Mean $\Delta F/F$ rel. to rew.')
ax.set_xlabel('Reward cell type')
ax.legend_.remove()
ax.set_xticklabels(['Pre', 'Post', 'Far pre', 'Far post'])
# Use the last axis to get handles/labels
handles, labels = ax.get_legend_handles_labels()
# Create a single shared legend with title "Trial type"
ax.legend(
    handles, labels,
    loc='upper left',
    bbox_to_anchor=(1.02, 1),
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

# 2) Post-hoc paired comparisons: correct vs incorrect within each cell_type
posthoc = []
for ct in cell_order:
    sub = bigdf[bigdf['cell_type']==ct]
    cor = sub[sub['trial_type']=='correct']['mean_dff']
    inc = sub[sub['trial_type']=='incorrect']['mean_dff']
    t, p_unc = scipy.stats.ttest_rel(cor, inc)
    posthoc.append({
        'cell_type': ct,
        't_stat':    t,
        'p_uncorrected': p_unc
    })

posthoc = pd.DataFrame(posthoc)
# Bonferroni
posthoc['p_bonferroni'] = np.minimum(posthoc['p_uncorrected'] * len(posthoc), 1.0)
print(posthoc)
# map cell_type → x-position
xpos = {ct: i for i, ct in enumerate(cell_order)}
for _, row in posthoc.iterrows():
    x = xpos[row['cell_type']]
    y = bigdf[
        (bigdf['cell_type']==row['cell_type'])
    ]['mean_dff'].quantile(.7) + 0.1  # just above the tallest bar
    p = row['p_bonferroni']
    stars = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
    ax.text(x, y, stars, ha='center', va='bottom', fontsize=42)
    if p>0.05:
        ax.text(x, y, f'p={p:.2g}', ha='center', va='bottom', fontsize=12)
# Assuming `axes` is a list of subplots and `ax` is the one with the legend (e.g., the last one)

# Example interpretation (fill in with your numbers)
# trial_type (Num DF = 1, Den DF = 9, F = 12.3, p = 0.006)
# -- There is a significant main effect of trial type: across all cell types, mean ΔF/F is different on correct vs. incorrect trials.

# cell_type (Num DF = 3, Den DF = 27, F = 8.7, p < 0.001)
# -- There is a significant main effect of cell type: some cell types have higher overall ΔF/F than others, regardless of trial outcome.

# trial_type × cell_type (Num DF = 3, Den DF = 27, F = 4.2, p = 0.014)
# -- The interaction is significant: the difference between correct vs. incorrect ΔF/F depends on which cell type you look at.

# Because the interaction is significant, you should then examine post-hoc tests (e.g., the paired comparisons you ran) to see for each cell type whether correct vs. incorrect is significant.
# plt.savefig(os.path.join(savedst, 'allcelltype_trialtype.svg'),bbox_inches='tight')
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
bigdf_avg = bigdf.groupby(['animal', 'cell_type'])['cosine_sim'].mean().reset_index()
bigdf = bigdf.groupby(['animal', 'cell_type']).mean().reset_index()
# Step 2: Check if data is balanced
pivoted = bigdf_avg.pivot(index='animal', columns='cell_type', values='cosine_sim')
if pivoted.isnull().any().any():
    print("⚠️ Warning: Data is unbalanced — some animals are missing data for some cell types.")
else:
    print("✅ Data is balanced.")

# Step 3: One-way repeated measures ANOVA
aov = AnovaRM(
    data=bigdf_avg,
    depvar='cosine_sim',
    subject='animal',
    within=['cell_type']
).fit()
print(aov)

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
ax.set_xlabel("Cell type")
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
fig.suptitle('Tuning properties')
plt.savefig(os.path.join(savedst, 'allcelltype_corr_v_incorr_cs.svg'),bbox_inches='tight')

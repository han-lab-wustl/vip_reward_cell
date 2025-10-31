"""
bo fit a model to get lick responsive neurons and compared those with reward and place cells
"""
#%%
import pickle, os, numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt,scipy
import matplotlib.backends.backend_pdf, matplotlib as mpl, sys
from statsmodels.stats.multitest import multipletests
import matplotlib.patches as mpatches
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.pyr_reward.placecell import intersect_arrays,make_tuning_curves_radians_by_trialtype, make_tuning_curves
from projects.pyr_reward.rewardcell import get_radian_position_first_lick_after_rew
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
plt.rc('font', size=20)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"

#%%
# get cell eg
fl = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper\from_bo\RRLR_Ratio'
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
with open(fl, "rb") as fp: #unpickle
   dct = pickle.load(fp)
rrlr_overlap =[np.nanmean(v) for k,v in dct.items()]

fl = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper\from_bo\PPLR_Ratio'
with open(fl, "rb") as fp: #unpickle
   dct = pickle.load(fp)
placelr_overlap =[np.nanmean(v) for k,v in dct.items()]

fl = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper\from_bo\Random_PPLR_Ratio'
with open(fl, "rb") as fp: #unpickle
   dct = pickle.load(fp)
shuf_placelr_overlap =[np.nanmean(v) for k,v in dct.items()]


fl = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper\from_bo\\Random_RRLR_Ratio'
with open(fl, "rb") as fp: #unpickle
   dct = pickle.load(fp)
shuf_rrlr_overlap =[np.nanmean(v) for k,v in dct.items()]

df=pd.DataFrame()
df['% Overlap'] = np.concatenate([rrlr_overlap,placelr_overlap,shuf_placelr_overlap,shuf_rrlr_overlap])
df['Cell type'] = np.concatenate([['Reward']*len(rrlr_overlap),['Place']*len(placelr_overlap),['Place']*len(shuf_placelr_overlap),['Reward']*len(shuf_rrlr_overlap)])
df['Condition'] = np.concatenate([['Real']*len(rrlr_overlap),['Real']*len(placelr_overlap),['Shuffle']*len(shuf_placelr_overlap),['Shuffle']*len(shuf_rrlr_overlap)])
df['Animal'] = [k for k,v in dct.items()]*4
palette = {
    'Real': 'mediumslateblue',
    'Shuffle': 'grey'
}
fig, ax = plt.subplots(figsize=(3,4))
s = 10

# Plot lines connecting real and shuffle for each animal and transition
for animal in df['Animal'].unique():
    for tr, trans in enumerate(df['Cell type'].unique()):
        d_an = df[(df['Animal'] == animal) & (df['Cell type'] == trans)]
        if len(d_an) == 2:  # has both real and shuffle
            # x positions adjusted for dodge:
            x_real = tr - 0.2
            x_shuffle = tr + 0.2
            y_real = d_an[d_an['Condition']=='Real']['% Overlap'].values[0]
            y_shuffle = d_an[d_an['Condition']=='Shuffle']['% Overlap'].values[0]
            ax.plot([x_real, x_shuffle], [y_real, y_shuffle], color='grey', linewidth=1.5, zorder=0,alpha=.5)

# Stripplot with dodge for side-by-side points
# sns.stripplot(data=df, x='Cell type', y='% Overlap', hue='Condition',
#               dodge=True, palette=palette, alpha=0.7, s=s, ax=ax)

# Barplot with dodge for side-by-side bars
sns.barplot(data=df, x='Cell type', y='% Overlap', hue='Condition',
            palette=palette, dodge=True, errorbar='se', fill=False, ax=ax)
 
# ax.legend_.remove()  # remove legend for now
# --- Run paired t-tests and annotate ---
# Step 1: Collect p-values for all transitions
p_vals = []
t_stats = []
transitions = df['Cell type'].unique()

for trans in transitions:
    real_vals = df[(df['Cell type'] == trans) & (df['Condition'] == 'Real')].sort_values('Animal')['% Overlap']
    shuf_vals = df[(df['Cell type'] == trans) & (df['Condition'] == 'Shuffle')].sort_values('Animal')['% Overlap']
    t_stat, p_val = scipy.stats.wilcoxon(real_vals, shuf_vals)
    t_stats.append(t_stat)
    p_vals.append(p_val)

# Step 2: Apply multiple comparisons correction
rejected, p_vals_corrected, _, _ = multipletests(p_vals, method='fdr_bh')

# Step 3: Function to get asterisks
def pval_to_asterisks(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return ''  # no label for ns

# Step 4: Add asterisks to plot
y_max = df['% Overlap'].max()
for i, (trans, p_corr) in enumerate(zip(transitions, p_vals_corrected)):
    stars = pval_to_asterisks(p_corr)
    # ax.text(i, y_max + y_max*0.05, stars, ha='center', va='bottom', fontsize=46)
    # ax.text(i, y_max + y_max*0.05, p_corr, ha='center', va='bottom', fontsize=8)
    ax.text(i, y_max + y_max*0.05, 'ns', ha='center', va='bottom', fontsize=14)
    ax.plot([i-.2,i-.2,i+.2,i+.2], [y_max, y_max + y_max*0.05,y_max + y_max*0.05,y_max],color='k')
    

ax.spines[['top','right']].set_visible(False)
ax.set_title('Real vs. Shuffle Lick cells')
plt.savefig(os.path.join(savedst, 'lick_overlap_place_rew.svg'), bbox_inches='tight')

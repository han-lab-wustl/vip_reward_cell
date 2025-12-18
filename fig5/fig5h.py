"""
decoding analysis from bo
"""
#%%

import pickle, os, numpy as np, pandas as pd
import matplotlib as mpl
# matplotlib.use('tkagg')  # or 'qt5agg' if you're using PyQt5
%matplotlib inline
import matplotlib.pyplot as plt# mpl.use('qt5agg')  # or 'qt5agg' if you have PyQt installed
import matplotlib.pyplot as plt
import scipy, seaborn as sns
from statsmodels.stats.multitest import multipletests
import matplotlib.patches as mpatches,sys
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.pyr_reward.placecell import intersect_arrays,make_tuning_curves_radians_by_trialtype, make_tuning_curves
from projects.pyr_reward.rewardcell import get_radian_position_first_lick_after_rew, wilcoxon_r
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
plt.rc('font', size=18)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"

fl = r"C:\Users\Han\Documents\MATLAB\vip_reward_cell_copy_w_raw_data\raw_data\exfig3h.p"
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper\panels_main_figures'
with open(fl, "rb") as fp: #unpickle
    dct = pickle.load(fp)
#%%
# The file contained a list of eight dictionaries, [PRV, APRV, POV, APOV, PRF, APRF, POF, APOF], (Pre validation error (prv) pred - truth, Absolute pre validation error (aprv), |pred - truth|, Post validation error (pov), Absolute post validation error (apov), Pre testing error (prf), Absolute pre testing error (aprf), Post testing error (pof),Absolute post testing error (apof)). V stands for validation, which are unseen successful trials, F stands for failure, which are unseen failed trials. 
# Each dictionary's keys are mouse id, and values are lists fo decoding error of this mouse in different epochs. Therefore, the mean of the list represents the mean decoding error of this animal. A paired t-test between PRV and POV over animals will show the difference between the pre and post position decoding, which reflects the animal in the pre-reward area mostly encodes the future position, while encoding the previous position (or fluctuation around exact location).

labels = ['PRV', 'APRV', 'POV', 'APOV', 
          'PRF', 'APRF', 'POF', 'APOF']
trtypes = ['Correct', 'Correct', 'Correct', 'Correct', 
           'Incorrect', 'Incorrect', 'Incorrect', 'Incorrect']

# We'll demonstrate with Pre vs Post reward position error on correct trials
prv_dict = dct[0]  # Pre-reward validation error
pov_dict = dct[2]  # Post-reward validation error
prf_dict = dct[4]  # Post-reward validation error
pof_dict = dct[6]  # Post-reward validation error
# Get list of mice common to both conditions
mice = list(set(prv_dict.keys()) & set(pov_dict.keys()))

# Prepare data
data = []
for mouse in mice:
    mean_prv = np.mean(prv_dict[mouse])
    mean_pov = np.mean(pov_dict[mouse])
    mean_prf = np.mean(prf_dict[mouse])
    mean_pof = np.mean(pof_dict[mouse])
    data.append({'mouse': mouse, 'condition': 'Pre-reward', 
                 'mean_error': mean_prv, 'trial_type': 'Correct'})
    data.append({'mouse': mouse, 'condition': 'Post-reward', 
                 'mean_error': mean_pov, 'trial_type': 'Correct'})
    data.append({'mouse': mouse, 'condition': 'Pre-reward', 
                 'mean_error': mean_prf, 'trial_type': 'Incorrect'})
    data.append({'mouse': mouse, 'condition': 'Post-reward', 
                 'mean_error': mean_pof, 'trial_type': 'Incorrect'})
from statsmodels.stats.anova import AnovaRM
df = pd.DataFrame(data)
# Run 2-way repeated measures ANOVA
aovrm = AnovaRM(df, depvar='mean_error', subject='mouse', within=['condition', 'trial_type'])
res = aovrm.fit()
print(res)

# Plotting
palette={'Correct':'seagreen', 'Incorrect': 'firebrick'}
s = 12
a=0.7
df['mean_error'] = df['mean_error']*270

fig, ax = plt.subplots(figsize=(4,5))
# sns.stripplot(x='condition', y='mean_error', data=df, alpha=a,
            #   hue='trial_type', dodge=True, size=s, palette=palette)
sns.barplot(x='condition', y='mean_error', data=df, 
              hue='trial_type', dodge=True,fill=False, palette=palette)

# Get the x positions for each hue/condition combination using the dodge offset
# This works assuming the seaborn internals haven't changed; 0.2 is seaborn's default dodge value
x_base = {'Pre-reward': -.2, 'Post-reward': .8}  # Shift Incorrect right to separate
# Add grey lines per trial type (no cross-connections)
for trial_type in ['Pre-reward', 'Post-reward']:
    base_x = x_base[trial_type]
    for mouse in mice:
        subdf = df[(df['mouse'] == mouse) & (df['condition'] == trial_type)]
        if len(subdf) == 2:
            y0 = subdf[subdf['trial_type'] == 'Correct']['mean_error'].values[0]
            y1 = subdf[subdf['trial_type'] == 'Incorrect']['mean_error'].values[0]
            ax.plot([base_x, base_x + .4], [y0, y1], color='grey', alpha=0.5)

ax.set_ylabel('Mean decoding error (cm)')
ax.set_xlabel('Position on track (cm)')
ax.get_legend()  # optional: hide legend if redundant
for tick in ax.get_xticklabels():
    tick.set_rotation(20)
# Pairwise comparisons
pairs = [
    (('Pre-reward', 'Correct'), ('Pre-reward', 'Incorrect')),
    (('Post-reward', 'Correct'), ('Post-reward', 'Incorrect')),
    (('Pre-reward', 'Correct'), ('Post-reward', 'Correct')),
]

pvals = []; rs=[]
for (cond1, ttype1), (cond2, ttype2) in pairs:
    vals1 = df[(df['condition'] == cond1) & (df['trial_type'] == ttype1)].sort_values('mouse')['mean_error']
    vals2 = df[(df['condition'] == cond2) & (df['trial_type'] == ttype2)].sort_values('mouse')['mean_error']
    stat, p = wilcoxon_r(vals1.values, vals2.values)
    pvals.append(p)
    rs.append(stat)

# Bonferroni correction
rej, pvals_corrected, _, _ = multipletests(pvals, method='fdr_bh')
# Helper for annotation
def get_bar_height(x1, x2, y, h=1):
    return (x1 + x2) / 2, y + h
# Define plot x positions
label_positions = {
    ('Pre-reward', 'Correct'): -0.2,
    ('Pre-reward', 'Incorrect'): 0.2,
    ('Post-reward', 'Correct'): 0.8,
    ('Post-reward', 'Incorrect'): 1.2,
}
def pval_to_asterisk(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return ''
# Annotate plot with corrected p-values
ymax = df['mean_error'].max()
step = 9  # vertical spacing between annotations
step2 = 3
# for i, ((g1, g2), pval_corr, show) in enumerate(zip(pairs, pvals_corrected, rej)):
#     if not show:
#         continue
#     x1 = label_positions[g1]
#     x2 = label_positions[g2]
#     y = ymax + step * i
#     ax.plot([x1, x1, x2, x2], [y, y+3, y+3, y], lw=1.5, color='black')
#     ax.text((x1+x2)/2, y+step2, f"p = {pval_corr:.3g}", ha='center', va='bottom', fontsize=10)
#     asterisk = pval_to_asterisk(pval_corr)
#     ax.text((x1+x2)/2, y, f"{asterisk}", ha='center', va='bottom', fontsize=46)

ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()

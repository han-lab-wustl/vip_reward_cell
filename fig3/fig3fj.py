"""
plt transition data
"""
#%%
import pickle, os, numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt,scipy
import matplotlib.backends.backend_pdf, matplotlib as mpl, sys
from statsmodels.stats.multitest import multipletests
import matplotlib.patches as mpatches
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
plt.rc('font', size=20)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"

def wilcoxon_r(x, y):
    # x, y are paired arrays (same subjects)
    W, p = scipy.stats.wilcoxon(x, y)
    diffs = x - y
    n = np.count_nonzero(diffs)  # exclude zero diffs
    if n == 0:
        return np.nan, p
    # Normal approximation for Wilcoxon (no ties/zeros correction here)
    mean_W = n * (n + 1) / 4
    sd_W = np.sqrt(n * (n + 1) * (2*n + 1) / 24)
    Z = (W - mean_W) / sd_W
    # Enforce direction from the actual mean difference
    Z = np.sign(np.nanmean(diffs)) * abs(Z)
    r = Z / np.sqrt(n)
    return r, p

df=pd.read_csv(r'C:\Users\Han\Documents\MATLAB\vip_reward_cell\raw_data\fig2j.csv')
fig, ax = plt.subplots(figsize=(5,5))
s = 10

# Stripplot with dodge for side-by-side points
# sns.stripplot(data=df, x='Transition', y='Value', hue='Group',
#               dodge=True, palette=palette, alpha=0.7, s=s, ax=ax)

# Barplot with dodge for side-by-side bars
sns.barplot(data=df[df.Group=='Real'], x='Transition', y='Value',  dodge=True, errorbar='se', fill=False, ax=ax,color='mediumslateblue')
sns.barplot(data=df[df.Group=='Shuffle'], x='Transition', y='Value',  label='shuffle', alpha=0.4, color='grey',err_kws={'color': 'grey'},errorbar=None,ax=ax)

ax.set_ylabel('% Cell transition')
plt.xticks(rotation=25)
# --- Run paired t-tests and annotate ---
# Step 1: Collect p-values for all transitions
p_vals = []
t_stats = []
transitions = df['Transition'].unique()

for trans in transitions:
    real_vals = df[(df['Transition'] == trans) & (df['Condition'] == 'Real')].sort_values('Animal')['Value']
    shuf_vals = df[(df['Transition'] == trans) & (df['Condition'] == 'Shuffle')].sort_values('Animal')['Value']
    t_stat, p_val = wilcoxon_r(real_vals.values, shuf_vals.values)
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
y_max = df['Value'].max()-10
for i, (trans, p_corr) in enumerate(zip(transitions, p_vals_corrected)):
    stars = pval_to_asterisks(p_corr)
    ax.text(i, y_max + y_max*0.05, stars, ha='center', va='bottom', fontsize=46)
    ax.text(i, y_max + y_max*0.05-5, f'({t_stats[i]:.3g},\n{p_corr:.3g})', ha='center', va='bottom', fontsize=8)

ax.spines[['top','right']].set_visible(False)
# ax.set_title('Real vs Shuffle Transitions')
ax.legend()
fig.tight_layout(rect=[0, 0, 0.85, 1])

#%%
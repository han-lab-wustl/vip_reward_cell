"""
get reward distance cells between opto and non opto conditions
oct 2024
mods in june 2025
control vs. opto epoch only
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
plt.rcParams["font.family"] = "Arial"

s=12
# save
realdf=pd.read_csv(r'C:\Users\Han\Documents\MATLAB\vip_reward_cell_copy_w_raw_data\raw_data\cell\fig7d.csv')
# compare between groups
dfbig=realdf.groupby(['animal', 'epoch_dur','condition','opto']).mean(numeric_only=True).reset_index()
dfbig['animals']=dfbig['animal']
dfagg=dfbig[dfbig.epoch_dur=='early']

# number of epochs vs. reward cell prop    
fig,axes = plt.subplots(ncols=2,figsize=(8,3),sharex=True,sharey=True)
ax=axes[0]
# av across mice
pl =['k','slategray']
df_plt =dfagg
sns.barplot(x='condition', y='goal_cell_prop',hue='opto',
        data=df_plt,
        palette=pl,
        fill=False,ax=ax, color='k', errorbar='se',legend=False)
# ----- connecting lines per animal -----
for cond in df_plt['condition'].unique():
    sub = df_plt[df_plt['condition']==cond]
    wide = sub.pivot(index='animals', columns='opto', values='goal_cell_prop').dropna()
    xpos = list(df_plt['condition'].unique()).index(cond)
    for _, row in wide.iterrows():
        ax.plot([xpos-0.2, xpos+0.2], [row[False], row[True]], color='gray', alpha=0.5, lw=1.5)
# ----- paired stats -----
stats_results = {}
for cond in df_plt['condition'].unique():
    sub = df_plt[df_plt['condition']==cond]
    wide = sub.pivot(index='animals', columns='opto', values='goal_cell_prop').dropna()
    if wide.shape[1]==2:
        t,p = scipy.stats.ttest_rel(wide[False], wide[True])
        stats_results[cond] = {'t':t, 'p':p, 'n':len(wide)}
    else:
        stats_results[cond] = None
# ----- annotate p-values -----
for i, cond in enumerate(df_plt['condition'].unique()):
    res = stats_results[cond]
    if res is not None:
        p = res['p']
        ymax = df_plt[df_plt['condition']==cond]['goal_cell_prop'].max()
        # choose stars
        if p < 0.001:
            stars = '***'
        elif p < 0.01:
            stars = '**'
        elif p < 0.05:
            stars = '*'
        else:
            stars = f"ns"
        ax.text(i, ymax*1.15, stars, ha='center', va='bottom', fontsize=14)
        ax.text(i, ymax*1.01, f't={res["t"]:.3g}\np={p:.3g}', ha='center', va='bottom', fontsize=9)

ax.spines[['top','right']].set_visible(False)
new_labels = {'ctrl': 'Control', 'vip': 'VIP\nInhibition', 'vip_ex': 'VIP\nExcitation'}
ax.set_xlabel('')
ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'], rotation=20)
ax.set_ylabel('Reward cell %')
ax.set_title('First 8 trials')

ax=axes[1]
### late
dfagg=dfbig[dfbig.epoch_dur=='late']
df_plt = dfagg
sns.barplot(x='condition', y='goal_cell_prop',hue='opto',
        data=df_plt,
        palette=pl,
        fill=False,ax=ax, color='k', errorbar='se',legend=False)
# ----- connecting lines per animal -----
for cond in df_plt['condition'].unique():
    sub = df_plt[df_plt['condition']==cond]
    wide = sub.pivot(index='animals', columns='opto', values='goal_cell_prop').dropna()
    xpos = list(df_plt['condition'].unique()).index(cond)
    for _, row in wide.iterrows():
        ax.plot([xpos-0.2, xpos+0.2], [row[False], row[True]], color='gray', alpha=0.5, lw=1.5)
# ----- paired stats -----
stats_results = {}
for cond in df_plt['condition'].unique():
    sub = df_plt[df_plt['condition']==cond]
    wide = sub.pivot(index='animals', columns='opto', values='goal_cell_prop').dropna()
    if wide.shape[1]==2:
        t,p = scipy.stats.ttest_rel(wide[False], wide[True])
        stats_results[cond] = {'t':t, 'p':p, 'n':len(wide)}
    else:
        stats_results[cond] = None
# ----- annotate p-values -----
for i, cond in enumerate(df_plt['condition'].unique()):
    res = stats_results[cond]
    if res is not None:
        p = res['p']
        ymax = df_plt[df_plt['condition']==cond]['goal_cell_prop'].max()
        # choose stars
        if p < 0.001:
            stars = '***'
        elif p < 0.01:
            stars = '**'
        elif p < 0.05:
            stars = '*'
        else:
            stars = f"ns"
        ax.text(i, ymax*1.15, stars, ha='center', va='bottom', fontsize=14)
        ax.text(i, ymax*1.01, f't={res["t"]:.3g}\np={p:.3g}', ha='center', va='bottom', fontsize=9)

ax.spines[['top','right']].set_visible(False)
new_labels = {'ctrl': 'Control', 'vip': 'VIP\nInhibition', 'vip_ex': 'VIP\nExcitation'}
ax.set_xlabel('')
ax.set_xticklabels(['Control', 'VIP\nInhibition', 'VIP\nExcitation'], rotation=20)
ax.set_title('Last 8 trials')

# --- Summary stats (mean ± SEM) ---
summary = (
    dfbig.groupby(["condition", "epoch_dur", "opto"])
      .agg(mean_goal=('goal_cell_prop', 'mean'),
           sem_goal=('goal_cell_prop', 'sem'),
           n=('goal_cell_prop', 'count'))
      .reset_index()
)
print("Summary by condition, epoch, and opto:")
print(summary)
#%%
# correlate with behavior

dfbig=realdf.groupby(['animal', 'epoch_dur','condition','opto']).mean(numeric_only=True).reset_index()
dfagg=dfbig[dfbig.epoch_dur=='late']
pl=['slategray','red','darkgoldenrod']
pivoted_avg = dfagg.pivot_table(
    index=['animal', 'condition'],
    columns='opto',
    values='goal_cell_prop'
).reset_index()
pivoted_avg.columns.name = None
pivoted_avg = pivoted_avg.rename(columns={False: 'goal_cell_prop_off', True: 'goal_cell_prop_on'})
pivoted_avg['difference'] = pivoted_avg['goal_cell_prop_on']-pivoted_avg['goal_cell_prop_off']
# pivoted_avg=pivoted_avg[pivoted_avg.cell_type=='early']
beh = pd.read_csv(r'C:\Users\Han\Documents\MATLAB\vip_reward_cell_copy_w_raw_data\raw_data\fig5f.csv')
beh=beh[(beh.animals.isin(realdf.animal.values))&(beh.days.isin(realdf.day.values))]
beh = beh.groupby(['animals', 'opto']).mean(numeric_only=True).reset_index()
beh=beh[beh.opto==True]
# Perform regression
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(beh.rates_diff.values, pivoted_avg.difference.values)
print(f"Correlation (r) = {r_value:.4f}, p-value = {p_value:.3g}")
# Plot scatter plot with regression line
fig,ax=plt.subplots(figsize=(4,4))
sns.scatterplot(x=beh.rates_diff.values, y=pivoted_avg.difference.values,hue=pivoted_avg.condition.values,s=300,alpha=.7,palette=pl,ax=ax)
ax.plot(beh.rates_diff.values, intercept + slope * beh.rates_diff.values, color='steelblue', linewidth=3)
ax.legend(['Control', 'VIP Inhibition', 'VIP Excitation'], fontsize='small')
ax.set_xlabel("$\Delta$ % Correct trials (LEDon-LEDoff)")
ax.set_ylabel("$\Delta$ Reward cell %")
ax.set_title(f"r={r_value:.3g}, p={p_value:.3g}",fontsize=16)
ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('Reward cell vs. performance')
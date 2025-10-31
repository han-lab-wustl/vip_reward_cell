
#%%
"""
% correct and lick traces per epoch
"""
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd, os
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"]=10
mpl.rcParams["ytick.major.size"]=10
plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.opto.behavior.behavior import get_success_failure_trials
from projects.pyr_reward.rewardcell import get_radian_position,performance_by_trialtype
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
#%%
# iterate through all animals
data_df = {}
for ii in range(len(conddf)):
    day = conddf.days.values[ii]
    animal = conddf.animals.values[ii]
    if (animal!='e217') & (conddf.optoep.values[ii]<2):
        if animal=='e145' or animal=='e139': pln=2 
        else: pln=0
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
        tcs_correct,tcs_fail,rates,rz_perm,lick_selectivity=performance_by_trialtype(params_pth, animal,bins=90)
        data_df[f'{animal}_{day:03d}_index{ii:03d}'] = [tcs_correct,tcs_fail,rates,rz_perm,lick_selectivity]
# lick selectivity
# 0 = early
# 1 = late
#%%

# get examples of correct vs. fail
# take the first epoch and first cell?
# v take all cells
# per day per animal
# settings
tcs_correct_all = [xx[0] for k,xx in data_df.items()]
tcs_fail_all = [xx[1] for k,xx in data_df.items()]
plt.rc('font', size=24) 
animals = [xx for ii, xx in enumerate(conddf.animals.values) if (xx != 'e217') & (conddf.optoep.values[ii] < 2)]
animals_test = np.unique(animals)
# animals_test=['z9']
# option to pick 'pre' or 'post' reward activity
activity_window = 'pre'  # options: 'pre' or 'post'

dff_correct_per_an = []
dff_fail_per_an = []

for animal in animals_test:
    dff_correct = []
    dff_fail = []
    tcs_correct = []
    bins = 90

    for ii, tcs_corr in enumerate(tcs_correct_all):
        if animals[ii] == animal and tcs_corr.shape[1] > 0:
            # remove nan epochs
            tc = tcs_corr#[np.nansum(np.isnan(tcs_corr),axis=1)!=tcs_corr.shape[1]]
            tcs_correct.append(tc)
            # choose pre or post reward
            if activity_window == 'pre':
                dff_correct.append(np.quantile(tc[:, int(bins/3):int(bins/2)], .9, axis=1))
            else:
                dff_correct.append(np.quantile(tc[:, int(bins/2):], .9, axis=1))

    tcs_fail = []
    for ii, tcs_f in enumerate(tcs_fail_all):
        if animals[ii] == animal and tcs_f.shape[1] > 0:
            tc = tcs_f#[np.nansum(np.isnan(tcs_corr),axis=1)!=tcs_corr.shape[1]]
            #np.vstack(np.nanmean(tcs_f, axis=0))
            tcs_fail.append(tc)
            if np.sum(np.isnan(tc)) == 0:                
                if activity_window == 'pre':
                    dff_fail.append(np.quantile(tc[:, int(bins/3):int(bins/2)], .9, axis=1))
                else:
                    dff_fail.append(np.quantile(tc[:, int(bins/2):], .9, axis=1))

    dff_correct_per_an.append(dff_correct)
    dff_fail_per_an.append(dff_fail)

    # plotting
    fig, axes = plt.subplots(
        ncols=3, nrows=2, figsize=(10, 12),
        gridspec_kw={'height_ratios': [2, 1], 'width_ratios':[1, 1, 0.05]},
        constrained_layout=True
    )
    axes = axes.flatten()

    # --- Heatmaps
    ax = axes[0]
    tc_org = np.vstack(tcs_correct) 
    tc = tc_org[~(np.nansum(np.isnan(tc_org),axis=1)==tc_org.shape[1])]
    vmin = 0
    vmax = np.nanquantile(tc, 0.999)  # 95th percentile
    peak_bins = np.argmax(tc, axis=1)
    sort_idx = np.argsort(peak_bins)
    im = ax.imshow(tc[sort_idx]**.6, vmin=vmin, vmax=vmax, aspect='auto')
    ax.axvline(bins//2, color='w', linestyle='--')
    ax.set_xticks(np.arange(0, bins, 30))
    ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi+.6, np.pi), 2), rotation=45)
    ax.set_ylabel('Epochs (sorted)')
    ax.set_xlabel('Reward-relative distance ($\Theta$)')
    ax.set_title(f'{animal}\nCorrect Trials')

    try:
        ax = axes[1]
        tc_f = np.vstack(tcs_fail)
        tc_f = tc_f[~(np.nansum(np.isnan(tc_org),axis=1)==tc_org.shape[1])]
        # sort by correct cells
        im2 = ax.imshow(tc_f[sort_idx]**.6, vmin=vmin, vmax=vmax, aspect='auto')
        ax.axvline(bins//2, color='w', linestyle='--')
        ax.set_xticks(np.arange(0, bins, 30))
        ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi+.6, np.pi), 2))
        ax.set_title('Incorrect Trials')
    except Exception as e:
        print(f"No failed trials for {animal}: {e}")

    # --- Colorbar
    cbar_ax = axes[2]
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar_ax.set_ylabel('$\Delta$ F/F', rotation=270, labelpad=15)
    cbar_ax.yaxis.set_label_position('left')
    cbar_ax.yaxis.tick_left()
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([f"{vmin:.2f}", f"{vmax:.2f}"])

    # --- Mean traces
    ax = axes[3]
    m = np.nanmean(np.vstack(tcs_correct), axis=0)
    vmin = 0
    vmax = np.nanmax(m)+np.nanmax(m)/2  # 95th percentile
    sem = scipy.stats.sem(np.vstack(tcs_correct), axis=0, nan_policy='omit')
    ax.plot(m, color='seagreen')
    ax.fill_between(np.arange(m.size), m - sem, m + sem, color='seagreen', alpha=0.5)
    ax.axvline(bins//2, color='k', linestyle='--')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylabel('$\Delta$ F/F')
    ax.set_ylim(vmin, vmax)
    ax.set_title('Correct Mean')

    try:
        ax = axes[4]
        m = np.nanmean(np.vstack(tcs_fail), axis=0)
        sem = scipy.stats.sem(np.vstack(tcs_fail), axis=0, nan_policy='omit')
        ax.plot(m, color='firebrick')
        ax.fill_between(np.arange(m.size), m - sem, m + sem, color='firebrick', alpha=0.5)
        ax.axvline(bins//2, color='k', linestyle='--')
        ax.set_xticks(np.arange(0, bins, 30))
        ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi+.6, np.pi), 2))
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlabel('Reward-relative distance ($\Theta$)')
        ax.set_ylabel('$\Delta$ F/F')
        ax.set_ylim(vmin, vmax)
        ax.set_title('Incorrect Mean')
    except Exception as e:
        print(f"No failed trials mean plot for {animal}: {e}")

    axes[5].axis('off')  # turn off the last unused axis (bottom-right)
    fig.suptitle('Licking behavior') 
#     plt.savefig(os.path.join(savedst, f'{animal}_pre_rew_correctvfail.svg'),bbox_inches='tight')

#%%
# performance average by transition
plt.rc('font', size=20)
rates=[xx[2] for k,xx in data_df.items()]
perms=[list(combinations(range(len(xx[0])), 2)) for k,xx in data_df.items()]
rz_perm=[xx[3] for k,xx in data_df.items()]
rates_perm = [[[rates[i][p[0]],rates[i][p[1]]] for p in perm] for i,perm in enumerate(perms)] 
# get different transition combinations?
perms = [[list(xx) for xx in perm] for perm in perms]
rz_perm = [[list(xx) for xx in perm] for perm in rz_perm]
pairs = np.concatenate(perms)
rz_pairs = np.concatenate(rz_perm)
df = pd.DataFrame(pairs)
df.columns = ['ep_transition_1', 'ep_transition_2']
df_org = conddf.copy()
df_org = df_org[((df_org.animals!='e217')) & (df_org.optoep<2)]

df['animal'] = np.concatenate([[xx]*len(perms[ii]) for ii,xx in enumerate(df_org.animals.values)])
df['performance_transition_1'] = np.concatenate(rates_perm)[:,0]
df['performance_transition_1'] = df['performance_transition_1']*100
df['performance_transition_2'] = np.concatenate(rates_perm)[:,1]
df['performance_transition_2'] = df['performance_transition_2']*100
df['rewzone_transition_1'] = np.concatenate(rz_perm)[:,0]
df['rewzone_transition_2'] = np.concatenate(rz_perm)[:,1]

df = df[(df.ep_transition_1==1) & (df.ep_transition_2==2)]
df = df.groupby(['animal','rewzone_transition_1','rewzone_transition_2']).mean(numeric_only=True)
df = df.reset_index()
df = df[(df.animal!='e189') & (df.animal!='z16')]
s=10
colors = ['k', 'slategray', 'darkcyan', 'darkgoldenrod', 'orchid']
rz_palette = 'Set1_r'#sns.color_palette(colors)  # Set custom palette

fig,ax = plt.subplots(figsize=(5,4))
# sns.stripplot(x='rewzone_transition_2',
#     y='performance_transition_2',hue='rewzone_transition_1',data=df,palette=rz_palette,
#     dodge=True,s=s,alpha=0.7)
sns.barplot(x='rewzone_transition_2',
    y='performance_transition_2',hue='rewzone_transition_1',data=df,palette=rz_palette,
    fill=False)
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(.8, 0.5), 
        title='Previous Reward Zone')
ax.set_ylabel('% Correct trials')
ax.set_xlabel('Reward Zone')
ax.spines[['top','right']].set_visible(False)

# Loop through each reward zone transition (current)
for rz2 in sorted(df['rewzone_transition_2'].unique()):
    df_rz = df[df['rewzone_transition_2'] == rz2]
    # Group performances by previous reward zone
    grouped = df_rz.groupby('rewzone_transition_1')['performance_transition_2'].apply(list)
    # Filter out groups with fewer than 2 entries
    valid_groups = [vals for vals in grouped if len(vals) > 1]
    valid_groups = [[xx for xx in vals if xx!=np.nan] for vals in valid_groups]

    if len(valid_groups) > 1:
        stat, p = scipy.stats.kruskal(*valid_groups)
        print(f"Kruskal-Wallis test for current reward zone {rz2}:")
        print(f"  H-statistic = {stat:.3f}, p = {p:.4f}")
    else:
        print(f"Not enough valid groups to test for reward zone {rz2}.")
# --- Draw connecting lines between reward zone transitions for each animal ---
from collections import defaultdict

# Get unique reward zones to establish consistent plotting positions
rz2_unique = sorted(df['rewzone_transition_2'].unique())
rz1_unique = sorted(df['rewzone_transition_1'].unique())

# Map (x, hue) positions used by seaborn for dodge logic
x_offset_map = defaultdict(dict)
dodge_amount = 0.2
for i, rz2 in enumerate(rz2_unique):
    for j, rz1 in enumerate(rz1_unique):
        x_pos = i - dodge_amount + j * (2 * dodge_amount / (len(rz1_unique)-1))  # calculate dodge position
        x_offset_map[rz2][rz1] = x_pos

# Draw lines per animal
for animal, df_animal in df.groupby('animal'):
    for rz2 in rz2_unique:
        df_sub = df_animal[df_animal['rewzone_transition_2'] == rz2]
        if len(df_sub) < 2:
            continue  # need at least 2 points to draw a line
        df_sub = df_sub.sort_values('rewzone_transition_1')  # sort by hue for line continuity
        x_vals = [x_offset_map[rz2][rz1] for rz1 in df_sub['rewzone_transition_1']]
        y_vals = df_sub['performance_transition_2'].values
        ax.plot(x_vals, y_vals, color='gray', alpha=0.5, linewidth=1.5)

savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper\panels_main_figures'
plt.savefig(os.path.join(savedst, 'rewzone_performance.svg'),bbox_inches='tight')


#%%
# licks selectivity by reward zone
# performance average by transition
rates=[[np.nanmean(yy[1]) for yy in xx[4]] for k,xx in data_df.items()]
perms=[list(combinations(range(len(xx[0])), 2)) for k,xx in data_df.items()]
rz_perm=[xx[3] for k,xx in data_df.items()]
rates_perm = [[[rates[i][p[0]],rates[i][p[1]]] for p in perm] for i,perm in enumerate(perms)] 
# get different transition combinations?
perms = [[list(xx) for xx in perm] for perm in perms]
rz_perm = [[list(xx) for xx in perm] for perm in rz_perm]
pairs = np.concatenate(perms)
rz_pairs = np.concatenate(rz_perm)
df = pd.DataFrame(pairs)
df.columns = ['ep_transition_1', 'ep_transition_2']
df_org = conddf.copy()
df_org = df_org[((df_org.animals!='e217')) & (df_org.optoep<2)]

df['animal'] = np.concatenate([[xx]*len(perms[ii]) for ii,xx in enumerate(df_org.animals.values)])
df['performance_transition_1'] = np.concatenate(rates_perm)[:,0]
df['performance_transition_1'] = df['performance_transition_1']
df['performance_transition_2'] = np.concatenate(rates_perm)[:,1]
df['performance_transition_2'] = df['performance_transition_2']
df['rewzone_transition_1'] = np.concatenate(rz_perm)[:,0]
df['rewzone_transition_2'] = np.concatenate(rz_perm)[:,1]

df = df[(df.ep_transition_1==1) & (df.ep_transition_2==2)]
df = df.groupby(['animal','rewzone_transition_1','rewzone_transition_2']).mean(numeric_only=True)
dflate = df.reset_index()
s=10
colors = ['k', 'slategray', 'darkcyan', 'darkgoldenrod', 'orchid']
rz_palette = 'Set1_r'#sns.color_palette(colors)  # Set custom palette

fig,ax = plt.subplots(figsize=(6,5))
sns.stripplot(x='rewzone_transition_2',
    y='performance_transition_2',hue='rewzone_transition_1',data=dflate,palette=rz_palette,
    dodge=True,s=s,alpha=0.7)
sns.barplot(x='rewzone_transition_2',
    y='performance_transition_2',hue='rewzone_transition_1',data=dflate,palette=rz_palette,
    fill=False)
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.0, 0.5), 
        title='Previous Reward Zone')
ax.set_ylabel('Lick Selectivity (last 5 trials)')
ax.set_xlabel('Reward Zone')
ax.spines[['top','right']].set_visible(False)

# not significant
df_rz1 = dflate[dflate['rewzone_transition_2'] == 1]  # make sure type matches if it's string
import statsmodels.api as sm
from statsmodels.formula.api import ols
# Fit model to see effect of previous reward zone (transition 1) on performance in reward zone 1
model = ols('performance_transition_2 ~ C(rewzone_transition_1) + C(animal)', data=df_rz1).fit()
anova_table = sm.stats.anova_lm(model)
print("ANOVA (Effect of previous reward zone on performance in current reward zone 1):\n")
print(anova_table)
groups = df_rz1.groupby('rewzone_transition_1')
data_by_prev = {k: v.set_index('animal')['performance_transition_2'] for k, v in groups}
plt.savefig(os.path.join(savedst, 'rewzone_lick_select_last5trials.svg'),bbox_inches='tight')
#%%
# licks selectivity by reward zone
# performance average by transition
rates=[[np.nanmean(yy[0]) for yy in xx[4]] for k,xx in data_df.items()]
perms=[list(combinations(range(len(xx[0])), 2)) for k,xx in data_df.items()]
rz_perm=[xx[3] for k,xx in data_df.items()]
rates_perm = [[[rates[i][p[0]],rates[i][p[1]]] for p in perm] for i,perm in enumerate(perms)] 
# get different transition combinations?
perms = [[list(xx) for xx in perm] for perm in perms]
rz_perm = [[list(xx) for xx in perm] for perm in rz_perm]
pairs = np.concatenate(perms)
rz_pairs = np.concatenate(rz_perm)
df = pd.DataFrame(pairs)
df.columns = ['ep_transition_1', 'ep_transition_2']
df_org = conddf.copy()
df_org = df_org[((df_org.animals!='e217')) & (df_org.optoep<2)]

df['animal'] = np.concatenate([[xx]*len(perms[ii]) for ii,xx in enumerate(df_org.animals.values)])
df['performance_transition_1'] = np.concatenate(rates_perm)[:,0]
df['performance_transition_1'] = df['performance_transition_1']
df['performance_transition_2'] = np.concatenate(rates_perm)[:,1]
df['performance_transition_2'] = df['performance_transition_2']
df['rewzone_transition_1'] = np.concatenate(rz_perm)[:,0]
df['rewzone_transition_2'] = np.concatenate(rz_perm)[:,1]

df = df[(df.ep_transition_1==1) & (df.ep_transition_2==2)]
df = df.groupby(['animal','rewzone_transition_1','rewzone_transition_2']).mean(numeric_only=True)
df = df.reset_index()

s=10
colors = ['k', 'slategray', 'darkcyan', 'darkgoldenrod', 'orchid']
rz_palette = 'Set1_r'#sns.color_palette(colors)  # Set custom palette

fig,ax = plt.subplots(figsize=(6,5))
sns.stripplot(x='rewzone_transition_2',
    y='performance_transition_2',hue='rewzone_transition_1',data=df,palette=rz_palette,
    dodge=True,s=s,alpha=0.7)
sns.barplot(x='rewzone_transition_2',
    y='performance_transition_2',hue='rewzone_transition_1',data=df,palette=rz_palette,
    fill=False)
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.0, 0.5), 
        title='Previous Reward Zone')
ax.set_ylabel('Lick Selectivity (early 5 trials)')
ax.set_xlabel('Reward Zone')
ax.spines[['top','right']].set_visible(False)

# not significant
df_rz1 = df[df['rewzone_transition_2'] == 1]  # make sure type matches if it's string
import statsmodels.api as sm
from statsmodels.formula.api import ols
# Fit model to see effect of previous reward zone (transition 1) on performance in reward zone 1
model = ols('performance_transition_2 ~ C(rewzone_transition_1) + C(animal)', data=df_rz1).fit()
anova_table = sm.stats.anova_lm(model)
print("ANOVA (Effect of previous reward zone on performance in current reward zone 1):\n")
print(anova_table)
groups = df_rz1.groupby('rewzone_transition_1')
data_by_prev = {k: v.set_index('animal')['performance_transition_2'] for k, v in groups}
plt.savefig(os.path.join(savedst, 'rewzone_lick_select_early5trials.svg'),bbox_inches='tight')

#%%
# recalculate tc
animals_unique = np.unique(animals)
df=pd.DataFrame()
correct = np.concatenate([np.concatenate(xx) for xx in dff_correct_per_an])
incorrect = np.concatenate([np.concatenate(xx) for xx in dff_fail_per_an])
df['mean_dff'] = np.concatenate([correct,incorrect])
df['trial_type']=np.concatenate([['correct']*len(correct),['incorrect']*len(incorrect)])
ancorr = np.concatenate([[animals_unique[ii]]*len(np.concatenate(xx)) for ii,xx in enumerate(dff_correct_per_an)])
anincorr = np.concatenate([[animals_unique[ii]]*len(np.concatenate(xx)) for ii,xx in enumerate(dff_fail_per_an)])
df['animal'] = np.concatenate([ancorr, anincorr])
bigdf=df
# average
bigdf=bigdf.groupby(['animal', 'trial_type']).mean(numeric_only=True)
bigdf=bigdf.reset_index()
s=13
fig,ax = plt.subplots(figsize=(2,5))
sns.stripplot(x='trial_type', y='mean_dff', data=bigdf,hue='trial_type',
        dodge=True,palette={'correct':'seagreen', 'incorrect': 'firebrick'},
        s=s,alpha=0.7)
sns.barplot(x='trial_type', y='mean_dff', data=bigdf,hue='trial_type',
        fill=False,palette={'correct':'seagreen', 'incorrect': 'firebrick'})

ax.spines[['top','right']].set_visible(False)
ax.set_ylabel('Post-reward mean tuning curve ($\Delta F/F$)')
ax.set_xlabel('Trial type')
cor = bigdf.loc[(bigdf.trial_type=='correct'), 'mean_dff']
incor = bigdf.loc[(bigdf.trial_type=='incorrect'), 'mean_dff']
t,pval = scipy.stats.wilcoxon(cor,incor)
ans = bigdf.animal.unique()
for i in range(len(ans)):
    ax = sns.lineplot(x='trial_type', y='mean_dff', 
    data=bigdf[bigdf.animal==ans[i]],
    errorbar=None, color='dimgray', linewidth=2, alpha=0.7,ax=ax)

# statistical annotation       
ii=0.5
y=.2
pshift=y/7
fs=30
if pval < 0.001:
        plt.text(ii, y, "***", ha='center', fontsize=fs)
elif pval < 0.01:
        plt.text(ii, y, "**", ha='center', fontsize=fs)
elif pval < 0.05:
        plt.text(ii, y, "*", ha='center', fontsize=fs)
ax.text(ii, y+pshift, f'p={pval:.2g}',rotation=45,fontsize=12)

ax.set_title('Licking behavior',pad=50)
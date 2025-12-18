"""
zahra
dec 2025
get trial by trial heatmap of rew cells
1) get cells near rew
2) in ctrl v. opto, see how many move with ep 2
3) how many stay or get decorr?
4) degree of backward vs forward shifting BTSP
"""
#%%
import numpy as np, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
plt.rc('font', size=12)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap, BoundaryNorm

df=pd.read_csv(r'C:\Users\Han\Documents\MATLAB\vip_reward_cell_copy_w_raw_data\raw_data\cell\sfig7.csv')

# df=df[abs(df.com_shift)<100]
# df=df[abs(df.rewshift)<100]
color=['slategray','red','darkgoldenrod']

df['condition'] = [xx if 'vip' in xx else 'ctrl' for xx in df.in_type]
# remove outlier
df=df[~((df.condition=='vip_ex') & (df.dir=='forward') & (df.cell_type=='goal') & (df.com_shift<-20))]
conds = ['ctrl','vip','vip_ex']
from statsmodels.stats.multitest import multipletests
# plot hist
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8,4))
fig.subplots_adjust(wspace=0.05)
# LEFT PANEL: x ≤ –50
ax1.hist(df[df.condition=='ctrl']['rewshift'], bins=30, density=True, alpha=0.3, color=color[0])
ax1.hist(df[df.condition=='vip']['rewshift'], bins=30, density=True, alpha=0.3, color=color[1])
ax1.hist(df[df.condition=='vip_ex']['rewshift'], bins=30, density=True, alpha=0.3, color=color[2])
ax1.set_xlim([-120, -50])
# RIGHT PANEL: x ≥ 50
ax2.hist(df[df.condition=='ctrl']['rewshift'], bins=30, density=True, alpha=0.3, color=color[0],label='Control')
ax2.hist(df[df.condition=='vip']['rewshift'], bins=30, density=True, alpha=0.3, color=color[1],label='VIP Inhibition')
ax2.hist(df[df.condition=='vip_ex']['rewshift'], bins=30, density=True, alpha=0.3, color=color[2],label='VIP Excitation')
ax2.set_xlim([50, 100])
# diagonal break marks
d = .015
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((1-d, 1+d), (-d, +d), **kwargs)
ax1.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
kwargs = dict(transform=ax2.transAxes, color='k', clip_on=False)
ax2.plot((-d, +d), (-d, +d), **kwargs)
ax2.plot((-d, +d), (1-d, 1+d), **kwargs)
ax2.legend()
fig.suptitle('Reward shift distributions')
ax1.set_ylabel('Session probability')
ax1.set_xlabel('Reward shift (epoch 2-epoch 1, cm)')
plt.tight_layout()


#%%
nms=['Control','VIP Inhibition','VIP Excitation']
conds = ['ctrl','vip','vip_ex']
color=['slategray','red','darkgoldenrod']
hue_order=['place','goal','other']
directions = ['backward','forward']
cell_types = ['place', 'goal','other']

fig,axes=plt.subplots(figsize=(10,6),ncols=3,sharex=True,sharey=True)
# df=df[abs(df.com_shift)<100]
# df=df[abs(df.rewshift)<100]

for c,con in enumerate(conds):
    ax = axes[c]
    dfcon = df[df.condition == con]
   #  dfcon=dfcon[abs(dfcon.rewshift)<90]
    # ----- Plot -----
    order=['backward','forward']
    legend = (c==0)

    sns.violinplot(
        x='com_shift', y='dir', hue='cell_type',
        data=dfcon, fill=False, order=order,
        color=color[c], hue_order=hue_order,
        ax=ax, legend=legend
    )

    ax.set_title(nms[c])
    ax.axvline(0,color='k',linestyle='--')
    ax.set_xlabel('COM shift (cm)')
    ax.set_ylabel('Reward location shift')
    ax.spines[['top','right']].set_visible(False)
    ax.set_yticklabels(['Backward','Forward'])

    # ----------------------------------------------------
    # ---------- Collect stats for correction ------------
    # ----------------------------------------------------
    test_info = []    # will store (cell_type, direction, t, p_raw, n)

    for dirs in directions:
        for ct in cell_types:
            data = dfcon[(dfcon.cell_type==ct) & (dfcon.dir==dirs)]['com_shift']
            n = len(data)

            if n > 1:
                tval, pval = scipy.stats.ttest_1samp(data, 0,nan_policy='omit')
            else:
                tval, pval = np.nan, np.nan

            test_info.append((ct, dirs, tval, pval, n))

    # extract vector of raw p-values (ignoring NaN)
    raw_pvals = np.array([ti[3] for ti in test_info])
    raw_mask = ~np.isnan(raw_pvals)

    # FDR correction on valid p-values
    corrected = np.full_like(raw_pvals, np.nan)
    if raw_mask.sum() > 0:
        _, pvals_fdr, _, _ = multipletests(raw_pvals[raw_mask], method='fdr_bh')
        corrected[raw_mask] = pvals_fdr

    # ----------------------------------------------------
    # ----------- Annotate corrected p-values ------------
    # ----------------------------------------------------
    # y positions: offset backward vs forward
    for d, dirs in enumerate(directions):
        y_positions = (np.arange(len(cell_types)) * 0.1) + d

        for yp, ct in zip(y_positions, cell_types):
            # find the test entry
            idx = [i for i,ti in enumerate(test_info) if ti[0]==ct and ti[1]==dirs][0]
            tval, p_raw, p_corr, n = test_info[idx][2], test_info[idx][3], corrected[idx], test_info[idx][4]

            msg = f"t={tval:.2g}, p={p_corr:.3g}, n={n}"
            ax.text(
                0.95, yp, msg,
                va='center', ha='right',
                fontsize=8,
                transform=ax.get_yaxis_transform()
            )

fig.suptitle('Last 3 trials')
plt.tight_layout()

#%% 
# compare to control
conds = ['ctrl','vip','vip_ex']
color=['slategray','red','darkgoldenrod']
# df=df[abs(df.rewshift)<90]
dfcon=df[df.cell_type=='goal']
# dfcon=dfcon.groupby(['animals','days','dir','cell_type']).mean(numeric_only=True)
order=['backward','forward']
fig,ax=plt.subplots(figsize=(3.5,4))
sns.violinplot(x='com_shift',y='dir',hue='condition',data=dfcon,fill=False,order=order,hue_order=conds,legend=True,ax=ax,palette=color)
ax.axvline(0,color='k',linestyle='--')
ax.set_xlabel('COM shift (cm)')
ax.set_ylabel('Reward location shift')
ax.spines[['top', 'right']].set_visible(False)
ax.set_yticklabels(['Backward','Forward'])
ax.set_title("Cells near reward (in epoch 1)")

#%%

pairs = [('ctrl','vip'),('ctrl','vip_ex')]
cllty=['goal']
direc=['backward','forward']
for p1,p2 in pairs:
   for cllt in cllty:
      # compare conds
      dfcon=df
      x1=dfcon[((dfcon.condition==p1) & (dfcon.cell_type==cllt) & (dfcon.dir=='forward'))].com_shift.values
      x2=dfcon[((dfcon.condition==p2) & (dfcon.cell_type==cllt) & (dfcon.dir=='forward'))].com_shift.values
      t,p=scipy.stats.ranksums(x1,x2,nan_policy='omit')
      print(f'{p1},{p2},{cllt},{t:3g},{p:.3g},{len(x1[~np.isnan(x1)])},{len(x2[~np.isnan(x2)])}')
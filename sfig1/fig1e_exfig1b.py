"""
lick selectivity across trials
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random
import matplotlib.patches as patches
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"

src = r'C:\Users\Han\Documents\MATLAB\vip_reward_cell\raw_data\fig1e.p'
with open(src, "rb") as fp: #unpickle
   dct = pickle.load(fp)
# plot
plt.rc('font', size=16)          # controls default text sizes
lick_tcs_lst = dct['lick_selectivity']
ans = dct['animals']

# ex fig 1b
# max 21 correct trials
fig,axes=plt.subplots(ncols=4,figsize=(10,3.5),sharex=True,sharey=True)
axes=axes.flatten()
for i in range(4):
    ax=axes[i]
    lick_tcs_ep_=[v[i] for v in lick_tcs_lst if len(v)>i]
    ans_ep = np.array([ans[jj] for jj,v in enumerate(lick_tcs_lst) if len(v)>i])
    ntrials=np.nanmax([len(xx) for xx in lick_tcs_ep_])
    lick_tcs_ep=np.ones((len(lick_tcs_ep_),ntrials))*np.nan
    for ll,ls in enumerate(lick_tcs_ep_):
        lick_tcs_ep[ll,:len(ls)]=ls
    for an in np.unique(ans_ep):
        lick_tcs_ep_an = lick_tcs_ep[ans_ep==an]
        ax.plot(np.nanmean(lick_tcs_ep_an,axis=0),label=an,alpha=0.5)
        # m=np.nanmean(lick_tcs_ep_an,axis=0)
        # ax.fill_between(range(0,ntrials), 
        #     m-scipy.stats.sem(lick_tcs_ep_an,axis=0,nan_policy='omit'),
        #     m+scipy.stats.sem(lick_tcs_ep_an,axis=0,nan_policy='omit'), alpha=0.2)
    ax.plot(np.nanmean(lick_tcs_ep,axis=0),color='k',label='Mean')

    ax.set_xticks([0,int((ntrials-1)/2),ntrials-1])
    ax.set_xticklabels([1,int(ntrials/2),ntrials])
    ax.spines[['top','right']].set_visible(False)    
    ax.axhline(0.8,color='grey',linestyle='--')
    ax.axvline(13,color='b',linewidth=3,label='Last 8 trials')
    if i==0: 
        ax.set_ylabel('Lick selectivity')
    if i==3: ax.set_xlabel('# correct trials across epoch')
    ax.set_title(f'Epoch {i+1}')
ax.legend(fontsize=10)
plt.tight_layout()

#%%
# fig 1
colors=['k','slategray','darkcyan','darkgoldenrod']
# max 21 correct trials
lick_tcs_lst = dct['lick_selectivity']

fig,ax=plt.subplots(figsize=(4,3.5),sharex=True,sharey=True)
for i in range(4):
    lick_tcs_ep_=[v[i] for v in lick_tcs_lst if len(v)>i]
    ans_ep = np.array([ans[jj] for jj,v in enumerate(lick_tcs_lst) if len(v)>i])
    ntrials=np.nanmax([len(xx) for xx in lick_tcs_ep_])
    # get trial num
    trials_len=np.array([len(xx) for xx in lick_tcs_ep_])
    lick_tcs_ep=np.ones((len(lick_tcs_ep_),ntrials))*np.nan
    for ll,ls in enumerate(lick_tcs_ep_):
        lick_tcs_ep[ll,:len(ls)]=ls
    lick_tcs_ep=lick_tcs_ep[trials_len>9]
    ax.plot(np.nanmean(lick_tcs_ep,axis=0),color=colors[i],label=f'Epoch {i+1}, {lick_tcs_ep.shape[0]} sessions')
    sem =scipy.stats.sem(lick_tcs_ep,axis=0,nan_policy='omit')
    m=np.nanmean(lick_tcs_ep,axis=0)
    ax.fill_between(range(len(m)),m-sem,m+sem,alpha=0.2,color=colors[i])
    ax.set_xticks([0,int((ntrials-1)/2),ntrials-1])
    ax.set_xticklabels([1,int(ntrials/2),ntrials])
    ax.spines[['top','right']].set_visible(False)    
    # ax.axhline(0.8,color='grey',linestyle='--')
    ax.set_xlim([-1,20])
ax.axvline(13,color='b',linewidth=3,label='Last 8 trials')
ax.set_ylabel('Lick selectivity')
ax.set_xlabel('# correct trials across epoch')
ax.legend(fontsize=10)
plt.tight_layout()


"""
zahra
july 2024
quantify reward-relative cells
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
# plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from placecell import make_tuning_curves_radians_by_trialtype, intersect_arrays
from rewardcell import get_radian_position,create_mask_from_coordinates,pairwise_distances,extract_data_rewcentric
from projects.opto.behavior.behavior import get_success_failure_trials
from projects.pyr_reward.circular import get_circular_data
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
savepth = os.path.join(savedst, 'circular_stats.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_rewardcentric_all.p"
with open(saveddataset, "rb") as fp: #unpickle
        radian_alignment_saved = pickle.load(fp)
#%%
# initialize var
# radian_alignment_saved = {} # overwrite
goal_cell_iind = []
goal_cell_prop = []
goal_cell_null = []
dist_to_rew = [] # per epoch
num_epochs = []
pvals = []
rates_all = []
total_cells = []
epoch_perm = []
meanangles_all = []
rvals_all = []
radian_alignment = {}
goal_cm_window = 20
lasttr=8 # last trials
bins=90
coms_mean_rewrel = []
coms_mean_abs = []
coms_mean_warp = []
tcs = []
# cm_window = [10,20,30,40,50,60,70,80] # cm
# iterate through all animals
for ii in range(len(conddf)):
        day = conddf.days.values[ii]
        animal = conddf.animals.values[ii]
        if (animal!='e217') & (conddf.optoep.values[ii]<2):
                if animal=='e145' or animal=='e139': pln=2 
                else: pln=0
                params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
                meanangles_abs,rvals_abs,meanangles_rad,rvals_rad,meanangles_warped,rvals_warped,\
                tc_mean,com_mean_rewrel,tcs_abs_mean,com_abs_mean,tcs_warped_mean,com_warped_mean,\
                        tcs_correct,tcs_correct_abs=get_circular_data(ii,params_pth,animal,day,bins,radian_alignment,
                        radian_alignment_saved,goal_cm_window,pdf,epoch_perm,goal_cell_iind,goal_cell_prop,num_epochs,
                        goal_cell_null,pvals,total_cells,
                        num_iterations=1000)
                meanangles_all.append([meanangles_abs,meanangles_rad,meanangles_warped])
                rvals_all.append([rvals_abs,rvals_rad,rvals_warped])
                coms_mean_rewrel.append(com_mean_rewrel)
                coms_mean_abs.append(com_abs_mean)
                coms_mean_warp.append(com_warped_mean)
                tcs.append([tcs_correct,tcs_correct_abs])

pdf.close()
#%%
# plot r val distributions as a function of reward relative distance vs. track distance
dfc = conddf.copy()
dfc = dfc[((dfc.animals!='e217')) & (dfc.optoep<2)]

# fix com mean
coms_mean_rewrel = np.concatenate([xx[0] for xx in coms_mean_rewrel])
coms_mean_abs = np.concatenate([xx[0] for xx in coms_mean_abs])

plt.hist(coms_mean_rewrel)
plt.hist(coms_mean_abs)
#%%
df = pd.DataFrame()
df['com_mean_rewardrel'] = np.concatenate(coms_mean_rewrel) # add pi to align with place map?
df['com_mean_abs'] = np.concatenate(coms_mean_abs)
df['com_mean_warp'] = np.concatenate(coms_mean_warp)
df['meanangles_abs'] = np.concatenate([xx[0] for xx in meanangles_all])
df['meanangles_rewardrel'] = np.concatenate([xx[1] for xx in meanangles_all])
df['meanangles_warp'] = np.concatenate([xx[2] for xx in meanangles_all])
# df['tc_abs'] = [xx[0] for xx in tcs]
# df['tc_rewrel'] = [xx[1] for xx in tcs]

df['rval_abs'] = np.concatenate([xx[0] for xx in rvals_all])
df['rval_rewardrel'] = np.concatenate([xx[1] for xx in rvals_all])
df['rval_warp'] = np.concatenate([xx[2] for xx in rvals_all])

df['animal'] = np.concatenate([[xx]*len(coms_mean_rewrel[ii]) for ii,xx in enumerate(dfc.animals.values)])
df['days'] = np.concatenate([[xx]*len(coms_mean_rewrel[ii]) for ii,xx in enumerate(dfc.days.values)])

df.to_csv(r'C:\Users\Han\Desktop\circular_stats_active_all_ep_thres02.csv')
#%%
# df = pd.read_csv(r'C:\Users\Han\Desktop\circular_stats_active_all_ep_thres02.csv')

# Create a 2D density plot
fig, ax = plt.subplots(figsize=(6,5))
sns.kdeplot(x='com_mean_rewardrel', y='rval_rewardrel', data=df, cmap="Purples", 
        fill=True, thresh=0)
ax.axvline(0, color='k', linestyle='--')
ax.axvline(-np.pi/4, color='r', linestyle='--')
ax.axvline(np.pi/4, color='r', linestyle='--')
ax.set_xlabel("Circular reward-relative distance")
ax.set_ylabel("r value")
ax.set_title(f"Circular reward-relative map")
plt.show()

fig, ax = plt.subplots(figsize=(6,5))
sns.kdeplot(x='com_mean_abs', y='rval_abs', data=df,cmap="Blues", fill=True, thresh=0)
ax.axvline(0, color='k', linestyle='--')
ax.axvline(270, color='k', linestyle='--')
plt.xlabel("Allocentric distance")
ax.set_ylabel("r value")
plt.title(f"Place map")
plt.show()

fig, ax = plt.subplots(figsize=(6,5))
sns.kdeplot(x='com_mean_warp', y='rval_warp', data=df,cmap="RdPu", fill=True, thresh=0)
ax.axvline(0, color='k', linestyle='--')
plt.xlabel("Reward-warped distance")
ax.set_ylabel("r value")
plt.title(f"Reward-warped map")
plt.show()

#%%
# even out sessions

animals=df.animal.unique()
session_nums = [[ii for ii,xx in enumerate(df.loc[df.animal==animal,'days'].unique())] for animal in animals]
days_unique =  [[xx for ii,xx in enumerate(df.loc[df.animal==animal,'days'].unique())] for animal in animals]
session_all = []
for kk,animal in enumerate(animals):
        session_an = []
        for ii,xx in enumerate(df.loc[df.animal==animal, 'days'].values):
                ind = np.where(xx==days_unique[kk])[0][0]
                session_an.append(session_nums[kk][ind])
        session_all.append(session_an) 
df['session_num']=np.concatenate(session_all)

df = df[df.session_num<10]
cmap=sns.color_palette()

for animal in df.animal.unique():
        # Create a 2D density plot
        fig, ax = plt.subplots(figsize=(6,5))
        sns.scatterplot(x='com_mean_rewardrel', y='rval_rewardrel', hue='days', data=df[df.animal==animal],alpha=0.5)
        ax.axvline(0, color='k', linestyle='--')
        ax.set_xlabel("Reward-relative distance")
        ax.set_ylabel("r value")
        ax.set_title(f"{animal}, Reward-relative map")
        plt.show()

        fig, ax = plt.subplots(figsize=(6,5))
        sns.scatterplot(x='com_mean_abs', y='rval_abs', hue='days',  data=df[df.animal==animal],alpha=.5)
        ax.axvline(0, color='k', linestyle='--')
        ax.axvline(270, color='k', linestyle='--')
        plt.xlabel("Allocentric distance")
        ax.set_ylabel("r value")
        plt.title(f"{animal}, Place map")
        plt.show()
        
#%%
# r value - reward vs. place
# sns.scatterplot(x='rval_abs', y='rval_rewardrel', data=df,alpha=0.1)

fig, ax = plt.subplots(figsize=(6,5))
sns.kdeplot(x='rval_rewardrel', y='rval_abs', data=df,cmap='Blues',fill=True, thresh=0)
ax.set_ylabel('r value, Reward-centric')
ax.set_xlabel('r value, Place-centric')
animals=df.animal.unique()

fig, ax = plt.subplots(figsize=(6,5))
sns.scatterplot(x='rval_rewardrel', y='rval_abs', data=df[(df.animal!='e216') & (df.animal!='e218')],
        alpha=0.2)
ax.set_ylabel('r value, Reward-centric')
ax.set_xlabel('r value, Place-centric')
animals=df.animal.unique()

# reward vs. place specific cell coms
fig, ax = plt.subplots(figsize=(6,5))
dfplt = df[(df. animal!='e216') & (df.animal!='e218') & (df.rval_abs<0.5)]
sns.scatterplot(x='rval_rewardrel', y='com_mean_rewardrel', data=dfplt,
        alpha=0.2)
ax.set_ylabel('r value, Reward-centric')
ax.set_xlabel('r value, Place-centric')
#%%
# get tuning curves of high reward, low place cells and vice versa
epochs = 4
fig1, axes1 = plt.subplots(nrows=1, ncols=epochs, sharex=True)
fig2, axes2 = plt.subplots(nrows=1, ncols=epochs, sharex=True)
fig3, axes3 = plt.subplots(nrows=1, ncols=epochs, sharex=True)
for ep in range(epochs):
        try:
                tc_rewc = np.vstack([(xx[0][ep,:,:]) for xx in tcs if len(xx[0])>=(ep+1)])
                tc_abs = np.vstack([(xx[1][ep,:,:]) for xx in tcs if len(xx[1])>=(ep+1)])
                r_abs = np.concatenate([xx[0] for ii,xx in enumerate(rvals_all) if len(tcs[ii][0])>=(ep+1)])
                r_rew = np.concatenate([xx[1] for ii,xx in enumerate(rvals_all) if len(tcs[ii][0])>=(ep+1)])
                ind = np.where((r_rew>0.9) & (r_abs<0.6))[0]
                tcs_rew = tc_rewc[ind]

                pc_ind = np.where((r_rew<0.6) & (r_abs>0.9))[0]
                tcs_pc = tc_abs[pc_ind]

                # in_btw_ind = np.where((df.rval_abs.values>0.7)
                #                 & (df.rval_rewardrel.values>0.4) & (df.rval_rewardrel.values<0.7))[0]
                # tcs_inbtw = tc_abs[in_btw_ind]
                highplacehighrew = np.where((r_abs>0.8)& (r_rew>0.9))[0]
                tcs_highplacehighrew = tc_abs[highplacehighrew]
                com_abs = np.concatenate([xx for ii,xx in enumerate(coms_mean_abs) if len(tcs[ii][0])>=(ep+1)])
                com_r=np.concatenate([xx for ii,xx in enumerate(coms_mean_rewrel) if len(tcs[ii][0])>=(ep+1)])
                axes1[ep].imshow(tcs_pc[np.argsort(com_abs[pc_ind])])
                # plt.plot(np.nanmean(tcs_pc,axis=0))
                axes2[ep].imshow(tcs_rew[np.argsort(com_r[ind])])
                # plt.plot(np.nanmean(tcs_rew,axis=0))
                # plt.imshow(tcs_inbtw[np.argsort(df.com_mean_abs.values[in_btw_ind])])
                # plt.show()
                # plt.plot(np.nanmean(tcs_inbtw,axis=0))
                axes3[ep].imshow(tcs_highplacehighrew[np.argsort(com_abs[highplacehighrew])])
                # plt.plot(np.nanmean(tcs_highplacehighrew,axis=0))
        except Exception as e:
                print(e)
fig1.suptitle('High place, low reward')
fig2.suptitle('High reward, low place')
fig3.suptitle('High place, high reward')

#%%
animals=df.animal.unique()
# animals=['e186']
# for animal in animals:  
#         fig, ax = plt.subplots(figsize=(6,5))     
#         sns.scatterplot(x='rval_abs', y='rval_rewardrel', 
#                         data=df[df.animal==animal],alpha=0.1)
#         # sns.kdeplot(x='rval_abs', y='rval_rewardrel', data=df[df.animal==animal],cmap='Blues',fill=True, thresh=0)
#         ax.set_ylabel('r value, Reward-centric')
#         ax.set_xlabel('r value, Place-centric')
#         ax.set_title(f"{animal}")
#         plt.show()

#%%
# Create a scatterplot
track_length=270
# animals=['e200', 'e189']
# for animal in animals:  
#         # sns.scatterplot(x='rval_rewardrel', y='rval_abs', hue='days',data=df[df.animal==animal], 
#         #         alpha=.5)
#         # # ax.axvline(0, color='k', linestyle='--')
#         # ax.set_ylabel('r value, Reward-centric')
#         # ax.set_xlabel('r value, Place-centric')
#         # ax.set_title(f"{animal}")
#         # plt.show()
#         # rvals per day
#         fig, ax = plt.subplots(figsize=(6,5))       
#         sns.stripplot(x='session_num', y='rval_rewardrel', hue='session_num',data=df[df.animal==animal], 
#                 alpha=.5)
#         sns.boxplot(x='session_num', y='rval_rewardrel',data=df[df.animal==animal],fill=False)

#         # ax.axvline(0, color='k', linestyle='--')
#         ax.set_ylabel('r value, Reward-centric')
#         ax.set_xlabel('Recording day')
#         ax.set_title(f"{animal}")
#         plt.show()
df = df[(df.animal!='e189')&(df.animal!='e190')&(df.animal!='e200')&(df.animal!='e139')&(df.animal!='e186')]
fig, ax = plt.subplots(figsize=(14,5))       
sns.stripplot(x='session_num', y='rval_abs', hue='animal',data=df, 
        alpha=.5,dodge=True)
# sns.barplot(x='session_num', y='rval_rewardrel',hue='animal',data=df,fill=False)
sns.lineplot(x='session_num', y='rval_abs',hue='animal',data=df)
# ax.axvline(0, color='k', linestyle='--')
ax.set_ylabel('r value, Reward-centric')
ax.set_xlabel('Recording day')
ax.set_xlim([-0.5,4.5])
plt.show()

#%%
# separate by com  (near vs. far reward)
# sns.barplot(x='session_num', y='rval_rewardrel',hue='animal',data=df,fill=False)
# fig, ax = plt.subplots(figsize=(14,5))       
bigplt = df[(abs(df.com_mean_rewardrel)<(np.pi/4)) & (df.session_num<5)]
bigplt = df[(abs(df.com_mean_rewardrel)<(np.pi/4))]
# bigplt = df[((df.com_mean_rewardrel)<(np.pi/4)) & (df.com_mean_rewardrel>0) & (df.session_num<5)]
sns.lmplot(x='session_num', y='rval_rewardrel',hue='animal',x_jitter=.5,
        data=bigplt,scatter_kws={'alpha':.2},
        height=5, aspect=2)
for ii,animal in enumerate(bigplt.animal.unique()):
        testdf=bigplt[bigplt.animal==animal]
        r, p = scipy.stats.pearsonr(testdf['rval_rewardrel'], testdf['session_num'])
        print(f'{animal}, r: {r:.3f}, p-value: {p:.10f}\n')
# ax.axvline(0, color='k', linestyle='--')
ax.set_ylabel('r value, Reward-centric')
ax.set_xlabel('Recording day')
# ax.set_xlim([-0.5,4.5])
plt.show()

#%%
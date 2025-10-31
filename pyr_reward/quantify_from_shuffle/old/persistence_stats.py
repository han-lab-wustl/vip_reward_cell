"""
persistence of goal cells vs. stats
dec 2024
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
from rewardcell import get_radian_position,create_mask_from_coordinates,pairwise_distances
from projects.opto.behavior.behavior import get_success_failure_trials
from projects.memory.dopamine import get_rewzones
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
savepth = os.path.join(savedst, 'persistence.pdf')
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
radian_alignment = {}
cm_window = 20
relative_coms = []
relative_coms_shufs=[]
#%%
# cm_window = [10,20,30,40,50,60,70,80] # cm
# iterate through all animals
for ii in range(len(conddf)):
    day = conddf.days.values[ii]
    animal = conddf.animals.values[ii]
    if (animal!='e217') & (conddf.optoep.values[ii]<2):
        if animal=='e145' or animal=='e139': pln=2 
        else: pln=0
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
        print(params_pth)
        fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
            'pyr_tc_s2p_cellind', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
            'stat'])
        VR = fall['VR'][0][0][()]
        scalingf = VR['scalingFACTOR'][0][0]
        try:
                rewsize = VR['settings']['rewardZone'][0][0][0][0]/scalingf        
        except:
                rewsize = 10
        ybinned = fall['ybinned'][0]/scalingf
        track_length=180/scalingf    
        forwardvel = fall['forwardvel'][0]    
        changeRewLoc = np.hstack(fall['changeRewLoc'])
        trialnum=fall['trialnum'][0]
        rewards = fall['rewards'][0]
        if animal=='e145':
                ybinned=ybinned[:-1]
                forwardvel=forwardvel[:-1]
                changeRewLoc=changeRewLoc[:-1]
                trialnum=trialnum[:-1]
                rewards=rewards[:-1]
        # set vars
        eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
        lasttr=8 # last trials
        bins=90
        rad = get_radian_position(eps,ybinned,rewlocs,track_length,rewsize) # get radian coordinates
        track_length_rad = track_length*(2*np.pi/track_length)
        bin_size=track_length_rad/bins 
        rz = get_rewzones(rewlocs,1/scalingf)       
        # get average success rate
        rates = []
        for ep in range(len(eps)-1):
                eprng = range(eps[ep],eps[ep+1])
                success, fail, str_trials, ftr_trials, ttr, \
                total_trials = get_success_failure_trials(trialnum[eprng], rewards[eprng])
                rates.append(success/total_trials)
        rates_all.append(np.nanmean(np.array(rates)))
        
        # added to get anatomical info
        # takes time
        fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
        Fc3 = fall_fc3['Fc3']
        dFF = fall_fc3['dFF']
        if 'bordercells' not in fall.keys(): # if no border cell var
                fall['bordercells'] = [np.zeros_like(fall['iscell'][:,0]).astype(bool)]
        Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
        dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
        skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
        # skew_filter = skew[((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
        # skew_mask = skew_filter>2
        Fc3 = Fc3[:, skew>2] # only keep cells with skew greateer than 2
        if f'{animal}_{day:03d}_index{ii:03d}' in radian_alignment_saved.keys():
            tcs_correct, coms_correct, tcs_fail, coms_fail, \
            com_goal, goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av = radian_alignment_saved[f'{animal}_{day:03d}_index{ii:03d}']            
        else:# remake tuning curves relative to reward        
            # 9/19/24
            # find correct trials within each epoch!!!!
            tcs_correct, coms_correct, tcs_fail, coms_fail = make_tuning_curves_radians_by_trialtype(eps,rewlocs,ybinned,rad,Fc3,trialnum,
                rewards,forwardvel,rewsize,bin_size)          
        fall_stat = scipy.io.loadmat(params_pth, variable_names=['stat','ops'])
        ops = fall_stat['ops']
        stat = fall_stat['stat']
        meanimg=np.squeeze(ops)[()]['meanImg']
        s2p_iind = np.arange(stat.shape[1])
        s2p_iind_filter = s2p_iind[((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
        s2p_iind_filter = s2p_iind_filter[skew>2]
        goal_window = cm_window*(2*np.pi/track_length) # cm converted to rad
        # change to relative value 
        coms_rewrel = np.array([com-np.pi for com in coms_correct])
        perm = list(combinations(range(len(coms_correct)), 2)) 
        rz_perm = [(int(rz[p[0]]),int(rz[p[1]])) for p in perm]   
        epoch_perm.append([perm,rz_perm]) 
        # if 4 ep
        # account for cells that move to the end/front
        # Define a small window around pi (e.g., epsilon)
        epsilon = .7 # 20 cm
        # Find COMs near pi and shift to -pi
        com_loop_w_in_window = []
        for pi,p in enumerate(perm):
            for cll in range(coms_rewrel.shape[1]):
                    com1_rel = coms_rewrel[p[0],cll]
                    com2_rel = coms_rewrel[p[1],cll]
                    # print(com1_rel,com2_rel,com_diff)
                    if ((abs(com1_rel - np.pi) < epsilon) and 
                    (abs(com2_rel + np.pi) < epsilon)):
                            com_loop_w_in_window.append(cll)
        # get abs value instead
        coms_rewrel[:,com_loop_w_in_window]=abs(coms_rewrel[:,com_loop_w_in_window])
        com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
        com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
        dist_to_rew.append(coms_rewrel)
        # get goal cells across all epochs        
        goal_cells = intersect_arrays(*com_goal)  
        # get other ep goal cells
        twoepgc = np.unique(np.concatenate(com_goal))
        twoepgcx = [xx for xx in twoepgc if xx not in goal_cells]    
        relcoms = [np.nanmedian(coms_rewrel[:,goal_cells],axis=0),
                    np.nanmean(coms_rewrel[:,twoepgcx],axis=0)]
        relative_coms.append(relcoms)
        # s2p_iind_goal_cells = s2p_iind_filter[goal_cells]
        # get per comparison
        goal_cells_p_per_comparison = [len(xx)/len(coms_correct[0]) for xx in com_goal]
        goal_cell_iind.append(goal_cells);goal_cell_p=len(goal_cells)/len(coms_correct[0])        
        goal_cell_prop.append([goal_cells_p_per_comparison,goal_cell_p]);num_epochs.append(len(coms_correct))
        colors = ['k', 'slategray', 'darkcyan', 'darkgoldenrod', 'orchid']
        if len(goal_cells)>0:
            rows = int(np.ceil(np.sqrt(len(goal_cells))))
            cols = int(np.ceil(len(goal_cells) / rows))
            fig, axes = plt.subplots(rows, cols, figsize=(30,30),sharex=True)
            if len(goal_cells) > 1:
                axes = axes.flatten()
            else:
                axes = [axes]
            for i,gc in enumerate(goal_cells):            
                for ep in range(len(coms_correct)):
                    ax = axes[i]
                    ax.plot(tcs_correct[ep,gc,:], label=f'rewloc {rewlocs[ep]}', color=colors[ep])
                    # if len(tcs_fail)>0:
                    #     ax.plot(tcs_fail[ep,gc,:], label=f'fail rewloc {rewlocs[ep]}', color=colors[ep], linestyle = '--')
                    ax.axvline((bins/2), color='k')
                    ax.set_title(f'cell # {gc}')
                    ax.spines[['top','right']].set_visible(False)
            ax.set_xticks(np.arange(0,bins+1,20))
            ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi, np.pi/2.25),3))
            ax.set_xlabel('Radian position (centered start rew loc)')
            ax.set_ylabel('Fc3')
        #     ax.legend()
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        # get shuffled iteration
        num_iterations = 1000; shuffled_dist = np.zeros((num_iterations))
        # max of 5 epochs = 10 perms
        goal_cell_shuf_ps_per_comp = np.ones((num_iterations,10))*np.nan; goal_cell_shuf_ps = []
        relative_coms_shuf=[]
        for i in range(num_iterations):
            # shuffle locations
            rewlocs_shuf = rewlocs #[random.randint(100,250) for iii in range(len(eps))]
            shufs = [list(range(coms_correct[ii].shape[0])) for ii in range(1, len(coms_correct))]
            [random.shuffle(shuf) for shuf in shufs]
            # first com is as ep 1, others are shuffled cell identities
            com_shufs = np.zeros_like(coms_correct); com_shufs[0,:] = coms_correct[0]
            com_shufs[1:1+len(shufs),:] = [coms_correct[ii][np.array(shufs)[ii-1]] for ii in range(1, 1+len(shufs))]
            # OR shuffle cell identities
            # relative to reward
            coms_rewrel = np.array([com-np.pi for ii, com in enumerate(com_shufs)])             
            perm = list(combinations(range(len(coms_correct)), 2)) 
            # account for cells that move to the end/front
            # Find COMs near pi and shift to -pi
            com_loop_w_in_window = []
            for pi,p in enumerate(perm):
                    for cll in range(coms_rewrel.shape[1]):
                            com1_rel = coms_rewrel[p[0],cll]
                            com2_rel = coms_rewrel[p[1],cll]
                            # print(com1_rel,com2_rel,com_diff)
                            if ((abs(com1_rel - np.pi) < epsilon) and 
                            (abs(com2_rel + np.pi) < epsilon)):
                                    com_loop_w_in_window.append(cll)
            # get abs value instead
            # cont.
            coms_rewrel[:,com_loop_w_in_window]=abs(coms_rewrel[:,com_loop_w_in_window])
            com_remap = np.array([(coms_rewrel[perm[jj][0]]-coms_rewrel[perm[jj][1]]) for jj in range(len(perm))])        
            # get goal cells across all epochs
            com_goal_shuf = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
            goal_cells_shuf_p_per_comparison = [len(xx)/len(coms_correct[0]) for xx in com_goal_shuf]
            goal_cells_shuf = intersect_arrays(*com_goal_shuf); shuffled_dist[i] = len(goal_cells_shuf)/len(coms_correct[0])
            # get other ep goal cells
            twoepgc = np.unique(np.concatenate(com_goal_shuf))
            twoepgcx = [xx for xx in twoepgc if xx not in goal_cells_shuf]    
            relative_coms_shuf.append([np.nanmedian(coms_rewrel[:,goal_cells_shuf],axis=0),np.nanmedian(coms_rewrel[:,twoepgcx],axis=0)])
            goal_cell_shuf_p=len(goal_cells_shuf)/len(com_shufs[0])
            goal_cell_shuf_ps.append(goal_cell_shuf_p)
            goal_cell_shuf_ps_per_comp[i, :len(goal_cells_shuf_p_per_comparison)] = goal_cells_shuf_p_per_comparison
        # save median of relative coms shuffle
        dimrelcomshuf = [len(xx[0]) for xx in relative_coms_shuf]
        relcomshuf_allep = [xx[0] for xx in relative_coms_shuf]
        dim=max(dimrelcomshuf)
        relative_coms_shuf_allep = np.ones((num_iterations,dim))*np.nan
        for jj in range(num_iterations):
            relative_coms_shuf_allep[jj,:len(relcomshuf_allep[jj])]=relcomshuf_allep[jj]
        relative_coms_shuf_allep_med=np.nanmedian(relative_coms_shuf_allep,axis=0)
        
        dimrelcomshuf = [len(xx[1]) for xx in relative_coms_shuf]
        relcomshuf_twoep = [xx[1] for xx in relative_coms_shuf]
        dim=max(dimrelcomshuf)
        relative_coms_shuf_twoep = np.ones((num_iterations,dim))*np.nan
        for jj in range(num_iterations):
            relative_coms_shuf_twoep[jj,:len(relcomshuf_twoep[jj])]=relcomshuf_twoep[jj]
        relative_coms_shuf_twoep_med=np.nanmedian(relative_coms_shuf_twoep,axis=0)
        relative_coms_shufs.append([relative_coms_shuf_twoep,relative_coms_shuf_twoep_med])
        # save median of goal cell shuffle
        goal_cell_shuf_ps_per_comp_av = np.nanmedian(goal_cell_shuf_ps_per_comp,axis=0)        
        goal_cell_shuf_ps_av = np.nanmedian(np.array(goal_cell_shuf_ps)[1])
        # compare to average of shuffle dist
        goal_cell_null.append([goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av])
        p_value = sum(shuffled_dist>goal_cell_p)/num_iterations
        pvals.append(p_value); 
        # print(f'{animal}, day {day}: significant goal cells proportion p-value: {p_value}')
        total_cells.append(len(coms_correct[0]))
        radian_alignment[f'{animal}_{day:03d}_index{ii:03d}'] = [tcs_correct, coms_correct, tcs_fail, coms_fail,
                        com_goal, goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av]

pdf.close()
# save pickle of dcts
# with open(saveddataset, "wb") as fp:   #Pickling
#     pickle.dump(radian_alignment, fp) 
#%%
plt.rc('font', size=16)          # controls default text sizes
# plot goal cells across epochs
inds = [int(xx[-3:]) for xx in radian_alignment.keys()]
df = conddf.copy()
df = df[((df.animals!='e217')) & (df.optoep<2) & (df.index.isin(inds))]
df['num_epochs'] = num_epochs
df['goal_cell_prop'] = [xx[1] for xx in goal_cell_prop]
df['opto'] = df.optoep.values>1
df['day'] = df.days
df['session_num_opto'] = np.concatenate([[xx-df[df.animals==an].days.values[0] for xx in df[df.animals==an].days.values] for an in np.unique(df.animals.values)])
df['session_num'] = np.concatenate([[ii for ii,xx in enumerate(df[df.animals==an].days.values)] for an in np.unique(df.animals.values)])
df['condition'] = ['vip' if xx=='vip' else 'ctrl' for xx in df.in_type.values]
df['p_value'] = pvals
df['goal_cell_prop_shuffle'] = [xx[1] for xx in goal_cell_null]

fig,ax = plt.subplots(figsize=(5,5))
ax = sns.histplot(data = df.loc[df.opto==False], x='p_value', 
                hue='animals', bins=40)
ax.spines[['top','right']].set_visible(False)
ax.axvline(x=0.05, color='k', linestyle='--')
sessions_sig = sum(df.loc[df.opto==False,'p_value'].values<0.05)/len(df.loc[df.opto==False])
ax.set_title(f'{(sessions_sig*100):.2f}% of sessions are significant')
ax.set_xlabel('P-value')
ax.set_ylabel('Sessions')
#%%
# TODO
# com of persistent cells across epochs
fig,ax = plt.subplots()
ax.hist(np.concatenate([xx[1] for xx in relative_coms]), color='k',alpha=.5,
    density=True,label='2 epochs')
ax.hist(np.concatenate([xx[0] for ii,xx in enumerate(relative_coms) if num_epochs[ii]==3]), 
    color='slategray',alpha=.5,density=True,label='3 epochs')
ax.hist(np.concatenate([xx[0] for ii,xx in enumerate(relative_coms) if num_epochs[ii]==4]), 
    color='darkcyan',alpha=.5,density=True,label='4 epochs')
# ax.hist(np.concatenate([xx[0] for ii,xx in enumerate(relative_coms) if num_epochs[ii]==5]), 
#     color='orange',alpha=.5,density=True)
ax.set_ylabel('Density of COMs')
ax.set_xlabel(f'Reward-relative distance ($\Theta$)')
ax.spines[['top','right']].set_visible(False)
ax.legend()
#%%
# Histogram of # of epochs / days a cell is a goal cell
fig,ax = plt.subplots()
ax.hist(np.concatenate([[2]*len(xx[0]) for ii,xx in enumerate(relative_coms) if num_epochs[ii]==2]), color='slategray',alpha=.7,
    label='2 epochs')
ax.hist(np.concatenate([[3]*len(xx[0]) for ii,xx in enumerate(relative_coms) if num_epochs[ii]==3]), 
    color='orchid',alpha=.4,label='3 epochs')
ax.hist(np.concatenate([[4]*len(xx[0]) for ii,xx in enumerate(relative_coms) if num_epochs[ii]==4]), 
    color='royalblue',alpha=.4,label='4 epochs')
# ax.hist(np.concatenate([xx[0] for ii,xx in enumerate(relative_coms) if num_epochs[ii]==5]), 
#     color='orange',alpha=.5,density=True)
ax.set_ylabel('Density of COMs of reward-map cells')
ax.set_xlabel('Reward-relative distance')
ax.spines[['top','right']].set_visible(False)
ax.legend()

#%%
#%%
# number of epochs vs. reward cell prop    
fig,ax = plt.subplots(figsize=(5,5))
df_plt = df
# av across mice
df_plt = df_plt.groupby(['animals','num_epochs']).mean(numeric_only=True)
sns.stripplot(x='num_epochs', y='goal_cell_prop',
        hue='animals',data=df_plt,
        s=10)
sns.barplot(x='num_epochs', y='goal_cell_prop',
        data=df_plt,
        fill=False,ax=ax, color='k', errorbar='se')
ax = sns.lineplot(data=df_plt, # correct shift
        x=df_plt.index.get_level_values('num_epochs')-2, y='goal_cell_prop_shuffle',color='grey', 
        label='shuffle')
ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.01, 1.05))

eps = [2,3,4]
for ep in eps:
    # rewprop = df_plt.loc[(df_plt.num_epochs==ep), 'goal_cell_prop']
    rewprop = df_plt.loc[(df_plt.index.get_level_values('num_epochs')==ep), 'goal_cell_prop']
    shufprop = df_plt.loc[(df_plt.index.get_level_values('num_epochs')==ep), 'goal_cell_prop_shuffle']
    t,pval = scipy.stats.wilcoxon(rewprop, shufprop)
    print(f'{ep} epochs, pval: {pval}')
#%%    
# include all comparisons 
df_perms = pd.DataFrame()
df_perms['epoch_comparison'] = [str(tuple(xx)) for xx in np.concatenate([yy[0] for yy in epoch_perm])]
df_perms['rewzone_comparison'] = [str(tuple(xx)) for xx in np.concatenate([yy[1] for yy in epoch_perm])]
goal_cell_perm = [xx[0] for xx in goal_cell_prop]
goal_cell_perm_shuf = [xx[0][~np.isnan(xx[0])] for xx in goal_cell_null]
df_perms['goal_cell_prop'] = np.concatenate(goal_cell_perm)
df_perms['goal_cell_prop_shuffle'] = np.concatenate(goal_cell_perm_shuf)
df_perm_animals = [[xx]*len(goal_cell_perm[ii]) for ii,xx in enumerate(df.animals.values)]
df_perms['animals'] = np.concatenate(df_perm_animals)
df_perm_days = [[xx]*len(goal_cell_perm[ii]) for ii,xx in enumerate(df.session_num.values)]
df_perms['session_num'] = np.concatenate(df_perm_days)
# df_perms = df_perms[df_perms.animals!='e189']
df_permsav = df_perms.groupby(['animals','rewzone_comparison']).mean(numeric_only=True)

fig,ax = plt.subplots(figsize=(7,5))
sns.stripplot(x='rewzone_comparison', y='goal_cell_prop',
        hue='animals',data=df_permsav,
        s=8,ax=ax)
sns.barplot(x='rewzone_comparison', y='goal_cell_prop',
        data=df_permsav,
        fill=False,ax=ax, color='k', errorbar='se')
ax = sns.lineplot(data=df_permsav, # correct shift
        x='rewzone_comparison', y='goal_cell_prop_shuffle',
        color='grey', label='shuffle')

ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.01, 1.05))

eps = df_permsav.index.get_level_values("rewzone_comparison").unique()
for ep in eps:
    # rewprop = df_plt.loc[(df_plt.num_epochs==ep), 'goal_cell_prop']
    rewprop = df_permsav.loc[(df_permsav.index.get_level_values('rewzone_comparison')==ep), 'goal_cell_prop'].values
    shufprop = df_permsav.loc[(df_permsav.index.get_level_values('rewzone_comparison')==ep), 'goal_cell_prop_shuffle'].values
    t,pval = scipy.stats.wilcoxon(rewprop, shufprop)
    print(f'{ep} epochs, pval: {pval}')

# take a mean of all epoch comparisons
df_perms['num_epochs'] = [2]*len(df_perms)
df_permsav2 = df_perms.groupby(['animals', 'num_epochs']).mean(numeric_only=True)
#%%
# compare to shuffle
df_plt2 = pd.concat([df_permsav2,df_plt])
df_plt2 = df_plt2[df_plt2.index.get_level_values('animals')!='e189']
df_plt2 = df_plt2[df_plt2.index.get_level_values('num_epochs')<5]
df_plt2 = df_plt2.groupby(['animals', 'num_epochs']).mean(numeric_only=True)
# number of epochs vs. reward cell prop incl combinations    
fig,ax = plt.subplots(figsize=(3,5))
# av across mice
sns.stripplot(x='num_epochs', y='goal_cell_prop',color='k',
        data=df_plt2,s=10,alpha=0.7)
sns.barplot(x='num_epochs', y='goal_cell_prop',
        data=df_plt2,
        fill=False,ax=ax, color='k', errorbar='se')
ax = sns.lineplot(data=df_plt2, # correct shift
        x=df_plt2.index.get_level_values('num_epochs').astype(int)-2, 
        y='goal_cell_prop_shuffle',color='grey', 
        label='shuffle')
ax.spines[['top','right']].set_visible(False)
ax.legend()#.set_visible(False)
ax.set_xlabel('# of reward loc. switches')
ax.set_ylabel('Reward-distance cell proportion')
eps = [2,3,4]
y = 0.3
pshift = 0.04
fs=36
for ii,ep in enumerate(eps):
        rewprop = df_plt2.loc[(df_plt2.index.get_level_values('num_epochs')==ep), 'goal_cell_prop']
        shufprop = df_plt2.loc[(df_plt2.index.get_level_values('num_epochs')==ep), 'goal_cell_prop_shuffle']
        t,pval = scipy.stats.wilcoxon(rewprop, shufprop)
        print(f'{ep} epochs, pval: {pval}')
        # statistical annotation        
        if pval < 0.001:
                plt.text(ii, y, "***", ha='center', fontsize=fs)
        elif pval < 0.01:
                plt.text(ii, y, "**", ha='center', fontsize=fs)
        elif pval < 0.05:
                plt.text(ii, y, "*", ha='center', fontsize=fs)
        ax.text(ii-0.5, y+pshift, f'p={pval:.3g}',fontsize=10,rotation=45)

# com
plt.savefig(os.path.join(savedst, 'reward_cell_prop_per_an.png'), 
        bbox_inches='tight')
#%%
# as a function of session/day
df_plt = df.groupby(['animals','session_num','num_epochs']).mean(numeric_only=True)
df_permsav2 = df_perms.groupby(['animals', 'session_num','num_epochs']).mean(numeric_only=True)
# compare to shuffle
df_plt2 = pd.concat([df_permsav2,df_plt])
# df_plt2 = df_plt2[df_plt2.index.get_level_values('animals')!='e189']
df_plt2 = df_plt2[df_plt2.index.get_level_values('num_epochs')==2]
df_plt2 = df_plt2.groupby(['animals', 'session_num','num_epochs']).mean(numeric_only=True)
# number of epochs vs. reward cell prop incl combinations    
fig,ax = plt.subplots(figsize=(7,5))
# av across mice
sns.stripplot(x='session_num', y='goal_cell_prop',hue='animals',
        data=df_plt2,s=10,alpha=0.7)
sns.barplot(x='session_num', y='goal_cell_prop',color='darkslateblue',
        data=df_plt2,fill=False,ax=ax, errorbar='se')
ax.set_xlim([-.5,9.5])
# ax = sns.lineplot(data=df_plt2, # correct shift
#         x=df_plt2.index.get_level_values('num_epochs').astype(int)-2, y='goal_cell_prop_shuffle',color='grey', 
#         label='shuffle')
ax.spines[['top','right']].set_visible(False)
# ax.legend().set_visible(False)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax.set_xlabel('# of sessions')
ax.set_ylabel('Reward-distance cell proportion')
df_reset = df_plt2.reset_index()
sns.regplot(x='session_num', y='goal_cell_prop',
        data=df_reset, scatter=False, color='k')
r, p = scipy.stats.pearsonr(df_reset['session_num'], 
        df_reset['goal_cell_prop'])
ax = plt.gca()
ax.text(.5, .8, 'r={:.3f}, p={:.3g}'.format(r, p),
        transform=ax.transAxes)
ax.set_title('2 epoch combinations')

#%%
df['recorded_neurons_per_session'] = total_cells
df_plt_ = df[(df.opto==False)&(df.p_value<0.05)]
df_plt_= df_plt_[(df_plt_.animals!='e200')&(df_plt_.animals!='e189')]
df_plt_ = df_plt_.groupby(['animals']).mean(numeric_only=True)

fig,ax = plt.subplots(figsize=(7,5))
sns.scatterplot(x='recorded_neurons_per_session', y='goal_cell_prop',hue='animals',
        data=df_plt_,
        s=150, ax=ax)
sns.regplot(x='recorded_neurons_per_session', y='goal_cell_prop',
        data=df_plt_,
        ax=ax, scatter=False, color='k'
)
r, p = scipy.stats.pearsonr(df_plt_['recorded_neurons_per_session'], 
        df_plt_['goal_cell_prop'])
ax = plt.gca()
ax.text(.5, .8, 'r={:.2f}, p={:.2g}'.format(r, p),
        transform=ax.transAxes)

ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.01, 1.05))
ax.set_xlabel('Av. # of neurons per session')
ax.set_ylabel('Reward cell proportion')

plt.savefig(os.path.join(savedst, 'rec_cell_rew_prop_per_an.svg'), 
        bbox_inches='tight')

#%%
df['success_rate'] = rates_all

# all animals
fig,ax = plt.subplots(figsize=(7,5))
sns.scatterplot(x='success_rate', y='goal_cell_prop',hue='animals',
        data=df,
        s=150, ax=ax)
sns.regplot(x='success_rate', y='goal_cell_prop',
        data=df,
        ax=ax, scatter=False, color='k'
)
r, p = scipy.stats.pearsonr(df['success_rate'], 
        df['goal_cell_prop'])
ax = plt.gca()
ax.text(.5, .8, 'r={:.2f}, p={:.2g}'.format(r, p),
        transform=ax.transAxes)

ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.01, 1.05))
ax.set_xlabel('Success rate')
ax.set_ylabel('Reward cell proportion')
#%%
an_nms = df.animals.unique()
rows = int(np.ceil(np.sqrt(len(an_nms))))
cols = int(np.ceil(np.sqrt(len(an_nms))))
fig,axes = plt.subplots(nrows=rows, ncols=cols,
            figsize=(10,10))
rr=0;cc=0
for an in an_nms:        
    ax = axes[rr,cc]
    sns.scatterplot(x='success_rate', y='goal_cell_prop',
            data=df[(df.animals==an)&(df.opto==False)&(df.p_value<0.05)],
            s=200, ax=ax)
    ax.spines[['top','right']].set_visible(False)
    ax.set_title(an)
    rr+=1
    if rr>=rows: rr=0; cc+=1    
fig.tight_layout()
#%%
# # #examples
fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
Fc3 = fall_fc3['Fc3']
dFF = fall_fc3['dFF']
Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
Fc3 = Fc3[:,(skew>2)] # only keep cells with skew greateer than 2
bin_size=3 # cm
# get abs dist tuning 
tcs_correct_abs, coms_correct_abs, tcs_fail, coms_fail = make_tuning_curves_by_trialtype(eps,rewlocs,ybinned,
Fc3,trialnum,rewards,forwardvel,rewsize,bin_size)

# # #plot example tuning curve
plt.rc('font', size=30)  
fig,axes = plt.subplots(1,3,figsize=(20,20), sharex = True)
for ep in range(3):
        axes[ep].imshow(tcs_correct_abs[ep,com_goal[0]][np.argsort(coms_correct_abs[0,com_goal[0]])[:60],:]**.3)
        axes[ep].set_title(f'Epoch {ep+1}')
        axes[ep].axvline((rewlocs[ep]-rewsize/2)/bin_size, color='w', linestyle='--', linewidth=4)
        axes[ep].set_xticks(np.arange(0,(track_length/bin_size)+bin_size,30))
        axes[ep].set_xticklabels(np.arange(0,track_length+bin_size*30,bin_size*30).astype(int))
axes[0].set_ylabel('Reward-distance cells')
axes[2].set_xlabel('Absolute distance (cm)')
plt.savefig(os.path.join(savedst, 'abs_dist_tuning_curves_3_ep.svg'), bbox_inches='tight')

# fig,axes = plt.subplots(1,4,figsize=(15,20), sharey=True, sharex = True)
# axes[0].imshow(tcs_correct[0,com_goal[0]][np.argsort(coms_correct[0,com_goal[0]])[:60],:]**.5)
# axes[0].set_title('Epoch 1')
# im = axes[1].imshow(tcs_correct[1,com_goal[0]][np.argsort(coms_correct[0,com_goal[0]])[:60],:]**.5)
# axes[1].set_title('Epoch 2')
# im = axes[2].imshow(tcs_correct[2,com_goal[0]][np.argsort(coms_correct[0,com_goal[0]])[:60],:]**.5)
# axes[2].set_title('Epoch 3')
# im = axes[3].imshow(tcs_correct[3,com_goal[0]][np.argsort(coms_correct[0,com_goal[0]])[:60],:]**.5)
# axes[3].set_title('Epoch 4')
# ax = axes[1]
# ax.set_xticks(np.arange(0,bins+1,10))
# ax.set_xticklabels(np.round(np.arange(-np.pi, np.pi+np.pi/4.5, np.pi/4.5),1))
# ax.axvline((bins/2), color='w', linestyle='--')
# axes[0].axvline((bins/2), color='w', linestyle='--')
# axes[2].axvline((bins/2), color='w', linestyle='--')
# axes[3].axvline((bins/2), color='w', linestyle='--')
# axes[0].set_ylabel('Reward distance cells')
# axes[3].set_xlabel('Reward-relative distance (rad)')
# fig.tight_layout()
# plt.savefig(os.path.join(savedst, 'tuning_curves_4_ep.png'), bbox_inches='tight')
# for gc in goal_cells:
#%%
gc = 51
plt.rc('font', size=24)  
fig2,ax2 = plt.subplots(figsize=(5,5))

for ep in range(3):        
        ax2.plot(tcs_correct_abs[ep,gc,:], label=f'rewloc {rewlocs[ep]}', color=colors[ep],linewidth=3)
        ax2.axvline(rewlocs[ep]/bin_size, color=colors[ep], linestyle='--',linewidth=3)
        
        ax2.spines[['top','right']].set_visible(False)
ax2.set_title(f'animal: {animal}, day: {day}\ncell # {gc}')
ax2.set_xticks(np.arange(0,(track_length/bin_size)+bin_size,30))
ax2.set_xticklabels(np.arange(0,track_length+bin_size*30,bin_size*30).astype(int))
ax2.set_xlabel('Absolute position (cm)')
ax2.set_ylabel('$\Delta$ F/F')
        
plt.savefig(os.path.join(savedst, f'rewardd_cell_{gc}_tuning_per_ep.svg'), bbox_inches='tight')

fig2,ax2 = plt.subplots(figsize=(5,5))
for ep in range(3):        
        ax2.plot(tcs_correct[ep,gc,:], label=f'rewloc {rewlocs[ep]}', color=colors[ep],linewidth=3)
        ax2.axvline(bins/2, color="k", linestyle='--',linewidth=3)
        
        ax2.spines[['top','right']].set_visible(False)
ax2.set_title(f'animal: {animal}, day: {day}\ncell # {gc}')
ax2.set_xticks(np.arange(0,bins+1,30))
ax2.set_xticklabels(np.round(np.arange(-np.pi, 
        np.pi+np.pi/1.5, np.pi/1.5),1))
ax2.set_xlabel('Radian position ($\Theta$)')
ax2.set_ylabel('$\Delta$ F/F')
plt.savefig(os.path.join(savedst, f'rewardd_cell_{gc}_aligned_tuning_per_ep.svg'), bbox_inches='tight')

"""
zahra
get tuning curves with dark time
get cells in 2, 3, or 4 epochs
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
plt.rc('font', size=20)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from reward_shuffle import get_com_v_persistence
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_rewardcentric_all.p"
with open(saveddataset, "rb") as fp: #unpickle
        radian_alignment_saved = pickle.load(fp)

#%%
####################################### RUN CODE #######################################
# initialize var
ep_dicts = []
bins = 90
goal_window_cm=20
# cm_window = [10,20,30,40,50,60,70,80] # cm
# iterate through all animals
for ii in range(len(conddf)):
    day = conddf.days.values[ii]
    animal = conddf.animals.values[ii]
    if (animal!='e217') & (conddf.optoep.values[ii]<2):
        if animal=='e145' or animal=='e139': pln=2 
        else: pln=0
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
        ep_dict = get_com_v_persistence(params_pth, animal, day, ii)
        ep_dicts.append(ep_dict)
pdf.close()
####################################### RUN CODE #######################################
#%%
# key = ep
from scipy.stats import gaussian_kde

# plot histograms
fig, axes = plt.subplots(nrows = 2,figsize=(6,7))
for ii in range(2):
    colors = ['k', 'slategray', 'darkcyan', 'darkgoldenrod', 'orchid']
    a = 0.2
    lw = 3
    ax=axes[ii]
    labels = [2,3,4]
    com_ep2_comb = [xx[1][ii] for xx in ep_dicts if 1 in xx.keys()]
    com_ep3_comb = [xx[2][ii] for xx in ep_dicts if 2 in xx.keys()]
    com_ep4_comb = [xx[3][ii] for xx in ep_dicts if 3 in xx.keys()]
    com_ep5_comb = [xx[4][ii] for xx in ep_dicts if 4 in xx.keys()]
    # Plot histogram and confidence intervals for each epoch
    data_sets = [com_ep2_comb, com_ep3_comb, com_ep4_comb]
    for i, data in enumerate(data_sets):
        all_data = np.concatenate(data)
        all_data = all_data[~np.isnan(all_data)]
        # Smooth Gaussian KDE
        kde = gaussian_kde(all_data)
        # x_vals = np.linspace(-np.pi, np.pi, 270)
        if ii==0:
            x_vals = np.linspace(-220,200, 270)
        else:
            x_vals = np.linspace(-np.pi, np.pi, 270)
        y_vals = kde(x_vals)

        # Plot the KDE line
        ax.plot(x_vals, y_vals, color=colors[i], linewidth=4,
                label=f'{labels[i]}, {len(all_data)} cells')
    # Style and labels
    ax.set_ylabel('Probability density')
    if ii==1:
        ax.set_xticks([-np.pi, -np.pi/4,0, np.pi/4,np.pi])
        ax.set_xticklabels(["$-\\pi$", '$-\\pi/4$', "0",  '$\\pi/4$', "$\\pi$"])
        ax.set_xlabel('Reward-centric distance ($\Theta$)')
    else:        
        ax.set_xlabel('Reward-centric distance (cm)')        
        # Legend
        h_strip, l_strip = ax.get_legend_handles_labels()
        if ax.legend_: ax.legend_.remove()
        ax.legend(
            h_strip, l_strip,
            title='# of epochs',
            loc='upper left',
            bbox_to_anchor=(.8, 1),
            borderaxespad=0.,
            fontsize=14,
            title_fontsize=14
        )
    ax.spines[['top', 'right']].set_visible(False)
    ax.axvline(0,linestyle='--',color='grey',linewidth=3)


plt.tight_layout()
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper\panels_main_figures'
plt.savefig(os.path.join(savedst, 'com_hist_across_ep.svg'),bbox_inches='tight')
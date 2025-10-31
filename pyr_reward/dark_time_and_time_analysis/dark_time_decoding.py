"""
plot decoding for supplemental
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
fl = r"C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper\from_bo\DarkTimePrediction"
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
with open(fl, "rb") as fp: #unpickle
    dct = pickle.load(fp)
    
fig, ax=plt.subplots()
ax.plot(dct[0], color='k',label='Real position')
ax.plot(dct[1]*270, color='goldenrod',label='Predicted')
ymin, ymax=-.2,540
for l in np.where(dct[0]>0.006)[0]:
    ax.plot([l,l], [ymin, ymax], color='gray')
ax.text(60,550,'Trial #',color='gray')
ax.spines[['top','right']].set_visible(False)
ax.set_xticks([0, len(dct[0])])
ax.set_xticklabels([0, round(len(dct[0])/31.25,1)])
ax.set_xlabel('Time (s)')
ax.set_ylabel('Track position (cm)')
fig.suptitle('Bayesian position decoding (delay period)')
ax.legend()
# plt.tight_layout()
plt.savefig(os.path.join(os.path.join(savedst, 'darktime_decoding.svg')))

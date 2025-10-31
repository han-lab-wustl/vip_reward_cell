
"""
zahra
april 2025
export to bo
"""
#%%

import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd, matplotlib.backends.backend_pdf
import pickle, seaborn as sns, random, math, os, matplotlib as mpl
from collections import Counter
from itertools import combinations, chain
import scikit_posthocs as sp
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
# plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.pyr_reward.placecell import make_tuning_curves_radians_by_trialtype, intersect_arrays
from projects.pyr_reward.rewardcell import get_days_from_cellreg_log_file, find_log_file, get_radian_position, \
    get_tracked_lut, get_tracking_vars, get_shuffled_goal_cell_indices, get_reward_cells_that_are_tracked
from projects.opto.behavior.behavior import get_success_failure_trials
# import condition df
animals = ['e218','e216','e201','e186',
        'e190', 'e145', 'z8', 'z9']
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
radian_tuning_dct = r'Z:\\saved_datasets\\radian_tuning_curves_rewardcentric_all.p'
with open(radian_tuning_dct, "rb") as fp: #unpickle
    radian_alignment_saved = pickle.load(fp)
celltrackpth = r'Y:\analysis\celltrack'
# cell tracked days
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
#%%
per_day_goal_cells_all = []
# defined vars
maxep = 5
shuffles = 1000
# redo across days analysis but init array per animal
coms_per_an = []
for animal in animals:
    # all rec days
    if animal=='e145': pln=2
    else: pln=0    
    tracked_lut, days= get_tracked_lut(celltrackpth,animal,pln)
    tracked_lut.index.name = 'tracked_cell_index'
    tracked_lut.to_csv(os.path.join(savedst, animal+'.csv'))
# %%

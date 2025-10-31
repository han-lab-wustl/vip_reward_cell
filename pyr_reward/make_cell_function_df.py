
#%%
"""
zahra
feb 2025
data structure
"""
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd, os
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["ytick.major.size"] = 8
# plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from placecell import make_tuning_curves_radians_by_trialtype, intersect_arrays
from projects.opto.behavior.behavior import get_success_failure_trials
from rewardcell import get_radian_position,extract_data_df
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_tracking.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
savepth = os.path.join(savedst, 'vip_opto_cell_charac.pdf')
#%%
goal_cm_window=20 # to search for rew cells
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
# saveddataset = rf'Z:\saved_datasets\radian_tuning_curves_rewardcentric_all.p'
# with open(saveddataset, "rb") as fp: #unpickle 
#     radian_alignment_saved = pickle.load(fp)
radian_alignment_saved = {} # overwrite
dfs = []
radian_alignment = {}
lasttr=8 #  last trials 
bins=90
#%%
# iterate through all animals
for ii in range(179,len(conddf)):
    if ii!=179:
        day = conddf.days.values[ii]
        animal = conddf.animals.values[ii]
        if animal=='e145' or animal=='e139': pln=2 
        else: pln=0
        params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
        df=extract_data_df(ii, params_pth, animal, day, radian_alignment, radian_alignment_saved, 
                    goal_cm_window, pdf, pln)
        dfs.append(df)
#%%
# concat bigdf
bigdf=pd.concat(dfs)
bigdf.to_csv(r"\\storage1.ris.wustl.edu\ebhan\Active\dzahra\vip_opto_inhib_excit_cell_features_pyr_goal.csv",index=None)
#%%
bigdf[(bigdf.animal=='e201')&(bigdf.tracked_cellid==2)] 
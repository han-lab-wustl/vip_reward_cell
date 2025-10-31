
"""
zahra
get tuning curves with dark time
get cells in 2, 3, or 4 epochs
only use spatially tuned!!!
july 2025
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
from projects.memory.behavior import consecutive_stretch
from projects.pyr_reward.placecell import get_tuning_curve, calc_COM_EH, make_tuning_curves_radians_by_trialtype,\
    make_tuning_curves_by_trialtype_w_darktime,get_radian_position_first_lick_after_rew_w_dt
from projects.pyr_reward.rewardcell import get_radian_position,\
    get_radian_position_first_lick_after_rew, get_rewzones, get_goal_cells, goal_cell_shuffle,wilcoxon_r
from projects.opto.behavior.behavior import get_success_failure_trials
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
rewdf=pd.read_csv(r'Z:\saved_datasets\rew_v_shuffle.csv')
placedf=pd.read_csv(r'Z:\saved_datasets\place_v_shuffle.csv')

from scipy.optimize import curve_fit

# Define the exponential decay function
def exponential_decay(t, A, tau):
    return A * np.exp(-t / tau)
tau_all = []; y_fit_all = []

# df_plt2=df_plt2.reset_index()
for an in rewdf.animals.unique():
   # Initial guesses for the optimization
   initial_guess = [4, 2]  # Amplitude guess and tau guess
   y = rewdf[rewdf.animals==an]
   t=np.array([2,3,4])
   # Fit the model to the data using curve_fit
   params, params_covariance = curve_fit(exponential_decay, t, y.goal_cell_prop.values, p0=initial_guess)
   # Extract the fitted parameters
   A_fit, tau_fit = params
   tau_all.append(tau_fit)
   # Generate the fitted curve using the optimized parameters
   y_fit = exponential_decay(t, A_fit, tau_fit)
   y_fit_all.append(y_fit)
rew_tau_all=tau_all
tau_all = []; y_fit_all = []
# df_plt2=df_plt2.reset_index()
for an in placedf.animals.unique():
   # Initial guesses for the optimization
   initial_guess = [4, 2]  # Amplitude guess and tau guess
   y = placedf[placedf.animals==an]
   t=np.array([2,3,4])
   # Fit the model to the data using curve_fit
   pc=y.place_cell_prop.values
   if len(pc)==3:
      params, params_covariance = curve_fit(exponential_decay, t, pc, p0=initial_guess)
      # Extract the fitted parameters
      A_fit, tau_fit = params
      tau_all.append(tau_fit)
      # Generate the fitted curve using the optimized parameters
      y_fit = exponential_decay(t, A_fit, tau_fit)
      y_fit_all.append(y_fit)

# %%

df=pd.DataFrame()
df['tau']=np.concatenate([tau_all,rew_tau_all])
df['celltype']=np.concatenate([['place']*len(tau_all),['reward']*len(rew_tau_all)])
# Get summary statistics including SEM
summary = df.groupby(['celltype'])['tau'].agg([
    'count', 
    'mean', 
    'std', 
    ('sem', lambda x: x.sem()),  # Standard Error of Mean
    'median', 
    'min', 
    'max'
])

print(summary)


"""
zahra
hmm model
april 2025
"""
#%%
import numpy as np, h5py, scipy, matplotlib.pyplot as plt, sys, pandas as pd
import pickle, seaborn as sns, random, math
from collections import Counter
from itertools import combinations, chain
import matplotlib.backends.backend_pdf, matplotlib as mpl
from sklearn.cluster import KMeans
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
# plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.pyr_reward.placecell import make_tuning_curves_radians_by_trialtype, intersect_arrays
from projects.pyr_reward.rewardcell import get_radian_position,create_mask_from_coordinates,pairwise_distances,extract_data_rewcentric,\
    get_radian_position_first_lick_after_rew, get_rewzones
from projects.opto.behavior.behavior import get_success_failure_trials
# import condition df
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper'
saveddataset = r"Z:\saved_datasets\radian_tuning_curves_rewardcentric_all.p"
with open(saveddataset, "rb") as fp: #unpickle
        radian_alignment_saved = pickle.load(fp)
savepth = os.path.join(savedst, 'pre_rew_assemblies.pdf')
pdf = matplotlib.backends.backend_pdf.PdfPages(savepth)
def interpolate_dff_nan(dff_array):
    """
    Interpolates NaNs in a 2D dF/F array (cells x time).
    Uses linear interpolation along time (axis=1).
    """
    interpolated = np.empty_like(dff_array)
    for i in range(dff_array.shape[0]):
        trace = pd.Series(dff_array[i])
        interpolated[i] = trace.interpolate(method='linear', limit_direction='both').values
    return interpolated

#%%
# initialize var
# radian_alignment_saved = {} # overwrite
tcs_rew = []
goal_cells_all = []
p_rewcells_in_assemblies=[]

bins = 90
goal_window_cm=20
epoch_perm=[]
assembly_cells_all_an=[]
# cm_window = [10,20,30,40,50,60,70,80] # cm
# iterate through all animals
ii=0
day = conddf.days.values[ii]
animal = conddf.animals.values[ii]
if animal=='e145' or animal=='e139': pln=2 
else: pln=0
params_pth = rf"Y:\analysis\fmats\{animal}\days\{animal}_day{day:03d}_plane{pln}_Fall.mat"
print(params_pth)
fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 
        'pyr_tc_s2p_cellind', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
        'stat', 'licks'])
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
licks=fall['licks'][0]
if animal=='e145':
    ybinned=ybinned[:-1]
    forwardvel=forwardvel[:-1]
    changeRewLoc=changeRewLoc[:-1]
    trialnum=trialnum[:-1]
    rewards=rewards[:-1]
    licks=licks[:-1]
# set vars
eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
rad = get_radian_position_first_lick_after_rew(eps, ybinned, licks, rewards, rewsize,rewlocs,
                trialnum, track_length) # get radian coordinates
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
rate=np.nanmean(np.array(rates))

# added to get anatomical info
# takes time
fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
Fc3 = fall_fc3['Fc3']
dFF = fall_fc3['dFF']
Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool))]
dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool))]
skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
Fc3 = Fc3[:, skew>2] # only keep cells with skew greateer than 2
tcs_correct, coms_correct, tcs_fail, coms_fail = make_tuning_curves_radians_by_trialtype(eps,rewlocs,ybinned,rad,Fc3,trialnum,
rewards,forwardvel,rewsize,bin_size)          
goal_window = goal_window_cm*(2*np.pi/track_length) # cm converted to rad
# change to relative value 
coms_rewrel = np.array([com-np.pi for com in coms_correct])
perm = list(combinations(range(len(coms_correct)), 2)) 
rz_perm = [(int(rz[p[0]]),int(rz[p[1]])) for p in perm]   
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
com_goal = np.unique(np.concatenate(com_goal))
from hmmlearn import hmm
# Assume `neural_data` is shape (n_timepoints, n_neurons)
# And you have timestamps of licks and movements
# Fit a Gaussian HMM with 2 hidden states (pre- and post-reward)
# start with 1 trial
eprng = np.arange(eps[0],eps[1])
trials = trialnum[eprng]
f = Fc3[eprng,:]
# f = f[:,com_goal]
# make trial structure
f__ = [f[trials==tr] for tr in np.unique(trials)]
maxtime = np.nanmax(np.array([len(xx) for xx in f__]))
dff = np.zeros((len(np.unique(trials)), f.shape[1], maxtime))*np.nan
unqtrials = np.unique(trials)
for tr_idx, tr in enumerate(unqtrials):
    f_ = f[trials == tr].T  # shape: (n_cells, trial_length)
    original_len = f_.shape[1]
    orig_time = np.linspace(0, 1, original_len)
    new_time = np.linspace(0, 1, maxtime)
    for cell in range(f.shape[1]):
        trace = f_[cell]
        if np.sum(~np.isnan(trace)) > 1:  # must have >1 valid point to interpolate
            interp_func = scipy.interpolate.interp1d(orig_time, trace, kind='linear', bounds_error=False, fill_value='extrapolate')
            dff[tr_idx, cell] = interp_func(new_time)

# Reshape for HMM: (trials * time, features)
n_trials, n_cells, n_timepoints = dff.shape
dff_clean = dff#np.nan_to_num(dff, nan=0.0)
# Reshape for HMM
X = dff_clean.transpose(0, 2, 1).reshape(-1, dff_clean.shape[1])
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
n_states = 4  # choose based on expected behavioral phases
model = hmm.GaussianHMM(
    n_components=n_states,
    covariance_type='diag',
    n_iter=500,
    tol=1e-4,     # slightly higher tolerance
    verbose=True
)
model.fit(X)

# Decode states for each trial
decoded_states = np.zeros((n_trials, n_timepoints), dtype=int)
for trial in range(n_trials):
    trial_data = dff[trial].T  # shape (n_timepoints, n_cells)
    trial_data = np.nan_to_num(trial_data)  # Ensure no NaNs
    states = model.predict(trial_data)  # shape (n_timepoints,)
    decoded_states[trial] = states
# align to lick/movement
lick = licks[eprng]
move = forwardvel[eprng]
lick_trial = np.zeros((len(np.unique(trials)), maxtime))
for tr in range(len(unqtrials)):
    lick_ = lick[trials==unqtrials[tr]]
    lick_trial[tr,:len(lick_)] = lick_
move_trial = np.zeros((len(np.unique(trials)), maxtime))
for tr in range(len(unqtrials)):
    move_ = move[trials==unqtrials[tr]]
    move_trial[tr,:len(move_)] = move_
# per trial plot

for t in range(len(lick_trial)):
    plt.figure()
    plt.plot(decoded_states[t])
    plt.plot(lick_trial[t])
    plt.plot(move_trial[t]/max(move_trial[t]))

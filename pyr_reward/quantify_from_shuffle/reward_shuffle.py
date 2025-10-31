
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
    make_tuning_curves_by_trialtype_w_darktime,get_radian_position_first_lick_after_rew_w_dt,\
        make_tuning_curves
from projects.pyr_reward.rewardcell import get_radian_position,\
    get_radian_position_first_lick_after_rew, get_rewzones, get_goal_cells, goal_cell_shuffle
from projects.opto.behavior.behavior import get_success_failure_trials

def get_com_v_persistence(params_pth, animal, day, ii,goal_window_cm=20):
    print(params_pth)
    fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 'timedFF',
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
    time=fall['timedFF'][0]
    if animal=='e145':
        ybinned=ybinned[:-1]
        forwardvel=forwardvel[:-1]
        changeRewLoc=changeRewLoc[:-1]
        trialnum=trialnum[:-1]
        rewards=rewards[:-1]
        licks=licks[:-1]
        time=time[:-1]
    # set vars
    eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
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
    # tc w/ dark time
    track_length_dt = 550 # cm estimate based on 99.9% of ypos
    track_length_rad_dt = track_length_dt*(2*np.pi/track_length_dt) # estimate bin for dark time
    bins_dt=150 
    bin_size_dt=track_length_rad_dt/bins_dt # typically 3 cm binswith ~ 475 track length
    tcs_correct_dt, coms_correct_dt, tcs_fail_dt, coms_fail_dt, ybinned_dt, rad = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,rewsize,ybinned,time,licks,
        Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
        bins=bins_dt)
    bin_size=3
    tcs_correct_abs, coms_correct_abs,_,__ = make_tuning_curves(eps,rewlocs,ybinned,Fc3,trialnum,rewards,forwardvel,rewsize,bin_size)
    coms_correct_abs_relrew = [com-rewlocs[ep] for ep,com in enumerate(coms_correct_abs)]
    coms_correct_abs_relrew=np.array(coms_correct_abs_relrew)
    goal_window = goal_window_cm*(2*np.pi/track_length) # cm converted to rad
    # in old track length coordinates, so its closer to ~40 cm
    goal_cells_dt, com_goal_postrew_dt, perm_dt, rz_perm_dt = get_goal_cells(rz, goal_window, coms_correct_dt, cell_type = 'all')
    # remove empty epochs
    com_goal_postrew_dt=[com for com in com_goal_postrew_dt if len(com)>0]
    # only get consecutive perms
    com_goal_postrew_dt = [xx for ii,xx in enumerate(com_goal_postrew_dt) if perm_dt[ii][0]-perm_dt[ii][1]==-1]
    perm_dt = [xx for ii,xx in enumerate(perm_dt) if xx[0]-xx[1]==-1]
    #only get perms with non zero cells
    # find dropped out cells 
    all_gc = np.concatenate(com_goal_postrew_dt) if len(com_goal_postrew_dt) else []
    ep_dict = {}
    if len(all_gc)>0:
        unique, counts = np.unique(all_gc, return_counts=True)
        # Combine into a dictionary if desired
        freq_dict = dict(zip(unique, counts))
        for ep in range(1,np.max(counts)+1):
            cells_ep = [k for k,v in freq_dict.items() if v==ep]
            # find which epochs this cell is considered reward cell
            rew_perm = [[perm_com for kk,perm_com in enumerate(perm_dt) if cll in com_goal_postrew_dt[kk]] for cll in cells_ep]
            rew_perm = [[list(yy) for yy in xx] for xx in rew_perm]
            rew_eps = [np.unique(xx) for xx in rew_perm]
            # av across epochs
            coms = [np.nanmean(coms_correct_dt[rew_ep,cells_ep[cll]],axis=0) for cll, rew_ep in enumerate(rew_eps)]
            coms_abs = [np.nanmean(coms_correct_abs_relrew[rew_ep,cells_ep[cll]],axis=0) for cll, rew_ep in enumerate(rew_eps)]
            # also get absoulte/reward subtracted com
            coms = np.array(coms)-np.pi
            ep_dict[ep]=[coms_abs,coms]
    
    return ep_dict


def get_com_v_persistence_place(params_pth, animal, day, ii,goal_window_cm=20):
    print(params_pth)
    fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 'timedFF',
            'putative_pcs', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
            'stat', 'licks'])
    pcs = np.vstack(np.array(fall['putative_pcs'][0]))
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
    time=fall['timedFF'][0]
    if animal=='e145':
        ybinned=ybinned[:-1]
        forwardvel=forwardvel[:-1]
        changeRewLoc=changeRewLoc[:-1]
        trialnum=trialnum[:-1]
        rewards=rewards[:-1]
        licks=licks[:-1]
        time=time[:-1]
    # set vars
    eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
    diff =np.insert(np.diff(eps), 0, 1e15)
    eps=eps[diff>2000]

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
    Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
    dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
    skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
    # looser restrictions
    pc_bool = np.sum(pcs,axis=0)>=1
    Fc3 = Fc3[:,((skew>2)&pc_bool)] # only keep cells with skew greateer than 2
    # if no cells pass these crit
    if Fc3.shape[1]==0:
            Fc3 = fall_fc3['Fc3']
            Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & (~fall['bordercells'][0].astype(bool)))]
            pc_bool = np.sum(pcs,axis=0)>=1
            Fc3 = Fc3[:,((skew>1.2)&pc_bool)]
    bin_size=3 # cm
    # get abs dist tuning 
    tcs_correct_abs, coms_correct_abs,_,__ = make_tuning_curves(eps,rewlocs,ybinned,
    Fc3,trialnum,rewards,forwardvel,
    rewsize,bin_size)
    # get cells that maintain their coms across at least 2 epochs
    place_window = 20 # cm converted to rad                
    perm = list(combinations(range(len(coms_correct_abs)), 2))     
    com_per_ep = np.array([(coms_correct_abs[perm[jj][0]]-coms_correct_abs[perm[jj][1]]) for jj in range(len(perm))])        
    compc = [np.where((comr<place_window) & (comr>-place_window))[0] for comr in com_per_ep]
    # only get consecutive perms
    com_goal_postrew_dt = [xx for ii,xx in enumerate(compc) if perm[ii][0]-perm[ii][1]==-1]
    perm_dt = [xx for ii,xx in enumerate(perm) if xx[0]-xx[1]==-1]
    #only get perms with non zero cells
    # find dropped out cells 
    all_gc = np.concatenate(com_goal_postrew_dt) if len(com_goal_postrew_dt) else []
    ep_dict = {}
    if len(all_gc)>0:
        unique, counts = np.unique(all_gc, return_counts=True)
        # Combine into a dictionary if desired
        freq_dict = dict(zip(unique, counts))
        for ep in range(1,np.max(counts)+1):
            cells_ep = [k for k,v in freq_dict.items() if v==ep]
            # find which epochs this cell is considered reward cell
            rew_perm = [[perm_com for kk,perm_com in enumerate(perm_dt) if cll in com_goal_postrew_dt[kk]] for cll in cells_ep]
            rew_perm = [[list(yy) for yy in xx] for xx in rew_perm]
            rew_eps = [np.unique(xx) for xx in rew_perm]
            # av across epochs
            coms = [np.nanmean(coms_correct_abs[rew_ep,cells_ep[cll]],axis=0) for cll, rew_ep in enumerate(rew_eps)]
            coms = np.array(coms)-np.pi
            ep_dict[ep]=coms
    
    return ep_dict

def allrewardsubtypes(params_pth,animal,day,ii,conddf,num_iterations=1000,bins=90):
    fall = scipy.io.loadmat(params_pth, variable_names=['coms', 'changeRewLoc', 'timedFF',
    'putative_pcs', 'ybinned', 'VR', 'forwardvel', 'trialnum', 'rewards', 'iscell', 'bordercells',
    'stat', 'licks'])
    pcs = np.vstack(np.array(fall['putative_pcs'][0]))
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
    time=fall['timedFF'][0]
    if animal=='e145':
        ybinned=ybinned[:-1]
        forwardvel=forwardvel[:-1]
        changeRewLoc=changeRewLoc[:-1]
        trialnum=trialnum[:-1]
        rewards=rewards[:-1]
        licks=licks[:-1]
        time=time[:-1]
    # set vars
    eps = np.where(changeRewLoc>0)[0];rewlocs = changeRewLoc[eps]/scalingf;eps = np.append(eps, len(changeRewLoc))
    # diff =np.insert(np.diff(eps), 0, 1e15)
    # eps=eps[diff>2000]
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
    rad = get_radian_position_first_lick_after_rew(eps, ybinned, licks, rewards, rewsize,rewlocs,trialnum, track_length) # get radian coordinates
    # added to get anatomical info
    # takes time
    fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
    Fc3 = fall_fc3['Fc3']
    dFF = fall_fc3['dFF']
    Fc3 = Fc3[:, ((fall['iscell'][:,0]).astype(bool) & ~(fall['bordercells'][0].astype(bool)))]
    dFF = dFF[:, ((fall['iscell'][:,0]).astype(bool) & ~(fall['bordercells'][0].astype(bool)))]
    skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
    #if pc in all but 1
    # looser restrictions
    pc_bool = np.sum(pcs,axis=0)>0
    Fc3 = Fc3[:,((skew>1.2)&pc_bool)] # only keep cells with skew greateer than 2
    # tc w/ dark time
    track_length_dt = 550 # cm estimate based on 99.9% of ypos
    track_length_rad_dt = track_length_dt*(2*np.pi/track_length_dt) # estimate bin for dark time
    bins_dt=150 
    bin_size_dt=track_length_rad_dt/bins_dt # typically 3 cm binswith ~ 475 track length
    tcs_correct, coms_correct, tcs_fail, coms_fail, ybinned_dt,raddt = make_tuning_curves_by_trialtype_w_darktime(eps,rewlocs,rewsize,ybinned,time,licks,
        Fc3,trialnum, rewards,forwardvel,scalingf,bin_size_dt,
        bins=bins_dt)
    goal_window = 20*(2*np.pi/track_length) # cm converted to rad
    # change to relative value 
    coms_rewrel = np.array([com-np.pi for com in coms_correct])
    perm = list(combinations(range(len(coms_correct)), 2)) 
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
    ######################### SPLIT INTO NEAR VS. FAR #########################
    bounds = [[-np.pi, -np.pi/4], [-np.pi/4,0], [0,np.pi/4], [np.pi/4, np.pi]]
    celltypes = ['far_pre', 'near_pre', 'near_post', 'far_post']
    ######################### PRE v POST #########################
    goal_cell_null_per_celltype=[]; goal_cell_prop_per_celltype=[]
    for kk,celltype in enumerate(celltypes):
        com_goal = [np.where((comr<goal_window) & (comr>-goal_window))[0] for comr in com_remap]
        print(celltype)
        com_goal_postrew = [[xx for xx in com if ((np.nanmedian(coms_rewrel[:,xx], axis=0)>=bounds[kk][0]) & (np.nanmedian(coms_rewrel[:,xx], axis=0)<bounds[kk][1]))] if len(com)>0 else [] for com in com_goal]
        #only get perms with non zero cells
        com_goal_postrew=[com for com in com_goal_postrew if len(com)>0]         
        if len(com_goal_postrew)>0:
            goal_cells = intersect_arrays(*com_goal_postrew); 
        else:
            goal_cells=[]
        ### test
        for gc in goal_cells:
            plt.figure()
            plt.title(celltype)
            plt.plot(tcs_correct[:,gc].T)
            plt.show()
        # get shuffled iteration
        shuffled_dist = np.zeros((num_iterations))
        # max of 5 epochs = 10 perms
        goal_cell_shuf_ps_per_comp = np.ones((num_iterations,10))*np.nan
        goal_cell_shuf_ps = []
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
            # in addition, com near but after goal
            com_goal_postrew_shuf = [[xx for xx in com if ((np.nanmedian(coms_rewrel[:,
                xx], axis=0)>=bounds[kk][0]) & (np.nanmedian(coms_rewrel[:,
                xx], axis=0)<bounds[kk][1]))] if len(com)>0 else [] for com in com_goal_shuf]
            # check to make sure just a subset
            # otherwise reshuffle
            while not sum([len(xx) for xx in com_goal_shuf])>=sum([len(xx) for xx in com_goal_postrew_shuf]):
                print('redo')
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
                # in addition, com near but after goal
                com_goal_postrew_shuf = [[xx for xx in com if ((np.nanmedian(coms_rewrel[:,
                    xx], axis=0)>=bounds[i][0]) & (np.nanmedian(coms_rewrel[:,
                    xx], axis=0)<bounds[i][1]))] if len(com)>0 else [] for com in com_goal_shuf]

            com_goal_postrew_shuf=[com for com in com_goal_postrew_shuf if len(com)>0]
            goal_cells_shuf_p_per_comparison = [len(xx)/len(coms_correct[0]) for xx in com_goal_postrew_shuf]
            if len(com_goal_postrew_shuf)>0:
                goal_cells_shuf = intersect_arrays(*com_goal_postrew_shuf); 
            else:
                goal_cells_shuf=[]
            shuffled_dist[i] = len(goal_cells_shuf)/len(coms_correct[0])
            goal_cell_shuf_p=len(goal_cells_shuf)/len(com_shufs[0])
            goal_cell_shuf_ps.append(goal_cell_shuf_p)
            goal_cell_shuf_ps_per_comp[i, :len(goal_cells_shuf_p_per_comparison)] = goal_cells_shuf_p_per_comparison

        goal_cell_shuf_ps_per_comp_av = np.nanmean(goal_cell_shuf_ps_per_comp,axis=0)        
        goal_cell_shuf_ps_av = np.nanmean(np.array(goal_cell_shuf_ps))
        goal_cell_p=len(goal_cells)/len(coms_correct[0]) 
        goal_cell_p_per_comp = [len(xx)/len(coms_correct[0]) for xx in com_goal_postrew]
        goal_cell_null_per_celltype.append([goal_cell_shuf_ps_per_comp_av,goal_cell_shuf_ps_av])
        goal_cell_prop_per_celltype.append([goal_cell_p,goal_cell_p_per_comp])
    goal_cell_prop.append(goal_cell_prop_per_celltype)
    goal_cell_null.append(goal_cell_null_per_celltype)
    datadct[f'{animal}_{day:03d}_index{ii:03d}'] = [tcs_correct, coms_correct, tcs_fail, coms_fail,  goal_cell_prop_per_celltype, goal_cell_null_per_celltype]

    return datadct
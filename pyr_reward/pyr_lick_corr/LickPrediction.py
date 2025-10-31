#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
from scipy import stats
import matplotlib.pyplot as plt
import scipy.io
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
from IPython.display import clear_output
np.set_printoptions(threshold=sys.maxsize)

# covert time series data from numpy to tensor
class CreateTimeSeriesData(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

import sys
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.opto.behavior.behavior import smooth_lick_rate

# enter the mouse id and day to get a list of Fc3 signals, dFF signals, lick rate, reward postions.
# interneuron trigger means the filter of skewness. False means skewness > 2
def GetData(mice, day, interneuron = False):
    if day < 10:
        day = '0' + str(day)
    try:
        mat = scipy.io.loadmat('Y:\\analysis\\fmats\\{}\\{}_day0{}_plane0_Fall.mat'.format(mice, mice, day))
    except:
        mat = scipy.io.loadmat('Y:\\analysis\\fmats\\{}\\days\\{}_day0{}_plane0_Fall.mat'.format(mice, mice, day))
    # print(mat.keys())
    # print(mat.keys())
    y = mat['ybinned'][0]
    VR = mat['VR'][0][0][()]
    scalingf = VR['scalingFACTOR'][0][0]
    try:
        rewsize = VR['settings']['rewardZone'][0][0][0][0]  
    except:
        rewsize = 10

    licks = np.hstack(mat['licks'])
    time = np.hstack(mat['timedFF'])
    dt=np.nanmedian(np.diff(time))
    licks = smooth_lick_rate(licks,dt)
    # the index of trials in each epoch
    Trialnum = mat['trialnum'][0]
    # The reward position in the track
    RewardPos = mat['changeRewLoc'][0]
    EpochPos = np.where(RewardPos > 0)[0]
    # start of rew loc
    reward_loc = RewardPos[EpochPos]-(rewsize/2)
    R = mat['rewards'][0]
    Sigs = np.transpose(mat['Fc3'])
    Sigs_dFF = np.transpose(mat['dFF'])
    skew = scipy.stats.skew(Sigs_dFF, nan_policy='omit', axis=1)
    iscell = mat['iscell']
    # print(skew[(mat['bordercells'][0] < 1) & (mat['iscell'][:,0] > 0)])
    try:
        if not interneuron:
            idx = np.where( (mat['bordercells'][0] < 1) & (mat['iscell'][:,0] > 0) & (skew > 2) )[0]
        else:
            idx = np.where( (mat['bordercells'][0] < 1) & (mat['iscell'][:,0] > 0) )[0]
        # & (skew < 2)
        # idx = np.where( (mat['iscell'][:,0] > 0) )[0]
    except:
        print('No bordercells.')
        idx = np.where( (mat['iscell'][:,0] > 0) & (skew > 2) )[0]
    og_idx = idx.copy()
    NeuronActivity = Sigs[idx,:]
    NeuronActivity_dFF = Sigs_dFF[idx,:]
    idx_new = np.where(abs(NeuronActivity.max(axis = 1)) < 20)[0]

    EpochPos = np.where(mat['changeRewLoc'][0]>0)[0]
    EpochSig = []
    EpochSig_dFF = []
    EpochY = []
    EpochLicks = []
    EpochLicksClass = []
    EpochTrialnum = []
    EpochR = []
    for i in range(len(EpochPos)-1):
        EpochSig.append(NeuronActivity[idx_new,EpochPos[i]:EpochPos[i+1]])
        EpochSig_dFF.append(NeuronActivity_dFF[idx_new,EpochPos[i]:EpochPos[i+1]])
        EpochY.append(y[EpochPos[i]:EpochPos[i+1]])
        EpochLicks.append(licks[EpochPos[i]:EpochPos[i+1]])
        EpochTrialnum.append(Trialnum[EpochPos[i]:EpochPos[i+1]])
        EpochR.append(R[EpochPos[i]:EpochPos[i+1]])
    EpochSig.append(NeuronActivity[idx_new,EpochPos[-1]:])
    EpochSig_dFF.append(NeuronActivity_dFF[idx_new,EpochPos[-1]:])
    EpochY.append(y[EpochPos[-1]:])
    EpochLicks.append(licks[EpochPos[-1]:])
    EpochTrialnum.append(Trialnum[EpochPos[-1]:])
    EpochR.append(R[EpochPos[-1]:])

    EpochTrialSigs = []
    EpochTrialSigs_dFF = []
    EpochTrialY = []
    EpochTrialLicks = []
    EpochTrialLicksClass = []
    EpochTrialR = []
    
    for epochs, epochs_dff, epochy, epocht, epochr, epochl in zip(EpochSig, EpochSig_dFF, EpochY, EpochTrialnum, EpochR, EpochLicks):
        temp_sig = []
        temp_sig_dFF = []
        temp_y = []
        temp_r = []
        temp_l = []
        for i in range(epocht[:-10].min(),epocht[10:].max()+1):
            if i >= 0:
                idx = np.where(epocht == i)[0]
                temp_sig.append(epochs[:, idx[0]:idx[-1]+1])
                temp_sig_dFF.append(epochs_dff[:, idx[0]:idx[-1]+1])
                temp_y.append(epochy[idx[0]:idx[-1]+1])
                temp_l.append(epochl[idx[0]:idx[-1]+1])
                if i < 3:
                    temp_r.append(-1)
                elif 1 in epochr[idx[0]:idx[-1]+1]:
                    temp_r.append(1)
                else:
                    temp_r.append(0)
            
        EpochTrialSigs.append(temp_sig)
        EpochTrialSigs_dFF.append(temp_sig_dFF)
        EpochTrialY.append(temp_y)
        EpochTrialLicks.append(temp_l)
        EpochTrialR.append(temp_r)

    return EpochTrialSigs, EpochTrialSigs_dFF, EpochTrialY, EpochTrialLicks, EpochTrialR, reward_loc, og_idx


# Define the neuron type from the feature table created by Zahra.
# Return RR index, PP index, others in terms of spatially tuned neurons, original index gives all spatially tuned indices from
# all neurons. total_num is the total number of neurons in the feature table of this animal and this day.
# the total number should match the neuron number after skewness filter in that day, i.e., Fc3[0][0].shape[0]
def FindNeuronType(cell_features, m, d, e1, e2):
    rr_com1 = cell_features['reward_relative_circular_com']\
    [(cell_features['animal'] == m) & (cell_features['recording_day'] == d) \
     & (cell_features['epoch'] == e1)].to_numpy()
#         rr_com1 = rr_com1[abs(temp) > np.pi/8]

    rr_com2 = cell_features['reward_relative_circular_com']\
    [(cell_features['animal'] == m) & (cell_features['recording_day'] == d) \
     & (cell_features['epoch'] == e2)].to_numpy()
#         rr_com2 = rr_com2[abs(temp) > np.pi/8]

    ac_com1 = cell_features['allocentric_com']\
    [(cell_features['animal'] == m) & (cell_features['recording_day'] == d) \
     & (cell_features['epoch'] == e1)].to_numpy()
    temp = rr_com1.copy()
#         ac_com1 = ac_com1[abs(temp) > np.pi/8]

    ac_com2 = cell_features['allocentric_com']\
    [(cell_features['animal'] == m) & (cell_features['recording_day'] == d) \
     & (cell_features['epoch'] == e2)].to_numpy()
    
#     RR_List = np.where(abs(rr_com1 - rr_com2)/2/np.pi < 0.05)[0]
#     PP_List = np.where(abs(ac_com1 - ac_com2)/2/135 < 0.05)[0]
    # RR_List = np.where(abs(rr_com1 - rr_com2)/2/np.pi < 20/270)[0]
    # only pre-reward
    prerew = np.array([np.nanmedian([xx,rr_com2[ii]]) for ii,xx in enumerate(rr_com1)])
    RR_List = np.where((abs(rr_com1 - rr_com2)/2/np.pi < 20/270) & (prerew<0) & (prerew>-np.pi/4))[0]
    
    PP_List = np.where(abs(ac_com1 - ac_com2)/2/135 < 20/270)[0]
    Null_List = np.array([i for i in range(len(rr_com1)) if (i not in RR_List) and (i not in PP_List)])

    temp = np.where((cell_features['animal'] == m) & (cell_features['recording_day'] == d) & (cell_features['epoch'] == e1))[0]
    # print(temp.shape)
    Ori_idx = np.arange(len(cell_features['spatially_tuned'][temp]))
    
    return RR_List, PP_List, Null_List, Ori_idx, temp.shape[0]

# change path to read your feature table
# epoch index in feature table is from 1 to n, epoch index in python is from 0 to n-1.
cell_features = pd.read_csv(r"\\storage1.ris.wustl.edu\ebhan\Active\dzahra\cell_features_pyr_goal.csv")

# Create the data for training, validating and testing. 
# input are dFF signals, Y position, licks, trial type, and epoch id for training.
# only use the pre-reward signals for training, validation, and testing. training and validation use successful trials, test uses failure trials.
# outputs are tensors of training/validation/testing data, labels.
def CreateData(Fc3, Y, L, Type, rl, epoch_num, shuffle = False):
    success_trial_id = [i for i in range(len(Type[epoch_num])) if Type[epoch_num][i] == 1]
    failure_trial_id = [i for i in range(len(Type[epoch_num])) if Type[epoch_num][i] == 0]
    print(success_trial_id)
    print(failure_trial_id)
    # EpochX = Fc3[0]
    EpochX = Fc3[epoch_num]
    # EpochX = loaded_data2[epoch_num]
    EpochY = Y[epoch_num]
    EpochL = L[epoch_num]
    # print(len(EpochX))
    
    TrainingTrial = []
    TrainingTrial = []
    ValidateTrial = []
    for i in range(0,len(success_trial_id)-1):
      if (i+2) % 3 == 0:
        ValidateTrial.append(success_trial_id[i])
      else:
        TrainingTrial.append(success_trial_id[i])
    # TestingTrial = TrainingTrial.copy()
    TestingTrial = failure_trial_id.copy()
    if len(failure_trial_id) == 0 or len(success_trial_id) == 0:
        return [], [], [], [], [], []
    # if max(failure_trial_id) < min(success_trial_id):
    #     return [], [], [], [], [], []
    # else:
    #     TestingTrial = [i for i in failure_trial_id if i > min(success_trial_id)]
    print(TestingTrial)
    # remove licks after rewloc pass
    y = EpochY[TrainingTrial[0]]
    num_per_bin, bins = np.histogram(y, np.arange(0,180,2))
    tmp_idx = np.where(num_per_bin > 10)[0]
    tmp_tr = set(np.arange(0,y.shape[0]))
    for bin_idx in tmp_idx:
        tmp_tr = tmp_tr & set(np.where( (EpochY[TrainingTrial[0]] < bins[bin_idx]) | (EpochY[TrainingTrial[0]] > bins[bin_idx+1]) )[0])
    tmp_tr = tmp_tr & set(np.where( (EpochY[TrainingTrial[0]] < rl[epoch_num]) )[0])
    tmp_tr = np.array(list(tmp_tr))
    tmp_tr = np.sort(tmp_tr)
    TrainingData = EpochX[TrainingTrial[0]][:, tmp_tr]
    TrainingLabel = EpochL[TrainingTrial[0]][tmp_tr].reshape(-1,1)

    y = EpochY[TestingTrial[0]]
    num_per_bin, bins = np.histogram(y, np.arange(0,180,2))
    tmp_idx = np.where(num_per_bin > 10)[0]
    tmp_tr = set(np.arange(0,y.shape[0]))
    for bin_idx in tmp_idx:
      tmp_tr = tmp_tr & set(np.where( (EpochY[TestingTrial[0]] < bins[bin_idx]) | (EpochY[TestingTrial[0]] > bins[bin_idx+1]) )[0])
    tmp_tr = tmp_tr & set(np.where( (EpochY[TestingTrial[0]] < rl[epoch_num]) )[0])
    tmp_tr = np.array(list(tmp_tr))
    tmp_tr = np.sort(tmp_tr)
    TestingData = EpochX[TestingTrial[0]][:, tmp_tr]
    TestingLabel = EpochL[TestingTrial[0]][tmp_tr].reshape(-1,1)
    # y = EpochY[TrainingTrial[0]]
    # tmp_tr = np.where(y < 3)[0]
    # TestingData = EpochX[TrainingTrial[0]][:, tmp_tr]
    # TestingLabel = EpochY[TrainingTrial[0]][tmp_tr].reshape(-1,1)
    
    y = EpochY[ValidateTrial[0]]
    num_per_bin, bins = np.histogram(y, np.arange(0,180,2))
    tmp_idx = np.where(num_per_bin > 10)[0]
    tmp_tr = set(np.arange(0,y.shape[0]))
    for bin_idx in tmp_idx:
      tmp_tr = tmp_tr & set(np.where( (EpochY[ValidateTrial[0]] < bins[bin_idx]) | (EpochY[ValidateTrial[0]] > bins[bin_idx+1]) )[0])
    tmp_tr = tmp_tr & set(np.where( (EpochY[ValidateTrial[0]] < rl[epoch_num]) )[0])
    tmp_tr = np.array(list(tmp_tr))
    tmp_tr = np.sort(tmp_tr)
    ValidateData = EpochX[ValidateTrial[0]][:, tmp_tr]
    ValidateLabel = EpochL[ValidateTrial[0]][tmp_tr].reshape(-1,1)
    
    for i in TrainingTrial[1:]:
        y = EpochY[i]
        num_per_bin, bins = np.histogram(y, np.arange(0,180,2))
        tmp_idx = np.where(num_per_bin > 10)[0]
        tmp_tr = set(np.arange(0,y.shape[0]))
        for bin_idx in tmp_idx:
            tmp_tr = tmp_tr & set(np.where( (EpochY[i] < bins[bin_idx]) | (EpochY[i] > bins[bin_idx+1]) )[0])
        tmp_tr = tmp_tr & set(np.where( (EpochY[i] < rl[epoch_num]) )[0])
        tmp_tr = np.array(list(tmp_tr))
        tmp_tr = np.sort(tmp_tr)
        # print(EpochX[i].shape, EpochY[i].shape)
        TrainingData = np.concatenate((TrainingData, EpochX[i][:, tmp_tr]), axis=1)
        TrainingLabel = np.concatenate((TrainingLabel, EpochL[i][tmp_tr].reshape(-1,1)), axis=0)
    
    for i in ValidateTrial[1:]:
        y = EpochY[i]
        num_per_bin, bins = np.histogram(y, np.arange(0,180,2))
        tmp_idx = np.where(num_per_bin > 10)[0]
        tmp_tr = set(np.arange(0,y.shape[0]))
        for bin_idx in tmp_idx:
            tmp_tr = tmp_tr & set(np.where( (EpochY[i] < bins[bin_idx]) | (EpochY[i] > bins[bin_idx+1]) )[0])
        tmp_tr = tmp_tr & set(np.where( (EpochY[i] < rl[epoch_num]) )[0])
        tmp_tr = np.array(list(tmp_tr))
        tmp_tr = np.sort(tmp_tr)
        ValidateData = np.concatenate((ValidateData, EpochX[i][:, tmp_tr]), axis=1)
        ValidateLabel = np.concatenate((ValidateLabel, EpochL[i][tmp_tr].reshape(-1,1)), axis=0)
    
    for i in TestingTrial[1:]:
    # for i in TrainingTrial[1:]:
        y = EpochY[i]
        # tmp_tr = np.where(y < 3)[0]
        num_per_bin, bins = np.histogram(y, np.arange(0,180,2))
        tmp_idx = np.where(num_per_bin > 10)[0]
        tmp_tr = set(np.arange(0,y.shape[0]))
        for bin_idx in tmp_idx:
            tmp_tr = tmp_tr & set(np.where( (EpochY[i] < bins[bin_idx]) | (EpochY[i] > bins[bin_idx+1]) )[0])
        tmp_tr = tmp_tr & set(np.where( (EpochY[i] < rl[epoch_num]) )[0])
        tmp_tr = np.array(list(tmp_tr))
        tmp_tr = np.sort(tmp_tr).astype(int)
        TestingData = np.concatenate((TestingData, EpochX[i][:, tmp_tr]), axis=1)
        TestingLabel = np.concatenate((TestingLabel, EpochL[i][tmp_tr].reshape(-1,1)), axis=0)
    
    th = 0.01
    TrainingData = torch.tensor(TrainingData).t().float()
    TrainingLabel = torch.tensor(TrainingLabel).float()
    TestingData = torch.tensor(TestingData).t().float()
    TestingLabel = torch.tensor(TestingLabel).float()
    ValidateData = torch.tensor(ValidateData).t().float()
    ValidateLabel = torch.tensor(ValidateLabel).float()
    # TrainingData, TrainingLabel = RemoveNan(TrainingData, TrainingLabel)
    # TestingData, TestingLabel = RemoveNan(TestingData, TestingLabel)
    # ValidateData, ValidateLabel = RemoveNan(ValidateData, ValidateLabel)
    indices = torch.randperm(TrainingLabel.size(0))
    if shuffle:
        TrainingLabel = TrainingLabel[indices]
    TrainingData[TrainingData < th] = 0
    TestingData[TestingData < th] = 0
    ValidateData[ValidateData < th] = 0
    print(TrainingData.shape, TrainingLabel.shape)
    print(TestingData.shape, TestingLabel.shape)
    print(ValidateData.shape, ValidateLabel.shape)

    return TrainingData, TrainingLabel, TestingData, TestingLabel, ValidateData, ValidateLabel


# Convert tensor to data loader
def Data2Dataloader(TrainingData, TrainingLabel, TestingData, TestingLabel, ValidateData, ValidateLabel, batch_size = 64):
    Train_dataset = CreateTimeSeriesData(TrainingData, TrainingLabel)
    Train_loader = DataLoader(dataset=Train_dataset, batch_size=batch_size, shuffle=True, drop_last = True)
    
    Test_dataset = CreateTimeSeriesData(TestingData, TestingLabel)
    Test_loader = DataLoader(dataset=Test_dataset, batch_size=batch_size, shuffle=False)
    
    Validate_dataset = CreateTimeSeriesData(ValidateData, ValidateLabel)
    Validate_loader = DataLoader(dataset=Validate_dataset, batch_size=batch_size, shuffle=False, drop_last = False)
    return Train_loader, Test_loader, Validate_loader

# Training process
def TrainModel(model, optimizer, device, Train_loader, Validate_loader, mse_loss, num_epochs = 1000):
    l = []
    val_l = []
    
    for epoch in range(num_epochs):
        train_loss = 0.0
        for inputs, targets in Train_loader:
            # Forward pass
            inputs, targets = inputs.to(device), targets.to(device)
            # inputs, targets = inputs.to(device), targets.squeeze(-1).to(torch.int64).to(device)
            outputs = model(inputs)
            # loss = ce_loss(outputs, targets)
            # mse = mae_loss(outputs, targets)
            loss = mse_loss(outputs, targets)
            # kl = kl_loss(model)
            # loss = ce + kl_weight*kl
    
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
    
        l.append(train_loss/len(Train_loader))
        if epoch % 50 == 0:
            print('Epoch [{}/{}], Train Loss: {:.4f}'.format(epoch+1, num_epochs, train_loss/len(Train_loader)))
            test_loss = 0
            for inputs, targets in Validate_loader:
              # Forward pass
              inputs, targets = inputs.to(device), targets.to(device)
              # inputs, targets = inputs.to(device), targets.squeeze(-1).to(torch.int64).to(device)
              outputs = model(inputs)
              loss = mse_loss(outputs, targets)
              # loss = ce_loss(outputs, targets)
              test_loss += loss.item()
            print('Epoch [{}/{}], Test Loss: {:.4f}'.format(epoch+1, num_epochs, test_loss/len(Test_loader)))
    return model
# find top 20 percent of neurons that contribute to the prediction based on the weight.
def top_percent_indices(vector, percent):
    vector = np.asarray(vector)
    n = len(vector)
    k = int(n * percent)
    if k == 0:
        return np.array([])
    indices = np.argpartition(vector, -k)[-k:]
    indices = indices[np.argsort(-vector[indices])]
    return indices
#%%
conddf = pd.read_csv(r"Z:\condition_df\conddf_pyr_goal_cells.csv", index_col=None)
datadct={}
#%%
for ii in range(12,len(conddf)):
    day = conddf.days.values[ii]
    animal = conddf.animals.values[ii]
    optoep=conddf.optoep.values[ii]
    in_type=conddf.in_type.values[ii]
    if (animal!='e217') & (conddf.optoep.values[ii]<2):
        print(animal,day)
        Fc3, X, Y, L, TP, rl, _ = GetData(animal, day, interneuron = False)
        print(X[0][0].shape)
        # Although function uses variable Fc3, real input is dFF signals, X. Fc3 has a much worse performance.
        ep=0
        TrD, TrL, TsD, TsL, VlD, VlL = CreateData(X, Y, L, TP, rl, epoch_num = ep)
        if len(TrD)==0:
            ep=ep+1
            TrD, TrL, TsD, TsL, VlD, VlL = CreateData(X, Y, L, TP, rl, epoch_num = ep)
            if len(TrD)==0:
                pass

        Train_loader, Test_loader, Validate_loader = Data2Dataloader(TrD, TrL, TsD, TsL, VlD, VlL)
        # training example
        input_size = TrD.shape[-1]
        output_size = 1
        # mse loss
        mse_loss = nn.MSELoss()
        # linear model
        model = nn.Sequential(nn.Linear(input_size, output_size), nn.ReLU())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        # TrD, TrL, TsD, TsL, VlD, VlL
        model = TrainModel(model, optimizer, device, Train_loader, Validate_loader, mse_loss, num_epochs = 1000)

        # check validation sets.
        pred_y = np.zeros([1,1])
        truth = np.zeros([1,1])
        l = 0
        rew_loc = rl[ep]/180
        # temp_idx = torch.arange(inputs.cpu().detach().size(1))
        for x,t in Validate_loader: # Test_loader for testing
            # for x,t in Train_loader:
            t = t.to(device)
            # x = x[:,temp_idx].to(device)
            x = x.to(device)
            y = model(x)
            # y =torch.argmax(y, dim = 1).unsqueeze(-1)
            pred_y = np.concatenate((pred_y,y.cpu().detach().numpy()), axis = 0)
            truth = np.concatenate((truth,t.cpu().detach().numpy()), axis = 0)
            loss = mse_loss(t, y)
            l += loss.item()
        plt.figure(figsize=(10, 3), dpi=80)
        plt.plot(truth[1:], label = 'truth')
        plt.plot(pred_y[1:], alpha = 0.8, label = 'predict')
        # plt.ylim(-0.05,1.05)
        plt.legend()
        plt.show()
        # get weights of linear model
        weights = model.state_dict()['0.weight'].detach().cpu().numpy()
        # plt.figure(figsize = (10, 2))
        # plt.imshow(weights, aspect = 'auto', interpolation = 'None')
        # plt.yticks([0])
        # check the number of RR, PP, Other cells in the lick cells.
        n_r = 0 # rr
        n_p = 0 # pp
        n_n = 0 # others
        weights = model.state_dict()['0.weight'].detach().cpu().numpy()
        top_20_percent_indices = top_percent_indices(weights[0], 0.2)
        cell_features = pd.read_csv(r"\\storage1.ris.wustl.edu\ebhan\Active\dzahra\cell_features_pyr_goal.csv")
        if ep == len(Fc3)-1:
            RR_List, PP_List, Null_List, Ori_idx, total_num = FindNeuronType (cell_features, animal, day, ep, ep+1)
        else:
            RR_List, PP_List, Null_List, Ori_idx, total_num = FindNeuronType (cell_features, animal, day, ep+1, ep+2)

        # for i in np.where(weights[0] > 0.025)[0]:
        for i in top_20_percent_indices:
            if i in Ori_idx[RR_List]:
                n_r += 1
                # print('{} in RR List'.format(i))
            if i in Ori_idx[PP_List]:
                n_p += 1
                # print('{} in PP List'.format(i))
            if i in Ori_idx[Null_List]:
                n_n += 1
                # print('{} in Null List'.format(i))
        # print(top_20_percent_indices.shape, RR_List.shape, PP_List.shape, n_r, n_p, n_n)
        print(f'####### lick contributing cells #######\nreward cell %: {n_r/top_20_percent_indices.shape[0]:.2g} \n place cell %: {n_p/top_20_percent_indices.shape[0]:.2g} \n other cell %: {n_n/top_20_percent_indices.shape[0]:.2g}')
        print(f'####### population enrichment #######\nreward cell %: {RR_List.shape[0]/Ori_idx.shape[0]:.2g} \n place cell %: {PP_List.shape[0]/Ori_idx.shape[0]:.2g} \n other cell %: {Null_List.shape[0]/Ori_idx.shape[0]:.2g}')
        print(f'####### enrichment-lick contribution #######\nreward cell %: {(n_r/top_20_percent_indices.shape[0])-(RR_List.shape[0]/Ori_idx.shape[0]):.2g} \n place cell %: {(n_p/top_20_percent_indices.shape[0])-(PP_List.shape[0]/Ori_idx.shape[0]):.2g} \n other cell %: {(n_n/top_20_percent_indices.shape[0])-(Null_List.shape[0]/Ori_idx.shape[0]):.2g}')
        
        datadct[f'{animal}_{day}']=(n_r/top_20_percent_indices.shape[0]),(RR_List.shape[0]/Ori_idx.shape[0]),(n_p/top_20_percent_indices.shape[0]),(PP_List.shape[0]/Ori_idx.shape[0]),(n_n/top_20_percent_indices.shape[0]),(Null_List.shape[0]/Ori_idx.shape[0])

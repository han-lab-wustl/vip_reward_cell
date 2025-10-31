"""ensemble functions
"""


from sklearn.decomposition import PCA, FastICA
import matplotlib.pyplot as plt, scipy
from sklearn.cluster import KMeans
import numpy as np, sys
from collections import Counter
from itertools import combinations, chain
from sklearn.preprocessing import StandardScaler
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
from projects.pyr_reward.placecell import make_tuning_curves_by_trialtype_w_darktime, intersect_arrays, make_tuning_curves
from projects.pyr_reward.rewardcell import get_radian_position,create_mask_from_coordinates,pairwise_distances,extract_data_rewcentric,\
    get_radian_position_first_lick_after_rew, get_rewzones
from projects.opto.behavior.behavior import get_success_failure_trials

from sklearn.metrics.pairwise import euclidean_distances
def elbow_method(ica_components, max_clusters=10):
    """
    Applies the elbow method to determine the optimal number of clusters (assemblies) for KMeans.
    
    Parameters:
    - ica_components (np.ndarray): ICA components (neurons x time_bins)
    - max_clusters (int): Maximum number of clusters to test
    
    Returns:
    - optimal_n_clusters (int): Optimal number of clusters based on the elbow method
    """
    wcss = []  # List to store WCSS for each number of clusters
    
    # Iterate over the range of clusters to calculate WCSS
    for n_clusters in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(ica_components.T)  # Fit KMeans on the transposed ICA components
        wcss.append(kmeans.inertia_)  # Inertia is the WCSS
    
    # Plot the WCSS to visualize the elbow
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_clusters + 1), wcss, marker='o', linestyle='-', color='b')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
    plt.show()
    
    # Determine the elbow point (optimal number of clusters)
    # This is a heuristic based on the "elbow" in the curve
    optimal_n_clusters = np.diff(wcss).argmin() + 2  # +2 because np.diff reduces the length by 1
    return optimal_n_clusters


def get_cells_by_assembly(labels):
    """
    Groups cell indices by their assigned assembly labels.
    
    Parameters:
    - labels (np.ndarray): Array of assembly membership for each neuron
    
    Returns:
    - assembly_cells (dict): Keys are assembly IDs, values are lists of neuron indices
    """
    assembly_cells = {}
    for idx, label in enumerate(labels):
        if label not in assembly_cells:
            assembly_cells[label] = []
        assembly_cells[label].append(idx)
    return assembly_cells

def cluster_neurons_from_ica(ica_components):
    """
    Clusters neurons based on ICA components using KMeans with optimal number of clusters
    determined by the elbow method.
    
    Parameters:
    - ica_components (np.ndarray): ICA components (neurons x time_bins)
    
    Returns:
    - labels (np.ndarray): Cluster labels for each neuron
    """
    # Apply the elbow method to find the optimal number of clusters
    n_assemblies = elbow_method(ica_components, max_clusters=10)
    print(f"Optimal number of assemblies (clusters) detected: {n_assemblies}")
    
    # Fit KMeans with the optimal number of clusters
    kmeans = KMeans(n_clusters=n_assemblies)
    kmeans.fit(ica_components.T)  # Transpose if necessary to have neurons as features
    
    return kmeans.labels_


def detect_assemblies_with_ica(spike_matrix, zscore=True, plot=False):
    """
    Detect cell assemblies using PCA + ICA with MP-based estimation of number of assemblies.

    Parameters:
    - spike_matrix (np.ndarray): neurons x time_bins
    - zscore (bool): Whether to z-score the spike matrix before analysis
    - plot (bool): Whether to plot results (not implemented here)

    Returns:
    - ica_components (np.ndarray): independent components, shape: (n_assemblies, neurons)
    - assembly_activities (np.ndarray): time courses of each assembly (n_assemblies x time_bins)
    - labels (np.ndarray): cluster labels for each neuron (neurons,)
    - n_assemblies (int): number of assemblies detected
    """
    n_neurons, n_time_bins = spike_matrix.shape

    # Optional z-scoring
    if zscore:
        mean = spike_matrix.mean(axis=1, keepdims=True)
        std = spike_matrix.std(axis=1, keepdims=True)
        std[std == 0] = 1
        Z = (spike_matrix - mean) / std
        Z = np.nan_to_num(Z)
    else:
        Z = spike_matrix

    # PCA
    pca = PCA()
    pcs = pca.fit_transform(Z.T)
    components = pca.components_
    eigenvalues = pca.explained_variance_

    # Marchenko–Pastur threshold
    q = n_time_bins / n_neurons
    lambda_max = (1 + 1 / np.sqrt(q))**2
    significant_idx = np.where(eigenvalues > lambda_max)[0]
    n_assemblies = len(significant_idx)

    if n_assemblies == 0:
        print("No significant components found using MP threshold.")
        return None, None, None, 0

    # Project onto significant PCA components
    Z_reduced = pcs[:, significant_idx]

    # ICA
    ica = FastICA(n_components=n_assemblies, random_state=0)
    ica_activations = ica.fit_transform(Z_reduced)  # shape: (time_bins, n_assemblies)
    ica_components = ica.components_  # shape: (n_assemblies, n_significant_PCs)
    
    # Backproject ICA components to neuron space
    assembly_patterns = np.dot(ica_components, components[significant_idx])

    # Assembly activity over time
    assembly_activities = np.dot(assembly_patterns, Z)

    # Cluster neurons based on ICA weights
    kmeans = KMeans(n_clusters=n_assemblies, n_init=10, random_state=0)
    labels = kmeans.fit_predict(assembly_patterns.T)

    return assembly_patterns, assembly_activities, labels, n_assemblies

def detect_cell_assemblies_AV(spike_matrix, plot=True,
                force_n_components=True):
    """
    Detects cell assemblies using the Assembly Vector (AV) method.
    
    Parameters:
    - spike_matrix (np.ndarray): neurons x time_bins
    - plot (bool): Whether to plot diagnostics

    Returns:
    - assembly_vectors (np.ndarray): shape (n_assemblies, n_neurons)
    - assemblies (dict): assembly_id -> list of neuron indices
    """
    n_neurons, n_time_bins = spike_matrix.shape

    # Z-score the spike matrix
    Z = (spike_matrix - spike_matrix.mean(axis=1, keepdims=True)) / spike_matrix.std(axis=1, keepdims=True)
    Z = np.nan_to_num(Z)

    # Correlation matrix
    C = np.corrcoef(Z)

    # PCA on the correlation matrix
    pca = PCA()
    pca.fit(C)
    eigenvalues = pca.explained_variance_
    pcs = pca.components_

    # Determine significant components via Marchenko–Pastur threshold
    q = n_neurons / n_time_bins
    lambda_max = (1 + 1 / np.sqrt(q)) ** 2
    significant_indices = np.where(eigenvalues > lambda_max)[0]
    P_sig = pcs[significant_indices, :]  # shape: (n_significant, n_neurons)
    if len(significant_indices) == 0 and force_n_components:
        print(f"Forcing {force_n_components} components.")
        significant_indices = np.arange(force_n_components)

    # Project each column of C into the assembly space
    PAS = P_sig.T @ P_sig  # Projection matrix (n_neurons x n_neurons)
    neuron_vectors = PAS @ C  # shape: (n_neurons, n_neurons)

    # Compute interaction matrix
    interaction_matrix = neuron_vectors @ neuron_vectors.T

    # Binarize using k-means
    kmeans = KMeans(n_clusters=2, random_state=0).fit(interaction_matrix.flatten()[:, None])
    threshold = np.mean([interaction_matrix.flatten()[kmeans.labels_ == 0].mean(),
                         interaction_matrix.flatten()[kmeans.labels_ == 1].mean()])
    binary_matrix = (interaction_matrix > threshold).astype(int)

    # Cluster neurons using the binary interaction matrix
    kmeans = KMeans(n_clusters=len(significant_indices), random_state=0).fit(binary_matrix)
    labels = kmeans.labels_

    # Group neurons into assemblies
    assemblies = {}
    for i in range(len(significant_indices)):
        assemblies[f'Assembly_{i+1}'] = np.where(labels == i)[0]

    # Compute AVs as mean of neuron vectors in each group
    assembly_vectors = []
    for idxs in assemblies.values():
        AV = neuron_vectors[idxs].mean(axis=0)
        AV /= np.linalg.norm(AV)  # normalize
        assembly_vectors.append(AV)
    assembly_vectors = np.array(assembly_vectors)

    if plot:
        plt.figure(figsize=(6, 5))
        plt.imshow(interaction_matrix, cmap='viridis')
        plt.title("Interaction Matrix")
        plt.colorbar(label="Inner Product")
        plt.tight_layout()
        plt.show()

    return assembly_vectors, assemblies

# pca
def detect_cell_assemblies(spike_matrix, significance_threshold=None, plot=True):
    """
    Detects cell assemblies using PCA and computes their activity over time.

    Parameters:
    - spike_matrix (np.ndarray): neurons x time_bins, z-scored spike data
    - significance_threshold (float): If None, uses Marchenko–Pastur threshold
    - plot (bool): Whether to plot PCA components and assembly activity

    Returns:
    - significant_components (list): List of significant PCA components (indices)
    - assembly_patterns (np.ndarray): significant PCs, shape: (neurons, n_significant)
    - assembly_activities (np.ndarray): time courses of each assembly (n_significant x time_bins)
    """

    n_neurons, n_time_bins = spike_matrix.shape

    # Robust z-scoring
    mean_activity = spike_matrix.mean(axis=1, keepdims=True)
    std_activity = spike_matrix.std(axis=1, keepdims=True)
    std_activity[std_activity == 0] = 1  # prevent division by zero

    Z = (spike_matrix - mean_activity) / std_activity
    Z = np.nan_to_num(Z, nan=0.0)  # convert NaNs to zero

    # PCA
    pca = PCA()
    pcs = pca.fit_transform(Z.T)  # shape: (time_bins, components)
    components = pca.components_  # shape: (n_components, n_neurons)
    eigenvalues = pca.explained_variance_

    # Marchenko–Pastur threshold
    if significance_threshold is None:
        q = n_time_bins / n_neurons
        lambda_max = (1 + (1 / np.sqrt(q))) ** 2
        significant_components = np.where(eigenvalues > lambda_max)[0]
    else:
        significant_components = np.where(eigenvalues > significance_threshold)[0]

    assembly_patterns = components[significant_components]  # shape: (n_significant, n_neurons)

    # Normalize each pattern and compute projection matrix with zeroed diagonals
    projection_matrices = []
    for w in assembly_patterns:
        w = w / np.linalg.norm(w)
        P = np.outer(w, w)
        np.fill_diagonal(P, 0)
        projection_matrices.append(P)

    # Compute assembly activity over time
    assembly_activities = []
    for P in projection_matrices:
        R = np.zeros(n_time_bins)
        for t in range(n_time_bins):
            z_t = Z[:, t]
            R[t] = z_t.T @ P @ z_t
        assembly_activities.append(R)
    assembly_activities = np.array(assembly_activities)

    if plot:
        # Plot eigenvalues
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(eigenvalues, marker='o')
        plt.axhline(y=lambda_max, color='r', linestyle='--', label='MP Threshold')
        plt.title("PCA Eigenvalues")
        plt.xlabel("Component")
        plt.ylabel("Eigenvalue")
        plt.legend()

        # Plot assembly activity
        plt.subplot(1, 2, 2)
        for i, activity in enumerate(assembly_activities):
            plt.plot(activity, label=f"Assembly {i+1}")
        plt.title("Assembly Activity Over Time")
        plt.xlabel("Time Bin")
        plt.ylabel("Activity")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return significant_components, assembly_patterns, assembly_activities

def get_all_ensemble_data(params_pth, animal, day, pdf, bins=90, goal_window_cm=20):
    # Load relevant variables
    fall = scipy.io.loadmat(params_pth, variable_names=[
        'coms', 'changeRewLoc', 'timedFF', 'ybinned', 'VR', 'forwardvel', 'trialnum', 
        'rewards', 'iscell', 'bordercells', 'stat', 'licks'
    ])
    VR = fall['VR'][0][0][()]
    scalingf = VR['scalingFACTOR'][0][0]
    try:
        rewsize = VR['settings']['rewardZone'][0][0][0][0] / scalingf        
    except:
        rewsize = 10
        
    ybinned = fall['ybinned'][0] / scalingf
    track_length = 180 / scalingf    
    forwardvel = fall['forwardvel'][0]    
    changeRewLoc = np.hstack(fall['changeRewLoc'])
    trialnum = fall['trialnum'][0]
    rewards = fall['rewards'][0]
    licks = fall['licks'][0]
    time = fall['timedFF'][0]
    # trim if needed
    if animal == 'e145':
        ybinned = ybinned[:-1]
        forwardvel = forwardvel[:-1]
        changeRewLoc = changeRewLoc[:-1]
        trialnum = trialnum[:-1]
        rewards = rewards[:-1]
        licks = licks[:-1]
        time = time[:-1]

    eps = np.where(changeRewLoc > 0)[0]
    rewlocs = changeRewLoc[eps] / scalingf
    eps = np.append(eps, len(changeRewLoc))
    rz = get_rewzones(rewlocs, 1 / scalingf)

    # Load and filter dF/F data
    fall_fc3 = scipy.io.loadmat(params_pth, variable_names=['Fc3', 'dFF'])
    Fc3 = fall_fc3['Fc3']
    dFF = fall_fc3['dFF']
    Fc3 = Fc3[:, (fall['iscell'][:, 0]).astype(bool)]
    dFF = dFF[:, (fall['iscell'][:, 0]).astype(bool)]
    skew = scipy.stats.skew(dFF, nan_policy='omit', axis=0)
    Fc3 = Fc3[:, skew > 2]
    # bin size based on dark time estimate
    bins_dt = 150
    track_length_dt = 550
    bin_size_dt = (2 * np.pi) / bins_dt
    # Get tuning curves for each trial type
    ########################## rew cells w/ dt ##########################
    tcs_correct, coms_correct, tcs_fail, coms_fail, ybinned_dt, rad = make_tuning_curves_by_trialtype_w_darktime(
        eps, rewlocs, rewsize, ybinned, time, licks, Fc3, trialnum, rewards, forwardvel, 
        scalingf, bin_size_dt, bins=bins_dt)
    # Define goal window
    goal_window = goal_window_cm * (2 * np.pi / track_length)
    # Recenter COMs around reward zone
    coms_rewrel = np.array([com - np.pi for com in coms_correct])
    perm = list(combinations(range(len(coms_correct)), 2))
    # restrict to consecutive epochs
    perm = [p for p in perm if (p[0]-p[1])<2]
    # Fix wrap-around
    epsilon = 0.7
    com_loop_w_in_window = []
    for pi, p in enumerate(perm):
        for cll in range(coms_rewrel.shape[1]):
            com1_rel = coms_rewrel[p[0], cll]
            com2_rel = coms_rewrel[p[1], cll]
            if ((abs(com1_rel - np.pi) < epsilon) and (abs(com2_rel + np.pi) < epsilon)):
                com_loop_w_in_window.append(cll)
    coms_rewrel[:, com_loop_w_in_window] = abs(coms_rewrel[:, com_loop_w_in_window])
    com_remap = np.array([(coms_rewrel[p[0]] - coms_rewrel[p[1]]) for p in perm])
    com_goal = [np.where((comr < goal_window) & (comr > -goal_window))[0] for comr in com_remap]
    # Separate cells into pre- and post-reward goal cells
    com_goal_prerew = [[xx for xx in com if np.nanmedian(coms_rewrel[:, xx]) < 0] if len(com) > 0 else [] for com in com_goal]
    com_goal_postrew = [[xx for xx in com if np.nanmedian(coms_rewrel[:, xx]) >= 0] if len(com) > 0 else [] for com in com_goal]
    dedicated_pre = intersect_arrays(*com_goal_prerew) if len(com_goal_prerew) > 0 else []
    dedicated_post = intersect_arrays(*com_goal_postrew) if len(com_goal_postrew) > 0 else []
    # use all cells in every epoch
    pre_goal_cells = np.concatenate(com_goal_prerew)
    post_goal_cells = np.concatenate(com_goal_postrew)
    ########################## place cells w/ dt ##########################
    # Place cell detection
    tcs_correct_abs, coms_correct_abs,_,_ = make_tuning_curves(eps, rewlocs, ybinned, Fc3, trialnum, rewards, forwardvel, rewsize, 3)
    place_window = 20
    perm_abs = list(combinations(range(len(coms_correct_abs)), 2))
    # restrict to consecutive epochs
    perm_abs = [p for p in perm_abs if (p[0]-p[1])==-1]
    com_per_ep = np.array([(coms_correct_abs[p[0]] - coms_correct_abs[p[1]]) for p in perm_abs])
    compc = [np.where((comr < place_window) & (comr>-place_window))[0] for comr in com_per_ep]
    dedicated_pcs = intersect_arrays(*compc) if len(compc) > 0 else []
    pcs_all = np.concatenate(compc)
    print(f'# pre rew: {len(pre_goal_cells)}\n\
          # post rew: {len(post_goal_cells)}\n\
            # place: {len(pcs_all)}')
    # Run ICA for each population
    print("\nRunning ICA on pre-reward goal cells...")
    pdf,assemblies_pre, frac_pre = run_ica_and_plot(pre_goal_cells,pre_goal_cells,'pre',Fc3,eps,tcs_correct,tcs_fail,pdf,animal, day)

    print("\nRunning ICA on post-reward goal cells...")
    pdf,assemblies_post, frac_post = run_ica_and_plot(post_goal_cells,post_goal_cells,'post',Fc3,eps,tcs_correct,tcs_fail,pdf,animal, day)

    print("\nRunning ICA on place cells...")
    # not sep into correct/incorr for place cells
    pdf,assemblies_place, frac_place = run_ica_and_plot(pcs_all,pcs_all,'place',Fc3,eps,tcs_correct_abs,np.zeros_like(tcs_correct_abs),pdf,animal, day)

    return {
        'pre': {'assemblies': assemblies_pre, 'fraction': frac_pre},
        'post': {'assemblies': assemblies_post, 'fraction': frac_post},
        'place': {'assemblies': assemblies_place, 'fraction': frac_place}
    }


# ICA function to run and save tuning curve plots
def run_ica_and_plot(cell_ids,dedicated_cells,label,Fc3,eps,tcs_correct,tcs_fail,pdf,animal, day):
    assemblies = {}; used_cells = set()
    try:
        if len(cell_ids) > 0:
            # all epochs v 1 epoch
            # eps[0]:eps[1]
            patterns, activities, labels_, n = detect_assemblies_with_ica(Fc3[:, cell_ids].T)
            print(f"{label}: {n} assemblies detected")
            labels_clusters = cluster_neurons_from_ica(patterns)
            cell_groups = get_cells_by_assembly(labels_clusters)
            for assembly_id, cells in sorted(cell_groups.items(), key=lambda x: len(x[1]), reverse=True):
                cells = np.array(cells)
                # leave max filter out
                # peak = np.nanmax(tcs_correct[0, cell_ids[cells], :], axis=1)
                # cells = cells[peak > 0.05]
                if len(cells) < 5:
                    continue
                ids = set(cell_ids[cells])
                if not ids.isdisjoint(used_cells):
                    continue
                used_cells.update(ids)
                fig, ax = plt.subplots()
                ax.plot(tcs_correct[0, cell_ids[cells], :].T)
                ax.set_title(f'{animal}, {day}, {label} Assembly {assembly_id}')
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
                assemblies[f'{label}_assembly_{assembly_id}'] = [
                    tcs_correct[:, cell_ids[cells], :], 
                    tcs_fail[:, cell_ids[cells], :]
                ]
    except Exception as e:
        print(e)
    return pdf, assemblies, len(used_cells) / len(dedicated_cells) if len(dedicated_cells) > 0 else 0

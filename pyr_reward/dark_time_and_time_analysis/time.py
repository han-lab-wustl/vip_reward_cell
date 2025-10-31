import numpy as np

def filter_cells_by_field_selectivity(tuning_curves, threshold=0.5, fraction_cutoff=0.5):
    """
    Filter cells based on whether most of their activity is in-field.
    
    Parameters:
        tuning_curves: np.array of shape (epochs, cells, bins)
        threshold: float, fraction of max to define in-field
        fraction_cutoff: float, minimum fraction of total activity that must be in-field
    
    Returns:
        selected_cells: list of cell indices that pass the filter
        in_field_fraction: (epochs x cells) array of in-field firing fractions
    """
    n_epochs, n_cells, n_bins = tuning_curves.shape
    in_field_fraction = np.full((n_epochs, n_cells), np.nan)

    for ep in range(n_epochs):
        for cell in range(n_cells):
            tc = tuning_curves[ep, cell, :]
            if np.isnan(tc).all():
                continue
            max_val = np.nanmax(tc)
            if max_val == 0:
                continue
            in_field_mask = tc >= threshold * max_val
            total_sum = np.nansum(tc)
            in_field_sum = np.nansum(tc[in_field_mask])
            if total_sum > 0:
                in_field_fraction[ep, cell] = in_field_sum / total_sum

    # Average across epochs (or use another rule)
    avg_in_field_fraction = np.nanmean(in_field_fraction, axis=0)
    selected_cells = np.where(avg_in_field_fraction >= fraction_cutoff)[0]

    return selected_cells, in_field_fraction

__all__ = ["regression_metrics"]

import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error

def regression_metrics(y_true, y_pred, metrics=["mae"], target_names=None):
    """
    metrics: list of metrics to compute ["mae", "r2"]. If None, computes all.
    target_names: list of names for each regression target
    """
    if metrics is None:
        metrics = ["mae", "r2"]
    
    n_targets = y_true.shape[1]
    results = {}
    
    if target_names is None:
        target_names = [str(i) for i in range(n_targets)]
    elif len(target_names) != n_targets:
        raise ValueError(f"target_names length {len(target_names)} must match y_true shape {n_targets}")
    
    for i in range(n_targets):
        y_true_i = y_true[:, i]
        y_pred_i = y_pred[:, i]
        
        mask_i = ~np.isnan(y_true_i)
        y_true_clean = y_true_i[mask_i]
        y_pred_clean = y_pred_i[mask_i]
        
        if len(y_true_clean) > 0:
            if "mae" in metrics:
                results[f"{target_names[i]}_mae"] = mean_absolute_error(y_true_clean, y_pred_clean)
            if "r2" in metrics:
                results[f"{target_names[i]}_r2"] = r2_score(y_true_clean, y_pred_clean)
        else:
            if "mae" in metrics:
                results[f"{target_names[i]}_mae"] = np.nan
            if "r2" in metrics:
                results[f"{target_names[i]}_r2"] = np.nan
    
    # Global metrics (unchanged)
    y_true_flat = y_true.ravel()
    y_pred_flat = y_pred.ravel()
    mask_flat = ~np.isnan(y_true_flat)
    y_true_clean = y_true_flat[mask_flat]
    y_pred_clean = y_pred_flat[mask_flat]
    
    if len(y_true_clean) > 0:
        if "mae" in metrics:
            results["mae"] = mean_absolute_error(y_true_clean, y_pred_clean)
        if "r2" in metrics:
            results["r2"] = r2_score(y_true_clean, y_pred_clean)
    else:
        if "mae" in metrics:
            results["mae"] = np.nan
        if "r2" in metrics:
            results["r2"] = np.nan    
    
    return results
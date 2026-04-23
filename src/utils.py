"""Utility functions shared among all training modes."""


import time
import torch
import torch.nn as nn
from tqdm import tqdm
import logging
from typing import Dict, Any
import numpy as np
import pandas as pd


# Import metrics from lifelines and scikit-survival
from lifelines.utils import concordance_index
from sksurv.metrics import brier_score, cumulative_dynamic_auc
from sklearn.metrics import roc_auc_score
from lifelines import CoxPHFitter
from scipy.interpolate import interp1d



def evaluate_rsf(model, data, client_id, config):
    """
    RSF evaluation with PAPER-STYLE C-index
    (matches FedSurF original implementation).

    C-index:
      - dense linspace time grid
      - cumulative log-risk
      - concordance_index_censored

    NOTE:
      AUC / IBS here are NOT paper-style and are optional.
    """

    import logging
    import numpy as np
    from scipy.interpolate import interp1d
    from sksurv.metrics import (
        concordance_index_censored,
        concordance_index_ipcw,
        cumulative_dynamic_auc,
        brier_score,
    )

    logger = logging.getLogger("main")
    logger.info(f"[Client {client_id}] Evaluating RSF (paper-style C-index)...")


    # Unpack the data
    X_test  = data["X_test"]
    y_test  = data["y_test"]
    y_train = data["y_train"]


    # 1) FedSurF prediction (tree-averaged survival curves)
    surv_fns = model.predict_survival_function_fedsurf(X_test)
    # surv_fns[i] = (times_i, surv_probs_i)


    # 2) PAPER-STYLE TIME GRID
    sorted_train_times = np.sort(np.unique(y_train["time"]))
    sorted_test_times  = np.sort(np.unique(y_test["time"]))

    # Guard against pathological cases
    if len(sorted_train_times) < 4 or len(sorted_test_times) < 4:
        logger.warning(
            f"[Client {client_id}] Not enough unique times for paper-style grid."
        )
        return {
            "C-index": np.nan,
            "IPCW_C-index": np.nan,
            "AUC": np.nan,
            "IBS": np.nan,
            "client_name": f"Client {client_id}",
        }

    eval_times = np.linspace(
        start=max(sorted_train_times[1], sorted_test_times[1]),
        stop=min(sorted_train_times[-2], sorted_test_times[-2]),
        num=100,
    )

    # DEBUG: print entire eval_times --> 
    print(
    f"\n[DEBUG][Client {client_id}] FULL eval_times ({len(eval_times)} points):"
    )
    print(np.array2string(eval_times, precision=6, separator=", "))
    print()

    # DEBIG: Train/test time ranges --> 
    print(
    f"[DEBUG][Client {client_id}] y_train time range: "
    f"[{y_train['time'].min():.6f}, {y_train['time'].max():.6f}]"
    )
    print(
        f"[DEBUG][Client {client_id}] y_test  time range: "
        f"[{y_test['time'].min():.6f}, {y_test['time'].max():.6f}]"
    )
    print()





    # 3) Interpolate survival curves onto paper grid
    n_samples = len(surv_fns)
    n_times   = len(eval_times)

    surv_probs = np.zeros((n_samples, n_times))

    for i, (t, s) in enumerate(surv_fns):
        f = interp1d(
            t,
            s,
            kind="previous",
            bounds_error=False,
            fill_value=(1.0, s[-1]),
        )
        surv_probs[i, :] = f(eval_times)

    # Numerical safety (exactly like paper)
    surv_probs = np.nan_to_num(surv_probs, nan=0.5)
    surv_probs[surv_probs < 0.0] = 0.0
    surv_probs[surv_probs > 1.0] = 1.0
    surv_probs = surv_probs * (1.0 - 1e-8) + 1e-8


    # 4) PAPER-STYLE RISK DEFINITION
    risks = -np.log(surv_probs)
    risk_scores = np.sum(risks, axis=1)


    # 5) PAPER-STYLE C-INDEX
    c_index = concordance_index_censored(
        y_test["event"],
        y_test["time"],
        risk_scores,
    )[0]


    # 6) Optional additional metrics (NOT paper-style)
    try:
        cindex_ipcw = concordance_index_ipcw(
            y_train, y_test, risk_scores
        )[0]
    except Exception:
        cindex_ipcw = np.nan




    # FedSurF++ STYLE CUMULATIVE AUC (time-dependent)
    # DEBGUG:
    print(f"[DEBUG][Client {client_id}] Checking event / risk counts per eval_time:")

    for i, t in enumerate(eval_times): 
        n_events = np.sum((y_test["event"] == 1) & (y_test["time"] <= t))
        n_at_risk = np.sum(y_test["time"] >= t)

        print(
            f"[DEBUG][Client {client_id}] t={t:.6f} | "
            f"events_up_to_t={n_events} | "
            f"at_risk={n_at_risk}"
        )

    print()

    try:
        # time-dependent risk: higher = more likely event
        risk_scores_t = 1.0 - surv_probs   # shape (n_samples, n_times)

        auc_t, mean_auc = cumulative_dynamic_auc(
            y_train,
            y_test,
            risk_scores_t,
            eval_times,
        )

        # FILTER INVALID TIMES: bc if not yield to nan due to high censoring
        valid = (
            ~np.isnan(auc_t)
        )

        print(
            f"[DEBUG][Client {client_id}] AUC valid times: "
            f"{valid.sum()} / {len(valid)}"
        )

        if valid.sum() < 3:
            mean_auc = np.nan
        else:
            mean_auc = np.trapz(
                auc_t[valid],
                eval_times[valid]
            ) / (eval_times[valid][-1] - eval_times[valid][0])
            
    except Exception as e:
        logger.warning(
            f"[Client {client_id}] Cumulative AUC failed: {e}"
        )
        mean_auc = np.nan


    try:
        _, bs_scores = brier_score(
            y_train, y_test, surv_probs, eval_times
        )
        ibs = np.trapz(bs_scores, eval_times) / (eval_times[-1] - eval_times[0])
    except Exception:
        ibs = np.nan


    # 7) Logging
    logger.info(
        f"[Client {client_id}] "
        f"C-index(paper)={c_index:.4f}, "
        f"IPCW={cindex_ipcw:.4f}, "
        f"AUC={mean_auc:.4f}, "
        f"IBS={ibs:.4f}"
    )

    return {
        "C-index": float(c_index),
        "IPCW_C-index": float(cindex_ipcw),
        "AUC": float(mean_auc),
        "IBS": float(ibs),
        "client_name": f"Client {client_id}",
    }









def evaluate_deepsurv(model, data, client_id, config):
    """
    DeepSurv evaluation using PAPER-STYLE approach (matching FedSurF).
    
    DeepSurv produces survival curves via baseline hazard estimation,
    allowing computation of all metrics: C-index, IPCW, AUC, and IBS.
    
    Parameters:
        model: DeepSurv model instance (must be fitted)
        data: dict with keys 'X_test', 'y_test', 'y_train'
        client_id: client identifier
        config: configuration object
        
    Returns:
        dict with evaluation metrics (same as FedSurF)
    """
    import logging
    import numpy as np
    from scipy.interpolate import interp1d
    from sksurv.metrics import (
        concordance_index_censored,
        concordance_index_ipcw,
        cumulative_dynamic_auc,
        brier_score,
    )
    
    logger = logging.getLogger("main")
    model_name = getattr(config, "model", "DeepSurv")
    logger.info(f"[Client {client_id}] Evaluating {model_name} (paper-style, full metrics)...")
    
    
    # Unpack the data
    X_test = data["X_test"]
    y_test = data["y_test"]
    y_train = data["y_train"]
    

    # 1) PAPER-STYLE TIME GRID (same as FedSurF)
    sorted_train_times = np.sort(np.unique(y_train["time"]))
    sorted_test_times = np.sort(np.unique(y_test["time"]))
    
    # Guard against pathological cases
    if len(sorted_train_times) < 4 or len(sorted_test_times) < 4:
        logger.warning(
            f"[Client {client_id}] Not enough unique times for paper-style grid."
        )
        return {
            "C-index": np.nan,
            "IPCW_C-index": np.nan,
            "AUC": np.nan,
            "IBS": np.nan,
            "client_name": f"Client {client_id}",
        }
    
    eval_times = np.linspace(
        start=max(sorted_train_times[1], sorted_test_times[1]),
        stop=min(sorted_train_times[-2], sorted_test_times[-2]),
        num=100,
    )
    
    # DEBUG: print eval_times info
    print(
        f"\n[DEBUG][Client {client_id}] {model_name} eval_times ({len(eval_times)} points):"
    )
    print(np.array2string(eval_times, precision=6, separator=", "))
    print()
    
    print(
        f"[DEBUG][Client {client_id}] y_train time range: "
        f"[{y_train['time'].min():.6f}, {y_train['time'].max():.6f}]"
    )
    print(
        f"[DEBUG][Client {client_id}] y_test  time range: "
        f"[{y_test['time'].min():.6f}, {y_test['time'].max():.6f}]"
    )
    print()
    

    # 2) Predict survival functions using baseline hazard
    surv_fns = model.predict_survival_function(X_test)
    
    if surv_fns is None:
        logger.error(f"[Client {client_id}] Failed to predict survival functions")
        return {
            "C-index": np.nan,
            "IPCW_C-index": np.nan,
            "AUC": np.nan,
            "IBS": np.nan,
            "client_name": f"Client {client_id}",
        }
    

    # 3) Interpolate survival curves onto paper grid (same as FedSurF)
    n_samples = len(surv_fns)
    n_times = len(eval_times)
    
    surv_probs = np.zeros((n_samples, n_times))
    
    for i, (t, s) in enumerate(surv_fns):
        f = interp1d(
            t,
            s,
            kind="previous",
            bounds_error=False,
            fill_value=(1.0, s[-1]),
        )
        surv_probs[i, :] = f(eval_times)
    
    # Numerical safety (exactly like FedSurF)
    surv_probs = np.nan_to_num(surv_probs, nan=0.5)
    surv_probs[surv_probs < 0.0] = 0.0
    surv_probs[surv_probs > 1.0] = 1.0
    surv_probs = surv_probs * (1.0 - 1e-8) + 1e-8
    

    # 4) PAPER-STYLE RISK DEFINITION (same as FedSurF)
    risks = -np.log(surv_probs)
    risk_scores = np.sum(risks, axis=1)
    

    # 5) PAPER-STYLE C-INDEX (same as FedSurF)
    c_index = concordance_index_censored(
        y_test["event"],
        y_test["time"],
        risk_scores,
    )[0]
    

    # 6) IPCW C-index (same as FedSurF)
    try:
        cindex_ipcw = concordance_index_ipcw(
            y_train, y_test, risk_scores
        )[0]
    except Exception:
        cindex_ipcw = np.nan
    

    # 7) FedSurF++ STYLE CUMULATIVE AUC (same as FedSurF)
    print(f"[DEBUG][Client {client_id}] Checking event / risk counts per eval_time:")
    
    for i, t in enumerate(eval_times):
        n_events = np.sum((y_test["event"] == 1) & (y_test["time"] <= t))
        n_at_risk = np.sum(y_test["time"] >= t)
        
        print(
            f"[DEBUG][Client {client_id}] t={t:.6f} | "
            f"events_up_to_t={n_events} | "
            f"at_risk={n_at_risk}"
        )
    
    print()
    
    try:
        # time-dependent risk: higher = more likely event
        risk_scores_t = 1.0 - surv_probs  # shape (n_samples, n_times)
        
        auc_t, mean_auc = cumulative_dynamic_auc(
            y_train,
            y_test,
            risk_scores_t,
            eval_times,
        )
        
        # FILTER INVALID TIMES (same as FedSurF)
        valid = ~np.isnan(auc_t)
        
        print(
            f"[DEBUG][Client {client_id}] AUC valid times: "
            f"{valid.sum()} / {len(valid)}"
        )
        
        if valid.sum() < 3:
            mean_auc = np.nan
        else:
            mean_auc = np.trapz(
                auc_t[valid],
                eval_times[valid]
            ) / (eval_times[valid][-1] - eval_times[valid][0])
    
    except Exception as e:
        logger.warning(
            f"[Client {client_id}] Cumulative AUC failed: {e}"
        )
        mean_auc = np.nan
    

    # 8) IBS (same as FedSurF)
    try:
        _, bs_scores = brier_score(
            y_train, y_test, surv_probs, eval_times
        )
        ibs = np.trapz(bs_scores, eval_times) / (eval_times[-1] - eval_times[0])
    except Exception:
        ibs = np.nan
    

    # 9) Logging (same format as FedSurF)
    logger.info(
        f"[Client {client_id}] "
        f"C-index(paper)={c_index:.4f}, "
        f"IPCW={cindex_ipcw:.4f}, "
        f"AUC={mean_auc:.4f}, "
        f"IBS={ibs:.4f}"
    )
    
    return {
        "C-index": float(c_index),
        "IPCW_C-index": float(cindex_ipcw),
        "AUC": float(mean_auc),
        "IBS": float(ibs),
        "client_name": f"Client {client_id}",
    }

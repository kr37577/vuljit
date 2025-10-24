import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, average_precision_score, precision_score, recall_score, matthews_corrcoef,
)
from typing import Tuple, List, Dict, Any, Optional
import settings
from model_definition_regression import get_regression_pipeline, get_regression_param_distribution
from sklearn.model_selection import RandomizedSearchCV


def evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, Any]:
    mae = mean_absolute_error(y_true, y_pred)
    # Support older scikit-learn versions that don't accept 'squared' kwarg
    try:
        rmse = mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
    r2 = r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else None
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': (None if r2 is None else float(r2)),
        'n_test': int(len(y_true)),
        'sum_true': float(np.sum(y_true)),
    }


def _safe_size_from_features(X: pd.DataFrame) -> Optional[pd.Series]:
    for cand in ('change_size', 'total_lines_changed'):
        if cand in X.columns:
            s = pd.to_numeric(X[cand], errors='coerce').fillna(0.0).astype(float)
            return s
    # Fallback: try lines_added + lines_deleted
    if 'lines_added' in X.columns and 'lines_deleted' in X.columns:
        s = (pd.to_numeric(X['lines_added'], errors='coerce').fillna(0.0).astype(float)
             + pd.to_numeric(X['lines_deleted'], errors='coerce').fillna(0.0).astype(float))
        return s
    return None


def _auc_effort_curve(size: np.ndarray, faults: np.ndarray, order: np.ndarray) -> float:
    """Compute AUC of cumulative faults vs cumulative size curve under a given order.
    x-axis: cumulative size fraction; y-axis: cumulative faults fraction.
    Returns AUC in [0,1].
    """
    S = np.asarray(size, float)
    F = np.asarray(faults, float)
    idx = np.asarray(order, int)
    S_tot = np.sum(S)
    F_tot = np.sum(F)
    if S_tot <= 0 or F_tot <= 0 or len(idx) == 0:
        return np.nan
    S_ord = S[idx]
    F_ord = F[idx]
    x = np.cumsum(S_ord) / S_tot
    y = np.cumsum(F_ord) / F_tot
    # prepend (0,0) to include origin
    x = np.concatenate([[0.0], x])
    y = np.concatenate([[0.0], y])
    # clip to [0,1]
    x = np.clip(x, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)
    return float(np.trapz(y, x))


def compute_norm_popt(
    size: np.ndarray,
    faults: np.ndarray,
    predicted_density: np.ndarray,
) -> Optional[float]:
    """Compute normalized Popt following effort-aware evaluation.
    - size: effort proxy (e.g., LOC or change_size)
    - faults: actual counts
    - predicted_density: scores to rank by (higher = riskier)
    Returns None if not computable.
    """
    S = np.asarray(size, float)
    F = np.asarray(faults, float)
    Dhat = np.asarray(predicted_density, float)
    # guard
    if np.sum(S) <= 0 or np.sum(F) <= 0 or S.shape[0] != F.shape[0] or F.shape[0] != Dhat.shape[0]:
        return None
    # model order: by predicted density desc
    ord_model = np.argsort(-Dhat, kind='mergesort')
    # optimal: by actual density desc; worst: asc
    dens_true = np.divide(F, np.maximum(S, 1e-12))
    ord_opt = np.argsort(-dens_true, kind='mergesort')
    ord_worst = np.argsort(dens_true, kind='mergesort')

    auc_model = _auc_effort_curve(S, F, ord_model)
    auc_opt = _auc_effort_curve(S, F, ord_opt)
    auc_worst = _auc_effort_curve(S, F, ord_worst)
    if any(np.isnan([auc_model, auc_opt, auc_worst])):
        return None
    denom = (auc_opt - auc_worst)
    if denom <= 1e-12:
        return None
    return float((auc_model - auc_worst) / denom)


def run_time_series_regression(
    X_project: pd.DataFrame,
    y_project: pd.Series,
    project_name: str,
    run_random_state: int,
    model_name: str = 'random_forest',
    target_type: str = 'count',
) -> Tuple[List[Dict[str, Any]], pd.Series]:
    """Run time-series CV for regression and return fold metrics and OOS predictions.
    - target_type: 'count' (predict counts) or 'density' (predict counts/size)
    OOS series always returns predicted counts (for downstream simulation),
    while fold metrics include Norm(Popt) if size is available.
    """
    tss = TimeSeriesSplit(n_splits=settings.N_SPLITS_TIMESERIES)
    fold_metrics: List[Dict[str, Any]] = []
    oos = pd.Series(index=X_project.index, dtype=float, name='predicted_count')

    # size proxy
    size_series = _safe_size_from_features(X_project)

    param_dist = get_regression_param_distribution(model_name)
    for i, (train_idx, test_idx) in enumerate(tss.split(X_project)):
        X_train, X_test = X_project.iloc[train_idx], X_project.iloc[test_idx]
        y_train, y_test = y_project.iloc[train_idx], y_project.iloc[test_idx]

        if target_type == 'density':
            # train on density = count / size
            s_tr = _safe_size_from_features(X_train)
            s_te = _safe_size_from_features(X_test)
            if s_tr is None or s_te is None:
                # fallback to count prediction if size missing
                y_train_use = y_train.values.astype(float)
                y_test_use = y_test.values.astype(float)
                target_is_density = False
            else:
                s_tr_np = np.maximum(s_tr.values.astype(float), 1.0)
                s_te_np = np.maximum(s_te.values.astype(float), 1.0)
                y_train_use = np.divide(y_train.values.astype(float), s_tr_np)
                y_test_use = np.divide(y_test.values.astype(float), s_te_np)
                target_is_density = True
        else:
            y_train_use = y_train.values.astype(float)
            y_test_use = y_test.values.astype(float)
            target_is_density = False

        pipe = get_regression_pipeline(run_random_state, model=model_name)

        best = pipe
        if getattr(settings, 'USE_HYPERPARAM_OPTIMIZATION', False) and param_dist:
            cv_inner = TimeSeriesSplit(n_splits=max(2, min(5, settings.RANDOM_SEARCH_CV)))
            search = RandomizedSearchCV(
                estimator=pipe,
                param_distributions=param_dist,
                n_iter=max(5, min(30, settings.RANDOM_SEARCH_N_ITER)),
                scoring='neg_mean_absolute_error',
                refit=True,
                cv=cv_inner,
                random_state=run_random_state,
                n_jobs=1,
                verbose=0,
            )
            search.fit(X_train, y_train_use)
            best = search.best_estimator_
        else:
            best.fit(X_train, y_train_use)

        y_pred_raw = best.predict(X_test)
        y_pred_raw = np.asarray(y_pred_raw, float)

        # Transform predictions back to counts if trained on density
        if target_is_density:
            s_te = _safe_size_from_features(X_test)
            if s_te is not None:
                s_te_np = np.maximum(s_te.values.astype(float), 1.0)
                y_pred_counts = y_pred_raw * s_te_np
                y_pred_density = y_pred_raw
            else:
                y_pred_counts = y_pred_raw
                y_pred_density = None
        else:
            y_pred_counts = y_pred_raw
            # derive density if size available
            s_te = _safe_size_from_features(X_test)
            y_pred_density = None
            if s_te is not None:
                s_te_np = np.maximum(s_te.values.astype(float), 1.0)
                y_pred_density = np.divide(y_pred_counts, s_te_np)

        # clip negative to zero for counts
        y_pred_counts = np.clip(y_pred_counts, 0.0, None)

        # compute standard regression metrics on counts (converted if needed)
        m = evaluate_regression(pd.Series(y_test_use if target_is_density else y_test.values, index=X_test.index),
                                y_pred_raw if target_is_density else y_pred_counts)

        # compute Norm(Popt) if size and predicted density are available
        norm_popt = None
        if size_series is not None:
            s_eval = _safe_size_from_features(X_test)
            if s_eval is not None:
                s_eval_np = np.maximum(s_eval.values.astype(float), 1.0)
                # actual faults = counts in test
                f_eval = y_test.values.astype(float)
                # predicted ranking by density
                if y_pred_density is None and s_eval is not None:
                    y_pred_density = np.divide(np.asarray(y_pred_counts, float), s_eval_np)
                if y_pred_density is not None:
                    norm_popt = compute_norm_popt(s_eval_np, f_eval, np.asarray(y_pred_density, float))

        if norm_popt is not None:
            m['norm_popt'] = float(norm_popt)
        else:
            m['norm_popt'] = None

        # classification-style metrics from regression scores
        # Binary ground-truth: any fault (>0) vs none
        try:
            y_true_bin = (y_test.values.astype(float) > 0).astype(int)
            y_score = y_pred_density if y_pred_density is not None else y_pred_counts
            y_score = np.asarray(y_score, float)
            # AUC metrics require both classes in truth
            if len(np.unique(y_true_bin)) > 1:
                try:
                    m['auc_roc'] = float(roc_auc_score(y_true_bin, y_score))
                except Exception:
                    m['auc_roc'] = None
                try:
                    ap = average_precision_score(y_true_bin, y_score)
                    m['auc_pr'] = float(ap)
                    m['average_precision'] = float(ap)
                except Exception:
                    m['auc_pr'] = None
                    m['average_precision'] = None
            else:
                m['auc_roc'] = None
                m['auc_pr'] = None
                m['average_precision'] = None

            # Thresholded metrics at a simple threshold (>0)
            try:
                y_pred_lbl = (y_score > 0.0).astype(int)
                # precision/recall are defined only if there is at least one predicted positive
                if y_pred_lbl.sum() > 0 and y_true_bin.sum() >= 0:
                    m['cls_precision'] = float(precision_score(y_true_bin, y_pred_lbl, zero_division=0))
                    m['cls_recall'] = float(recall_score(y_true_bin, y_pred_lbl, zero_division=0))
                else:
                    m['cls_precision'] = None
                    m['cls_recall'] = None
                # MCC defined when both classes present in truth and prediction has at least two classes
                if (len(np.unique(y_true_bin)) > 1) and (len(np.unique(y_pred_lbl)) > 1):
                    m['cls_mcc'] = float(matthews_corrcoef(y_true_bin, y_pred_lbl))
                else:
                    m['cls_mcc'] = None
            except Exception:
                m['cls_precision'] = None
                m['cls_recall'] = None
                m['cls_mcc'] = None
        except Exception:
            m['auc_roc'] = None
            m['auc_pr'] = None
            m['average_precision'] = None
            m['cls_precision'] = None
            m['cls_recall'] = None
            m['cls_mcc'] = None
        m['fold'] = i
        m['target_type'] = target_type
        m['model'] = model_name
        fold_metrics.append(m)

        test_indices = X_project.iloc[test_idx].index
        oos.loc[test_indices] = y_pred_counts

    return fold_metrics, oos

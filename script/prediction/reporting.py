# reporting.py
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any, Tuple

def get_avg_metric_for_class(metrics_list: List[Dict], report_key: str, class_name_key: str, metric_name: str) -> Optional[float]:
    """クラスごとの特定の評価指標の平均を計算する。"""
    values = [m.get(report_key, {}).get(class_name_key, {}).get(metric_name) for m in metrics_list]
    valid_values = [v for v in values if v is not None]
    return np.mean(valid_values) if valid_values else None

def summarize_project_results(metrics_list: List[Dict], importances_df_list: List[pd.DataFrame], project_name: str) -> Tuple[Optional[Dict], Optional[pd.DataFrame]]:
    """プロジェクトごとの結果を集計・表示する。"""
    if not metrics_list:
        print(f"  プロジェクト '{project_name}': 成功したFoldがなかったため、結果を集計できません。")
        return None, None

    # 'accuracy' は classification_report のトップレベルにあるため、そこから直接取得します。
    accuracies = [m.get('classification_report_dict', {}).get('accuracy') for m in metrics_list if m.get('classification_report_dict')]
    valid_accuracies = [acc for acc in accuracies if acc is not None]
    avg_accuracy_val = np.mean(valid_accuracies) if valid_accuracies else None

    avg_metrics = {key: np.mean([m[key] for m in metrics_list if m.get(key) is not None]) for key in ["mcc", "auc_roc"]}
    # 'accuracy' をトップレベルのキーとして追加
    avg_metrics['accuracy'] = avg_accuracy_val

    avg_metrics["classification_report_dict"] = {
        'class_1': {
            'accuracy': avg_accuracy_val, # 全体の accuracy をここにも設定
            'precision': get_avg_metric_for_class(metrics_list, 'classification_report_dict', 'class_1', 'precision'),
            'recall': get_avg_metric_for_class(metrics_list, 'classification_report_dict', 'class_1', 'recall'),
            'f1-score': get_avg_metric_for_class(metrics_list, 'classification_report_dict', 'class_1', 'f1-score'),
            'roc_auc': np.mean([m.get('auc_roc') for m in metrics_list if m.get('auc_roc') is not None]),
            'pr_auc': np.mean([m.get('auc_pr') for m in metrics_list if m.get('auc_pr') is not None]),
            'mcc': np.mean([m.get('mcc') for m in metrics_list if m.get('mcc') is not None]),
        }
    }
    
    print(f"\n  プロジェクト '{project_name}' の平均評価結果 ({len(metrics_list)} Folds):")
    for key, val in avg_metrics.get('classification_report_dict', {}).get('class_1', {}).items():
        print(f"    平均 {key} (class_1): {val:.4f}" if val is not None else f"    平均 {key} (class_1): N/A")
        

    # 特徴量重要度をDataFrameとして平均化
    avg_importances_df = None
    if importances_df_list:
        # 全てのfoldのDataFrameを結合
        combined_importances = pd.concat(importances_df_list)
        # feature ごとに平均を計算
        avg_importances_df = combined_importances.groupby('feature')['importance'].mean().reset_index()
        print(f"  プロジェクト '{project_name}' の平均特徴量重要度:")
        print(avg_importances_df.sort_values(by='importance', ascending=False).head(30))
        # プロジェクト名を追加して、後で識別できるようにする
        avg_importances_df['project'] = project_name

    return avg_metrics, avg_importances_df


# === 追加: 予測確率列を所定の命名で一括付与するユーティリティ ===
def add_predicted_risk_columns(
    df: pd.DataFrame,
    oos_pred_series_by_exp: Dict[str, pd.Series],
    exp_to_name: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    oos_pred_series_by_exp: {"exp1": Series, "exp2": Series, ...} のようなアウトオブサンプル予測確率
    exp_to_name: {"exp1": "Kamei", "exp2": "Kamei_ProjCommit", ...} のマッピング
    既定の命名は prediction_code_backup と同一。
    """
    if exp_to_name is None:
        exp_to_name = {
            "exp1": "Kamei",
            "exp2": "Kamei_ProjCommit",
            "exp3": "Kamei_ProjTotal",
            "exp4": "Kamei_ProjAll",
            "exp5": "VCCFinder",
            "exp6": "VCCFinder_ProjCommit",
            "exp7": "VCCFinder_ProjTotal",
            "exp8": "VCCFinder_ProjAll",
            "exp9": "ProjCommit",
            "exp10": "ProjTotal",
            "exp11": "ProjAll",
        }
    for exp_key, exp_name in exp_to_name.items():
        s = oos_pred_series_by_exp.get(exp_key)
        if s is None:
            continue
        col = f"predicted_risk_{exp_name}"
        # indexをreindexして厳密にアラインし、floatで格納
        df[col] = pd.to_numeric(s, errors='coerce').reindex(df.index).astype(float)
    return df


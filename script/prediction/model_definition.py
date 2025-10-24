# model_definition.py
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import RandomizedSearchCV
import settings
import numpy as np  

def get_pipeline(run_random_state: int,scale_pos_weight: float = 1.0) -> ImbPipeline:
    """モデルと前処理（サンプリング）のパイプラインを定義して返す"""
    # 小さな離散探索空間を定義し、run_random_stateに基づき1組を決定的にサンプル
    # rng = np.random.RandomState(run_random_state)

    if settings.SELECTED_MODEL == 'random_forest':
        # rf_space = {
        #     "n_estimators": [300, 600, 900],
        #     "max_depth": [None, 10, 20],
        #     "min_samples_leaf": [1, 2, 4],
        #     "max_features": ['sqrt', 0.5, None],
        # }
        # picked = {k: rng.choice(v) for k, v in rf_space.items()}
        classifier = RandomForestClassifier(
            n_jobs = 1,
            n_estimators=1500,
            max_depth=None,
            min_samples_leaf=1,
            max_features='sqrt',
            min_samples_split=2,
            bootstrap=True,
            random_state=run_random_state,
            # class_weight='balanced_subsample'
        )
    elif settings.SELECTED_MODEL == 'xgboost':
        # xgb_space = {
        #     "n_estimators": [300, 600, 1000],
        #     "max_depth": [4, 6, 8],
        #     "learning_rate": [0.01, 0.1, 0.2],
        #     "subsample": [0.7, 1.0],
        #     "colsample_bytree": [0.7, 1.0],
        #     "min_child_weight": [1, 3, 5],
        # }
        # picked = {k: rng.choice(v) for k, v in xgb_space.items()}
        classifier = XGBClassifier(
            # ★★★ 実行ごとの random_state を使用 ★★★
            random_state=run_random_state,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=1,
            # 不均衡補正
            # scale_pos_weight=scale_pos_weight,
            tree_method='hist',
            verbosity=0,
            # n_estimators=picked["n_estimators"],
            # max_depth=picked["max_depth"],
            # learning_rate=picked["learning_rate"],
            # subsample=picked["subsample"],
            # colsample_bytree=picked["colsample_bytree"],
            # min_child_weight=picked["min_child_weight"],
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            reg_alpha=1.0,
            gamma=1.0,
            max_bin=256,
            objective='binary:logistic',
        )
    elif settings.SELECTED_MODEL == 'random':
        classifier = DummyClassifier(strategy=getattr(settings, "RANDOM_BASELINE_STRATEGY", "stratified"),
                                      random_state=run_random_state)
        return ImbPipeline([('classifier', classifier)])

    else:
        raise ValueError(f"不明なモデルが選択されています: {settings.SELECTED_MODEL}")

    if settings.SAMPLING_METHOD == 'random_under':
        sampler = RandomUnderSampler(
            # ★★★ 実行ごとの random_state を使用 ★★★
            random_state=run_random_state,
            sampling_strategy=settings.RANDOM_UNDER_SAMPLING_STRATEGY
        )
        sampler_name = 'rus'
    elif settings.SAMPLING_METHOD == 'smote':
        # ★★★ SMOTEにも実行ごとの random_state を適用 ★★★
        sampler = SMOTE(random_state=run_random_state)
        sampler_name = 'smote'
    else:
        print(f"警告: 不明なサンプリング手法 '{settings.SAMPLING_METHOD}' です。サンプリングなしで続行します。")
        return ImbPipeline([('classifier', classifier)])

    pipeline = ImbPipeline([(sampler_name, sampler), ('classifier', classifier)])
    return pipeline


def get_param_distribution() -> dict:
    """ハイパーパラメータ探索のためのパラメータ空間を返す（狭い堅実な範囲）"""
    if settings.SELECTED_MODEL == 'random_forest':
        param_dist = {
                "classifier__n_estimators": [200, 500, 1000, 1500],
                "classifier__max_depth": [None, 5, 10, 20, 30],
                "classifier__min_samples_split": [2, 5, 10, 20, 50],
                "classifier__min_samples_leaf": [1, 2, 5, 10, 20],
                "classifier__max_features": ["sqrt", "log2", 0.3, 0.5, 0.8, 1.0],
                "classifier__bootstrap": [True, False],
                # "classifier__class_weight": [None, "balanced"]
            }
    elif settings.SELECTED_MODEL == 'xgboost':
        param_dist = {
                'classifier__n_estimators': [200, 500, 800, 1200],
                'classifier__max_depth': [3, 4, 6, 8, 10],
                'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
                'classifier__subsample': [0.5, 0.7, 0.9, 1.0],
                'classifier__colsample_bytree': [0.5, 0.7, 0.9, 1.0],
                'classifier__min_child_weight': [1, 3, 5, 10, 20],
                'classifier__gamma': [0, 1, 5, 10],
                'classifier__reg_alpha': [0, 1, 5, 10],
                'classifier__reg_lambda': [1, 5, 10, 20],
                # 'classifier__scale_pos_weight': [1, 2, 5, 10]
            }
    elif settings.SELECTED_MODEL == 'random':
        param_dist = {
            "classifier__strategy": [settings.RANDOM_BASELINE_STRATEGY]
        }
    else:
        raise ValueError(f"不明なモデルが選択されています: {settings.SELECTED_MODEL}")
    return param_dist

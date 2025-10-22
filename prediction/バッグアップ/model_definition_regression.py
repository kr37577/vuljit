from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

try:
    from xgboost import XGBRegressor  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    XGBRegressor = None


def get_regression_pipeline(run_random_state: int, model: str = 'random_forest') -> Pipeline:
    """Return a simple regression pipeline for predicting counts/densities.
    - model: 'random_forest' | 'linear' | 'cart' | 'dummy'
    """
    if model == 'random_forest':
        reg = RandomForestRegressor(
            n_estimators=1500,
            max_depth=None,
            min_samples_leaf=1,
            min_samples_split=2,
            max_features='sqrt',
            n_jobs=1,
            bootstrap=True,
            random_state=run_random_state,
        )
        return Pipeline([('regressor', reg)])
    if model == 'linear':
        # Simple linear regression with standardization
        reg = LinearRegression()
        return Pipeline([
            ('scaler', StandardScaler(with_mean=False)),
            ('regressor', reg),
        ])
    if model == 'cart':
        reg = DecisionTreeRegressor(
            random_state=run_random_state,
        )
        return Pipeline([('regressor', reg)])
    if model == 'xgboost':
        if XGBRegressor is None:
            raise ValueError("XGBoost (xgboost) が見つかりません。'pip install xgboost' を行うか、別モデルを選択してください。")
        reg = XGBRegressor(
            random_state=run_random_state,
            n_jobs=1,
            tree_method='hist',
            verbosity=0,
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
            objective='reg:squarederror',
        )
        return Pipeline([('regressor', reg)])
    elif model == 'dummy':
        reg = DummyRegressor(strategy='mean')
        return Pipeline([('regressor', reg)])
    else:
        raise ValueError(f"Unknown regression model: {model}")


def get_regression_param_distribution(model: str = 'random_forest') -> dict:
    """Provide a small hyper-parameter space for optional tuning (kept narrow)."""
    if model == 'random_forest':
        return {
            'regressor__n_estimators': [300, 600, 1000, 1500],
            'regressor__max_depth': [None, 10, 20, 30],
            'regressor__min_samples_split': [2, 5, 10],
            'regressor__min_samples_leaf': [1, 2, 5],
            'regressor__max_features': ['sqrt', 'log2', 0.5, 0.8, 1.0],
        }
    if model == 'linear':
        # LinearRegression has few hyperparams; provide only scaler toggle to keep space tiny
        return {
            # placeholder for potential feature selection/scaling choices; keep empty for now
        }
    if model == 'cart':
        return {
            'regressor__max_depth': [None, 5, 10, 20, 30],
            'regressor__min_samples_split': [2, 5, 10, 20],
            'regressor__min_samples_leaf': [1, 2, 5, 10],
        }
    if model == 'xgboost':
        return {
            'regressor__n_estimators': [300, 600, 1000, 1500],
            'regressor__max_depth': [3, 4, 6, 8, 10],
            'regressor__learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
            'regressor__subsample': [0.6, 0.8, 1.0],
            'regressor__colsample_bytree': [0.6, 0.8, 1.0],
            'regressor__min_child_weight': [1, 3, 5, 10],
            'regressor__gamma': [0, 1, 5],
            'regressor__reg_alpha': [0, 1, 5],
            'regressor__reg_lambda': [1, 5, 10],
        }
    if model == 'dummy':
        return {}
    else:
        raise ValueError(f"Unknown regression model: {model}")

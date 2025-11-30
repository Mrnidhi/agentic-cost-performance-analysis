import optuna
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


class HyperparameterOptimizer:
    """Optimize model hyperparameters using Optuna."""

    def __init__(self, model_type: str = 'xgb', n_trials: int = 50):
        self.model_type = model_type
        self.n_trials = n_trials
        self.best_params = None
        self.best_model = None

    def _xgb_objective(self, trial, X, y):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42
        }
        model = XGBRegressor(**params)
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        return scores.mean()

    def _lgbm_objective(self, trial, X, y):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42,
            'verbose': -1
        }
        model = LGBMRegressor(**params)
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        return scores.mean()

    def optimize(self, X, y):
        """Run hyperparameter optimization."""
        if self.model_type == 'xgb':
            objective = lambda trial: self._xgb_objective(trial, X, y)
        elif self.model_type == 'lgbm':
            objective = lambda trial: self._lgbm_objective(trial, X, y)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        self.best_params = study.best_params
        return self.best_params

    def get_best_model(self):
        """Return model with best hyperparameters."""
        if self.best_params is None:
            raise ValueError("Run optimize() first.")

        if self.model_type == 'xgb':
            self.best_model = XGBRegressor(**self.best_params, random_state=42)
        elif self.model_type == 'lgbm':
            self.best_model = LGBMRegressor(**self.best_params, random_state=42, verbose=-1)

        return self.best_model


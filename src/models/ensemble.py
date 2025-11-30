import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


class EnsembleModel:
    """Ensemble model combining multiple regressors for agent performance prediction."""

    def __init__(self, weights: dict = None):
        self.models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'xgb': XGBRegressor(n_estimators=100, random_state=42),
            'lgbm': LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        }
        self.weights = weights or {'rf': 0.25, 'gb': 0.25, 'xgb': 0.25, 'lgbm': 0.25}
        self.fitted = False

    def fit(self, X, y):
        """Fit all models in the ensemble."""
        for name, model in self.models.items():
            model.fit(X, y)
        self.fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        """Generate weighted ensemble predictions."""
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        predictions = np.zeros(len(X))
        for name, model in self.models.items():
            predictions += self.weights[name] * model.predict(X)
        return predictions

    def get_feature_importance(self, feature_names: list) -> dict:
        """Get averaged feature importance across all models."""
        importance = np.zeros(len(feature_names))
        for name, model in self.models.items():
            importance += self.weights[name] * model.feature_importances_
        return dict(zip(feature_names, importance))


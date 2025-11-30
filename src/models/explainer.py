"""
Model explainability using SHAP and feature importance analysis.

This module provides interpretability tools for understanding
model predictions and feature contributions.

Course: DATA 230 (Data Visualization) at SJSU
"""

from typing import Optional, Any
from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd

from src.core.protocols import ModelExplainer, PredictionExplanation
from src.core.exceptions import PredictionError


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureContribution:
    """Feature contribution to a prediction."""
    
    feature_name: str
    contribution: float
    feature_value: float
    direction: str  # "positive" or "negative"
    
    @property
    def abs_contribution(self) -> float:
        """Absolute contribution value."""
        return abs(self.contribution)


class SHAPExplainer:
    """
    SHAP-based model explainer.
    
    Provides feature importance and contribution analysis using
    SHAP (SHapley Additive exPlanations) values.
    
    Example:
        >>> explainer = SHAPExplainer(model, feature_names)
        >>> explanation = explainer.explain_prediction(features, prediction)
    """
    
    def __init__(
        self,
        model: Any,
        feature_names: list[str],
        background_data: Optional[np.ndarray] = None
    ) -> None:
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model to explain.
            feature_names: List of feature names.
            background_data: Optional background data for SHAP.
        """
        self._model = model
        self._feature_names = list(feature_names)
        self._background_data = background_data
        self._shap_explainer = None
        
        self._initialize_explainer()
        
        logger.info(
            "Initialized SHAPExplainer",
            extra={"n_features": len(feature_names)}
        )
    
    def _initialize_explainer(self) -> None:
        """Initialize the SHAP explainer."""
        try:
            import shap
            
            # Try TreeExplainer for tree-based models
            if hasattr(self._model, "get_booster") or hasattr(self._model, "estimators_"):
                self._shap_explainer = shap.TreeExplainer(self._model)
            elif self._background_data is not None:
                self._shap_explainer = shap.KernelExplainer(
                    self._model.predict,
                    self._background_data
                )
            else:
                logger.warning("Could not initialize SHAP explainer")
                
        except ImportError:
            logger.warning("shap not available, explanations will be limited")
    
    def explain_prediction(
        self,
        features: np.ndarray,
        prediction: float
    ) -> PredictionExplanation:
        """
        Generate explanation for a prediction.
        
        Args:
            features: Input features for the prediction.
            prediction: Model prediction value.
            
        Returns:
            Explanation with feature contributions.
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Get SHAP values if available
        if self._shap_explainer is not None:
            try:
                shap_values = self._shap_explainer.shap_values(features)
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                shap_values = shap_values.flatten()
            except Exception as e:
                logger.warning(f"SHAP calculation failed: {e}")
                shap_values = np.zeros(len(self._feature_names))
        else:
            shap_values = np.zeros(len(self._feature_names))
        
        # Build contribution dictionary
        contributions = dict(zip(self._feature_names, shap_values))
        
        # Identify top positive and negative features
        sorted_features = sorted(
            contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        top_positive = tuple(
            name for name, val in sorted_features[:5] if val > 0
        )
        top_negative = tuple(
            name for name, val in sorted_features[-5:] if val < 0
        )
        
        # Calculate base value
        if self._shap_explainer is not None and hasattr(self._shap_explainer, "expected_value"):
            base_value = float(self._shap_explainer.expected_value)
            if isinstance(base_value, np.ndarray):
                base_value = float(base_value[0])
        else:
            base_value = prediction - sum(shap_values)
        
        # Calculate confidence based on feature contribution consistency
        contribution_std = np.std(shap_values)
        confidence = 1 / (1 + contribution_std)
        
        return PredictionExplanation(
            prediction=prediction,
            base_value=base_value,
            feature_contributions=contributions,
            top_positive_features=top_positive,
            top_negative_features=top_negative,
            confidence=confidence
        )
    
    def get_feature_contributions(
        self,
        features: np.ndarray
    ) -> dict[str, float]:
        """
        Get feature contributions to prediction.
        
        Args:
            features: Input features.
            
        Returns:
            Dictionary mapping feature names to contribution values.
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        if self._shap_explainer is not None:
            try:
                shap_values = self._shap_explainer.shap_values(features)
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                return dict(zip(self._feature_names, shap_values.flatten()))
            except Exception as e:
                logger.warning(f"SHAP calculation failed: {e}")
        
        return {name: 0.0 for name in self._feature_names}
    
    def get_global_importance(
        self,
        X: np.ndarray
    ) -> dict[str, float]:
        """
        Calculate global feature importance across dataset.
        
        Args:
            X: Feature matrix.
            
        Returns:
            Dictionary of mean absolute SHAP values per feature.
        """
        if self._shap_explainer is None:
            return {name: 0.0 for name in self._feature_names}
        
        try:
            shap_values = self._shap_explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            return dict(zip(self._feature_names, mean_abs_shap))
            
        except Exception as e:
            logger.warning(f"Global importance calculation failed: {e}")
            return {name: 0.0 for name in self._feature_names}


class FeatureImportanceAnalyzer:
    """
    Analyze feature importance using multiple methods.
    
    Combines model-based importance, permutation importance,
    and correlation analysis for comprehensive understanding.
    
    Example:
        >>> analyzer = FeatureImportanceAnalyzer(model, feature_names)
        >>> importance = analyzer.analyze(X, y)
    """
    
    def __init__(
        self,
        model: Any,
        feature_names: list[str]
    ) -> None:
        """
        Initialize feature importance analyzer.
        
        Args:
            model: Trained model.
            feature_names: List of feature names.
        """
        self._model = model
        self._feature_names = list(feature_names)
        
        logger.info("Initialized FeatureImportanceAnalyzer")
    
    def analyze(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> dict[str, dict[str, float]]:
        """
        Perform comprehensive feature importance analysis.
        
        Args:
            X: Feature matrix.
            y: Target values.
            
        Returns:
            Dictionary with importance scores from different methods.
        """
        results = {}
        
        # Model-based importance
        results["model_importance"] = self._get_model_importance()
        
        # Correlation with target
        results["correlation"] = self._calculate_correlation(X, y)
        
        # Permutation importance (simplified)
        results["permutation"] = self._permutation_importance(X, y)
        
        # Combined ranking
        results["combined_rank"] = self._calculate_combined_rank(results)
        
        logger.info(
            "Feature importance analysis completed",
            extra={"n_features": len(self._feature_names)}
        )
        
        return results
    
    def _get_model_importance(self) -> dict[str, float]:
        """Get model-based feature importance."""
        if hasattr(self._model, "feature_importances_"):
            importance = self._model.feature_importances_
            return dict(zip(self._feature_names, importance))
        
        if hasattr(self._model, "coef_"):
            coef = np.abs(self._model.coef_).flatten()
            return dict(zip(self._feature_names, coef))
        
        return {name: 0.0 for name in self._feature_names}
    
    def _calculate_correlation(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> dict[str, float]:
        """Calculate correlation between features and target."""
        correlations = {}
        
        for i, name in enumerate(self._feature_names):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            correlations[name] = abs(corr) if not np.isnan(corr) else 0.0
        
        return correlations
    
    def _permutation_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_repeats: int = 5
    ) -> dict[str, float]:
        """Calculate simplified permutation importance."""
        baseline_score = self._score(X, y)
        importance = {}
        
        for i, name in enumerate(self._feature_names):
            scores = []
            
            for _ in range(n_repeats):
                X_permuted = X.copy()
                np.random.shuffle(X_permuted[:, i])
                permuted_score = self._score(X_permuted, y)
                scores.append(baseline_score - permuted_score)
            
            importance[name] = np.mean(scores)
        
        return importance
    
    def _score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate model score."""
        if hasattr(self._model, "score"):
            return self._model.score(X, y)
        
        predictions = self._model.predict(X)
        mse = np.mean((y - predictions) ** 2)
        return -mse  # Negative MSE so higher is better
    
    def _calculate_combined_rank(
        self,
        results: dict[str, dict[str, float]]
    ) -> dict[str, float]:
        """Calculate combined importance ranking."""
        combined = {name: 0.0 for name in self._feature_names}
        
        for method, importance in results.items():
            if method == "combined_rank":
                continue
            
            # Normalize importance to 0-1
            values = list(importance.values())
            if max(values) > 0:
                normalized = {
                    k: v / max(values)
                    for k, v in importance.items()
                }
            else:
                normalized = importance
            
            # Add to combined score
            for name, value in normalized.items():
                combined[name] += value
        
        # Normalize combined scores
        max_combined = max(combined.values()) if combined.values() else 1
        if max_combined > 0:
            combined = {k: v / max_combined for k, v in combined.items()}
        
        return combined


import pandas as pd
import numpy as np


class BusinessMetrics:
    """Create business-focused features from agent performance data."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def create_performance_scores(self) -> pd.DataFrame:
        """Create composite performance metrics."""
        self.df['overall_performance_score'] = (
            self.df['success_rate'] + 
            self.df['accuracy_score'] + 
            self.df['efficiency_score']
        ) / 3

        self.df['resource_efficiency_score'] = (
            self.df['efficiency_score'] / 
            (self.df['memory_usage_mb'] * self.df['cpu_usage_percent'])
        ) * 1000

        self.df['cost_effectiveness'] = (
            self.df['overall_performance_score'] / 
            self.df['cost_per_task_cents']
        )

        self.df['weighted_quality_score'] = (
            self.df['success_rate'] * 0.4 + 
            self.df['accuracy_score'] * 0.4 + 
            self.df['data_quality_score'] * 0.2
        )

        return self.df

    def create_interaction_features(self) -> pd.DataFrame:
        """Create features that capture variable interactions."""
        self.df['complexity_autonomy_ratio'] = (
            self.df['autonomy_level'] / self.df['task_complexity']
        )

        self.df['success_autonomy_interaction'] = (
            self.df['success_rate'] * self.df['autonomy_level']
        )

        self.df['latency_per_operation'] = (
            self.df['response_latency_ms'] / self.df['execution_time_seconds']
        )

        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        return self.df

    def create_benchmark_features(self) -> pd.DataFrame:
        """Create features comparing agents to group benchmarks."""
        arch_median = self.df.groupby('model_architecture')['overall_performance_score'].transform('median')
        self.df['arch_performance_benchmark'] = self.df['overall_performance_score'] - arch_median

        cost_per_complexity = self.df['cost_per_task_cents'] / self.df['task_complexity']
        self.df['env_avg_cost_per_complexity'] = self.df.groupby('deployment_environment')[
            'cost_per_task_cents'
        ].transform('mean') / self.df.groupby('deployment_environment')[
            'task_complexity'
        ].transform('mean')

        return self.df

    def transform(self) -> pd.DataFrame:
        """Apply all business metric transformations."""
        self.create_performance_scores()
        self.create_interaction_features()
        self.create_benchmark_features()
        return self.df


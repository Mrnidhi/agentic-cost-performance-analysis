import pandas as pd


class TemporalFeatures:
    """Extract time-based features from timestamp data."""

    def __init__(self, df: pd.DataFrame, timestamp_col: str = 'timestamp'):
        self.df = df.copy()
        self.timestamp_col = timestamp_col

    def extract_time_components(self) -> pd.DataFrame:
        """Extract hour and day-of-week from timestamp."""
        self.df[self.timestamp_col] = pd.to_datetime(self.df[self.timestamp_col])
        self.df['hour_of_day'] = self.df[self.timestamp_col].dt.hour
        self.df['day_of_week'] = self.df[self.timestamp_col].dt.dayofweek
        return self.df

    def create_weekend_flag(self) -> pd.DataFrame:
        """Create binary flag for weekend days."""
        if 'day_of_week' not in self.df.columns:
            self.extract_time_components()
        self.df['is_weekend'] = self.df['day_of_week'] >= 5
        return self.df

    def create_time_periods(self) -> pd.DataFrame:
        """Categorize hours into business periods."""
        if 'hour_of_day' not in self.df.columns:
            self.extract_time_components()

        def get_period(hour):
            if 6 <= hour < 12:
                return 'morning'
            elif 12 <= hour < 18:
                return 'afternoon'
            elif 18 <= hour < 22:
                return 'evening'
            else:
                return 'night'

        self.df['time_period'] = self.df['hour_of_day'].apply(get_period)
        return self.df

    def transform(self) -> pd.DataFrame:
        """Apply all temporal transformations."""
        self.extract_time_components()
        self.create_weekend_flag()
        self.create_time_periods()
        return self.df


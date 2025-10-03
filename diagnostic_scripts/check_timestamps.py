import pandas as pd
from datetime import datetime, timedelta

# Simulate your data
start = datetime(2019, 11, 11, 6, 0, 0)
end = datetime(2019, 11, 14, 5, 59, 0)

# Create timestamps every minute
timestamps = pd.date_range(start=start, end=end, freq='1min')
print(f"Number of timestamps: {len(timestamps)}")
print(f"Duration in minutes: {(end - start).total_seconds() / 60}")
print(f"Duration + 1: {(end - start).total_seconds() / 60 + 1}")
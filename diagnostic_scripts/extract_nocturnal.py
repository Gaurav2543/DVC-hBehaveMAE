import numpy as np
import pandas as pd

# Load the data
data = np.load('embeddings_4weeks_level1.npz', allow_pickle=True)
embeddings = data['embeddings']
timestamps = data['timestamps']
frame_map = data['frame_map'].item()

# Create a DataFrame for easy filtering
df = pd.DataFrame(embeddings)
df['timestamp'] = timestamps

# Filter for a specific night
start_night = pd.to_datetime('2020-02-15 19:00:00')
end_night = pd.to_datetime('2020-02-16 07:00:00')

night_df = df[(df['timestamp'] >= start_night) & (df['timestamp'] < end_night)]

# nocturnal_embeddings are now ready for your downstream model
nocturnal_embeddings = night_df.drop(columns=['timestamp']).values
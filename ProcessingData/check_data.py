import numpy as np
import pandas as pd

flow = np.load("flow.npy")

# Extract everything for Station 0
# Shape becomes (87672, 6) -> (Time, Feature)
station_0_data = flow[:, 0, :]

# Define your features based on the SAWS lists in your pipeline
features = ["Temperature", "Humidity", "Pressure", "Rain", "Wind Direction", "Wind Speed"]

# Create a DataFrame
df = pd.DataFrame(station_0_data, columns=features)

# Save to CSV
df.to_csv("station_0_flow.csv", index=False)
print("Saved station 0 data to station_0_flow.csv!")

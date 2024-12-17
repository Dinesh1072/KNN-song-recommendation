import pandas as pd
import pickle

# Load dataset
file_path = "spnew.csv"
data = pd.read_csv(file_path)

# Save dataset as a pickle file
with open("track_data.pkl", "wb") as f:
    pickle.dump(data, f)

print("Dataset saved to 'track_data.pkl'")

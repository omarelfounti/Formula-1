import pandas as pd

# Load data
data_2024 = pd.read_csv('data/Formula1_2024season_raceResults.csv')
data_2025 = pd.read_csv('data/Formula1_2025Season_RaceResults.csv')

# Combine datasets
combined_data = pd.concat([data_2024, data_2025], ignore_index=True)

# Save combined data
combined_data.to_csv('data/F1_2024_2025_Combined.csv', index=False)
print("Data combined and saved successfully.")


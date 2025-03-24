import pandas as pd

# Load the dataset
df = pd.read_excel("//Volumes/HDD/Stammering_identification/Stammering_Identification_With_LargeLanguageModels/jalsakar20.xlsx")

# List of label columns
label_cols = ['Prolongation', 'Block', 'SoundRep', 'WordRep', 'DifficultToUnderstand', 'Interjection']

# Filter out rows with no labels (all zeros)
df = df[df[label_cols].sum(axis=1) > 0]

# Create a 'Combination' column to represent the exact set of active labels (sorted)
df['Combination'] = df[label_cols].apply(
    lambda row: tuple(sorted(row[row == 1].index.tolist())),
    axis=1
)

# Group by combinations and sample up to 18 per group
sampled_df = df.groupby('Combination', group_keys=False).apply(
    lambda x: x.sample(n=min(18, len(x)), random_state=42)
)

# Drop the temporary 'Combination' column
sampled_df = sampled_df.drop(columns=['Combination'])

# Save to a new Excel file
sampled_df.to_excel("balanced_dataset_with_all_combinations11.xlsx", index=False)
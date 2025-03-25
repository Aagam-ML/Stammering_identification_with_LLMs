import pandas as pd


def process_files(label_file, transcript_file):
    # Process label file
    label_df = pd.read_excel(label_file)

    # Identify feature columns (excluding Counter)
    features = ['Prolongation', 'Block', 'SoundRep', 'WordRep', 'DifficultToUnderstand', 'Interjection']

    # Find all-zero rows
    zero_mask = (label_df[features] == 0).all(axis=1)
    zero_df = label_df[zero_mask]
    non_zero_df = label_df[~zero_mask]

    # Sample 500 zero rows (with replacement if needed)
    sampled_zero = zero_df.sample(n=500, replace=True, random_state=42)

    # Get counters to delete
    deleted_counters = zero_df[~zero_df.index.isin(sampled_zero.index)]['Counter'].tolist()

    # Create new label file
    new_label_df = pd.concat([non_zero_df, sampled_zero]).sort_index()
    new_label_df.to_excel('updated_labels.xlsx', index=False)

    # Process transcript file
    transcript_df = pd.read_excel(transcript_file)

    # Remove matching numbers
    new_transcript_df = transcript_df[~transcript_df['Number'].isin(deleted_counters)]
    new_transcript_df.to_excel('updated_transcript.xlsx', index=False)

    return deleted_counters


# Usage example
deleted = process_files('output_data.xlsx', 'transcription.xlsx')
print(f"Deleted counters: {deleted}")
import pandas as pd
import numpy as np


def process_feature_balance(label_file, transcript_file):
    """
    Process label and transcript files to balance feature-specific rows.
    - For each feature, ensure exactly 500 rows where only that feature is active.
    - Update both label and transcript files accordingly.
    - Skips first column (Counter) during feature analysis
    """
    # Load data and keep first column but exclude from feature processing
    label_df = pd.read_excel(label_file)
    transcript_df = pd.read_excel(transcript_file)

    # Define feature columns (exclude first column 'Counter')
    features = ['Prolongation', 'Block', 'SoundRep', 'WordRep',
                'DifficultToUnderstand', 'Interjection']

    # Dictionary to track changes
    all_operations = {
        'deleted_counters': [],
        'added_entries': pd.DataFrame()
    }

    # Process each feature
    for feature in features:
        # Find rows where only this feature is 1 and all others are 0 (excluding Counter column)
        mask = (
                (label_df[feature] == 1) &
                (label_df.drop(columns=['Counter', feature]).sum(axis=1) == 0)
        )
        feature_df = label_df[mask]

        # Determine sampling needed
        current_count = len(feature_df)

        if current_count == 0:
            print(f"No rows found for feature: {feature}")
            continue

        print(f"Processing feature: {feature}, Rows found: {current_count}")

        if current_count > 500:
            # Undersample
            keep = feature_df.sample(n=500, random_state=42)
            discard = feature_df.drop(keep.index)

            # Track deleted counters
            all_operations['deleted_counters'].extend(discard['Counter'].tolist())

            # Update label dataframe
            label_df = label_df.drop(discard.index)

            print(f"Undersampled {current_count - 500} rows for feature: {feature}")

        elif current_count < 500:
            # Oversample
            needed = 500 - current_count
            duplicates = feature_df.sample(n=needed, replace=True, random_state=42)

            # Generate new counter values
            max_counter = label_df['Counter'].max()
            duplicates = duplicates.reset_index(drop=True)
            duplicates['Counter'] = [max_counter + i + 1 for i in range(len(duplicates))]

            # Add to label dataframe
            label_df = pd.concat([label_df, duplicates], ignore_index=True)

            # Create corresponding transcript entries
            transcript_copies = pd.merge(duplicates[['Counter']], transcript_df,
                                         left_on='Counter', right_on='Number',
                                         how='left').drop('Counter', axis=1)
            all_operations['added_entries'] = pd.concat([all_operations['added_entries'], transcript_copies])

            print(f"Oversampled {needed} rows for feature: {feature}")

    # Apply all changes to transcript
    transcript_df = transcript_df[~transcript_df['Number'].isin(all_operations['deleted_counters'])]
    transcript_df = pd.concat([transcript_df, all_operations['added_entries']], ignore_index=True)

    # Save results (preserving Counter column in output)
    label_df.to_excel('balanced_labels.xlsx', index=False)
    transcript_df.to_excel('balanced_transcripts.xlsx', index=False)

    return {
        'deleted_counters': all_operations['deleted_counters'],
        'added_count': len(all_operations['added_entries'])
    }


# Usage
result = process_feature_balance('updated_labels.xlsx', 'updated_transcript.xlsx')
print(f"Deleted {len(result['deleted_counters'])} counters")
print(f"Added {result['added_count']} new transcript entries")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from openpyxl.drawing.image import Image
from openpyxl import Workbook
import os

# Load data and skip first column
df = pd.read_excel('updated_labels.xlsx').iloc[:, 1:]  # Skip first column


# Helper function to get feature columns
def get_features(df):
    return df.columns.tolist()


# Modified functions to skip first column
def calculate_ratios(df):
    ratios = {}
    features = get_features(df)
    for column in features:
        total = len(df)
        ones = df[column].sum()
        zeros = total - ones
        ratio = ones / zeros if zeros != 0 else float('inf')
        ratios[column] = ratio
    return ratios


def analyze_features(df):
    # Calculate ratios
    ratios = calculate_ratios(df)

    print("Ratios of 1s to 0s for each column:")
    for column, ratio in ratios.items():
        print(f"{column}: {ratio:.4f}")

    # Count no-feature rows
    no_feature_rows = df[(df == 0).all(axis=1)].shape[0]
    print(f"\nNumber of rows with no features: {no_feature_rows}")

    # Plot ratios
    plt.figure(figsize=(10, 6))
    plt.bar(ratios.keys(), ratios.values(), color='skyblue')
    plt.xlabel('Features')
    plt.ylabel('Ratio (1s/0s)')
    plt.title('Feature Ratio Analysis')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for i, (col, ratio) in enumerate(ratios.items()):
        if ratio == float('inf'):
            plt.text(i, 0.5, 'All 1s', rotation=90, ha='center', va='bottom', color='red')

    plt.tight_layout()
    plt.show()


def correlation_analysis(df):
    features = get_features(df)

    # Pairwise correlations
    co_matrix = pd.DataFrame(index=features, columns=features)
    for f1, f2 in combinations(features, 2):
        count = df[(df[f1] == 1) & (df[f2] == 1)].shape[0]
        co_matrix.loc[f1, f2] = count
        co_matrix.loc[f2, f1] = count
    co_matrix = co_matrix.fillna(0).astype(int)

    # Save correlations
    co_df = pd.DataFrame([(f1, f2, co_matrix.loc[f1, f2])
                          for f1, f2 in combinations(features, 2)],
                         columns=['Feature1', 'Feature2', 'Count'])
    co_df.to_excel('pairwise_correlations.xlsx', index=False)

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(co_matrix.astype(float), annot=True, fmt=".0f",
                cmap="YlGnBu", cbar_kws={'label': 'Co-occurrence Count'})
    plt.title('Feature Co-occurrence Heatmap')
    plt.tight_layout()
    plt.show()

    # All combinations
    combo_counts = {}
    for r in range(2, len(features) + 1):
        for combo in combinations(features, r):
            count = df[list(combo)].all(axis=1).sum()
            combo_counts[' & '.join(combo)] = count

    combo_df = pd.DataFrame(list(combo_counts.items()),
                            columns=['Combination', 'Count'])
    combo_df.to_excel('all_combinations.xlsx', index=False)

    return co_df, combo_df


# Run analyses
analyze_features(df)
pairwise_df, combinations_df = correlation_analysis(df)

print("Pairwise correlations:")
print(pairwise_df)
print("\nAll combinations:")
print(combinations_df)
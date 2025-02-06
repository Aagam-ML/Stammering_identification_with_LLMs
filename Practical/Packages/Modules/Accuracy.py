import pandas as pd

def find_Accuracy(Assumption_file , GroundTruth_file):
    df1 = pd.read_excel(Assumption_file)
    df2 = pd.read_excel(GroundTruth_file)

    columns_to_compare = ["Prolongation", "Block", "SoundRep", "WordRep", "DifficultToUnderstand",
                          "Interjection"]  # Replace with your column names

    if len(df1) != len(df2):
        print("The files have different numbers of rows.")
    elif len(df1.columns) != len(df2.columns):
        print("The files have different numbers of columns.")
    else:
        # Initialize counters
        total_matches = 0
        total_values = 0

        for column in columns_to_compare:
            if column in df1.columns and column in df2.columns:
                # Count the number of matching values in this column
                matches = (df1[column] == df2[column]).sum()
                total_matches += matches
                total_values += len(df1[column])

                # Print column-wise accuracy
                column_accuracy = matches / len(df1[column]) * 100
                print(f"Accuracy for {column}: {column_accuracy:.2f}%")
            else:
                print(f"Column '{column}' not found in one of the files.")
        print(total_matches)
        print(total_values)
        # Calculate overall accuracy
        overall_accuracy = total_matches / total_values * 100
        print(f"Overall accuracy: {overall_accuracy:.2f}%")

find_Accuracy("/Volumes/HDD/Stammering_identification/Stammering_identification_with_LLMs/Practical/Packages/small_labels/New_Tiny_Stammering_Features_Analysis.xlsx","/Volumes/HDD/Stammering_identification/Stammering_identification_with_LLMs/Practical/SEP-28k_label copy.xlsx")
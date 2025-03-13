import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, hamming_loss, classification_report
import pandas as pd
from sklearn.metrics import hamming_loss, f1_score, classification_report


# Step 1: Load the BERT features
def load_features(file_path):
    return torch.load(file_path)


# Step 2: Prepare your labels (this part needs your actual labels data)
# Assuming labels are in a suitable format like a NumPy array or a list of lists
def load_labels(file_path):
    # This should be replaced with how you actually load your labels
    # For example: return np.load('your_labels.npy')
    df = pd.read_excel(file_path)
    # Select the first 5001 rows and convert to a NumPy array
    labels = df.iloc[0:5001].values

    return labels


# Main function to execute the workflow
def main():
    file_path = '/Volumes/HDD/Stammering_identification/Stammering_identification_with_LLMs/Practical/BERT/bert_features_1.pt'
    label_path ="/Volumes/HDD/Stammering_identification/Stammering_identification_with_LLMs/Practical/Random_Forest/SEP-28k_label.xlsx"
    features = load_features(file_path)
    labels = load_labels(label_path)

    # Convert tensor to numpy if not already done
    if isinstance(features, torch.Tensor):
        features = features.numpy()

    # Step 3: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Step 4: Initialize and train the multi-label model
    model = MultiOutputClassifier(RandomForestClassifier(n_estimators=6, random_state=42))
    model.fit(X_train, y_train)

    # Step 5: Make predictions and evaluate the model
    y_pred = model.predict(X_test)

    from sklearn.metrics import hamming_loss, f1_score, classification_report

    # Check the shape and type of y_test and y_pred
    print("y_test shape:", y_test.shape)
    print("y_pred shape:", y_pred.shape)

    # Ensure both are numpy arrays (sometimes they might be lists or other formats)
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    try:
        print("Hamming Loss:", hamming_loss(y_test, y_pred))
        print("F1 Score (Micro):", f1_score(y_test, y_pred, average='micro'))
        print("F1 Score (Macro):", f1_score(y_test, y_pred, average='macro'))
        print("Classification Report:\n", classification_report(y_test, y_pred))
    except Exception as e:
        print("Error in metric calculation:", e)
        # Further investigation into array shapes and data types
        print("y_test dtype:", y_test.dtype)
        print("y_pred dtype:", y_pred.dtype)
        print("Unique values in y_test:", np.unique(y_test))
        print("Unique values in y_pred:", np.unique(y_pred))


if __name__ == "__main__":
    main()

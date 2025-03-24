import pandas as pd
import torch
import numpy as np
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2


def get_class_weights(y):
    """Compute class weights dynamically to handle class imbalance."""
    weights = []
    for i in range(y.shape[1]):
        counts = np.bincount(y[:, i], minlength=2)  # Ensure length 2
        counts = np.clip(counts, 1, None)  # Avoid division by zero
        weights.append({
            0: len(y) / (2 * counts[0]),  # Weight for class 0
            1: len(y) / (2 * counts[1])   # Weight for class 1
        })
    return weights


def weighted_bce(class_weights):
    """Custom loss function to handle per-label weights."""
    def loss(y_true, y_pred):
        total_loss = 0.0
        for i in range(len(class_weights)):
            w0 = class_weights[i][0]
            w1 = class_weights[i][1]
            y_t = y_true[:, i]
            y_p = y_pred[:, i]
            bce = tf.keras.losses.binary_crossentropy(y_t, y_p)
            weights = w1 * y_t + w0 * (1 - y_t)  # Apply weights based on true class
            total_loss += tf.reduce_mean(bce * weights)
        return total_loss / len(class_weights)
    return loss


def train_model(X_train, y_train, class_weights, X_test, y_test):
    """Train a neural network model with dynamic class weights and evaluation."""
    # Define the neural network
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(6, activation='sigmoid')  # Output layer for 6 labels
    ])

    # Compile the model with custom loss and metrics
    model.compile(
        optimizer=Nadam(learning_rate=0.001),
        loss=weighted_bce(class_weights),
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )

    # Set up callbacks for early stopping and learning rate reduction
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=1000,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, lr_scheduler]
    )

    # Evaluate the model on the test set
    loss, accuracy, precision, recall = model.evaluate(X_test, y_test, batch_size=32)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Test Precision: {precision * 100:.2f}%")
    print(f"Test Recall: {recall * 100:.2f}%")

    # Get predictions and apply thresholding
    predictions = model.predict(X_test, batch_size=32)
    binary_predictions = (predictions >= 0.5).astype(int)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, binary_predictions))

    return model, history, binary_predictions


def main():
    # Load BERT features
    try:
        features = torch.load(
            '/Volumes/HDD/Stammering_identification/Stammering_identification_with_LLMs/Practical/DeepNeuralNetworkApproach/bert_features_779.pt'
        ).numpy()  # Convert to numpy array
    except Exception as e:
        print(f"Error loading BERT features: {e}")
        return

    # Load labels
    try:
        labels_df = pd.read_excel(
            "/Volumes/HDD/Stammering_identification/Stammering_identification_with_LLMs/Practical/Data_imbalance/balanced_dataset_with_all_combinations.xlsx"
        )
    except Exception as e:
        print(f"Error loading labels: {e}")
        return

    # Extract label columns
    label_columns = ['Prolongation', 'Block', 'SoundRep', 'WordRep', 'DifficultToUnderstand', 'Interjection']
    y = labels_df[label_columns].values

    # Ensure features and labels have the same number of samples
    if features.shape[0] != y.shape[0]:
        print(f"Mismatch in number of samples between features ({features.shape[0]}) and labels ({y.shape[0]})")
        return

    # Split data using multi-label stratified split
    try:
        X_train, y_train, X_test, y_test = iterative_train_test_split(features, y, test_size=0.2)
    except Exception as e:
        print(f"Error during train-test split: {e}")
        return

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Compute class weights dynamically
    class_weights = get_class_weights(y_train)

    # Train the model
    model, history, binary_predictions = train_model(X_train, y_train, class_weights, X_test, y_test)


if __name__ == "__main__":
    main()
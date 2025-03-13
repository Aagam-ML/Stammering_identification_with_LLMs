import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, hamming_loss, f1_score, classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
import warnings

warnings.filterwarnings("ignore")


# Improved Model Architecture with BatchNorm and Deeper Layers
class EnhancedClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.layer2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.layer3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.output = nn.Linear(128, output_size)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.bn1(self.layer1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.layer2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.layer3(x)))
        return self.sigmoid(self.output(x))


# Focal Loss for Class Imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.BCELoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


# Data Loading with Enhanced Validation
def load_data(feature_path, label_path):
    # Load features
    features = torch.load(feature_path).numpy()

    # Load and analyze labels
    df = pd.read_excel(label_path)
    labels = df.iloc[10001:15001].values

    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Split into train (60%), validation (20%), test (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# Threshold Optimization using Youden's J Index
def optimize_thresholds(y_true, y_probs):
    thresholds = []
    for i in range(y_true.shape[1]):
        fpr, tpr, thresh = roc_curve(y_true[:, i], y_probs[:, i])
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        thresholds.append(thresh[optimal_idx])
    return np.array(thresholds)


def main():
    # Load and prepare data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(
        '/Volumes/HDD/Stammering_identification/Stammering_identification_with_LLMs/Practical/BERT/bert_features_3.pt',
        "/Volumes/HDD/Stammering_identification/Stammering_identification_with_LLMs/Practical/Random_Forest/SEP-28k_label.xlsx"
    )

    # Convert to PyTorch tensors
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                torch.tensor(y_val, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                 torch.tensor(y_test, dtype=torch.float32))

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)
    test_loader = DataLoader(test_dataset, batch_size=128)

    # Initialize model with class weights
    model = EnhancedClassifier(X_train.shape[1], y_train.shape[1])
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train.ravel())
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    criterion = FocalLoss(alpha=0.25, gamma=2)

    # Optimizer and Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                              steps_per_epoch=len(train_loader),
                                              epochs=50)

    # Training with Early Stopping
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 7
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(50):
        # Training
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        # Update history
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        print(f"Epoch {epoch + 1:02d}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Early Stopping Check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))

    # Threshold Optimization
    model.eval()
    with torch.no_grad():
        y_probs = model(torch.tensor(X_test, dtype=torch.float32)).numpy()

    optimal_thresholds = optimize_thresholds(y_test, y_probs)
    y_pred = (y_probs > optimal_thresholds).astype(int)

    # Evaluation Metrics
    print("\nOptimized Class Thresholds:", optimal_thresholds)
    print("Test Hamming Loss:", hamming_loss(y_test, y_pred))
    print("Micro F1 Score:", f1_score(y_test, y_pred, average='micro'))
    print("Macro F1 Score:", f1_score(y_test, y_pred, average='macro'))
    print("Subset Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Plot Training History
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title("Training History")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
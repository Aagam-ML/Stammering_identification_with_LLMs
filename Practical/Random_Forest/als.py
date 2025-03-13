import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import hamming_loss, f1_score, classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from collections import defaultdict

warnings.filterwarnings("ignore")


# -------------------- Data Loading --------------------
def load_features(file_path):
    features = torch.load(file_path)
    return features.numpy() if isinstance(features, torch.Tensor) else features


def load_labels(file_path):
    df = pd.read_excel(file_path)
    labels = df.iloc[10001:15001].values

    # Class distribution analysis
    class_counts = np.sum(labels, axis=0)
    total_samples = labels.shape[0]
    class_percentages = (class_counts / total_samples) * 100

    class_dist = pd.DataFrame({
        'Class': [f'Class {i}' for i in range(6)],
        'Count': class_counts,
        'Percentage': class_percentages
    })

    print("\nClass Distribution Analysis:")
    print(class_dist)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_dist['Class'], class_dist['Count'])
    plt.title('Class Distribution')
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)

    for bar, percentage in zip(bars, class_dist['Percentage']):
        plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                 f'{percentage:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    return labels


# -------------------- Enhanced Model --------------------
class ImprovedClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        self.block2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.residual = nn.Linear(input_size, 256)
        self.output = nn.Linear(256, output_size)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        res = self.residual(x)[:, :256]
        x = x2 + res
        return torch.sigmoid(self.output(x))


# -------------------- Focal Loss --------------------
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


# -------------------- Main Pipeline --------------------
def main():
    # Load and preprocess data
    features = load_features(
        '/Volumes/HDD/Stammering_identification/Stammering_identification_with_LLMs/Practical/BERT/bert_features_3.pt')
    labels = load_labels(
        "/Volumes/HDD/Stammering_identification/Stammering_identification_with_LLMs/Practical/Random_Forest/SEP-28k_label.xlsx")

    # Multilabel stratified splitting
    mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_index, test_index = next(mskf.split(features, labels))
    X_train_val, X_test = features[train_index], features[test_index]
    y_train_val, y_test = labels[train_index], labels[test_index]

    # Further split train/val
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                      test_size=0.2, random_state=42)

    # Normalization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Initialize model and training
    model = ImprovedClassifier(X_train.shape[1], y_train.shape[1])
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)

    # Enhanced class weighting
    label_counts = np.sum(y_train.numpy(), axis=0)
    class_weights = torch.tensor(1.0 / np.sqrt(label_counts + 1e-6), dtype=torch.float32)
    criterion = FocalLoss(alpha=class_weights)

    # Create weighted sampler
    sample_weights = y_train @ class_weights.numpy()
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # Modified DataLoader with sampler
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)

    # Training loop with early stopping
    best_f1 = 0
    patience = 7
    no_improve = 0
    history = defaultdict(list)

    for epoch in range(100):
        model.train()
        epoch_loss = 0.0

        # Training
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
            val_preds = (val_outputs > 0.5).float()
            val_f1 = f1_score(y_val, val_preds, average='micro')

        scheduler.step(val_loss)

        # Early stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            no_improve = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Track history
        history['train_loss'].append(epoch_loss / len(train_loader))
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)

        print(
            f"Epoch {epoch + 1}: Train Loss {history['train_loss'][-1]:.4f} | Val Loss {val_loss:.4f} | Val F1 {val_f1:.4f}")

    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))

    # Class-specific threshold tuning
    model.eval()
    with torch.no_grad():
        val_probs = model(X_val).numpy()

    thresholds = []
    for i in range(6):
        fpr, tpr, thresh = roc_curve(y_val[:, i], val_probs[:, i])
        optimal_idx = np.argmax(tpr - fpr)
        thresholds.append(thresh[optimal_idx])

    print("\nOptimal Class Thresholds:", [f"{t:.3f}" for t in thresholds])

    # Final evaluation
    with torch.no_grad():
        test_probs = model(X_test).numpy()

    y_pred = np.zeros_like(test_probs)
    for i in range(6):
        y_pred[:, i] = (test_probs[:, i] > thresholds[i]).astype(int)

    print("\nFinal Evaluation:")
    print("Hamming Loss:", hamming_loss(y_test, y_pred))
    print("Micro F1:", f1_score(y_test, y_pred, average='micro'))
    print("Macro F1:", f1_score(y_test, y_pred, average='macro'))
    print("\nClassification Report:\n", classification_report(y_test, y_pred,
                                                              target_names=["0", "1", "2", "3", "4", "5"]))

    # Plot metrics
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training History')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['val_f1'], label='Validation F1')
    plt.title('Validation F1 Score')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Confusion matrices
    fig, axes = plt.subplots(1, 6, figsize=(15, 3), sharey=True)
    fig.suptitle('Confusion Matrices for Each Label')
    for i in range(6):
        ax = axes[i]
        cm = confusion_matrix(y_test[:, i], y_pred[:, i])
        sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, ax=ax, cbar=False)
        ax.set_title(f"Class {i}")
        ax.set_xlabel('Predicted')
        if i == 0: ax.set_ylabel('True')
        ax.set_xticklabels(['False', 'True'])
        ax.set_yticklabels(['False', 'True'])
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    main()
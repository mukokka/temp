import os
import sys
import time
from pathlib import Path
import numpy as np
import mne
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, LeaveOneGroupOut, StratifiedKFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils import shuffle

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', buffering=1, encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

class CNN_EEG(nn.Module):
    def __init__(self):
        super(CNN_EEG, self).__init__()

        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 2), stride=1, padding='same'),
            nn.ELU(),
            nn.Conv2d(16, 16, kernel_size=(1, 2), stride=1, padding='same'),
            nn.ELU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, 2), stride=1, padding='same'),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=1, padding='same'),
            nn.ELU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(4, 1), stride=1, padding='same'),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=(4, 1), stride=1, padding='same'),
            nn.ELU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))
        )

        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(4, 1), stride=1, padding='same'),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=(4, 1), stride=1, padding='same'),
            nn.ELU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))
        )

        # Fully Connected layers
        self.fc1 = nn.Linear(4736, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 2)

    def forward(self, x):
        #print(f"Input shape: {x.shape}")         torch.Size([16, 1, 8, 600])

        x = self.block1(x)
        #print(f"After Block 1: {x.shape}")       torch.Size([16, 16, 4, 600])

        x = self.block2(x)
        #print(f"After Block 2: {x.shape}")       torch.Size([16, 32, 2, 600])

        x = self.block3(x)
        #print(f"After Block 3: {x.shape}")       torch.Size([16, 32, 2, 150])

        x = self.block4(x)
        #print(f"After Block 4: {x.shape}")       torch.Size([16, 64, 2, 37])

        x = x.view(x.size(0), -1)  # Flatten
        #print(f"After Flatten: {x.shape}")       torch.Size([16, 4736])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x_cl = F.relu(self.fc3(x))
        outputs = self.out(x_cl)

        return outputs, x_cl

def print_label_distribution(y, name=""):
    values, counts = np.unique(y, return_counts=True)
    print(f"{name} labels:")
    for v, c in zip(values, counts):
        print(f"  label {v}: {c} samples")

def sliding_window_epochs(X, y, window_size, step_size):
    X_new, y_new = [], []
    for xi, yi in zip(X, y):
        length = xi.shape[1]
        for start in range(0, length - window_size + 1, step_size):
            segment = xi[:, start:start + window_size]
            X_new.append(segment)
            y_new.append(yi)
    return np.stack(X_new), np.array(y_new)

def train_one_fold(model, train_loader, val_loader, num_epochs, lr, patience):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    softmax_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Training
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_loss = np.inf

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs, cl_features = model(inputs)

            loss_softmax = softmax_criterion(outputs, targets)
            loss = loss_softmax

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

        avg_train_loss = train_loss / total
        train_acc = correct / total

        # validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, cl_features = model(inputs)

                loss_softmax = softmax_criterion(outputs, targets)
                loss = loss_softmax

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, dim=1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)

        avg_val_loss = val_loss / total
        val_acc = correct / total

        print(f"Epoch [{epoch}/{num_epochs}]  "
              f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2%} | "
              f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2%}")

        # save optimized model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        #if avg_val_loss < best_val_loss:
        #    best_val_loss = avg_val_loss
        #    best_model_state = model.state_dict()
        #    patience_counter = 0
        #else:
        #    patience_counter += 1

        # early stopping
        if patience_counter >= patience:
            print(f" Early stopping at epoch {epoch} — best val acc: {best_val_acc:.2%}")
            break

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

    return best_model_state, best_val_acc, train_losses, val_losses, train_accs, val_accs

def test_model(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    model = model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs,_ = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc = correct / total
    return test_acc

def plot_train_val_curves(train_values, val_values, title, ylabel, save_path):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_values) + 1)

    plt.plot(epochs, train_values, label='Train', linewidth=2)
    plt.plot(epochs, val_values, label='Validation', linewidth=2)

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved figure to {save_path}")
    else:
        plt.show()

    plt.close()

def main():
    WorkingDir = Path(r'C:\Users\USER\python_files\test\EEG+fNIRS\02 M2NN')
    log_path = WorkingDir / 'train_log.txt'
    sys.stdout = Logger(log_path)

    number = 29
    sub_list = ['sub' + f'{i:02d}' for i in range(1, number + 1)]

    save_dir = 'results/figures'
    os.makedirs(save_dir, exist_ok=True)
    figures_dir = WorkingDir / 'results/figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_test_accs=[]
    all_test_accs_std=[]

    data_root = Path(r"C:\Users\USER\python_files\test\EEG+fNIRS\02 M2NN\preprocessed_epochs")
    for sub in sub_list:
        file_path = data_root / sub / "epochs_eeg-epo.fif"
        if not file_path.exists():
            print(f"file doesn't exist: {file_path}")
            continue

        print(f"load {sub} HBR data...")
        epochs = mne.read_epochs(file_path, preload=True)
        X = epochs.get_data()        # shape: X=(60, 24, 101)
        y = epochs.events[:, -1]     # shape: y=(60,)

        print(f"shape of {sub} eeg: X={X.shape}, y={y.shape}")

        X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

        X_test_win, y_test_win = sliding_window_epochs(X_test, y_test, window_size=600, step_size=200)
        X_test_tensor = torch.tensor(X_test_win, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test_win, dtype=torch.long)

        #X_test_tensor = X_test_tensor.permute(0, 2, 1)
        X_test_tensor = X_test_tensor.unsqueeze(1)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=16)

        print_label_distribution(y, name="total")
        print_label_distribution(y_train_full, name="train_full")
        print_label_distribution(y_test, name="test")

        test_accs=[]
        # 9-fold stratified CV on training set
        print("9-fold Stratified CV on training set:")
        skf = StratifiedKFold(n_splits=9, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
            print(f"📂 Fold {fold + 1}")
            X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]
            y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]
            print_label_distribution(y_train, name="train")
            print_label_distribution(y_val, name="val")
            X_train_win, y_train_win = sliding_window_epochs(X_train, y_train, window_size=600, step_size=200)
            X_val_win, y_val_win = sliding_window_epochs(X_val, y_val, window_size=600, step_size=200)

            print(X_train.shape)
            print(X_val.shape)
            print(X_test.shape)

            print(X_train_win.shape)
            print(X_val_win.shape)
            print(X_test_win.shape)

            print(y_train_win.shape)
            print(y_val_win.shape)
            print(y_test_win.shape)

            X_tr_tensor, X_val_tensor = torch.tensor(X_train_win, dtype=torch.float32), torch.tensor(X_val_win, dtype=torch.float32)
            y_tr_tensor, y_val_tensor = torch.tensor(y_train_win, dtype=torch.long), torch.tensor(y_val_win, dtype=torch.long)

            #X_tr_tensor = X_tr_tensor.permute(0, 2, 1)
            #X_val_tensor = X_val_tensor.permute(0, 2, 1)

            X_tr_tensor = X_tr_tensor.unsqueeze(1)                        # (B, 1, 24, 30)
            X_val_tensor = X_val_tensor.unsqueeze(1)
            # https://github.com/jitalo333/2DCNN-emotion-classifier/blob/main/CNN_classifier.ipynb

            tr_dataset = TensorDataset(X_tr_tensor, y_tr_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

            train_loader = DataLoader(tr_dataset, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=16)


            EEG_2DCNN = CNN_EEG()

            best_model_state, best_val_acc, train_losses, val_losses, train_accs, val_accs = train_one_fold(EEG_2DCNN, train_loader, val_loader, num_epochs=200, lr=1e-3, patience=35)

            plot_train_val_curves(train_losses, val_losses, 'Loss Curve', 'Loss', figures_dir / f'{sub} loss_curve_fold{fold + 1}.png')
            plot_train_val_curves(train_accs, val_accs, 'Accuracy Curve', 'Accuracy', figures_dir / f'{sub} acc_curve_fold{fold + 1}.png')

            EEG_2DCNN.load_state_dict(best_model_state)
            test_acc= test_model(EEG_2DCNN, test_loader)

            test_accs.append(test_acc)

            print(f"Fold {fold + 1} Test Accuracy: {test_acc:.2%}")

        avg_test_accs = np.mean(test_accs)
        std_test_accs = np.std(test_accs)
        print("All Fold Test Accuracies:", [f"{acc:.2%}" for acc in test_accs])
        print(f"Average Test Accuracy across folds: {avg_test_accs:.2%} ± {std_test_accs:.2%}")
        all_test_accs.append(avg_test_accs)
        all_test_accs_std.append(std_test_accs)

    avg_test_acc_for_all_subs = np.mean(all_test_accs)
    std_test_acc_for_all_subs = np.std(all_test_accs)
    print("All Fold Test Accuracies:", [f"{acc:.2%}" for acc in all_test_accs])
    print(f"Average Test Accuracy across folds: {avg_test_acc_for_all_subs:.2%} ± {std_test_acc_for_all_subs:.2%}")

    plt.figure(figsize=(10, 6))
    x_labels = [str(i + 1) for i in range(29)]
    y_values = all_test_accs

    plt.bar(range(len(y_values)), y_values, yerr=all_test_accs_std, color='skyblue', alpha=0.8)
    plt.xticks(range(len(y_values)), x_labels, rotation=45, ha='right', fontsize=10)
    plt.xlabel('Subjects')
    plt.ylabel('Average Test Accuracy (%)')
    plt.title('Average Test Accuracy for Each subject')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

if __name__ == "__main__":
    main()

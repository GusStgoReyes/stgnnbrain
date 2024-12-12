from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import torch
import torch.nn.functional as F

def calculate_metrics(y_true, y_pred, y_prob):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_prob[:, 1])
    }


def train(model, train_loader, criterion, optimizer, device, num_epochs=100, min_delta = 0.001, patience = 10):
    best_train_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_losses = []

        for batch in train_loader:
            optimizer.zero_grad()
            # Move batch to the device
            batch = batch.to(device)

            # Forward pass
            output = model(batch.x, batch.edge_index, batch.edge_attr)

            # Compute loss
            loss = criterion(output, batch.y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # Early stopping check
        if avg_train_loss < best_train_loss - min_delta:
            best_train_loss = avg_train_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping...')
                break

        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {round(avg_train_loss, 3)}')

    return avg_train_loss

def evaluate(model, val_loader, criterion, device):
    # Switch model to evaluation mode
    model.eval()

    # Placeholders for metrics
    all_true_labels = []
    all_pred_probs = []
    all_pred_labels = []
    val_losses = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            # Forward pass
            output = model(batch.x, batch.edge_index, batch.edge_attr)

            val_loss = criterion(output, batch.y)
            val_losses.append(val_loss.item())

            # Predictions
            val_pred_probs = F.softmax(output, dim=1)
            val_pred_labels = torch.argmax(val_pred_probs, dim=1)

            # Collect outputs
            all_true_labels.append(batch.y.cpu().numpy())
            all_pred_probs.append(val_pred_probs.cpu().numpy())
            all_pred_labels.append(val_pred_labels.cpu().numpy())

    # Aggregate results
    all_true_labels = np.concatenate(all_true_labels, axis=0)
    all_pred_probs = np.concatenate(all_pred_probs, axis=0)
    all_pred_labels = np.concatenate(all_pred_labels, axis=0)

    # Compute metrics
    metrics = calculate_metrics(all_true_labels, all_pred_labels, all_pred_probs)
    avg_val_loss = np.mean(val_losses)

    return avg_val_loss, metrics


# gnnsyn/train/trainer.py

import torch
import torch.nn.functional as F

def train_one_epoch_full(model, dataloader, criterion, optimizer, device='cpu'):
    model.train()
    total_loss = 0
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data)  # full-batch forward
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

def eval_model_full(model, dataloader, criterion, device='cpu'):
    model.eval()
    total_loss = 0
    correct = 0
    total_nodes = 0
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            total_loss += loss.item()

            pred = out.argmax(dim=-1)
            correct += pred.eq(data.y).sum().item()
            total_nodes += data.y.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total_nodes if total_nodes > 0 else 0.0
    return avg_loss, accuracy

def train_one_epoch_ns(model, x_all, y_all, train_loader, optimizer, device='cpu'):
    model.train()
    total_loss = 0
    for batch_size, n_id, adjs in train_loader:
        adjs = [adj.to(device) for adj in adjs]
        x_mb = x_all[n_id].to(device)
        y_mb = y_all[n_id[:batch_size]].to(device)

        optimizer.zero_grad()
        out = model.forward_ns(x_mb, adjs)  # neighbor sampling forward
        loss = F.cross_entropy(out, y_mb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)

def eval_model_ns(model, x_all, y_all, loader, device='cpu'):
    model.eval()
    total_loss = 0
    correct = 0
    total_nodes = 0
    with torch.no_grad():
        for batch_size, n_id, adjs in loader:
            adjs = [adj.to(device) for adj in adjs]
            x_mb = x_all[n_id].to(device)
            y_mb = y_all[n_id[:batch_size]].to(device)

            out = model.forward_ns(x_mb, adjs)
            loss = F.cross_entropy(out, y_mb)
            total_loss += loss.item()

            pred = out.argmax(dim=-1)
            correct += pred.eq(y_mb).sum().item()
            total_nodes += batch_size

    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
    accuracy = correct / total_nodes if total_nodes > 0 else 0.0
    return avg_loss, accuracy

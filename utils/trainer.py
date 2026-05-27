import torch


def train_model(model, train_loader, val_loader, criterion, optimizer, ckpt_dir, epochs, device):

    best = 0.0

    for epoch in range(epochs):

        model.train()
        total = 0

        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            preds, feats = model(imgs)
            loss, *_ = criterion(preds, feats, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item()

        torch.save(model.state_dict(), f"{ckpt_dir}/last.pth")

        print(f"Epoch {epoch}: loss {total/len(train_loader):.4f}")
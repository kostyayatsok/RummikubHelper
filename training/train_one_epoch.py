def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    
    for images, targets in data_loader:
        images = [image.to(device) for image in images]
        targets = [{k : v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

def train():
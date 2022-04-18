from data_utils import FasterRCNNSynthecticDataset
from data_utils.coco_format_dataset import CocoFormatDataset
from training.engine import train_one_epoch
from training.evaluate import evaluate
from data_utils import SyntheticConfig
from training import utils
import torch
from models import FasterRCNN
import wandb

torch.manual_seed(42)

LOCAL = True
SYNTHETIC = False

batch_size = 16 if not LOCAL else 2

if SYNTHETIC:
    num_epochs = 1000
    lr_step_size = 160

    train_data_config = SyntheticConfig()
    train_data_config.dataset_size = 512 if not LOCAL else 4
    dataset = FasterRCNNSynthecticDataset(train_data_config)
else:
    num_epochs = 100
    lr_step_size = 40

    dataset = CocoFormatDataset(
        annotation_path="images/rummy-6-censored/train/_annotations.coco.json",
        images_dir="images/rummy-6-censored/train",
        img_sz=640
    )    
# test_data_config = SyntheticConfig()
# test_data_config.dataset_size = 64 if not LOCAL else 4
# dataset_test = FasterRCNNSynthecticDataset(test_data_config)

dataset_test = CocoFormatDataset(
    annotation_path="images/coco-test/_annotations.coco.json",
    images_dir="images/coco-test/",
    img_sz=640
)


data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, num_workers=2,
    collate_fn=utils.collate_fn
)
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=batch_size, num_workers=2,
    collate_fn=utils.collate_fn
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = FasterRCNN()
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.02,
                            momentum=0.9, weight_decay=0.0001)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=lr_step_size,
                                               gamma=0.1)
wandb.init(project="Rummy", name="FasterRCNN-v&c-real")

best_precision = 0
for epoch in range(num_epochs):
    metrics = train_one_epoch(
        model, optimizer, data_loader,
        device, epoch, print_freq=500
    )

    wandb_log = {"epoch":epoch}
    for name, value in metrics.meters.items():
        wandb_log.update({name: value.avg})
    wandb.log(wandb_log)

    lr_scheduler.step()
    val_log = evaluate(model, data_loader_test, device, epoch)

    if val_log[f"precision@0.95"] > best_precision:
        torch.save(model.state_dict(), "model.pt")
        best_precision = val_log[f"precision@0.95"]
    if epoch % 10:
        wandb.save("model.pt", policy="now")
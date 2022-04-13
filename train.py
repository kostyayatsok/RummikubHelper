from data_utils import FasterRCNNSynthecticDataset
from training.engine import train_one_epoch, evaluate
from data_utils import SyntheticConfig
from training import utils
import torch
from models import FasterRCNN

torch.manual_seed(42)

train_data_config = SyntheticConfig()
train_data_config.dataset_size = 100
dataset = FasterRCNNSynthecticDataset(train_data_config)

test_data_config = SyntheticConfig()
test_data_config.dataset_size = 10
dataset_test = FasterRCNNSynthecticDataset(test_data_config)

data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn, pin_memory=True
)
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn, pin_memory=True
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = FasterRCNN()
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

num_epochs = 10
for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    lr_scheduler.step()
    evaluate(model, data_loader_test, device=device)
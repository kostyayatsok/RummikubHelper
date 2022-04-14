from data_utils import FasterRCNNSynthecticDataset
from training.engine import train_one_epoch
from training.evaluate import evaluate
from data_utils import SyntheticConfig
from training import utils
import torch
from models import FasterRCNN, LightFasterRCNN
import wandb

torch.manual_seed(42)

test_data_config = SyntheticConfig()
test_data_config.dataset_size = 2
dataset_test = FasterRCNNSynthecticDataset(test_data_config)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, num_workers=0,
    collate_fn=utils.collate_fn, pin_memory=False
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = LightFasterRCNN()
model.to(device)

wandb.init(project="Rummy", name="debug")
evaluate(model, data_loader_test, device, 0)
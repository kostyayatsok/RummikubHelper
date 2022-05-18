from data_utils import FasterRCNNSynthecticDataset, CocoFormatDataset
from training.engine import train_one_epoch
from training.evaluate import evaluate
from data_utils import SyntheticConfig
from training import utils
import torch
from models import FasterRCNN
import wandb
import pandas as pd

torch.manual_seed(42)

# test_data_config = SyntheticConfig()
# test_data_config.dataset_size = 2
# dataset_test = FasterRCNNSynthecticDataset(test_data_config)

dataset_test = CocoFormatDataset(
    # annotation_path="images/coco-test-640/_annotations.coco.json",
    annotation_path="images/coco-test-640/_one_class.coco.json",
    # annotation_path="images/coco-test-640/_colors.coco.json",
    # annotation_path="images/coco-test-1280/_values.coco.json",
    images_dir="images/coco-test-640/",
    img_sz=640,
    relabel=False
)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=4, num_workers=2,
    collate_fn=utils.collate_fn, pin_memory=False
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# weights_file = "checkpoints/FasterRCNN-values-and-colours-1604.pt"
# weights_file = "checkpoints/FasterRCNN-values-and-colours-1604.pt"
# weights_file = wandb.restore('model.pt', run_path="kostyayatsok/Rummy/1je8atim", replace=True).name
# weights_file = wandb.restore('model.pt', run_path="kostyayatsok/Rummy/1qhtyzl2", replace=True).name
if False:
    weights_file = wandb.restore('model.pt', run_path="kostyayatsok/Rummy/1l7luhf0", replace=True).name

    model = FasterRCNN()
    model.load_state_dict(torch.load(weights_file, map_location=device))
    model.to(device)

    wandb.init(project="RummyTest", name="FasterRCNN-v&c-mix-960")
    evaluate(model=model, loader=data_loader_test, device=device, epoch=-1)
else:
    name = 'tiles_yolox_tiny'
    model_out = pd.read_json(f"predictions/{name}_predictions.bbox.json")
    wandb.init(project="RummyTest", name=name)
    evaluate(model_out=model_out, loader=data_loader_test, device=device, epoch=-1)

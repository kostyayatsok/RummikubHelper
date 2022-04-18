import torch
from torch.utils.data import Dataset
from data_utils.coco_format_dataset import CocoFormatDataset
from data_utils.fasterRCNN_synthetic_dataset import FasterRCNNSynthecticDataset

from data_utils.synthetic_dataset import SyntheticConfig

class MixDataset(Dataset):
    def __init__(self, annotation_path, images_dir, img_sz=640, synth_config=SyntheticConfig()):
        self.real_dataset = CocoFormatDataset(annotation_path, images_dir, img_sz)
        self.synth_dataset = FasterRCNNSynthecticDataset(synth_config)
    def __len__(self):
        return self.synth_dataset.__len__()*2
    def __getitem__(self, idx):
        if idx % 2 == 0:
            return self.real_dataset.__getitem__(torch.randint(0, self.real_dataset.__len__()-1, (1,)))
        else:
            return self.synth_dataset.__getitem__(torch.randint(0, self.synth_dataset.__len__()-1, (1,)))
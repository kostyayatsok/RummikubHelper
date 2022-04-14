if __name__ == "__main__":
    import sys
    sys.path.append('.')

import torch
import numpy as np
from data_utils.synthetic_dataset import SyntheticConfig, SyntheticDataset
from utils.plot import plot

class FasterRCNNSynthecticDataset(SyntheticDataset):
    def __init__(self, config=SyntheticConfig()):
        super().__init__(config)
    def __getitem__(self, idx: int):
        image, labels = super().__getitem__(idx)
        img_sz = image.size(-1)

        labels = {
            "boxes": torch.tensor([
                [
                    int(label["x0"]*img_sz),
                    int(label["y0"]*img_sz),
                    int((label["x0"]+label["w"])*img_sz),
                    int((label["y0"]+label["h"])*img_sz)
                ] for label in labels
            ]),
            "labels": torch.tensor([
                label["label"] for label in labels
            ]),
            "image_id": torch.tensor([
                idx
            ]),
            "area": torch.tensor([
                label["w"]*img_sz * label["h"]*img_sz \
                                        for label in labels
            ]),
            "iscrowd": torch.tensor(
                [False] * len(labels)
            ),
        }
        return image, labels


if __name__ == "__main__":
    torch.manual_seed(42)

    dataset = FasterRCNNSynthecticDataset()
    id2label = {v:k for k, v in dataset.to_id.items()}
    for image, label in dataset: pass
        # label = np.array([[id_, box[0], box[1], box[2], box[3]] \
        #             for id_, box in zip(label["labels"], label["boxes"])], dtype=int)
        # plot(image.numpy().transpose((1, 2, 0)), label, id2label)
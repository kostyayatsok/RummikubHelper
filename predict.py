import argparse
from glob import glob
import torch
from models import FasterRCNN
from torchvision.io import read_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="images/coco-test/")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--output", type=str, default="prediction.json")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = FasterRCNN()
    model.load_state_dict(torch.load(args.chekpoint))
    model.eval()

    for image_path in glob(f"{args.path}/*.jpg"):
        image = read_image(image_path).to(device)/255.
        out = model(image)
        out[""]
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import argparse
import glob
import os
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    parser.add_argument("--n_examples", type=int, default=25)
    parser.add_argument("--output")
    parser.add_argument("--nrow", type=int, default=5)
    
    args = parser.parse_args()

    os.makedirs('/'.join(args.output.split('/')[:-1]), exist_ok=True)
    imgs = [torch.from_numpy(plt.imread(path).transpose((2, 0, 1))) \
                    for path in glob.glob(args.path + "/*")[:args.n_examples]]
    grid = make_grid(torch.stack(imgs, dim=0), nrow=args.nrow)
    plt.imsave(args.output, grid.numpy().transpose((1, 2, 0)))


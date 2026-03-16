#coding=gbk
import argparse
import os
import torch

from train_utils import evaluate
from my_dataset import DriveDataset
import transforms as T


class SegmentationPresetEval:
    def __init__(self, base_size, test_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.RandomResize(base_size, base_size),
	    T.CenterCrop(test_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    assert os.path.exists(args.weights), f"weights {args.weights} not found."

    # segmentation nun_classes + background
    num_classes = args.num_classes + 1
    base_size =  320
    test_size = 256


    # 用来保存训练以及验证过程中信息
    #results_file = "text_resultcto.txt"
    test_dataset = DriveDataset(args.data_path,
                               flag='test',
                               transforms=SegmentationPresetEval(base_size=base_size,test_size=test_size))

    num_workers = 0
    val_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=test_dataset.collate_fn)

    model = build_model(num_classes)
    model.load_state_dict(torch.load(args.weights, map_location=device)['model'])
    model.to(device)

    confmat, dice = evaluate(model, val_loader, device=device, num_classes=num_classes)
    print(confmat)
    print(f"Dice: {dice:.2f}\n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="pytorch ISIC test")

    parser.add_argument("--data-path", default="/userA02/lyp/datasets/ISIC2017", help="Datasets root")
    parser.add_argument("--weights", default="./save_weights/best_modelctoedg.pth")
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--device", default="cuda:2", help="training device")
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')

    args = parser.parse_args()
    main(args)

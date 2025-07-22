import os
import argparse
import torch
from torch.utils import model_zoo
import torch.nn as nn
import torchvision.transforms as transforms

from timm.models import create_model
from easyrobust.benchmarks import *
from easyrobust.models import *


parser = argparse.ArgumentParser(description='PyTorch ImageNet-CNS Inference')
parser.add_argument('--model', '-m', metavar='MODEL', default='',
                    help='model architecture')
parser.add_argument('--data_dir', default='benchmarks/data', 
                    help='benchmark datasets')
parser.add_argument('--ckpt_path', default='', type=str, 
                    help='model checkpoint for evaluation')
parser.add_argument('--output_dir', default='./evaluations',
                    help='path where to save csv files, empty for no saving')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--interpolation', default=3, type=int,
                    help='1: lanczos 2: bilinear 3: bicubic')
parser.add_argument('--input-size', default=224, type=int, 
                    help='images input size')
parser.add_argument('--crop-pct', default=0.875, type=float,
                metavar='N', help='input image center crop percent (for validation only)')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='number classes in dataset')
parser.add_argument('--global_pool', action='store_true', default=False,
                    help='use global pool for classification')
parser.add_argument('--class_mask', action='store_true', default=False,
                    help='mask classes that do not exist')
parser.add_argument('--use_ema', action='store_true', default=False,
                    help='use use_ema model state_dict')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='number of GPUS to use')


def main():
    args = parser.parse_args()

    #############################################################
    #         Load Model
    #############################################################
    print(f"Creating model: {args.model}")
    if args.ckpt_path:
        model = vit.__dict__[args.model](
            pretrained=False,
            global_pool=args.global_pool,
            num_classes=args.num_classes
        )
        if args.ckpt_path.startswith('http'):
            ckpt = model_zoo.load_url(args.ckpt_path)
        else:
            ckpt = torch.load(args.ckpt_path, map_location='cpu')
        if args.use_ema:
            assert 'state_dict_ema' in ckpt.keys() and ckpt['state_dict_ema'] is not None, 'no ema state_dict found!'
            state_dict = ckpt['state_dict_ema']
        else:
            if 'state_dict' in ckpt.keys():
                state_dict = ckpt['state_dict']
            elif 'model' in ckpt.keys():
                state_dict = ckpt['model']
            else:
                state_dict = ckpt
        model.load_state_dict(state_dict)
        model.eval()
    else:
        model = create_model(
            args.model,
            pretrained=True,
            num_classes=args.num_classes
        )
    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
    else:
        model = model.cuda()

    #############################################################
    #         Define Data Transform
    #############################################################
    test_transform = transforms.Compose([
        transforms.Resize(int(args.input_size/args.crop_pct), interpolation=args.interpolation),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    #############################################################
    #         Evaluate on Benchmarks
    #############################################################
    # evaluate_imagenet_val(model, 'benchmarks/data/imagenet-val', test_batchsize=args.batch_size, test_transform=test_transform)
    evaluate_imagenet_cns(model, args.data_dir, args, test_batchsize=args.batch_size, test_transform=test_transform)


if __name__ == '__main__':
    main()

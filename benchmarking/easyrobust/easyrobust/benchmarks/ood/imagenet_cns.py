import os
import re
import torch
import pandas as pd
from tqdm import tqdm

from torchvision import transforms, datasets
from timm.utils import AverageMeter, reduce_tensor, accuracy


def evaluate_imagenet_cns(model, data_dir, args, test_batchsize=128, test_transform=None, dist=False):
    if not os.path.exists(data_dir):
        print('{} is not exist. skip')
        return

    classes = sorted(entry.name for entry in os.scandir(data_dir) if entry.is_dir())
    cls_idx = [int(re.findall(r'\d+', c)[0]) for c in classes]
    
    if dist:
        assert torch.distributed.is_available() and torch.distributed.is_initialized()

    device = next(model.parameters()).device
    result_dict = {}

    if test_transform is None:
        insd_transform = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    else:
        insd_transform = test_transform

    dataset_insd = datasets.ImageFolder(data_dir, transform=insd_transform)

    sampler = None
    img_paths = [s[0] for s in dataset_insd.samples]
    if dist:
        sampler = torch.utils.data.DistributedSampler(dataset_insd, shuffle=False)

    insd_data_loader = torch.utils.data.DataLoader(
                    dataset_insd, sampler=sampler,
                    batch_size=test_batchsize,
                    num_workers=args.workers,
                    pin_memory=True,
                    drop_last=False
                )
            
    top1_m = AverageMeter()
    model.eval()
    df_all = pd.DataFrame({'pred_class': [], 'gt_class': [], 'img_dir': []})
    i_count = 0
    for input, target in tqdm(insd_data_loader):
        bs = len(input)
        input = input.to(device)
        with torch.no_grad():
            output = model(input)
        if args.class_mask:
            target = target.to(device)
            acc1, _ = accuracy(output[:, cls_idx], target, topk=(1, 5))
            _, pred = output[:, cls_idx].topk(1)
        else:
            target = [cls_idx[tgt] for tgt in target]
            target = torch.tensor(target)
            target = target.to(device)
            acc1, _ = accuracy(output, target, topk=(1, 5))
            _, pred = output.topk(1)
        if dist:
            acc1 = reduce_tensor(acc1, torch.distributed.get_world_size())
            torch.cuda.synchronize()
        top1_m.update(acc1.item(), output.size(0))
        img_paths_i = img_paths[i_count:i_count+bs]
        df = pd.DataFrame({'pred_class': pred[:, 0].tolist(), 'gt_class': target.tolist(), 'img_dir': img_paths_i})
        df_all = pd.concat([df_all, df], axis=0)
        i_count += bs

    if args.output_dir:
        csv_path = os.path.join(args.output_dir, f"{args.model}.csv")
        df_all.to_csv(csv_path)
        print(f"Saved per image classification results in '{csv_path}'")
    print(f"Top1 Accuracy on the ImageNet-CNS: {top1_m.avg:.1f}%")
    return top1_m.avg

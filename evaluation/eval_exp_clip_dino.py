import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from torchvision.datasets import ImageFolder
import yaml
import argparse
import time
import sys

sys.path.append(".")
sys.path.append("evaluation")

from similiarity_computer import ClipSimilarity, DinoImgSimilarity, ClipImgSimilarity,EvaluatorBase


def main(args):

    start_time = time.time()

    id_dir = f"id_{args.exp_id:04d}"
    args.dataset_dir = f"{args.exp_dir}/{id_dir}"
    
    print('args', args)

    dataset = ImageFolder(args.dataset_dir)
    classes = [c.split('_')[-1] for c in dataset.classes]

    print(f"Found {len(dataset)} images in {args.dataset_dir} with {len(classes)} classes")

    if args.compute_type == "clip":
        clip_sim = ClipSimilarity(batch_size=args.batch_size)
    elif args.compute_type == "dino_ref":
        dino_ref_sim = DinoImgSimilarity(batch_size=args.batch_size)
    elif args.compute_type == "dino_ref_no_head":
        dino_ref_sim = DinoImgSimilarity(batch_size=args.batch_size, imagenet_mode=False)
    elif args.compute_type == "clip_ref":
        clip_ref_sim = ClipImgSimilarity(batch_size=args.batch_size)
    else:
        raise ValueError(f"Compute type {args.compute_type} not supported.")

    classifier = EvaluatorBase(batch_size=args.batch_size, dataset_dir=args.dataset_dir, num_samples=None,random_sampling=False)

    img_inds_names = [(i, iname[1], iname[0].split('/')[-1]) for i, iname in enumerate(dataset.imgs)]

    num_scales = args.num_scales

    df_all = None

    results_dir = f'{args.exp_dir}/eval/{id_dir}/{args.compute_type}/'
    os.makedirs(results_dir, exist_ok=True)

    print('classes:',classifier.dataset.classes)
    
    for sel_class, class_name in enumerate(classifier.dataset.classes): 
        if args.select_one_class != -1:
            if sel_class != args.select_one_class: continue

        print(f"===========================")
        print(f'Processing class {sel_class}')
        if args.startnoise != -1:
            img_inds_names = [(i, iname[0].split('/')[-1]) for i, iname in enumerate(dataset.imgs) if (iname[1]==sel_class) and (f"startnoise_{args.startnoise:04d}" in iname[0])]
        else:
            img_inds_names = [(i, iname[0].split('/')[-1]) for i, iname in enumerate(dataset.imgs) if (iname[1]==sel_class)]
        
        print(f"Found {len(img_inds_names)} images for class {sel_class}: {classes[sel_class]}")
        sel_img_name_pairs = [(iname,dataset[i][0]) for i, iname in img_inds_names]
        sel_imgs = [dataset[i][0] for i, _ in img_inds_names]
        
        i_class_imgnet = classifier.target_2_imgnet_class_idx[sel_class].item()

        use_imagenet_labels = True
        if use_imagenet_labels:
            class_name_prompt = classifier.imagenet_idx_2_label[classifier.target_2_imgnet_class_idx[sel_class].item()].replace("_"," ")
        else:
            class_name_prompt = class_name.split("_")[1]
        
        img_dirs = [f"{args.dataset_dir}/{class_name}/{img_name_i}" for img_name_i,_ in sel_img_name_pairs]

       
        if args.compute_type == "clip":
            print(f'Computing CLIP scores for {len(sel_imgs)} images. Variation={args.variation},class={class_name_prompt}...')
            slider_scores = clip_sim.compute(sel_imgs, len(sel_imgs) * [f"a photo in {args.variation}",])
            class_in_slider_scores = clip_sim.compute(sel_imgs, len(sel_imgs) * [f"a photo of a {class_name_prompt} in {args.variation}",])
            class_scores = clip_sim.compute(sel_imgs, len(sel_imgs) * [f"a photo of a {class_name_prompt}",])
            df = pd.DataFrame({"i_class": i_class_imgnet, "class": classes[sel_class], 
                                   "CLIP_class": class_scores, "CLIP_shift": slider_scores,  "CLIP_class_shift": class_in_slider_scores,
                                    "img_dir": img_dirs})
            
        elif args.compute_type == "dino_ref_no_head":
            dino_ref_sims = dino_ref_sim.compute(sel_imgs,num_scales=num_scales)
            df = pd.DataFrame({"i_class": i_class_imgnet, "class": classes[sel_class], 
                                "DINO_ref_no_head": dino_ref_sims,
                                "img_dir": img_dirs})
        elif args.compute_type == "dino_ref":
            dino_ref_sims = dino_ref_sim.compute(sel_imgs,num_scales=num_scales)
            df = pd.DataFrame({"i_class": i_class_imgnet, "class": classes[sel_class], 
                                "DINO_ref": dino_ref_sims,
                                "img_dir": img_dirs})
        elif args.compute_type == "clip_ref":
            clip_ref_sims = clip_ref_sim.compute(sel_imgs,num_scales=num_scales)
            df = pd.DataFrame({"i_class": i_class_imgnet, "class": classes[sel_class], 
                                "CLIP_ref": clip_ref_sims,
                                "img_dir": img_dirs})

        # Load characteristics
        ch_path = f"{args.dataset_dir}/characteristics_{i_class_imgnet:03d}_noise_{args.startnoise:04d}.csv"

        c_ch = pd.read_csv(ch_path)
        if "var_2" not in c_ch.columns:
            c_ch["var_2"] = 0
        
        c_ch = c_ch.reset_index(drop=True)

        df = pd.concat([c_ch, df], axis=1)

        df.to_csv(f"{results_dir}/eval_variation_{i_class_imgnet:03d}.csv")

        df_all = pd.concat([df_all, df], ignore_index=True) if df_all is not None else df

        print("len(df):", len(df_all))


    print(f"Total run time: {time.time()-start_time:.2f} seconds")

if __name__ == "__main__":

    # Create argparse parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, default="../generation/results/shifts/")
    parser.add_argument('--exp_id', type=int, default=23)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--select_one_class', type=int, default=-1)
    parser.add_argument('--variation', type=str, default='snow')
    parser.add_argument('--distribution', type=str, default='IN')
    parser.add_argument('--startnoise', type=int, default=-1)
    parser.add_argument('--compute_type', type=str, default='clip')
    parser.add_argument('--num_scales', type=int, default=6)
    
    args_parsed = parser.parse_args()

    main(args_parsed)
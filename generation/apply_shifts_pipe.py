import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
import yaml
import argparse
from diffusers.pipelines import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer


def main(args):

    args.prompt = f"a picture of a {args.target}"
    args.exp_id = f"id_{args.exp_id:04d}"
    args.target = " ".join(args.target.split('_'))
    
    args.weight_dtype = getattr(torch, args.weight_dtype)

    all_seeds = list(range(1,args.num_seeds+1,1))

    print(args)

    pipe = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path,
                                                   requires_safety_checker=False, safety_checker=None,torch_dtype=args.weight_dtype)
    pipe = pipe.to(args.device)
    
    
    if args.use_imagenet_star:
        print("=== Using ImageNet star ===")
        model_name = args.imagenet_star_dir
        tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder", torch_dtype=args.weight_dtype).to("cuda")
        tokens_dict = torch.load(f"{model_name}/tokens.pt")
        tokens = tokens_dict['tokens']
        class_names = tokens_dict['initializer_words']
        pipe.text_encoder = text_encoder
        pipe.tokenizer = tokenizer
        args.prompt = f"a picture of a {tokens[args.class_int]}"
        print('Prompt:',args.prompt)
       
    results_dir = f"{args.results_dir}/{args.exp_id}/{args.class_int:03d}_{args.target}"
    args_dir = f"{args.results_dir}/{args.exp_id}"
    os.makedirs(f"{args_dir}", exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    d = {k: str(v) for k, v in args.__dict__.items()}
    with open(f"{args_dir}/args_{args.class_int:04d}.yaml", "w") as f:
        yaml.dump(d, f)
    print(f'Saved args: {d} in results/{args.exp_id}/args.yaml' )

    args.slider_alphas = np.array([0,1,])
    characteristics = pd.DataFrame(columns=['seed','start_noise','scale','distribution'])
    characteristics['var_1'] = np.tile(args.slider_alphas,len(all_seeds))
    characteristics['seed'] = np.array(all_seeds).repeat(len(args.slider_alphas))
    characteristics['distribution'] = 'IN' if args.use_imagenet_star else 'SD'
    characteristics.to_csv(f"{args_dir}/characteristics_{args.class_int:03d}.csv", index=False)

    args.shift = args.shift.replace("_"," ")
    for scale in args.slider_alphas:
        if scale == 0:
            apply_prompt = args.prompt
        elif scale == 1:
            apply_prompt = f"{args.prompt}{args.shift}"
        else:
            raise ValueError(f"scale {scale} not supported")
        for i_seed in range(0,len(all_seeds),args.batch_size):
            seeds = all_seeds[i_seed:i_seed+args.batch_size]
            print('->',scale,i_seed,i_seed/len(all_seeds))
            n = int(len(seeds))
            prompt_batch = n * [apply_prompt,]
            neg_prompt_batch = n * [args.negative_prompt,]
            generator_batch = [torch.Generator().manual_seed(seed_i) for seed_i in seeds]
            gen_images = pipe(prompt_batch,negative_prompt=neg_prompt_batch, 
                        generator=generator_batch,
                        guidance_scale=args.guidance_scale,
                        num_inference_steps=args.num_inference_steps).images
            if type(gen_images) is not list:
                gen_images = [gen_images]
            
            for i, gen_image in enumerate(gen_images):    
                gen_image.resize((256,256)).save(f"{results_dir}/seed_{seeds[i]:04d}_startnoise_{0:04d}_scale_{scale}.png")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--results_dir", type=str, default="results/shifts")
    parser.add_argument("--exp_id", type=int, default=22)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--weight_dtype", type=str, default='bfloat16')
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-2")
    parser.add_argument("--scheduler", type=str, default='Standard')
    parser.add_argument("--class_int", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--target", type=str, default="minivan")
    parser.add_argument("--negative_prompt", type=str, default="blurry,text,caption,lowquality,lowresolution,low res,grainy,ugly")
    parser.add_argument("--shift", type=str, default="in heavy snow")
    parser.add_argument('--use_imagenet_star', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--imagenet_star_dir", type=str, default="encoder_root_imagenet")

    args = parser.parse_args()

    print(args)

    main(args)
# ref:
# - https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L566
# - https://huggingface.co/spaces/baulab/Erasing-Concepts-In-Diffusion/blob/main/train.py
# - https://github.com/rohitgandikota/sliders/blob/main/trainscripts/textsliders/train_lora.py

from typing import List, Optional
import argparse
import ast
from pathlib import Path
import gc

import torch
from tqdm import tqdm
import yaml

from lora import LoRANetwork, DEFAULT_TARGET_REPLACE, UNET_TARGET_REPLACE_MODULE_CONV
import train_util
import model_util
import prompt_util
from prompt_util import PromptEmbedsCache, PromptEmbedsPair, PromptSettings
import debug_util
import config_util
from config_util import RootConfig

from transformers import CLIPTextModel, CLIPTokenizer
import wandb


def flush():
    torch.cuda.empty_cache()
    gc.collect()


def train(
    config: RootConfig,
    prompts: List[PromptSettings],
    device: int,
    args
):

    metadata = {
        "prompts": ",".join([prompt.json() for prompt in prompts]),
        "config": config.json(),
    }
    
    print(f"Saving to {config.save.path}")

    save_path = Path(config.save.path)
    save_path.mkdir(parents=True, exist_ok=True)

    with open(save_path / 'metadata.yaml', 'w') as file:
        yaml.dump(metadata, file)
    print(f"Metadata saved to {save_path}.")


    modules = DEFAULT_TARGET_REPLACE
    if config.network.type == "c3lier":
        modules += UNET_TARGET_REPLACE_MODULE_CONV

    if config.logging.verbose:
        print(metadata)

    if config.logging.use_wandb:
        wandb.init(project=f"LECO_{config.save.name}", config=metadata)

    weight_dtype = config_util.parse_precision(config.train.precision)
    save_weight_dtype = config_util.parse_precision(config.train.precision)

    tokenizer, text_encoder, unet, noise_scheduler = model_util.load_models(
        config.pretrained_model.name_or_path,
        scheduler_name=config.train.noise_scheduler,
        v2=config.pretrained_model.v2,
        v_pred=config.pretrained_model.v_pred,
    )

    if args.use_imagenet_star:
        tokenizer = CLIPTokenizer.from_pretrained(args.imagenet_star_dir, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(args.imagenet_star_dir, subfolder="text_encoder", torch_dtype=weight_dtype).to("cuda")

    text_encoder.to(device, dtype=weight_dtype)
    text_encoder.eval()

    unet.to(device, dtype=weight_dtype)
    unet.enable_xformers_memory_efficient_attention()
    unet.requires_grad_(False)
    unet.eval()

    config.network.training_method = args.training_method
    print(f"Training method: {config.network.training_method}")

    network = LoRANetwork(
        unet,
        rank=config.network.rank,
        multiplier=1.0,
        alpha=config.network.alpha,
        train_method=config.network.training_method,
    ).to(device, dtype=weight_dtype)

    optimizer_module = train_util.get_optimizer(config.train.optimizer)
    optimizer_kwargs = {}
    if config.train.optimizer_args is not None and len(config.train.optimizer_args) > 0:
        for arg in config.train.optimizer_args.split(" "):
            key, value = arg.split("=")
            value = ast.literal_eval(value)
            optimizer_kwargs[key] = value
    
    optimizer = optimizer_module(network.prepare_optimizer_params(), lr=config.train.lr, **optimizer_kwargs)
    lr_scheduler = train_util.get_lr_scheduler(
        config.train.lr_scheduler,
        optimizer,
        max_iterations=config.train.iterations,
        lr_min=config.train.lr / 100,
    )
    criteria = torch.nn.MSELoss()

    print("Prompts")
    for settings in prompts:
        print(settings)

    debug_util.check_requires_grad(network)
    debug_util.check_training_mode(network)

    cache = PromptEmbedsCache()
    prompt_pairs: list[PromptEmbedsPair] = []

    with torch.no_grad():
        for settings in prompts:
            print(settings)
            for prompt in [
                settings.target,
                settings.positive,
                settings.neutral,
                settings.unconditional,
            ]:
                print(prompt)
                if isinstance(prompt, list):
                    if prompt == settings.positive:
                        key_setting = 'positive'
                    else:
                        key_setting = 'attributes'
                    if len(prompt) == 0:
                        cache[key_setting] = []
                    else:
                        if cache[key_setting] is None:
                            cache[key_setting] = train_util.encode_prompts(
                                tokenizer, text_encoder, prompt
                            )
                else:
                    if cache[prompt] == None:
                        cache[prompt] = train_util.encode_prompts(
                            tokenizer, text_encoder, [prompt]
                        )

            prompt_pairs.append(
                PromptEmbedsPair(
                    criteria,
                    cache[settings.target],
                    cache[settings.positive],
                    cache[settings.unconditional],
                    cache[settings.neutral],
                    settings,
                )
            )

    del tokenizer
    del text_encoder

    flush()

    pbar = tqdm(range(config.train.iterations))

    for i in pbar:
        with torch.no_grad():


            noise_scheduler.set_timesteps(
                config.train.max_denoising_steps, device=device
            )

            optimizer.zero_grad()

            prompt_pair: PromptEmbedsPair = prompt_pairs[
                torch.randint(0, len(prompt_pairs), (1,)).item()
            ]

            timesteps_to = torch.randint(
                1, config.train.max_denoising_steps, (1,)
            ).item()

            height, width = (
                prompt_pair.resolution,
                prompt_pair.resolution,
            )
            if prompt_pair.dynamic_resolution:
                height, width = train_util.get_random_resolution_in_bucket(
                    prompt_pair.resolution
                )

            if config.logging.verbose:
                print("guidance_scale:", prompt_pair.guidance_scale)
                print("resolution:", prompt_pair.resolution)
                print("dynamic_resolution:", prompt_pair.dynamic_resolution)
                if prompt_pair.dynamic_resolution:
                    print("bucketed resolution:", (height, width))
                print("batch_size:", prompt_pair.batch_size)

            latents = train_util.get_initial_latents(
                noise_scheduler, prompt_pair.batch_size, height, width, 1
            ).to(device, dtype=weight_dtype)

            with network: 
                denoised_latents = train_util.diffusion(
                    unet,
                    noise_scheduler,
                    latents, 
                    train_util.concat_embeddings(
                        prompt_pair.unconditional,
                        prompt_pair.target,
                        prompt_pair.batch_size,
                    ),
                    start_timesteps=0,
                    total_timesteps=timesteps_to,
                    guidance_scale=7.5, 
                )

            noise_scheduler.set_timesteps(1000)

            current_timestep = noise_scheduler.timesteps[
                int(timesteps_to * 1000 / config.train.max_denoising_steps)
            ]

            guidance_scale = 1
            print(f"Guidance scale: {guidance_scale}")
            positive_latents = train_util.predict_noise(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents,
                train_util.concat_embeddings(
                    prompt_pair.unconditional,
                    prompt_pair.positive,
                    prompt_pair.batch_size,
                ),
                guidance_scale=guidance_scale, 
            ).to(device, dtype=weight_dtype)
            
            neutral_latents = train_util.predict_noise(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents,
                train_util.concat_embeddings(
                    prompt_pair.unconditional,
                    prompt_pair.neutral,
                    prompt_pair.batch_size,
                ),
                guidance_scale=guidance_scale,
            ).to(device, dtype=weight_dtype)

            unconditional_latents = train_util.predict_noise(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents,
                train_util.concat_embeddings(
                    prompt_pair.unconditional,
                    prompt_pair.unconditional,
                    prompt_pair.batch_size,
                ),
                guidance_scale=guidance_scale,
            ).to(device, dtype=weight_dtype)
            
            
            if config.logging.verbose:
                print("positive_latents:", positive_latents[0, 0, :5, :5])
                print("neutral_latents:", neutral_latents[0, 0, :5, :5])
                print("unconditional_latents:", unconditional_latents[0, 0, :5, :5])

        with network:
            target_latents = train_util.predict_noise(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents,
                train_util.concat_embeddings(
                    prompt_pair.unconditional,
                    prompt_pair.target,
                    prompt_pair.batch_size,
                ),
                guidance_scale=args.guidance_lora_post,
            ).to(device, dtype=weight_dtype)
            

            if config.logging.verbose:
                print("target_latents:", target_latents[0, 0, :5, :5])

        positive_latents.requires_grad = False
        neutral_latents.requires_grad = False
        unconditional_latents.requires_grad = False

        loss = prompt_pair.loss(
            target_latents=target_latents,
            positive_latents=positive_latents,
            neutral_latents=neutral_latents,
            unconditional_latents=unconditional_latents,
        )
             
        pbar.set_description(f"Loss*1k: {loss.item()*1000:.4f}")
        if config.logging.use_wandb:
            wandb.log(
                {"loss": loss, "iteration": i, "lr": lr_scheduler.get_last_lr()[0], f"loss_{timesteps_to}":loss} 
            )

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        del (
            positive_latents,
            neutral_latents,
            unconditional_latents,
            target_latents,
            latents,
        )
        flush()

        if (
            i % config.save.per_steps == 0
            and i != 0
            and i != config.train.iterations - 1
        ):
            print("Saving...")
            save_path.mkdir(parents=True, exist_ok=True)
            network.save_weights(
                save_path / f"{config.save.name}_{i}steps.pt",
                dtype=save_weight_dtype,
            )

    print("Saving...")
    save_path.mkdir(parents=True, exist_ok=True)
    network.save_weights(
        save_path / f"{config.save.name}_last.pt",
        dtype=save_weight_dtype,
    )

    del (
        unet,
        noise_scheduler,
        loss,
        optimizer,
        network,
    )

    flush()

    print("Done.")


def main(args):
    print('args',args)
    config_file = args.config_file

    config = config_util.load_config_from_yaml(config_file)
    if args.name is not None:
        config.save.name = args.name
    attributes = []
    if args.attributes is not None:
        attributes = args.attributes.split(',')
        attributes = [a.strip() for a in attributes]
    
    config.network.alpha = args.alpha
    config.network.rank = args.rank
    config.save.name += f'_alpha{args.alpha}'
    config.save.name += f'_rank{config.network.rank}'
    config.save.name += f'_{config.network.training_method}'
    config.save.path += f'/{config.save.name}'

    prompts = prompt_util.load_prompts_from_yaml(config.prompts_file, attributes)
    device = torch.device(f"cuda:{args.device}")

    if args.use_imagenet_star:
        tokens_dict = torch.load(f"{args.imagenet_star_dir}/tokens.pt")
        tokens = tokens_dict['tokens']
        class_names = tokens_dict['initializer_words']

    for i_p in range(len(prompts)):
        if args.target is not None:
            if args.use_imagenet_star:
                args.target = tokens[args.class_int]
            args.target = " ".join(args.target.split("_"))
            prompts[i_p].target = args.target
            prompts[i_p].unconditional = args.target
            prompts[i_p].neutral = args.target
        if args.variation is not None:
            args.variation = " ".join(args.variation.split('_'))
            prompts[i_p].positive = f"{args.target}, in {args.variation}" # definition of shift
        if args.pos_prompt is not None:
            prompts[i_p].positive = args.pos_prompt

        print(f"Using Positive prompt: {prompts[0].positive}")

        # Config for training
        prompts[i_p].action = "enhance"
        prompts[i_p].guidance_scale = 4
        prompts[i_p].resolution = 512
        prompts[i_p].dynamic_resolution = False
        prompts[i_p].batch_size = args.batch_size


    print("=================")
    print("Prompts:", prompts)
    print("Config:", config)
    print("=================")

    train(config=config, prompts=prompts, device=device,args=args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        required=False,
        default = 'trainscripts/textsliders/data/config.yaml',
        help="Config file for training.",
    )
    # config_file 'data/config.yaml'
    parser.add_argument(
        "--alpha",
        type=float,
        required=False,
        help="LoRA weight.",
        default=1.,
    )
    # --alpha 1.0
    parser.add_argument(
        "--rank",
        type=int,
        required=False,
        help="Rank of LoRA.",
        default=4,
    )
    # --rank 4
    parser.add_argument(
        "--device",
        type=int,
        required=False,
        default=0,
        help="Device to train on.",
    )
    # --device 0
    parser.add_argument(
        "--name",
        type=str,
        required=False,
        default=None,
        help="Device to train on.",
    )
    # --name 'eyesize_slider'
    parser.add_argument(
        "--attributes",
        type=str,
        required=False,
        default=None,
        help="attritbutes to disentangle (comma seperated string)",
    )
    # --attributes 'male, female'
    parser.add_argument(
        "--target",
        type=str,
        required=False,
        default=None,
        help="Object to consider.",
    )
    # --name 'minivan'
    parser.add_argument(
        "--variation",
        type=str,
        required=False,
        default=None,
        help="Variation (in xxx)",
    )
    parser.add_argument(
        "--pos_prompt",
        type=str,
        required=False,
        default=None,
        help="Sentence that describes the variation",
    )
    # --name 'heavy snow'
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        help="Batch size.",
        default=1,
    )
    # --rank 4
    parser.add_argument(
        "--save_path",
        type=str,
        required=False,
        default=None,
        help="",
    )
    # --name 'heavy snow'
    parser.add_argument(
        "--class_int",
        type=int,
        required=False,
        default=0,
        help="",
    )
    # --class_int 0
    parser.add_argument(
        '--use_imagenet_star', 
        default=False, 
        action=argparse.BooleanOptionalAction)
    # --use_imagenet_star
    parser.add_argument(
        "--lr",
        type=float,
        required=False,
        help="Learning rate.",
        default=0.0002,
    )
    parser.add_argument(
        "--training_method",
        type=str,
        required=False,
        default='noxattn',
        help="What to optimize LoRA for.",
    )
    parser.add_argument(
        "--guidance_lora_post",
        type=float,
        required=False,
        help="Guidance.",
        default=1.,
    )
    # 
    parser.add_argument(
        "--imagenet_star_dir",
        type=str,
        required=False,
        default="encoder_root_imagenet",
    )
    
    
    args = parser.parse_args()

    main(args)

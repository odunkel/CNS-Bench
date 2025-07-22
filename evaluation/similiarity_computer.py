from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.multimodal.clip_score import _clip_score_update as compute_clip_scores
import torch
from torch import nn
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from typing import List, Callable
from PIL import Image
from torchvision.datasets import ImageFolder
import json


class EvaluatorBase():
    def __init__(self, device: str = "cuda", batch_size: int = 32, transforms=None, dataset_dir: str = None, num_samples: int = 100, random_sampling: bool = False, filter_class: int = None, filter_str_callable: callable = None, subset_sampler: str = None):
        if dataset_dir is None:
            raise ValueError("dataset_dir must be provided")
        
        self.device = device
        self.batch_size = batch_size
        self.dataset = ImageFolder(dataset_dir, transform=transforms)
        imagenet_class_index = json.load(open('utils/imagenet_class_index.json'))
        self.imagenet_idx_2_label = {int(k): v[1] for k, v in imagenet_class_index.items()}
        self.num_samples = len(self.dataset) if num_samples is None else num_samples
        self.dloader = self.filter_imgs_and_return_dloader(filter_str_callable, subset_sampler, random_sampling, filter_class)
        self.target_2_imgnet_class_idx = self.get_label_mappings(imagenet_classes=False, functor_to_get_labels=lambda c: int(c.split("-")[0]) if "-" in c else int(c.split("_")[0]))

    def filter_imgs_and_return_dloader(self, filter_str_callable: callable, subset_sampler: str = None,random_sampling: bool = True, filter_class: int = None, num_workers: int = 0):
        subset_sampler_obj = None
        if subset_sampler == 'random':
            subset_sampler_obj = torch.utils.data.RandomSampler(self.dataset, num_samples=self.num_samples)
            dataset = self.dataset
        elif subset_sampler == 'subset':
            if filter_class is not None:
                indices = torch.where(torch.tensor(self.dataset.targets)==filter_class)[0]
            else:
                indices = torch.arange(len(self.dataset))
            if filter_str_callable is not None:
                imgs_names = [(i, img[0].split("/")[-1]) for i,img in enumerate(self.dataset.imgs) if i in indices]
                indices = torch.tensor([i for i, img_name in imgs_names if filter_str_callable(img_name)])
            if random_sampling:
                print("Random sampling is activated.")
                indices = indices[torch.randperm(indices.shape[0])]
            indices = indices[:self.num_samples]
            dataset = torch.utils.data.Subset(self.dataset, indices)
        else:
            dataset = self.dataset  
        dloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers, sampler=subset_sampler_obj)
        return dloader
    
    def get_label_mappings(self, imagenet_classes: bool = True, functor_to_get_labels: callable = lambda c: int(c.split("_")[0])):
        if imagenet_classes:
            considered_inds = list(self.imagenet_idx_2_label.keys())
        else:
            idx_to_imgnet_class_idx = {i:functor_to_get_labels(c) for c,i in self.dataset.class_to_idx.items()}
            considered_inds = list(idx_to_imgnet_class_idx.values())
        imgnet_inds = torch.tensor(considered_inds)
        return imgnet_inds

class ClassSimilarityComputer:
    def __init__(self, batch_size: int = 20):
        self.batch_size = batch_size
        self.metric = None

    def compute_metric(self, images: List[Image.Image], prompts: List[str] = None, gt_imagenet_classes: List[int] = None, return_argmax: bool = False):
        raise NotImplementedError("This method should be implemented in the subclass")
    
    def compute(self, images: list, prompts: list = None, gt_imagenet_classes: List[int] = None, return_argmax: bool = False):
        if prompts is not None and len(images) != len(prompts):
            raise ValueError("Number of images and prompts should be the same")
        if gt_imagenet_classes is not None and len(images) != len(gt_imagenet_classes):
            raise ValueError("Number of images and gt_imagenet_classes should be the same")
        scores = torch.zeros(len(images))
        class_inds = torch.zeros(len(images))
        for i_batch in tqdm(range(0, len(images), self.batch_size)):
            img_batch = images[i_batch:i_batch+self.batch_size]
            text_batch = prompts[i_batch:i_batch+self.batch_size] if prompts is not None else None
            gt_classes = gt_imagenet_classes[i_batch:i_batch+self.batch_size] if gt_imagenet_classes is not None else None
            if return_argmax:
                scores[i_batch:i_batch+self.batch_size],class_inds[i_batch:i_batch+self.batch_size] = self.compute_metric(img_batch, text_batch, gt_classes, return_argmax=return_argmax)
            else:
                scores[i_batch:i_batch+self.batch_size] = self.compute_metric(img_batch, text_batch, gt_classes, return_argmax=return_argmax)
        
        if return_argmax:
            return scores, class_inds
        return scores


class ClipSimilarity(ClassSimilarityComputer):

    def __init__(self,batch_size: int = 20):
        super().__init__(batch_size=batch_size)
        self.metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")

    def compute_metric(self, images: list, prompts: list, gt_imagenet_class: List[int] = None, return_argmax: bool = False):
        if type(images) == list:
            images = torch.from_numpy(np.stack([np.array(img) for img in images])).permute(0,3,1,2)
        return compute_clip_scores(images, prompts, self.metric.model, self.metric.processor)[0].detach()

  
class DinoSimilarity(ClassSimilarityComputer):

    def __init__(self, dino_model: str = "vitl14", device: str ='cuda', batch_size: int = 20, imagenet_mode: bool = True):
        super().__init__(batch_size=batch_size)

        self.device = device
        load_str = f'dinov2_{dino_model}_reg'
        if imagenet_mode:
            load_str += '_lc'
        self.model = torch.hub.load('facebookresearch/dinov2', load_str).to(device)
    
        self.batch_size = batch_size

        self.transform = self._get_transform()

    def _get_transform(self):
        transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
                ])
        return transform
    
    def preprocess_images(self, pil_images: List[Image.Image]):
        images = [self.transform(img) for img in pil_images]
        images = torch.stack(images).to(self.device)
        return images
    
    def compute_metric(self, images: List[Image.Image], prompts: List = None, gt_imagenet_classes: List[int] = None, return_argmax: bool = False):
        if gt_imagenet_classes is None:
            raise ValueError("GT ImageNet class is required")
        images = self.preprocess_images(images)
        with torch.no_grad():
            clstokens = self.model(images)
        sm_out = clstokens.softmax(-1)
        class_dino_sim = sm_out[torch.arange(len(sm_out)), gt_imagenet_classes]
        if return_argmax:
            return class_dino_sim, sm_out.argmax(-1)
        return class_dino_sim
    
    def compute_top_k_score(self, probs, k: int = 1):
        top_k = torch.topk(probs.cpu(), k=k, dim=1)
        top_k = top_k.indices.t()
        return top_k



class ImgSimilarityComputer:
    def __init__(self, batch_size: int = 20):
        self.batch_size = batch_size
        self.metric = None

    def compute_metric(self, images: list, prompts: list = None, gt_imagenet_classes: List[int] = None, return_argmax: bool = False):
        raise NotImplementedError("This method should be implemented in the subclass")
    

    def compute(self, images: List[Image.Image], reference_img_idx: int = 0, num_scales: int = 6):
        scores = torch.zeros(len(images))
        bs = self.batch_size
        for i_batch in tqdm(range(0, len(images), bs)):
            img_batch = images[i_batch:i_batch+bs]
            scores[i_batch:i_batch+bs] = self.compute_metric(img_batch, reference_img_idx, num_scales)
        return scores


class DinoImgSimilarity(ImgSimilarityComputer):

    def __init__(self, dino_model: str = "vitl14", device: str ='cuda', batch_size: int = 20, imagenet_mode: bool = True, return_argmax: bool = False):
        super().__init__(batch_size=batch_size)

        self.device = device
        load_str = f'dinov2_{dino_model}_reg'
        if imagenet_mode:
            load_str += '_lc'
        self.model = torch.hub.load('facebookresearch/dinov2', load_str).to(device)
    
        self.batch_size = batch_size

        self.transform = self._get_transform()

    def _get_transform(self):
        transform = transforms.Compose([           
            transforms.Resize(520),
            transforms.CenterCrop(518),
            transforms.ToTensor(), 
            transforms.Normalize(mean=0.5, std=0.2)
            ])

        return transform
    
    def preprocess_images(self, pil_images: List[Image.Image]):
        images = [self.transform(img) for img in pil_images]
        images = torch.stack(images).to(self.device)
        return images
    
    def compute_metric(self, images: List[Image.Image], reference_img_idx: int, num_scales: int):
        images = self.preprocess_images(images)

        with torch.no_grad():
            clstokens = self.model(images).softmax(-1)
        clstokens = clstokens.reshape(-1,num_scales,clstokens.shape[-1])
        batch_class_sim = []
        for i in range(clstokens.shape[0]):
            clstoken_ref = clstokens[i, reference_img_idx].unsqueeze(0)
            class_sim = torch.einsum('ij,ij->i', clstoken_ref, clstokens[i]) / (torch.norm(clstoken_ref, dim=-1) * torch.norm(clstokens[i], dim=-1))
            batch_class_sim.append(class_sim)
        class_sim = torch.cat(batch_class_sim,dim=0)
        return class_sim
    

class ClipImgSimilarity(ImgSimilarityComputer):
    
        def __init__(self,batch_size: int = 20):
            super().__init__(batch_size=batch_size)
            import clip
            self.device = "cuda:0"
            self.model, self.preprocess = clip.load("ViT-B/16", device=self.device)
    
        def compute_metric(self, images: List[Image.Image], reference_img_idx: int,  num_scales: int):
            images = torch.stack([self.preprocess(img) for img in images]).to(self.device)
            with torch.no_grad():
                images_feat = self.model.encode_image(images)
            images_feat = images_feat.reshape(-1,num_scales,images_feat.shape[-1])
            batch_class_sim = []
            for i in range(images_feat.shape[0]):
                ref_img_feat = images_feat[i, reference_img_idx].unsqueeze(0)
                class_sim = torch.einsum('ij,ij->i', ref_img_feat, images_feat[i]) / (torch.norm(ref_img_feat, dim=-1) * torch.norm(images_feat[i], dim=-1))
                batch_class_sim.append(class_sim)
            class_sim = torch.cat(batch_class_sim,dim=0)
            return class_sim

<h2 align="center">CNS-Bench: Benchmarking Image Classifier Robustness Under Continuous Nuisance Shifts</h2>
<div align="center"> 
    <a href="https://odunkel.github.io" target="_blank">Olaf Dünkel</a>, 
    <a href="https://artur.jesslen.ch//">Artur Jesslen*</a>,</span>
    <a href="https://jiahao000.github.io/">Jiaohao Xie*</a>,</span>
    <a href="https://people.mpi-inf.mpg.de/~theobalt" target="_blank">Christian Theobalt</a>,
    <a href="https://chrirupp.github.io/" target="_blank">Christian Rupprecht</a>,
    <a href="https://genintel.mpi-inf.mpg.de/" target="_blank">Adam Kortylewski</a>
</div>
<br>

<div align="center">

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://genintel.github.io/CNS)
[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)]()
[![Data](https://img.shields.io/badge/Data-Available-green)](https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.MAGNKP)

</div>


Code for **CNS-Bench** for training of LoRA shifting sliders (generation), dataset preparation (evaluation), and benchmarking.
- [Benchmarking](#benchmarking)
- [Evaluation](#evaluation)
- [Generation](#generation)
- [Citation](#citation)


## 🏁 Benchmarking with CNS-Bench
This section covers classifier evaluation and benchmarking.

### 📁 Dataset
To acquire dataset samples, download `benchmark/cns_bench.zip` from [Edmond](https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.MAGNKP).
Loading of an example dataset image is illustrated in `benchmarking/loading.ipynb`.

### 📊 Pre-computed results
The performances of models evaluated with the `easyrobust` pipeline are contained in `benchmark/results.zip` from [Edmond](https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.MAGNKP). Download them to `results`.
The notebook `benchmarking/results.ipynb` provides functionality to compute metrics, such as accuracies, accuarcy drops, and failure points.

To evaluate different classifiers, we refer to the related README in `benchmarking/easyrobust`.
Running easyrobust will save `.csv` files. These files are processed using `benchmarking/prepare_csv_files.py` for model benchmarking analysis.


## 📏 Evaluation
This section covers evaluation and out-of-class filtering.

### 🧹 Filtering of image samples

Our out-of-class filtering requires
- text alignments of the images to the class and shift, 
- the DINO cosine similarity of the class tokens to the reference image, and
- the CLIP image similarity to the reference image.

The following scripts evaluate these quantities for the images for an example experiment:
```bash
cd evaluation
python eval_exp_clip_dino.py --exp_dir ../generation/results/shifts --exp_id 0 --variation cartoon_style --compute_type clip --startnoise 750 --num_scales 6 --batch_size 180
python eval_exp_clip_dino.py --exp_dir ../generation/results/shifts --exp_id 0 --variation cartoon_style --compute_type dino_ref_no_head --startnoise 750 --num_scales 6 --batch_size 180
python eval_exp_clip_dino.py --exp_dir ../generation/results/shifts --exp_id 0 --variation cartoon_style --compute_type clip_ref --startnoise 750 --num_scales 6 --batch_size 180
```

The jupyter notebook `evaluation/sliders_filter.ipynb` loads the previously generated pandas dataframes and performs filtering.

## 🖼️ Generation of benchmarking images

This section covers LoRA adapters and generation of shifted images.

To generate images with varying shift scales, either use the pre-trained LoRA sliders (`sliders/lora_adapters.zip` from [Edmond](https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.MAGNKP)) or train your own LoRA sliders for the desired nuisance shift.

#### ⚙️ Installation
To create an environment to perform the shift, run the following:
```bash
conda create -n shifting python=3.10
conda activate shifting

conda install -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda  -y
conda install -c conda-forge transformers diffusers matplotlib accelerate pandas -y

# For filtering
conda install -c conda-forge torchmetrics -y
pip install git+https://github.com/openai/CLIP.git

```

#### 🚀 Usage of LoRA adapters for shifting

To perform a shift, load the LoRA adapters and use the diffusers StableDiffusion pipeline with a custom call implementation that supports the LoRA adapters.
For ImageNet* support, download `imagenet_star/model.safetensors` from [Edmond](https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.MAGNKP) and save the file in `generation/encoder_root_imagenet/text_encoder`.

The images of various scales are generated using the following command:
```bash
python apply_slider_shifts_pipe.py --exp_id <experiment ID> \
                                                                        --slider_name <name of slider> --target <class name> --class_int <ImageNet class index>  --slider_start_noise <number of active noise steps> \
                                                                        --slider_dir <directory of sliders> \
                                                                        --pretrained_model_name_or_path "stabilityai/stable-diffusion-2" \
                                                                        --use_imagenet_star \
                                                                        --num_inference_steps 50 --num_seeds 50 --batch_size 25
# Example
python apply_slider_shifts_pipe.py --exp_id 0 \
                                                                        --slider_name cartoon_style_class_002 --target great_white_shark --class_int 2  --slider_start_noise 750 \
                                                                        --slider_dir results/sliders \
                                                                        --pretrained_model_name_or_path "stabilityai/stable-diffusion-2" \
                                                                        --use_imagenet_star \
                                                                        --num_inference_steps 50 --num_seeds 50 --batch_size 25
```



### 🏋️ Training of custom LoRA adapters
We provide code for training LoRA adapters in generation/trainscripts with slight modification from the implementation of Gandikota et. al (2023).



<details>
    <summary>Click for more details.</summary>
    
We include the following main modifications:
1) We train sliders using the dataset interfaces provided by Vendrow et. al (2023) in https://github.com/MadryLab/dataset-interfaces.
2) We use a higher guidance scale (7.5) when computing the `denoised_latents` since we only compute unidirectional shifts.


**Installation**
Follow the installation guidelines in https://github.com/rohitgandikota/sliders/tree/main to install all dependencies for training the LoRA adapters.

Then, activate the environment:
```bash
conda activate sliders
cd generation
export WANDB_API_KEY=<your-key>
```

**Training**
Train the sliders with the desired configuration:
```bash
python trainscripts/textsliders/train_lora.py --attributes '' \
                                --name <name of slider> --class_int <ImageNet class index> --target <class name> --variation <shift> --batch_size 16 \
                             --use_imagenet_star
# Example
python trainscripts/textsliders/train_lora.py --attributes '' \
                                --name cartoon_style_class_002 --class_int 2 --target great_white_shark --variation cartoon_style --batch_size 16 \
                             --use_imagenet_star
```

</details> 


## 📚 Citation

If you find our work useful, please consider giving a star ⭐ and a citation.

```bibtex
@inproceedings{duenkel2025cns,
        title = {CNS-Bench: Benchmarking Image Classifier Robustness Under Continuous Nuisance Shifts},
        author = {D{\"u}nkel, Olaf and Jesslen, Artur and Xie, Jiaohao and Theobalt, Christian and Rupprecht, Christian and Kortylewski, Adam},
        booktitle = {ICCV},
        year = {2025}
    }
```

## 🙏 Acknowledgement
Our work heavily relies on components of [Concept Sliders](https://github.com/rohitgandikota/sliders), [Dataset Interfaces](https://github.com/MadryLab/dataset-interfaces), and [easyrobust](https://github.com/alibaba/easyrobust). We thank them for open-sourcing their works.


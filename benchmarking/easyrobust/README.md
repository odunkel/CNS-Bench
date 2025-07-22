# Benchmarking Vision Models Under Generative Continuous Nuisance Shifts

Our evaluation code is based on [EasyRobust](https://github.com/alibaba/easyrobust) framework.

## Installation
```bash
cd benchmarking/easyrobust
conda create -n easyrobust python=3.8
pip install -e .
```

Download the dataset and place into `/path/to/dataset`.

## Evaluation
To evaluate a given model on our benchmark, run:

```bash
bash benchmarks/run_cns_bench.sh <model-name> <dataset-path> <output-path> <optional-arguments>
```

For example, to evaluate the `CLIP ViT-B/16` model on the `cartoon_style` nuisance shift, run:

```bash
bash benchmarks/run_cns_bench.sh clip_vit_base_patch16_224 /path/to/dataset/cartoon_style evaluations
```

We also provide an example to run all available models with one script:

```bash
bash benchmarks/run_cns_bench_multi.sh <dataset-path> <output-path> <nuisance-shift-tag>
```

For example, to evaluate all available models on the `cartoon_style` nuisance shift, run:

```bash
bash benchmarks/run_cns_bench_multi.sh /path/to/dataset/cartoon_style evaluations cartoon_style
```

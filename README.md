# SONA

This repository is the official release code for the AAAI 2025 accepted paper
"Diffusion-based Semantic Outlier Generation via Nuisance Awareness for Out-of-Distribution Detection".

This repository contains the official implementation for the paper ["SONA"](https://arxiv.org/pdf/2408.14841).
It provides two major components:

- `generator/` – utilities and pipelines for generating Semantic Outlier Generation via Nuisance Awareness (SONA) samples using Stable Diffusion.
- `classifier/` – training code for the image classifier used in the paper.

## Installation

The code requires Python 3.8+ and PyTorch.  Install dependencies with

```bash
pip install -r requirements.txt
```

The generator additionally depends on `diffusers` and `transformers`.

## Generating SONA Samples

SONA images are generated using the Stable Diffusion pipeline defined in
`generator/pipelines/pipeline_sona.py`.  A convenience script is provided:

```bash
python generator/generate_sona.py --dataset imagenet \
    --datadir <image-root> \
    --pretrained stabilityai/stable-diffusion-2-base \
    --save_dir samples
```

The script reads class names from `generator/utils/classnames.txt` and saves
edited images under the directory given by `--save_dir`.

### Prompt Utilities

Prompt generation utilities live in `generator/utils/prompt_utils.py`.  These
provide helper functions for deterministic prompt creation and OOD word
selection.

## Training a Classifier

The classifier can be trained on ImageNet‑200 as follows:

```bash
python classifier/train.py --dataset in200 \
    --id_train_dir <path-to-id-train> \
    --id_train_vae_dir <path-to-id-vae-train> \
    --id_test_dir <path-to-id-val> \
    --ood_dir <path-to-generated-sona> \
    --save <output-dir>
```

Adjust the dataset paths to your environment.  Checkpoints are saved to the directory specified by `--save`.

## Repository Structure

```
classifier/              # ResNet‑18 classifier and training script
    network.py
    train.py

generator/               # Stable Diffusion based generator
    datasets.py          # Dataset loading helpers
    generate_sona.py     # Entry point for image generation
    pipelines/           # Diffusion pipelines
    schedulers/          # Custom scheduler implementations
    utils/               # Utility scripts and resources
```

Both `classifier` and `generator` are Python packages and can be imported.

## License

This project is released under the Apache 2.0 License.

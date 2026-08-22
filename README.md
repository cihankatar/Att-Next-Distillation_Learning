# TopoDistill

Official research implementation for **“TopoDistill for Skin Lesion Segmentation with Persistent Homology-Guided Pseudo-Masks.”**

TopoDistill combines topology-guided pseudo-mask supervision with token-level teacher–student self-distillation. It is designed to learn segmentation-aware representations from dermoscopic images and transfer the pretrained Att-Next encoder to skin-lesion segmentation when only a small labeled subset is available.

## Method overview

The repository implements three related experiments:

1. **TopoDistill pretraining** — `train_dino.py` trains a student encoder using token-level self-distillation and an auxiliary segmentation head supervised by persistent-homology pseudo-masks. An exponential-moving-average copy of the student is used as the teacher.
2. **Downstream segmentation** — `train_ssl_pretrained.py` loads the pretrained encoder, freezes it, and trains the Att-Next decoder and segmentation head on a labeled subset.
3. **Supervised baseline** — `train_random.py` trains Att-Next from random initialization using the same downstream data protocol.

The detached monitor head used during pretraining is supervised by ground-truth masks for evaluation only; its gradients do not update the encoder.

## Repository layout

```text
augmentation/                 paired image/mask augmentations
data/                         datasets, transforms, and data loaders
models/                       Att-Next encoder, decoder, and optional baselines
models/mednext/               optional MedNeXt implementation
utils/                        losses, metrics, heads, and pseudo-mask helpers
cubical_complex_main.py       topology-guided pseudo-mask generation
train_dino.py                 TopoDistill/self-distillation pretraining
train_ssl_pretrained.py       labeled downstream training with a frozen encoder
train_random.py               supervised random-initialization baseline
test.py                       checkpoint evaluation
plotting.py                   qualitative and topology visualizations
config.yaml                   human-readable reference defaults
wandb_init.py                 runtime arguments and experiment tracking
```

`plot_test_images.py`, the root-level `data_loader_ssl_pretrained.py`, `models/FAT_NET.py`, and `utils/Test_Train_Split.py` are retained as legacy experiment utilities. They are not required by the main TopoDistill pipeline and may refer to architectures or data layouts that are not included in this release.

## Installation

Python 3.10 or newer is recommended. Install a PyTorch build suitable for your CUDA version first when using a GPU, then install the remaining dependencies:

```bash
git clone https://github.com/cihankatar/Att-Next-Distillation_Learning.git
cd Att-Next-Distillation_Learning
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Copy the environment template and replace its example paths:

```bash
cp .env.example .env
set -a
source .env
set +a
```

Do not commit `.env`; it is ignored because it may contain a W&B API key.

## Data layout

Set `ML_DATA_ROOT` to the directory containing the datasets. A dataset used for all stages should follow this layout:

```text
ML_DATA_ROOT/
└── isic_2018_1/
    ├── train/
    │   ├── images/
    │   ├── masks/
    │   └── pmasks/
    ├── val/
    │   ├── images/
    │   ├── masks/
    │   └── pmasks/
    └── test/
        ├── images/
        ├── masks/
        └── pmasks/
```

Supported dataset identifiers are `isic_2018_1`, `isic_2016_1`, `PH2Dataset`, `kvasir_1`, and `ham_1` (stored in `HAM10000_1/`). Images, masks, and pseudo-masks are paired by their sorted filenames, so keep all three lists aligned.

Checkpoints are written under `ML_DATA_OUTPUT` on CUDA systems and `ML_DATA_OUTPUT_LOCAL` otherwise. Setting both variables to the same directory is valid.

## Generate topology-guided pseudo-masks

The generator writes binary PNG masks to `<dataset>/<split>/pmasks`. The default `--start-index 1` preserves the original experiment setting; pass `0` to process every image.

```bash
python cubical_complex_main.py \
  --dataset isic_2018_1 \
  --split train \
  --start-index 0
```

Use `--limit 10` for a short run and `--show` to display the pseudo-mask candidates, overlap metrics, and persistence diagram. Repeat the command for `val` and `test` when those splits are used during pretraining.

## Training

Select the dataset and seed through environment variables:

```bash
export TOPODISTILL_DATASET=isic_2018_1
export TOPODISTILL_SEED=932
```

Run TopoDistill pretraining:

```bash
python train_dino.py --epochs 300 --bsize 8 --lrate 0.0001
```

Run downstream segmentation with the pretrained encoder:

```bash
python train_ssl_pretrained.py --epochs 503 --sratio 0.1
```

By default, the downstream script reconstructs the encoder checkpoint name from the SSL defaults. For a custom pretraining run, provide its path explicitly:

```bash
export TOPODISTILL_ENCODER_CHECKPOINT=/absolute/path/to/encoder_checkpoint.pth
python train_ssl_pretrained.py --epochs 503 --sratio 0.1
```

Run the random-initialization supervised baseline:

```bash
python train_random.py --epochs 450 --sratio 0.1
```

All available runtime options can be listed with `python <script>.py --help`. `config.yaml` mirrors the default values for readability; `wandb_init.py` remains the runtime source of truth.

## Evaluation

`TOPODISTILL_DATASET` selects the dataset-specific checkpoint directory and `TOPODISTILL_TEST_DATASET` selects the test set. An explicit model path is recommended:

```bash
export TOPODISTILL_DATASET=isic_2018_1
export TOPODISTILL_TEST_DATASET=PH2Dataset
export TOPODISTILL_MODEL_CHECKPOINT=/absolute/path/to/segmentation_checkpoint.pth
python test.py --bsize 8
```

The evaluation reports IoU/Jaccard, Dice/F1, recall, precision, and pixel accuracy.

## Weights & Biases

Training logs to Weights & Biases. For a local run without network logging:

```bash
export WANDB_MODE=offline
```

For online logging, set `WANDB_MODE=online`, configure `WANDB_API_KEY`, and set `WANDB_DIR` to a writable location. No credentials are stored in this repository.

## Reproducibility notes

- The downstream labeled subset is shuffled deterministically using `TOPODISTILL_SEED`.
- Model filenames include the runtime arguments and seed. Changing an argument therefore produces a different checkpoint name.
- Pseudo-mask quality depends on image preprocessing and persistent-homology thresholds; inspect representative samples with `--show` before a full run.
- Dataset files and pretrained weights are not distributed in this repository.

## Citation

If you use this code, please cite the accompanying manuscript. A complete BibTeX entry will be added when the publication metadata is available.

## License

No open-source license has been added yet. Until one is provided, reuse and redistribution require permission from the author.

# CNN Nested NER

Implementation of **An Embarrassingly Easy but Strong Baseline for Nested Named Entity Recognition**.

This repository provides a span-based nested named entity recognition model built on top of pretrained Transformers. It represents candidate entity spans in a two-dimensional token-pair grid and refines span scores with a lightweight CNN, making it suitable for datasets where entities may overlap or be nested.

Paper: [An Embarrassingly Easy but Strong Baseline for Nested Named Entity Recognition](https://arxiv.org/abs/2208.04534)

## Overview

Nested NER differs from flat NER because entities can overlap. For example, a biomedical sentence may contain both a short entity and a larger entity that contains it. This code handles that setting by predicting entity labels over token spans instead of assigning one label per token.

The repository includes:

- preprocessing scripts for ACE2004, ACE2005, and GENIA
- document split files for ACE2004, ACE2005, and GENIA
- processed GENIA JSONLines files
- a CNN-based span classification model
- training and evaluation code using `fastNLP`, PyTorch, and Hugging Face Transformers


## Repository Structure

| Path | Description |
| --- | --- |
| `train.py` | Main training script. |
| `model/model.py` | `CNNNer` model definition. |
| `model/cnn.py` | Masked CNN layers used to refine span logits. |
| `model/multi_head_biaffine.py` | Multi-head biaffine span scoring module. |
| `model/metrics.py` | Nested NER evaluation metric. |
| `data/ner_pipe.py` | JSONLines loader, tokenizer alignment, and span-matrix construction. |
| `data/padder.py` | Custom 3D matrix padding for span labels. |
| `preprocess/` | Dataset preprocessing scripts and split files. |
| `preprocess/outputs/genia/` | Processed GENIA train/dev/test files included in the repository. |
| `requirements` | Python dependencies. |

## Installation

Install the listed dependencies:

```bash
pip install -r requirements
```

The requirements file contains:

```text
fastNLP>=1.0.0beta
pytorch>=1.8.0
torch_scatter>=2.0.9
fitlog
transformers==4.18.0
```

Depending on your environment, you may need to install `torch` and `torch_scatter` with CUDA-specific wheels.


## Data

The training script expects datasets under:

```text
preprocess/outputs/<dataset_name>/
  train.jsonlines
  dev.jsonlines
  test.jsonlines
```

Supported dataset names in `train.py`:

- `genia`
- `ace2004`
- `ace2005`

GENIA processed files are already included. ACE2004 and ACE2005 must be created from the licensed LDC corpora.

## Preprocessing

See [preprocess/readme.md](preprocess/readme.md) for detailed preprocessing instructions.

In short:

```bash
cd preprocess
python process_genia.py
python process_ace2004.py
python process_ace2005.py
```

ACE corpora are not included because they require LDC licenses:

- ACE2004: LDC2005T09
- ACE2005: LDC2006T06

The preprocessing scripts use the provided split files in:

```text
preprocess/splits/
  ace2004/
  ace2005/
  genia/
```


## Training

The script chooses a default pretrained encoder based on the dataset:

- `genia`: `dmis-lab/biobert-v1.1`
- `ace2004`, `ace2005`: `roberta-base`
- custom names: `roberta-base` unless `--model_name` is provided

GENIA:

```bash
python train.py \
  -d genia \
  -n 5 \
  -b 8 \
  --lr 7e-6 \
  --cnn_dim 200 \
  --biaffine_size 400 \
  --n_head 4 \
  --cnn_depth 3 \
  --logit_drop 0
```

ACE2004:

```bash
python train.py \
  -d ace2004 \
  -n 50 \
  -b 48 \
  --lr 2e-5 \
  --cnn_dim 120 \
  --biaffine_size 200 \
  --n_head 5 \
  --cnn_depth 2 \
  --logit_drop 0.1
```

ACE2005:

```bash
python train.py \
  -d ace2005 \
  -n 50 \
  -b 48 \
  --lr 2e-5 \
  --cnn_dim 120 \
  --biaffine_size 200 \
  --n_head 5 \
  --cnn_depth 2 \
  --logit_drop 0
```

You can override the pretrained encoder with:

```bash
python train.py -d genia --model_name dmis-lab/biobert-v1.1
```







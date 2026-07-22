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



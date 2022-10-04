# CLIP-knows-IA

In this repository we provide the source code neccessary to recreate our experiments.
We mainly use PyTorch for our experiments.

## Code

This repository contains code for in total three different experiment steps.

**`Fine-Tuning/`**:

This directory contains the scripts and files for fine-tuning the CLIP model for IAA on the AVA Dataset.

**`Linear-Regression/`**:

This directory contains the scripts and files for performing linear probing experiments.

**`Probing/`**:

This directory contains the scripts and files for performing zero-shot probing for IAA on the AVA Dataset.

## Additional Code

**`MLSP/`**:

In this folder we provide the source code for the MLSP model, which has been adapted from the [original code](https://github.com/subpic/ava-mlsp) to work with Python3 and TensorFlow2.
This code was used to calculate the performance of the MLSP model on our split of the AVA dataset.

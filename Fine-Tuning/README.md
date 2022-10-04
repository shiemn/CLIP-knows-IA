# Overview

The following files can be used to perform the same finetuning as in the paper:

We mainly use PyTorch-Ignite and Tensorboard in the following files. All necessary requirements are also listed in the `requirements.txt` file.

**`finetune_clip.py`**, **`finetune_vit.py`**:

These files can be used to finetune the CLIP/ViT model on the AVA Dataset

**`predict_clip.py`**, **`finetune_vit.py`**:

With these files and a trained model from the previous scripts you can analyse the performance of the model and create csv-files with the predictions of the model. 
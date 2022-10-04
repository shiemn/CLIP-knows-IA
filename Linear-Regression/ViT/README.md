# Overview

The following files can be used to perform the same experiments as in Section 4.2 of the paper:

We mainly use PyTorch and scikit-learn for these experiments. All necessary requirements are also listed in the `requirements.txt` file.

**`clip_embeddings.py`**, **`in21_embeddings.py`**:

These files can be used to create embeddings for the images of the AVA Dataset using either a CLIP model (**`clip_embeddings.py`**) or a ViT trained on IN21k (**`in21_embeddings.py`**).

**`regression.ipynb`**:

After creating the embedding files using the previously mentioned scripts, you can use this short notebook to perform LinearRegression and evaluate the performance of LinearProbing on the AVA Dataset.

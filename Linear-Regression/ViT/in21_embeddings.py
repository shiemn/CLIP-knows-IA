import sys
import time
import wandb

import torch
from torch.utils.data import DataLoader


import timm
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config


sys.path.append('/app/dataset/')

from AVA import AVADataset_with_ids as AVADataset

#Config, only used for experiment tracking in WANDB
config = {
    'model-type':'ViT32-IN21k'
}


#Set the paths correctly for your environment
CACHE_PATH = "~/.cache"
TRAIN_CSV_PATH = '/app/dataset/AVA_train.csv'
TEST_CSV_PATH = '/app/dataset/AVA_test.csv'
AVA_ROOT_DIR = '/memoryDataset/AVA2/images/'
SAVE_PATH = "/app/encodings/"

SUMMARY_PATH = 'runs/'

torch.hub.set_dir(CACHE_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


wandb.init(project="CLIP-IAA", entity="shiemn", config=config)

#Load the ViT model
model = timm.create_model('vit_base_patch32_224_in21k', pretrained=True)

model.to(device)
model.eval()

config = resolve_data_config({}, model=model)
transform = create_transform(**config)



#Create embeddings for each of the images in the test and train dataset and save them as a .pt file
for split in ['train', 'test']:
    print(f"Creating Embeddings for Split: {split}")

    if split == 'train':
        dataset = AVADataset(csv_file = TRAIN_CSV_PATH, root_dir = AVA_ROOT_DIR, transform = preprocess)
    else:
        dataset = AVADataset(csv_file = TEST_CSV_PATH, root_dir = AVA_ROOT_DIR, transform = preprocess)

    print("dataset loaded")
    
    dataloader = DataLoader(dataset, batch_size = 512, shuffle = False, pin_memory=True)
    print("dataloader ready")


    encodings = []
    ids = []
    ratings = []
    count = 0
    start = time.time()

    for i_batch, sample_batch in enumerate(dataloader):
        img, rating, id = sample_batch
        img = img.to(device)

        with torch.no_grad():
            image_features = model.forward_features(img)
        
        
        encodings.append(image_features.cpu())
        ids.append(id)
        ratings.append(rating.cpu())
        
        end = time.time()
        count += 1
        print(f"The {count}. Batch of 512 images has been processed in {end-start} s")
        start = time.time()
        
    
    torch.save(torch.cat(encodings), SAVE_PATH + split + '_encodings')   
    torch.save(ids, SAVE_PATH + split + '_ids')
    torch.save(torch.cat(ratings), SAVE_PATH + split + '_ratings')

    wandb.save(SAVE_PATH + split + '_encodings')
    wandb.save(SAVE_PATH + split + '_ids')
    wandb.save(SAVE_PATH + split + '_ratings')
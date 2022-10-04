import sys
import time
import wandb

import torch
import torch.nn as nn
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

sys.path.append('/app/dataset/')

from AVA import AVADataset_with_ids as AVADataset

#Config, only used for experiment tracking in WANDB
config = {
    'model-type':'ResNet50'
}

#Set the paths correctly for your environment
CACHE_PATH = "~/.cache"
CLIP_VARIANT = 'ResNet50-ImageNet21k'
TRAIN_CSV_PATH = '/app/dataset/AVA_train.csv'
TEST_CSV_PATH = '/app/dataset/AVA_test.csv'
AVA_ROOT_DIR = '/memoryDataset/AVA2/images/'
#Specify path to the model weights. The weights are available from https://github.com/Alibaba-MIIL/ImageNet21K
MODEL_PATH = '/app/.cache/resnet50_miil_21k.pth'
SAVE_PATH = "/app/encodings/"

SUMMARY_PATH = 'runs/'

torch.hub.set_dir(CACHE_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


def _convert_to_rgb(image):
    return image.convert('RGB')

def load_model_weights(model, model_path):
    #Correctly initialize the weights for the ResNet model

    state = torch.load(model_path, map_location='cpu')
    for key in model.state_dict():
        if 'num_batches_tracked' in key:
            continue
        p = model.state_dict()[key]
        if key in state['state_dict']:
            ip = state['state_dict'][key]
            if p.shape == ip.shape:
                p.data.copy_(ip.data)  # Copy the data of parameters
            else:
                print(
                    'could not load layer: {}, mismatch shape {} ,{}'.format(key, (p.shape), (ip.shape)))
        else:
            print('could not load layer: {}, not in checkpoint'.format(key))
    return model


wandb.init(project="CLIP-IAA", entity="shiemn", config=config)

#Load the RN50 model using the timm library
model = timm.create_model('resnet50', pretrained=False, num_classes=11221)
config = resolve_data_config({}, model=model)
preprocess = create_transform(**config)
preprocess = Compose([preprocess.transforms[0], preprocess.transforms[1], _convert_to_rgb, preprocess.transforms[2], preprocess.transforms[3]])
model = load_model_weights(model, MODEL_PATH)

#Remove the classification head
model.fc = nn.Identity()
model = model.to(device)


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
            image_features = model(img)
        
        
        encodings.append(image_features.cpu())
        ids.append(id)
        ratings.append(rating.cpu())
        
        end = time.time()
        count += 1
        print(f"The {count}. Batch of 512 images has been processed in {end-start} s")
        start = time.time()
        
    
    torch.save(torch.cat(encodings), SAVE_PATH + split + '_encodings.pt')   
    torch.save(ids, SAVE_PATH + split + '_ids.pt')
    torch.save(torch.cat(ratings), SAVE_PATH + split + '_ratings.pt')

    wandb.save(SAVE_PATH + split + '_encodings.pt')
    wandb.save(SAVE_PATH + split + '_ids.pt')
    wandb.save(SAVE_PATH + split + '_ratings.pt')
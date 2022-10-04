from argparse import ArgumentParser


from ignite.engine import Events, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss


from torch.utils.data import DataLoader


import torch.nn as nn

import numpy as np

import torch 
import sys 
sys.path.append('./helpers')
sys.path.append('./dataset')
from IgniteEMD import EMD_Loss
from AVA_vit import AVADataset
from Pearson import PearsonCorr
from Spearman import SpearmanR

import timm
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config

import pandas as pd


# Specify the paths to the image folder of the AVA Dataset and the Test/Train csv-files.
TRAIN_CSV_PATH = 'dataset/AVA_train.csv'
TEST_CSV_PATH = 'dataset/AVA_test.csv'
AVA_ROOT_DIR = '/memoryDataset/AVA2/images/'

#Specify the model-file to be used
MODEL_SAVE_PATH = 'models/ViT_model_20220215_104653.pt'

CSV_PATH = 'runs/prediction-vit.csv'




class Model(nn.Module):

    def __init__(self, base_model):
        super(Model, self).__init__()
        self.base = base_model
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.0),
            nn.Linear(in_features=768, out_features=10),
            nn.Softmax(dim=1))

    def forward(self, x):
        x = self.base(x)
        return self.classifier(x)
        



def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_model = timm.create_model('vit_base_patch32_224', pretrained=True)
    config = resolve_data_config({}, model=base_model)
    preprocess = create_transform(**config)

    base_model = base_model.to(device)
    base_model.head = nn.Identity()
    model = Model(base_model).float().to(device)

    print(f"Running on {device}")
    
    
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    #MODEL_SAVE_PATH = "models/ViT_model_" + time.strftime("%Y%m%d_%H%M%S") + ".pt"
    print(f"ModelSavePath : {MODEL_SAVE_PATH}")

    print(model)

    test_dataset = AVADataset(TEST_CSV_PATH, AVA_ROOT_DIR, preprocess)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=16)

    print("DataLoaders ready")

    #criterion = EarthMoverDistance()
    criterion = EMD_Loss()

    def binary_transform(output):
        y_pred, y = output
        
        y_pred = y_pred @ torch.tensor([1,2,3,4,5,6,7,8,9,10], dtype=torch.float32, device=device) > 5.0
        y = y @ torch.tensor([1,2,3,4,5,6,7,8,9,10], dtype=torch.float32, device=device) > 5.0

        return y_pred, y

    def continuous_transform(output):
        y_pred, y = output
        
        y_pred = y_pred @ torch.tensor([1,2,3,4,5,6,7,8,9,10], dtype=torch.float32, device=device)
        y = y @ torch.tensor([1,2,3,4,5,6,7,8,9,10], dtype=torch.float32 ,device=device)

        return y_pred, y

    

    val_metrics = {
        "EMD": Loss(criterion),
        "Accuracy": Accuracy(output_transform=binary_transform),
        "LCC": PearsonCorr(output_transform=continuous_transform),
        "SRCC": SpearmanR(output_transform=continuous_transform),
    }

    run.best_loss = np.Infinity #torch.tensor[float('inf')]
    run.early_stopping_counter = 0

    evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)


    def log_test_results():
        print("Run test Evaluator")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
       
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        print(f"Test Results - Avg loss: {metrics['EMD']:.4f},  Avg Accuracy: {metrics['Accuracy']:.4f}, SRCC: {metrics['SRCC']:.4f}, LCC: {metrics['LCC']:.4f}")

    
    def create_csv():

        print("Start creating CSV-File")

        predictions = []
        ground_truths = []
        image_ids = []

        with torch.no_grad():
            model.eval()

            for (idx, batch) in enumerate(test_loader):
                img = batch[0].to(device)
                pred = model(img)

                predictions.append(pred.cpu())
                ground_truths.append(batch[1].cpu())
                image_ids.extend(batch[2])

                if idx % 10 == 0:
                    print(f"TestDataset: {idx}/{len(test_loader)}")


            predictions = torch.cat(predictions) @ torch.tensor([1,2,3,4,5,6,7,8,9,10], dtype=torch.float32, device='cpu')
            ground_truths = torch.cat(ground_truths) @ torch.tensor([1,2,3,4,5,6,7,8,9,10], dtype=torch.float32, device='cpu')

            data = {'prediction': predictions.squeeze().numpy(), 'ground_truth': ground_truths.squeeze().numpy(), 'image_id': image_ids}
            print(data)
            df = pd.DataFrame(data)
            df.to_csv(CSV_PATH)  

        print("CSV Done...")

    create_csv()
    log_test_results()



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--modelpath", type=str, action='store', help="Path to model File"
    )

    args = parser.parse_args()

    run(args)

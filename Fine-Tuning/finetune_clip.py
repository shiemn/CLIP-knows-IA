from argparse import ArgumentParser
from re import X
import time

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

from scipy.stats import pearsonr, spearmanr

from torch.utils.data import DataLoader


from tensorboardX import SummaryWriter

import torch.nn as nn
from torch.optim import SGD

import numpy as np

import clip
import torch 
import sys 
sys.path.append('./helpers')
sys.path.append('./dataset')
from IgniteEMD import EarthMoverDistance, EMD_Loss
from AVA import AVADataset, get_train_val
from Pearson import PearsonCorr
from Spearman import SpearmanR


#Hyperparameters
DROPOUT = 0.0
LR_VISION = 1e-4
LR_CLASSIFIER = 1e-2
CLIP_VARIANT = 'ViT-B/32'
EARLY_STOPPING_PATIENCE = 10


# Specify the paths to the image folder of the AVA Dataset and the Test/Train csv-files.
TRAIN_CSV_PATH = 'dataset/AVA_train.csv'
TEST_CSV_PATH = 'dataset/AVA_test.csv'
AVA_ROOT_DIR = '/memoryDataset/AVA2/images/'

#Specify where to save the tensorboard files.
SUMMARY_PATH = 'runs/'





class Model(nn.Module):

    def __init__(self, base_model):
        super(Model, self).__init__()
        self.base = base_model.visual
        self.base.proj = None
        self.classifier = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            nn.Linear(in_features=768, out_features=10),
            nn.Softmax(dim=1))

    def forward(self, x):
        x = self.base(x)
        return self.classifier(x)
        



def run(train_batch_size, val_batch_size, epochs, momentum, log_interval):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    #Load the CLIP-Model from OpenAI
    base_model, preprocess = clip.load(CLIP_VARIANT, device, download_root=".cache")
    base_model = base_model.to(device)
    model = Model(base_model).float().to(device)

    #Initializing Tensorboard and specifying the path for the model
    print(f"Running on {device}")
    writer = SummaryWriter()
    MODEL_SAVE_PATH = "models/model_" + time.strftime("%Y%m%d_%H%M%S") + ".pt"
    print(f"ModelSavePath : {MODEL_SAVE_PATH}")

    writer.add_text('Mode', '768 Dimension CLIP')
    writer.add_text('DROPOUTRATE', str(DROPOUT))
    writer.add_text('EARLY_STOPPING_PATIENCE', str(EARLY_STOPPING_PATIENCE))
    writer.add_text('LR_VISION', str(LR_VISION))
    writer.add_text('LR_CLASSIFIER', str(LR_CLASSIFIER))
    writer.add_text('MODEL_SAVE_PATH', str(MODEL_SAVE_PATH))
    writer.add_text('CLIP_VARIANT', str(CLIP_VARIANT))
    
    print(f"LR_VISION: {LR_VISION}, LR_HEAD: {LR_CLASSIFIER}")
    print(model)


    #creating the Dataset and DataLoader objects
    train_dataset, val_dataset =  get_train_val(TRAIN_CSV_PATH, AVA_ROOT_DIR, preprocess, [9,1])
    test_dataset = AVADataset(TEST_CSV_PATH, AVA_ROOT_DIR, preprocess)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=16)
    val_loader= DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=16)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=16)

    print("DataLoaders ready")

    #Initializing the Optimizer with different learning rates for the classification head and the rest of the model
    optimizer = SGD([
            {'params': model.base.parameters(), 'lr': LR_VISION},
            {'params': model.classifier.parameters(), 'lr': LR_CLASSIFIER}
        ], momentum=momentum)

    criterion = EMD_Loss()

    #preparing the pytorch-ignite training loop
    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    print("Trainer ready")

    def binary_transform(output):
        """
        Transforms the rating distribution into a binary representation for the binary classification tasks. Only used during the evaluation.
        """
        y_pred, y = output
        
        y_pred = y_pred @ torch.tensor([1,2,3,4,5,6,7,8,9,10], dtype=torch.float32, device=device) > 5.0
        y = y @ torch.tensor([1,2,3,4,5,6,7,8,9,10], dtype=torch.float32, device=device) > 5.0

        return y_pred, y

    def continuous_transform(output):
        """
        Transforms the rating distribution into a single score between 1 and 10 for the continuous tasks. Only used during the evaluation.
        """

        y_pred, y = output
        
        y_pred = y_pred @ torch.tensor([1,2,3,4,5,6,7,8,9,10], dtype=torch.float32, device=device)
        y = y @ torch.tensor([1,2,3,4,5,6,7,8,9,10], dtype=torch.float32, device=device)

        return y_pred, y

    #Defining the validation metrics
    val_metrics = {
        "EMD": Loss(criterion),
        "Accuracy": Accuracy(output_transform=binary_transform),
        "LCC": PearsonCorr(output_transform=continuous_transform),
        "SRCC": SpearmanR(output_transform=continuous_transform),
    }

    #Setup of the early stopping
    run.best_loss = np.Infinity 
    run.early_stopping_counter = 0

    evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)


    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(trainer):
        print(f"Epoch[{trainer.state.epoch}] Loss: {trainer.state.output:.4f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        print(f"Training Results - Epoch: {trainer.state.epoch}  Avg loss: {metrics['EMD']:.4f},  Avg Accuracy: {metrics['Accuracy']:.4f}, SRCC: {metrics['SRCC']:.4f}, LCC: {metrics['LCC']:.4f}")

        writer.add_scalar('Loss/Train', metrics["EMD"], trainer.state.epoch)
        writer.add_scalar('Accuracy/Train', metrics["Accuracy"], trainer.state.epoch)
        writer.add_scalar('Pearson/Train', metrics["LCC"], trainer.state.epoch)
        writer.add_scalar('Spearman/Train', metrics["SRCC"], trainer.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        print(f"Validation Results - Epoch: {trainer.state.epoch}  Avg loss: {metrics['EMD']:.4f},  Avg Accuracy: {metrics['Accuracy']:.4f}, SRCC: {metrics['SRCC']:.4f}, LCC: {metrics['LCC']:.4f}")

        writer.add_scalar('Loss/Val', metrics["EMD"], trainer.state.epoch)
        writer.add_scalar('Accuracy/Val', metrics["Accuracy"], trainer.state.epoch)
        writer.add_scalar('Pearson/Val', metrics["LCC"], trainer.state.epoch)
        writer.add_scalar('Spearman/Val', metrics["SRCC"], trainer.state.epoch)
        

        if metrics["EMD"] <= run.best_loss:
            run.best_loss = metrics['EMD'] 
            run.early_stopping_counter = 0

            torch.save(model.state_dict(), MODEL_SAVE_PATH)

        else:
            run.early_stopping_counter += 1
            print(f"EarlyStopping: {run.early_stopping_counter}/{EARLY_STOPPING_PATIENCE}")

            if run.early_stopping_counter >= EARLY_STOPPING_PATIENCE:
                print('EarlyStopping: Stop training')
                trainer.terminate()


    endEvents = Events.COMPLETED | Events.TERMINATE

    @trainer.on(endEvents)
    def log_test_results(trainer):
        print("Test best Model...")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))

        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        print(f"Test Results - Epoch: {trainer.state.epoch}  Avg loss: {metrics['EMD']:.4f},  Avg Accuracy: {metrics['Accuracy']:.4f}, SRCC: {metrics['SRCC']:.4f}, LCC: {metrics['LCC']:.4f}")

        writer.add_scalar('Loss/Test', metrics["EMD"], trainer.state.epoch)
        writer.add_scalar('Accuracy/Test', metrics["Accuracy"], trainer.state.epoch)
        writer.add_scalar('Pearson/Test', metrics["LCC"], trainer.state.epoch)
        writer.add_scalar('Spearman/Test', metrics["SRCC"], trainer.state.epoch)


    trainer.run(train_loader, max_epochs=epochs)

    log_test_results(trainer)

    writer.flush()
    writer.close()

    


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128, help="input batch size for training (default: 128)")
    parser.add_argument(
        "--val_batch_size", type=int, default=128, help="input batch size for validation (default: 128)"
    )
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs to train (default: 50)")
    parser.add_argument("--momentum", type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument(
        "--log_interval", type=int, default=300, help="how many batches to wait before logging training status"
    )

    args = parser.parse_args()

    run(args.batch_size, args.val_batch_size, args.epochs, args.momentum, args.log_interval)

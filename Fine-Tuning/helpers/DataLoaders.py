import torch
from dataset.AVA import AVADataset

def get_data_loaders(csv_path: str, root_dir: str, transform = None, splits: list = [1,0]):

    dataset = AVADataset(csv_path, root_dir, transform)
    train, validation = torch.utils.data.random_split( dataset, splits, generator=torch.Generator().manual_seed(42))
    
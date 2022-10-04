from typing import Tuple
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch 
from PIL import Image
import os
import numpy as np

import warnings



class AVADataset(Dataset):

    def __init__(self, csv_file, root_dir, transform = None):
        """
        Args:
            csv_file (string): Path to the csv file with ratings.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.ratings_df = pd.read_csv(csv_file)
        self.root_dir = root_dir

        self.transform = transform
        
    def __len__(self):
        return len(self.ratings_df)
    
    def __getitem__(self, index):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if torch.is_tensor(index):
                index = index.tolist()
            
            try:
                id = str(int(self.ratings_df.iloc[index, 0]))
                img_name = os.path.join(self.root_dir, id + '.jpg')
                #print(img_name)
                image = Image.open(img_name)
                ratings = self.ratings_df.iloc[index, 1:]
                ratings = np.array([ratings])
                ratings = ratings.astype('float')
                
                ratings = ratings / ratings.sum()
                
                if self.transform:
                    image = self.transform(image)

                image = image.float()
                ratings = torch.tensor(ratings).float()

                #sample = {'image': image, 'ratings': ratings, 'id': int(self.ratings_df.iloc[index, 0])}
                sample = (image, ratings.squeeze())
                
                #print(sample[0].shape, sample[1].shape)
                return sample
            
            except OSError:
                
                #traceback.print_exc()
                print(f"Error when loading Image with the id {id}")
                return self.__getitem__(index-10)

    def get_target(self, index):

        if torch.is_tensor(index):
            index = index.tolist()
                
        id = str(int(self.ratings_df.iloc[index, 0]))
        ratings = self.ratings_df.iloc[index, 1:]
        ratings = np.array([ratings])
        ratings = ratings.astype('float')
                
        ratings = ratings / ratings.sum()
        ratings = torch.tensor(ratings).float()

                
        return ratings.squeeze()
            

class AVADataset_with_ids(Dataset):

    def __init__(self, csv_file, root_dir, transform = None):
        """
        Args:
            csv_file (string): Path to the csv file with ratings.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.ratings_df = pd.read_csv(csv_file)
        self.root_dir = root_dir

        self.transform = transform
        
    def __len__(self):
        return len(self.ratings_df)
    
    def __getitem__(self, index):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if torch.is_tensor(index):
                index = index.tolist()
            
            try:
                id = str(int(self.ratings_df.iloc[index, 0]))
                img_name = os.path.join(self.root_dir, id + '.jpg')
                #print(img_name)
                image = Image.open(img_name)
                ratings = self.ratings_df.iloc[index, 1:]
                ratings = np.array([ratings])
                ratings = ratings.astype('float')
                
                ratings = ratings / ratings.sum()
                
                if self.transform:
                    image = self.transform(image)

                image = image.float()
                ratings = torch.tensor(ratings).float()

                #sample = {'image': image, 'ratings': ratings, 'id': int(self.ratings_df.iloc[index, 0])}
                sample = (image, ratings.squeeze(), id)
                
                #print(sample[0].shape, sample[1].shape)
                return sample
            
            except OSError:
                
                #traceback.print_exc()
                print(f"Error when loading Image with the id {id}")
                return self.__getitem__(index-10)

    def get_target(self, index):

        if torch.is_tensor(index):
            index = index.tolist()
                
        id = str(int(self.ratings_df.iloc[index, 0]))
        ratings = self.ratings_df.iloc[index, 1:]
        ratings = np.array([ratings])
        ratings = ratings.astype('float')
                
        ratings = ratings / ratings.sum()
        ratings = torch.tensor(ratings).float()

                
        return ratings.squeeze() 




def get_train_val(csv_path: str, root_dir: str, transform = None, splits: float = 0.8):
    """[summary]

    Args:
        csv_path (str): [description]
        root_dir (str): [description]
        transform ([type], optional): [description]. Defaults to None.
        splits (float, optional): Share of the Train-Split. Defaults to 0.8.

    Returns:
        [type]: [description]
    """    

    
    dataset = AVADataset(csv_path, root_dir, transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train, validation = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    return (train, validation)



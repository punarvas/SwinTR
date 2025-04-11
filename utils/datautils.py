import pandas as pd # type: ignore
import os
import mrcfile
import numpy as np
from torch.utils.data import Dataset, DataLoader, DistributedSampler, TensorDataset
import torch
from monai.transforms import RandSpatialCropd, Compose



def one_hot_encode(array, output_dim, roi_size):
    n_classes = output_dim
    one_hot = np.zeros((n_classes, roi_size, roi_size, roi_size), dtype=np.int8)
    for i in range(n_classes):
        one_hot[i] = (array == i).astype(np.int8)
    return one_hot


class TomogramDataset(Dataset):
    def __init__(self, X, y, roi_size: int, input_dim: int, output_dim: int,
                 scale_x:bool=True, to_ohe_y:bool=True, monai_transform: bool = True):
        self.X = X
        self.y = y
        assert len(self.X) == len(self.y)
        self.length = len(self.X)
        self.X_min = np.min(self.X)
        self.X_max = np.max(self.X)
        self.scale_x = scale_x
        self.roi_size = roi_size
        self.to_ohe_y = to_ohe_y

        self.input_dim, self.output_dim = input_dim, output_dim

        self.input_shape = self.X[0].shape
        self.output_shape = self.y[0].shape
        print("Input shape:", self.input_shape)
        print("Output shape:", self.output_shape)

        self.monai_transform = monai_transform
        if self.monai_transform:
            self.transform = Compose([
                RandSpatialCropd(keys=["image", "label"], 
                                 roi_size=(roi_size, roi_size, roi_size), random_size=False)
            ])

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if self.scale_x:
            x = (x - self.X_min) / (self.X_max - self.X_min)
        if self.monai_transform:   # Data augmentation
            inputs = {"image": x.reshape(-1, *self.input_shape), 
                      "label": y.reshape(-1, *self.output_shape)}
            transformed_inputs = self.transform(inputs)
            x = transformed_inputs["image"]
            y = transformed_inputs["label"]
            # print("MONAI shape:", x.shape, y.shape)
        if self.to_ohe_y:
            y = one_hot_encode(y, self.output_dim, self.roi_size)
            # print("one_hot_encoded y shape:", y.shape)

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.int8)
        x = x.reshape(self.input_dim, self.roi_size, self.roi_size, self.roi_size)
        y = y.reshape(self.output_dim, self.roi_size, self.roi_size, self.roi_size)

        # print("Final shape:", x.shape, y.shape)
        return x, y
    
    def __len__(self):
        return self.length


def prepare_dataloader(dataset: Dataset, batch_size: int, distributed: bool = True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset) if distributed else None
    )

def load_tomodataset(t: pd.DataFrame, roi_size: int, input_dim: int, output_dim: int, 
                     monai_transform:bool=True, use_tomodataset: bool = True):
    X = []
    y = []
    for i in range(t.shape[0]):
        input_vol_path = t.iloc[i, 0]
        target_vol_path = t.iloc[i, 1]
        assert os.path.exists(input_vol_path) == True
        assert os.path.exists(target_vol_path) == True

        with mrcfile.open(input_vol_path, permissive=True) as mrc:
            source = mrc.data
        with mrcfile.open(target_vol_path, permissive=True) as mrc:
            target = mrc.data

        X.append(source)
        y.append(target)

    if use_tomodataset:
        dataset = TomogramDataset(X, y, roi_size=roi_size, input_dim=input_dim, 
                              output_dim=output_dim, monai_transform=monai_transform)
    else:
        X, y = np.array(X), np.array(y)
        n_samples = len(X)
        image_shape = X[0].shape
        X = X.reshape(n_samples, input_dim, *image_shape)
        X = torch.from_numpy(X).type(torch.float32)
        y = torch.from_numpy(y).type(torch.int8)  
        dataset = TensorDataset(X, y)
    return dataset 

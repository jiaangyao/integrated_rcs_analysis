import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset

import model.torch_model.torch_utils as ptu


class NeuralDataset(Dataset):
    def __init__(
        self,
        features: npt.NDArray | None,
        labels: npt.NDArray | None,
        transform=None,
        target_transform=None,
    ):
        # sanity check
        assert (
            features is not None and labels is not None
        ), "Need to initialize with either features and labels"
                
        # obtain the features and labels
        self.features = ptu.from_numpy(features)
        self.labels = ptu.from_numpy(labels).long()
        assert torch.is_tensor(self.features)
        assert torch.is_tensor(self.labels)

        # obtain the transforms
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        # convert the index to a list
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # apply transform to features
        features_sample = torch.clone(self.features[idx, ...])
        if self.transform:
            features_sample = self.transform(features_sample)

        # apply transform to labels
        label_sample = torch.clone(self.labels[idx, ...])
        if self.target_transform:
            label_sample = self.target_transform(label_sample)

        return features_sample, label_sample


class NeuralDatasetTest(Dataset):
    def __init__(self, features: npt.NDArray):
        # obtain the features and labels
        self.features = ptu.from_numpy(features)
        assert torch.is_tensor(self.features)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        # convert the index to a list
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # apply transform to features
        features_sample = torch.clone(self.features[idx, ...])
        return features_sample

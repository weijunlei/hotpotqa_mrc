import torch
import numpy as np
from torch.utils.data import Dataset


class LazyLoadTensorDataset(Dataset):

    def __init__(self, features, is_training):
        """
        Args:
            features: feature list
        """
        self.features = features
        self.is_training = is_training

    def __getitem__(self, index):
        feature = self.features[index]
        input_ids = torch.tensor(feature.input_ids, dtype=torch.long)
        input_mask = torch.tensor(feature.input_mask, dtype=torch.long)
        segment_ids = torch.tensor(feature.segment_ids, dtype=torch.long)
        example_index = torch.tensor(index, dtype=torch.long)
        tensors = [input_ids,
                   input_mask,
                   segment_ids,
                   example_index]

        if self.is_training:
            label = torch.tensor(feature.label, dtype=torch.long)
            tensors.append(label)
        return tensors

    def __len__(self):
        return len(self.features)

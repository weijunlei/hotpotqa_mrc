import torch
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
        cls_mask = torch.tensor(feature.cls_mask, dtype=torch.long)
        cls_weight = torch.tensor(feature.cls_weight, dtype=torch.float)

        tensors = [input_ids,
                   input_mask,
                   segment_ids,
                   cls_mask,
                   cls_weight,
                   ]

        if self.is_training:
            cls_label = torch.tensor(feature.cls_label, dtype=torch.long)
            tensors.append(cls_label)
            example_index = torch.tensor(index, dtype=torch.long)
            tensors.append(example_index)
        else:
            example_index = torch.tensor(index, dtype=torch.long)
            tensors.append(example_index)
        return tensors

    def __len__(self):
        return len(self.features)

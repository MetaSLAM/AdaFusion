"""Pairwise Margin-based loss and hard mining strategy"""

from pathlib import Path
import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from .read_dataset import Robotcar, NCLT, SubT, Pittsburgh

dataset_dict = {
    'robotcar': Robotcar,
    'nclt': NCLT,
    'subT': SubT,
    'pittsburgh': Pittsburgh,
}


class PairwiseData(Dataset):
    """Generate pairwise data (pos_pair, neg_pair) for training and testing.
    Supported datasets are Oxford Robotcar and SubT.
       Due to the strange property of `torch.utils.data.DataLoader` (class members
    cannot be saved), the hard mining part is moved to another class ``.
    """

    def __init__(self, config):
        """Initialize variables.
        Args:
            config: The *.yaml config EasyDict object.
        Returns: None.
        """
        # load pair relation from the pickle file
        config = config.dataset
        pair_file = config.train_pickle
        base_path = Path(config.base_path)
        with open(base_path / Path(pair_file), 'rb') as file:
            self.pairs = pickle.load(file)

        # data reader
        ReaderName = dataset_dict[config.name]
        self.reader = ReaderName(
            self.pairs['file_indices'],
            base_path,
            config.image_size,
            config.voxel_shape,
            config.augmentation,
        )
        self.posNum = len(self.pairs['pos_pairs'])
        self.negNum = len(self.pairs['neg_pairs'])
        self.random_num = torch.randint(200, 7000, (1,)).item()

    def __len__(self):
        """Return the length of the dataset."""
        return self.posNum

    def __getitem__(self, index):
        """Get the index-th item of the data, return {'pos_pair':(img_pc1, img_pc2),
        'neg_pair': (img_pc3, img_pc4)}, where img_pcx = (image, point cloud).
        """
        pos_indices = self.pairs['pos_pairs'][index]
        neg_indices = self.pairs['neg_pairs'][(index * self.random_num) % self.negNum]

        result = {
            'pos_pair': (
                self.reader.read_data(pos_indices[0]),
                self.reader.read_data(pos_indices[1]),
            ),
            'neg_pair': (
                self.reader.read_data(neg_indices[0]),
                self.reader.read_data(neg_indices[1]),
            ),
        }
        return result

    def shuffle_data(self, number):
        self.random_num = number


class PairwiseMarginLoss(nn.Module):
    def __init__(self, a: float, m: float, Lp: float):
        """Provide the pairwise margin-based loss in [1].
        Args:
            a, m: Parameters for margin. The distance of positive pairs is below
                  m-a and the distance of negative pairs is above m+a. Float.
            Lp: Use Lp-norm as the distance metric.
        Returns: None.
        """
        super(PairwiseMarginLoss, self).__init__()
        self.a = a
        self.m = m
        self.distance = nn.PairwiseDistance(p=Lp, keepdim=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, y: int):
        """Calculate the pairwise margin-based loss
           L(yij, Xi, Xj) = (α + yij · (d(Xi, Xj) − m))
        Args:
            x1, x2: Feature representation of the pair, [N, D]
            y: Label indicating positive pair (1) and negative pair (-1).
               Other values are not allowed.
        Returns:
            loss: The required loss, torch.Tensor of size [N]
        """
        # if y not in {1, -1}:
        #     raise ValueError(f'Label y for loss must be in {1, -1}. y={y}')
        loss = y * (self.distance(x1, x2) - self.m) + self.a
        return self.relu(loss)


def train_loader_generator(train_loader):
    """Helper function to convert dataloader to a generator. For `hard_mining()`
    This helper function (generator) can avoid sampling the same data.
    Args:
        train_loader: torch.utils.data.DataLoader
    Returns:
        train_loader_generator: The generator type.
    """
    while True:
        for pairs in train_loader:
            yield pairs


def hard_mining(train_loader, net, criterion, device, config):
    """Sample and make hard mining batch.
    Args:
        train_loader: can be torch.utils.data.DataLoader or generator
            from function `train_loader_generator()`
    Returns:
        hard_pair: The sampled hard negative pair
        loss_1 : The top 1 loss of the sample in the hard negative pair.
    """
    x1, x2 = [], []  # store features representation x1 and x2
    image1, pc1, image2, pc2 = [], [], [], []

    net.eval()  # change to 'evaluate' stage
    with torch.no_grad():
        for batch_index, pairs in enumerate(train_loader):
            image, pc = pairs['neg_pair'][0]
            image, pc = image.to(device), pc.to(device)
            image1.append(image)
            pc1.append(pc)
            x1.append(net(image, pc))
            #
            image, pc = pairs['neg_pair'][1]
            image, pc = image.to(device), pc.to(device)
            image2.append(image)
            pc2.append(pc)
            x2.append(net(image, pc))

            if batch_index + 1 == config.hardM.nBatch:
                break

        loss = criterion(torch.cat(x1), torch.cat(x2), -1)  # of size [N]

    # concatenate along Batch-axis N (dim=0)
    image1 = torch.cat(image1)
    pc1 = torch.cat(pc1)
    image2 = torch.cat(image2)
    pc2 = torch.cat(pc2)

    loss_ind = loss.argsort(descending=True)[: config.train_batch_size]
    hard_pair = (
        (image1[loss_ind], pc1[loss_ind]),
        (image2[loss_ind], pc2[loss_ind]),
    )

    # enlarge nBatch if loss[config.train_batch_size] becomes small
    # if loss[loss_ind[0]] < config.hardM.enlarge_thres:
    #     config.hardM.nBatch += 2

    net.train()  # change to 'train' stage
    return hard_pair, loss[loss_ind[0]]

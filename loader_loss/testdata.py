from pathlib import Path
import pickle
from torch.utils.data import Dataset

from .read_dataset import Robotcar, NCLT, SubT, Pittsburgh

dataset_dict = {
    'robotcar': Robotcar,
    'nclt': NCLT,
    'subT': SubT,
    'pittsburgh': Pittsburgh,
}


class TestDataRoboNclt(Dataset):
    """Class `TestData` for `Robotcar and NCLT`, testing and evaluation.
      The task of it is to simply retern X=(image, point cloud). Since test
    data has the same form for all types of losses, we make a new class to
    provide it.
    """

    def __init__(self, config):
        """Initialize variables.
        Args:
            config: The *.yaml config EasyDict object.
        Returns: None.
        """
        # load test items matching relation from the pickle file
        config = config.dataset
        pickle_file = Path(config.test_pickle)
        base_path = Path(config.base_path)
        with open(base_path / pickle_file, 'rb') as f:
            self.items_pickle = pickle.load(f)

        # file_indices = {0:['2014-07-14-14-49-50/1417534542711435',...],...}
        file_indices = self.items_pickle['file_indices']
        self.num_of_each_run = [len(file_indices[i]) for i in range(len(file_indices))]

        # combine each seperate run
        self.combine_indices = []
        for i in range(len(file_indices)):
            self.combine_indices.extend(file_indices[i])

        # data reader
        ReaderName = dataset_dict[config.name]
        self.reader = ReaderName(
            self.combine_indices,
            base_path,
            config.image_size,
            config.voxel_shape,
            augment=False,
        )

    def __len__(self):
        """Return the length of the dataset."""
        return sum(self.num_of_each_run)

    def __getitem__(self, index):
        """Get the index-th item of the data, return X = (image, point cloud)."""
        return self.reader.read_data(index)

    def get_pos_items(self):
        """Return {(0,1):[item1, item2, ....], (0,2):[],...}, itemx = [id1, id2, ...]"""
        return self.items_pickle['pos_items']

    def get_num_of_each_run(self):
        """number of data of each run, e.g. [212, 334, ...]"""
        return self.num_of_each_run

    def get_all_test_file_names(self):
        """All the test file names, ['xxx/xxxxxx', 'xxx/xxxxxxx']"""
        return self.combine_indices


class TestDataPitts(Dataset):
    """Class `TestData` of Pittsburgh dataset for testing and evaluation.
      The task of it is to simply retern X=(image, point cloud). Since test
    data has the same form for all types of losses, we make a new class to
    provide it.
    """

    def __init__(self, config):
        """Initialize variables.
        Args:
            config: The *.yaml config EasyDict object.
        Returns: None.
        """
        # load test items matching relation from the pickle file
        config = config.dataset
        pickle_file = Path(config.test_pickle)
        base_path = Path(config.base_path)
        with open(base_path / pickle_file, 'rb') as f:
            self.items_pickle = pickle.load(f)

        # file_indices = ['train_1/xxxx.pcd',...]
        self.combine_indices = self.items_pickle['file_indices']
        self.num_of_each_run = [0, len(self.combine_indices)]

        # data reader
        ReaderName = dataset_dict[config.name]
        self.reader = ReaderName(
            self.combine_indices,
            base_path,
            config.image_size,
            config.voxel_shape,
            augment=False,
        )

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.combine_indices)

    def __getitem__(self, index):
        """Get the index-th item of the data, return X = (image, point cloud)."""
        return self.reader.read_data(index)

    def get_pos_items(self):
        """Return {(0,1):[item1, item2, ....], (0,2):[],...}, itemx = [id1, id2, ...]"""
        return {(0, 1): self.items_pickle['pos_items']}

    def get_num_of_each_run(self):
        """number of data of each run, e.g. [212, 334, ...]"""
        return self.num_of_each_run

    def get_all_test_file_names(self):
        """All the test file names, ['xxx/xxxxxx', 'xxx/xxxxxxx']"""
        return self.combine_indices


def get_test_set(config):
    """Select a proper test set according to different dataset.
    Args:
        config: The *.yaml config EasyDict object.
    Returns:
        test set, either `TestDataRoboNclt` or `TestDataPitts`
    """
    if config.dataset.name == 'pittsburgh':
        return TestDataPitts(config)
    else:
        return TestDataRoboNclt(config)

"""This file provides classes that represent each dataset. All dataset
classes have the same interface.
   For basic functions please see the base class `baseDataset`
"""

from typing import List, Tuple

import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torchvision.transforms as transforms
import open3d as o3d


class BaseDataset:
    """Base class for all dataset.
    It can parse the file name `file_indices` and return the data
    (image, pointcloud), where the result is torch.Tensor and can be
    fed to the network directly.
    """

    def __init__(
        self,
        file_indices: List[str],
        basePath: str,
        image_size: Tuple[int],
        voxel_shape: Tuple[int],
        augment: bool = True,
    ):
        """Initialize variables.
        Args:
            file_indices: List[str], contains file info for the index
            basePath: Path to the dataset, e.g. '~/Disk1/robotcar-dataset'
            image_size: The size of the desired input image, (HxW)
            voxel_shape: The shape of the total voxel grid, (X,Y,Z)
            augment: decide whether to augment the data
        Returns: None
        """
        self.file_indices = file_indices
        self.basePath = Path(basePath)
        self.voxel_shape = np.array(voxel_shape, np.int32)
        self.augment = augment

        # transform operation
        if augment:
            self.image_jitter = transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2
            )
            self.pc_sigma = 0.02
            self.pc_clip = 0.05
        self.image_normalizer = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def data_augment(self, image, pc):
        """Data augment for both image and point cloud.
           For image: change of brightness, contrast, saturation
           For point cloud: jitter point position
        Args & Returns:
            image: Image data, PIL image
            pc: Point cloud data, numpy.ndarray
        """
        image = self.image_jitter(image)
        jitter_cloud = np.clip(
            self.pc_sigma * np.random.randn(*pc.shape).astype(np.float32),
            -1.0 * self.pc_clip,
            self.pc_clip,
        )  # jitter point cloud
        return image, jitter_cloud + pc

    def read_data(self, ind: int):
        """Read raw data, perform augmentation (if required), and return the
        standard torch.Tensor that can be fed to network directly.
           **Interface. Subclass must implement this method.**
        Args:
            ind: Index of the data tuple in the file list.
        Returns:
            image: torch.Tensor, the required image
            pc: torch.Tensor, the required point cloud
        """
        raise NotImplementedError('read_data() is not implemented.')


class Robotcar(BaseDataset):
    """Oxford robotcar dataset. For more info see class `BaseDataset`"""

    def __init__(
        self,
        file_indices: List[str],
        basePath: str,
        image_size: Tuple[int],
        voxel_shape: Tuple[int],
        augment: bool = True,
    ):
        super(Robotcar, self).__init__(
            file_indices, basePath, image_size, voxel_shape, augment
        )

        self.PC_FOLDER = Path('pointcloud_20m_10overlap')
        self.IMG_FOLDER = Path('image_20m_10overlap')

        self.VOXEL_LOWER_VALUE = np.array((-0.8, -0.4, -0.2), np.float32)
        self.VOXEL_UPPER_VALUE = np.array((0.8, 0.4, 0.4), np.float32)  # (x,y,z)
        self.VOXEL_PER_VALUE = self.voxel_shape / (
            self.VOXEL_UPPER_VALUE - self.VOXEL_LOWER_VALUE
        )  # (x,y,z)

    def read_data(self, ind: int):
        """Read raw data, perform augmentation (if required), and return the
        standard torch.Tensor that can be fed to network directly.
           **@override from the base class**
        Args:
            ind: Index of the data tuple in the file list.
        Returns:
            image_tensor: torch.Tensor, the required image, [3xHxW] in [-1.0,1.0]
            pc_tensor: torch.Tensor, the required point cloud, [1xXxYxZ] in {0,1.0}
        """
        # parse file names
        folder, timestamp = self.file_indices[ind].split('/')
        folder = self.basePath / Path(folder)
        image_path = folder / self.IMG_FOLDER / Path(timestamp + '.png')
        pc_path = folder / self.PC_FOLDER / Path(timestamp + '.bin')

        # read data
        image_PIL = Image.open(image_path)
        pc_ndarray = np.fromfile(pc_path, dtype=np.float64).astype(np.float32)
        pc_ndarray.resize(pc_ndarray.shape[0] // 3, 3)  # (N,3)
        pc_ndarray = pc_ndarray[:, [1, 0, 2]]  # to xyz order
        pc_ndarray[:, [0, 1]] = -pc_ndarray[:, [0, 1]]

        # data augmentation, do NOT change the type
        if self.augment:
            image_PIL, pc_ndarray = self.data_augment(image_PIL, pc_ndarray)

        # prepare output
        image_tensor = self.image_normalizer(image_PIL)
        pc_tensor = self.__pc_array_to_voxel(pc_ndarray)
        return image_tensor, pc_tensor

    def __pc_array_to_voxel(self, pc_ndarray: np.ndarray):
        """Convert (N,3) np.ndarray point cloud to voxel grid with shape
        `self.voxel_shape`. Point boundary is determined `self.VOXEL_LOWER_POINT`
        and `self.VOXEL_UPPER_POINT`.
        Args:
            pc_ndarray: The input (N,3) np.ndarray point cloud
        Returns:
            pc_tensor: The output point cloud voxel of type torch.Tensor
        """
        voxel_index = (
            pc_ndarray - np.tile(self.VOXEL_LOWER_VALUE, (pc_ndarray.shape[0], 1))
        ) * self.VOXEL_PER_VALUE
        voxel_index = np.round(voxel_index).astype(np.int32)  # raw index

        # filter out out-of-boundary points
        valid_mask = (voxel_index >= np.array([0, 0, 0], np.int32)) & (
            voxel_index < self.voxel_shape
        )  # only True in (x,y,z) means valid index
        valid_mask = valid_mask[:, 0] & valid_mask[:, 1] & valid_mask[:, 2]
        voxel_index = voxel_index[valid_mask]  # valid voxel index(inside voxel range)

        # deal with voxel according to index
        pc_voxel = np.zeros(self.voxel_shape, np.float32)
        pc_voxel[voxel_index[:, 0], voxel_index[:, 1], voxel_index[:, 2]] = 1.0
        pc_tensor = torch.unsqueeze(torch.from_numpy(pc_voxel), 0)  # insert channel 1
        return pc_tensor


class NCLT(BaseDataset):
    """The North Campus Long Term (NCLT) dataset. For more info see class `BaseDataset`
    Website: http://robots.engin.umich.edu/nclt/"""

    def __init__(
        self,
        file_indices: List[str],
        basePath: str,
        image_size: Tuple[int],
        voxel_shape: Tuple[int],
        augment: bool = True,
    ):
        super(NCLT, self).__init__(
            file_indices, basePath, image_size, voxel_shape, augment
        )

        self.PC_FOLDER = Path('pointclouds')
        self.IMG_FOLDER = Path('images')

        self.VOXEL_LOWER_VALUE = np.array((-20.0, -20.0, 0.0), np.float32)
        self.VOXEL_UPPER_VALUE = np.array((20.0, 20.0, 5.0), np.float32)  # (x,y,z)
        self.VOXEL_PER_VALUE = self.voxel_shape / (
            self.VOXEL_UPPER_VALUE - self.VOXEL_LOWER_VALUE
        )  # (x,y,z)

    def read_data(self, ind: int):
        """Read raw data, perform augmentation (if required), and return the
        standard torch.Tensor that can be fed to network directly.
           **@override from the base class**
        Args:
            ind: Index of the data tuple in the file list.
        Returns:
            image_tensor: torch.Tensor, the required image, [3xHxW] in [-1.0,1.0]
            pc_tensor: torch.Tensor, the required point cloud, [1xXxYxZ] in {0,1.0}
        """
        # parse file names
        folder, timestamp = self.file_indices[ind].split('/')
        folder = self.basePath / Path(folder)
        image_path = folder / self.IMG_FOLDER / Path(timestamp + '.jpg')
        pc_path = folder / self.PC_FOLDER / Path(timestamp + '.bin')

        # read data
        image_PIL = Image.open(image_path)
        pc_ndarray = np.fromfile(pc_path, dtype=np.float32)
        pc_ndarray.resize(pc_ndarray.shape[0] // 3, 3)  # (N,3)

        # data augmentation, do NOT change the type
        if self.augment:
            image_PIL, pc_ndarray = self.data_augment(image_PIL, pc_ndarray)

        # prepare output
        image_tensor = self.image_normalizer(image_PIL)
        pc_tensor = self.__pc_array_to_voxel(pc_ndarray)
        return image_tensor, pc_tensor

    def __pc_array_to_voxel(self, pc_ndarray: np.ndarray):
        """Convert (N,3) np.ndarray point cloud to voxel grid with shape
        `self.voxel_shape`. Point boundary is determined `self.VOXEL_LOWER_POINT`
        and `self.VOXEL_UPPER_POINT`.
        Args:
            pc_ndarray: The input (N,3) np.ndarray point cloud
        Returns:
            pc_tensor: The output point cloud voxel of type torch.Tensor
        """
        voxel_index = (
            pc_ndarray - np.tile(self.VOXEL_LOWER_VALUE, (pc_ndarray.shape[0], 1))
        ) * self.VOXEL_PER_VALUE
        voxel_index = np.round(voxel_index).astype(np.int32)  # raw index

        # filter out out-of-boundary points
        valid_mask = (voxel_index >= np.array([0, 0, 0], np.int32)) & (
            voxel_index < self.voxel_shape
        )  # only True in (x,y,z) means valid index
        valid_mask = valid_mask[:, 0] & valid_mask[:, 1] & valid_mask[:, 2]
        voxel_index = voxel_index[valid_mask]  # valid voxel index(inside voxel range)

        # deal with voxel according to index
        pc_voxel = np.zeros(self.voxel_shape, np.float32)
        pc_voxel[voxel_index[:, 0], voxel_index[:, 1], voxel_index[:, 2]] = 1.0
        pc_tensor = torch.unsqueeze(torch.from_numpy(pc_voxel), 0)  # insert channel 1
        return pc_tensor


class SubT(BaseDataset):
    """CMU SubT dataset. For more info see class `BaseDataset`"""

    def __init__(self):
        raise NotImplementedError('Class `SubT` is not implemented yet.')


class Pittsburgh(BaseDataset):
    """The Pittsburgh large-scale city dataset, for map merging system.
    Only has point cloud data.
    """

    def __init__(
        self,
        file_indices: List[str],
        basePath: str,
        image_size: Tuple[int],
        voxel_shape: Tuple[int],
        augment: bool = False,
    ):
        super(Pittsburgh, self).__init__(
            file_indices, basePath, image_size, voxel_shape, augment
        )

        self.VOXEL_LOWER_VALUE = np.array((-40.0, -40.0, -1.5), np.float32)
        self.VOXEL_UPPER_VALUE = np.array((40.0, 40.0, 7.0), np.float32)  # (x,y,z)
        self.VOXEL_PER_VALUE = self.voxel_shape / (
            self.VOXEL_UPPER_VALUE - self.VOXEL_LOWER_VALUE
        )  # (x,y,z)

    def read_data(self, ind: int):
        """Read raw data, perform augmentation (if required), and return the
        standard torch.Tensor that can be fed to network directly.
           **@override from the base class**
        Args:
            ind: Index of the data tuple in the file list.
        Returns:
            image_tensor: torch.Tensor, the required image, [3xHxW] in [-1.0,1.0]
            pc_tensor: torch.Tensor, the required point cloud, [1xXxYxZ] in {0,1.0}
        """
        # parse file names
        pc_path = Path(self.basePath) / Path(self.file_indices[ind])

        # read data
        pc_o3d = o3d.io.read_point_cloud(str(pc_path))
        pc_ndarray = np.asarray(pc_o3d.points, dtype=np.float32)  # (N,3)

        # prepare output
        image_tensor = torch.zeros((3, 2, 2), dtype=torch.float32)
        pc_tensor = self.__pc_array_to_voxel(pc_ndarray)
        return image_tensor, pc_tensor

    def __pc_array_to_voxel(self, pc_ndarray: np.ndarray):
        """Convert (N,3) np.ndarray point cloud to voxel grid with shape
        `self.voxel_shape`. Point boundary is determined `self.VOXEL_LOWER_POINT`
        and `self.VOXEL_UPPER_POINT`.
        Args:
            pc_ndarray: The input (N,3) np.ndarray point cloud
        Returns:
            pc_tensor: The output point cloud voxel of type torch.Tensor
        """
        voxel_index = (
            pc_ndarray - np.tile(self.VOXEL_LOWER_VALUE, (pc_ndarray.shape[0], 1))
        ) * self.VOXEL_PER_VALUE
        voxel_index = np.round(voxel_index).astype(np.int32)  # raw index

        # filter out out-of-boundary points
        valid_mask = (voxel_index >= np.array([0, 0, 0], np.int32)) & (
            voxel_index < self.voxel_shape
        )  # only True in (x,y,z) means valid index
        valid_mask = valid_mask[:, 0] & valid_mask[:, 1] & valid_mask[:, 2]
        voxel_index = voxel_index[valid_mask]  # valid voxel index(inside voxel range)

        # deal with voxel according to index
        pc_voxel = np.zeros(self.voxel_shape, np.float32)
        pc_voxel[voxel_index[:, 0], voxel_index[:, 1], voxel_index[:, 2]] = 1.0
        pc_tensor = torch.unsqueeze(torch.from_numpy(pc_voxel), 0)  # insert channel 1
        return pc_tensor

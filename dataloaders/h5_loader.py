import h5py
from pathlib import Path

import numpy as np
import torch
from torch.utils import data



class TetrominoesDataset(data.Dataset):
    """Represents an abstract HDF5 dataset.

    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        recursive: If True, searches for h5 files in subdirectories.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).
    """

    def __init__(self, file_path, recursive, load_data, data_cache_size=3, transform=None):
        super().__init__()
        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size
        self.transform = transform

        # Search for all h5 files
        p = Path(file_path)
        assert (p.is_file())
        self.file_path=p
        self._load_data(p)

    def __getitem__(self, index):
        # get data
        x = self.dataset["imgs"][index]
        if self.transform:
            x = self.transform(x.permute(2,0,1)).permute(1,2,0)

        # get label
        y = self.dataset["masks"][index]
        return {"image": x, "mask": y}

    def __len__(self):
        return self.len

    def _load_data(self, file_path):
        """Load data to the cache given the file
        path and update the cache index in the
        data_info structure.
        """
        self.dataset = {}
        self.len = 0
        with h5py.File(file_path) as h5_file:
            for gname, group in h5_file.items():
                for dname, ds in group.items():
                    # add data to the data cache and retrieve
                    # the cache index
                    self.dataset[dname] = torch.from_numpy(np.array(ds))
                    self.len = ds.shape[0]


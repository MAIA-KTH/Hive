import os
import sys
from typing import Callable, Sequence, Union, List, Optional, Dict

import numpy as np
from monai.data import CacheDataset, load_decathlon_datalist, load_decathlon_properties
from monai.transforms import Randomizable
from monai.utils import ensure_tuple


class LungLobeDataset(Randomizable, CacheDataset):
    def __init__(
            self,
            dataset_dir: str,
            section: str,
            transform: Union[Sequence[Callable], Callable] = (),
            seed: int = 0,
            val_frac: float = 0.2,
            cache_num: int = sys.maxsize,
            cache_rate: float = 1.0,
            num_workers: int = None,
    ) -> None:
        if not os.path.isdir(dataset_dir):
            raise ValueError("Dataset directory dataset_dir must be a directory.")
        self.section = section
        self.val_frac = val_frac
        self.set_random_state(seed=seed)

        if not os.path.exists(dataset_dir):
            raise RuntimeError(
                f"Cannot find dataset directory: {dataset_dir}, please use download=True to download it.")
        self.indices: np.ndarray = np.array([])
        data = self._generate_data_list(dataset_dir)
        # as `release` key has typo in Task04 config file, ignore it.
        property_keys = [
            "name",
            "description",
            "reference",
            "licence",
            "tensorImageSize",
            "modality",
            "labels",
            "numTraining",
            "numTest",
        ]
        self._properties = load_decathlon_properties(os.path.join(dataset_dir, "dataset.json"), property_keys)

        CacheDataset.__init__(self, data, transform, cache_num=cache_num, cache_rate=cache_rate,
                              num_workers=num_workers)

    def get_indices(self) -> np.ndarray:
        """
        Get the indices of datalist used in this dataset.
        """
        return self.indices

    def randomize(self, data: List[int]) -> None:
        self.R.shuffle(data)

    def get_properties(self, keys: Optional[Union[Sequence[str], str]] = None):
        """
        Get the loaded properties of dataset with specified keys.
        If no keys specified, return all the loaded properties.
        """
        if keys is None:
            return self._properties
        if self._properties is not None:
            return {key: self._properties[key] for key in ensure_tuple(keys)}
        return {}

    def _generate_data_list(self, dataset_dir: str) -> List[Dict]:
        section = "training" if self.section in ["training", "validation"] else "test"
        datalist = load_decathlon_datalist(os.path.join(dataset_dir, "dataset.json"), True, section)
        return self._split_datalist(datalist)

    def _split_datalist(self, datalist: List[Dict]) -> List[Dict]:
        if self.section == "test":
            return datalist
        length = len(datalist)
        indices = np.arange(length)
        self.randomize(indices)

        val_length = int(length * self.val_frac)
        if self.section == "training":
            self.indices = indices[val_length:]
        else:
            self.indices = indices[:val_length]

        return [datalist[i] for i in self.indices]

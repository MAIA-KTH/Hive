import json
import os
import sys
from pathlib import Path
from typing import Callable, Sequence, Union, List, Optional, Dict

import numpy as np
from monai.data import CacheDataset, load_decathlon_datalist, load_decathlon_properties
from monai.data import select_cross_validation_folds
from monai.transforms import Randomizable
from monai.utils import ensure_tuple
from sklearn.model_selection import KFold


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
            raise RuntimeError(f"Cannot find dataset directory: {dataset_dir}, please use download=True to download it.")
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

        CacheDataset.__init__(self, data, transform, cache_num=cache_num, cache_rate=cache_rate, num_workers=num_workers)

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
        with open(os.path.join(dataset_dir, "dataset.json"), "r") as dataset_file:
            dataset_dict = json.load(dataset_file)

        n_modalities = len(dataset_dict["modality"])
        for data in datalist:
            if n_modalities == 1:
                data["image"] = data["image"][: -len(".nii.gz")] + "_0000.nii.gz"
            else:
                for modality_idx in range(n_modalities):
                    data["image_{}".format(modality_idx)] = data["image"][: -len(".nii.gz")] + "_{0:04d}.nii.gz".format(
                        modality_idx
                    )
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


def partition_dataset(datalist: List[Dict], num_partitions: int, shuffle: bool, seed: int) -> List[List[Dict]]:
    """
    Given a dataset in the format ´´List->{"key_0":filename_0,"key_n", filename_n}´´, returns the partitioned dataset
    according to the given number of partitions and the seed.

    Parameters
    ----------
    datalist : List[Dict]
        Dataset list to be partitioned.
    num_partitions : int
        Number of partitions to split the dataset into.
    shuffle : bool
        Shuffle or not dataset before splitting.
    seed : int
        Seed value to use in the shuffling.

    Returns
    -------
    List[List[Dict]]
        List of partition for the input dictionary.
    """
    data_id = []
    key = list(datalist[0].keys())[0]
    for data in datalist:
        data_id.append(Path(data[key]).name)

    data_id = np.sort(data_id)
    kfold = KFold(n_splits=num_partitions, shuffle=shuffle, random_state=seed)

    partition_list = []
    for train_idx, test_idx in kfold.split(range(len(data_id))):
        partition = []
        for idx in test_idx:
            partition.append(datalist[idx])
        partition_list.append(partition)

    return partition_list


class CrossValidationDataset:
    """
    Cross validation dataset based on the general dataset which must have `_split_datalist` API.

    Args:
        dataset_cls: dataset class to be used to create the cross validation partitions.
            It must have `_split_datalist` API.
        nfolds: number of folds to split the data for cross validation.
        seed: random seed to randomly shuffle the datalist before splitting into N folds, default is 0.
        dataset_params: other additional parameters for the dataset_cls base class.

    Example of 5 folds cross validation training::

        cvdataset = CrossValidation(
            dataset_cls=DecathlonDataset,
            nfolds=5,
            seed=12345,
            root_dir="./",
            task="Task09_Spleen",
            section="training",
            download=True,
        )
        dataset_fold0_train = cvdataset.get_dataset(folds=[1, 2, 3, 4])
        dataset_fold0_val = cvdataset.get_dataset(folds=0)
        # execute training for fold 0 ...

        dataset_fold1_train = cvdataset.get_dataset(folds=[1])
        dataset_fold1_val = cvdataset.get_dataset(folds=[0, 2, 3, 4])
        # execute training for fold 1 ...

        ...

        dataset_fold4_train = ...
        # execute training for fold 4 ...

    """

    def __init__(
        self,
        dataset_cls: Callable,
        nfolds: int = 5,
        seed: int = 0,
        **dataset_params,
    ) -> None:
        if not hasattr(dataset_cls, "_split_datalist"):
            raise ValueError("dataset class must have _split_datalist API.")
        self.dataset_cls = dataset_cls
        self.nfolds = nfolds
        self.seed = seed
        self.dataset_params = dataset_params

    def get_dataset(self, folds: Union[Sequence[int], int]):
        """
        Generate dataset based on the specified fold indice in the cross validation group.

        Args:
            folds: index of folds for training or validation, if a list of values, concatenate the data.

        """
        nfolds = self.nfolds
        seed = self.seed

        class _NsplitsDataset(self.dataset_cls):  # type: ignore
            def _split_datalist(self, datalist: List[Dict]) -> List[Dict]:
                data = partition_dataset(datalist=datalist, num_partitions=nfolds, shuffle=True, seed=seed)
                return select_cross_validation_folds(partitions=data, folds=folds)

        return _NsplitsDataset(**self.dataset_params)

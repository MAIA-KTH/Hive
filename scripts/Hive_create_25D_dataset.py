#!/usr/bin/env python

import datetime
import json
import os
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from textwrap import dedent

from Hive.monai.apps.datasets import LungLobeDataset
from Hive.monai.transforms import OrientToRAId, Save2DSlicesd
from Hive.utils.log_utils import add_verbosity_options_to_argparser, get_logger, log_lvl_from_verbosity_args
from monai.apps import CrossValidation
from monai.transforms import LoadImaged, Compose
from tqdm import tqdm

TIMESTAMP = "{:%Y-%m-%d_%H-%M-%S}".format(datetime.datetime.now())

DESC = dedent(
    """
    Preprocess a dataset for a 2.5D experiment. A 3D dataset is sliced along the specified directions ( [``axial``,
    ``coronal``, ``sagittal``] ) and the set of 2D slices are stored as PNG, Numpy (*.npy*) or compressed Numpy
    (*.npz*) files. For each fold and each orientation, a JSON configuration file is also saved, which can be used in later
    training stages in the Monai environment.
    """  # noqa: E501
)
EPILOG = dedent(
    """
    Example call:
    ::
        {filename} --config-file LungLobeSeg_2.5D_config.json
        {filename} --config-file LungLobeSeg_2.5D_config.json --n-workers 4
        {filename} --config-file LungLobeSeg_2.5D_config.json --n-cache 10
    """.format(  # noqa: E501
        filename=Path(__file__).name
    )
)


def get_arg_parser():
    pars = ArgumentParser(description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter)

    pars.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Configuration JSON file with experiment and dataset parameters ",
    )

    pars.add_argument(
        "--n-workers",
        type=int,
        required=False,
        default=10,
        help="Number of worker threads to use. (Default: 10)",
    )

    pars.add_argument(
        "--n-cache",
        type=int,
        required=False,
        default=30,
        help="Number of items to be cached. (Default: 30)",
    )

    add_verbosity_options_to_argparser(pars)

    return pars


def main():
    parser = get_arg_parser()
    arguments = vars(parser.parse_args())

    logger = get_logger(  # NOQA: F841
        name=Path(__file__).name,
        level=log_lvl_from_verbosity_args(arguments),
    )

    with open(arguments["config_file"]) as json_file:
        config_dict = json.load(json_file)

    preprocessing_path = config_dict["preprocessing_folder"]
    orientations = list(config_dict["slice_size_2d"].keys())
    n_fold = config_dict["n_folds"]
    slice_2d_extension = config_dict["save_2D_slices_as"]
    file_extension = config_dict["FileExtension"]
    random_seed = config_dict["Seed"]
    n_workers = arguments["n_workers"]
    n_cache = arguments["n_cache"]
    dataset_folder = Path(config_dict["base_folder"]).joinpath(
        "Task" + config_dict["Task_ID"] + "_" + config_dict["Task_Name"])
    transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            OrientToRAId(keys=["image", "label"]),
            Save2DSlicesd(
                keys=["image", "label"],
                output_folder=preprocessing_path,
                file_extension=file_extension,
                slicing_axes=orientations,
                slices_2d_filetype=slice_2d_extension,
            ),
        ]
    )

    cv_dataset = CrossValidation(
        dataset_cls=LungLobeDataset,
        nfolds=n_fold,
        seed=random_seed,
        dataset_dir=dataset_folder,
        section="training",
        transform=transforms,
        num_workers=n_workers,
        cache_num=n_cache,
    )

    for fold in tqdm(range(n_fold), desc="Preprocessing Folds"):

        training_folds = list(range(n_fold))
        training_folds.remove(fold)
        dataset_fold_train = cv_dataset.get_dataset(folds=training_folds)
        dataset_fold_val = cv_dataset.get_dataset(folds=fold)

        section_dict = {
            "validation": dataset_fold_val,
            "training": dataset_fold_train,
        }

        for section in section_dict:
            dataset = section_dict[section]
            for orientation in orientations:
                filename_dict = [
                    {**image_filename, **label_filename}
                    for data in dataset
                    for image_filename, label_filename in zip(
                        data["image_meta_dict"]["filenames_{}".format(orientation)],
                        data["label_meta_dict"]["filenames_{}".format(orientation)],
                    )
                ]
                config_dict = {
                    "Fold": fold,
                    "orientation": orientation,
                    "{}_2D".format(section): filename_dict,
                    "num{}_2D".format(section.capitalize()): len(filename_dict),
                    "{}_3D".format(section): dataset.data,
                    "num{}_3D".format(section.capitalize()): len(dataset.data),
                }

                config_filepath = os.path.join(
                    preprocessing_path, orientation, "data",
                    "dataset_fold_{}_{}_{}.json".format(fold, section, orientation)
                )

                json_file = open(config_filepath, "w")
                json.dump(config_dict, json_file)


if __name__ == "__main__":
    main()

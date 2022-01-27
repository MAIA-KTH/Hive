#!/usr/bin/env python

import datetime
import json
from argparse import ArgumentParser, RawTextHelpFormatter
from multiprocessing import Pool
from pathlib import Path
from textwrap import dedent

from monai.transforms import LoadImaged, Compose
from tqdm import tqdm

from Hive.monai.apps.datasets import LungLobeDataset, CrossValidationDataset
from Hive.monai.transforms import OrientToRAId, Save2DSlicesd
from Hive.utils.file_utils import generate_dataset_json
from Hive.utils.log_utils import add_verbosity_options_to_argparser, get_logger, log_lvl_from_verbosity_args

TIMESTAMP = "{:%Y-%m-%d_%H-%M-%S}".format(datetime.datetime.now())

DESC = dedent(
    """
    Preprocess a dataset for a 2.5D experiment. A 3D dataset is sliced along the specified directions ( [``axial``,
    ``coronal``, ``sagittal``] ) and the set of 2D slices are stored as PNG,NIFTI (*.nii.gz*), Numpy (*.npy*) or compressed Numpy
    (*.npz*) files. For each fold and each orientation, a JSON dataset summary file is also saved, which can be used in later
    training stages.
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

    orientations = config_dict["orientations"]
    n_fold = config_dict["n_folds"]
    slice_2d_extension = config_dict["save_2D_slices_as"]
    file_extension = config_dict["FileExtension"]
    random_seed = 12345  # config_dict["Seed"]
    n_workers = arguments["n_workers"]
    n_cache = arguments["n_cache"]
    dataset_folder = Path(config_dict["base_folder"]).joinpath(
        "nnUNet_raw_data", "Task" + config_dict["Task_ID"] + "_" + config_dict["Task_Name"]
    )
    transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            OrientToRAId(keys=["image", "label"]),
            Save2DSlicesd(
                keys=["image", "label"],
                output_folder=str(dataset_folder),
                rescale_to_png={"image": True, "label": True},
                file_extension=file_extension,
                slicing_axes=orientations,
                slices_2d_filetype=slice_2d_extension,
            ),
        ]
    )

    cv_dataset = CrossValidationDataset(
        dataset_cls=LungLobeDataset,
        nfolds=n_fold,
        seed=random_seed,
        dataset_dir=dataset_folder,
        section="training",
        num_workers=n_workers,
        cache_num=n_cache,
    )

    training_folds = list(range(n_fold))

    dataset = cv_dataset.get_dataset(folds=training_folds)
    pool = Pool(n_workers)
    transformed_data = []
    for data in dataset:
        transformed_data.append(
            (data),
        )

    res = pool.starmap_async(transforms, transformed_data)
    pool.close()

    [res.get() for _ in tqdm(transformed_data, desc="2D Slicing")]
    pool.join()

    for orientation in orientations:
        generate_dataset_json(
            str(Path(dataset_folder).joinpath(orientation, "dataset.json")),
            str(Path(dataset_folder).joinpath(orientation, "imagesTr")),
            None,
            list(config_dict["Modalities"].values()),
            config_dict["label_dict"],
            config_dict["DatasetName"] + "_2D_{}".format(orientation),
        )


if __name__ == "__main__":
    main()

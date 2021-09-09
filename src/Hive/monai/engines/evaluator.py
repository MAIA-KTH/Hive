import importlib
import logging
import os
from os import PathLike
from pathlib import Path
from typing import Union, Dict, Any, Callable, List

import monai
import torch
from Hive.monai import ORIENTATION_MAP
from Hive.monai.data import HiveNiftiSaver
from Hive.monai.transforms import ToListMemorySaver
from Hive.utils.file_utils import subfiles
from ignite.engine import Engine, Events
from monai.data import decollate_batch, list_data_collate
from monai.handlers import CheckpointLoader
from monai.inferers import SimpleInferer
from monai.transforms import (
    Compose,
    LoadImaged,
    AsDiscrete,
    AsChannelFirstD,
    ToTensord,
    VoteEnsemble,
    MeanEnsemble,
    KeepLargestConnectedComponent,
)
from monai.transforms import Transform
from tqdm import tqdm


class Hive2DTo3DEvaluator(Engine):
    """
    Hive Engine for evaluation. The engine receives network configurations and parameters, experiment settings
    and automatically creates an evaluator used for inferring 2D networks to create 3D volumes.
    The 2D inference is performed N times, according to the given orientations. For each run, the corresponding saved model
    is restored: **model path** can be either a folder path ( with the following structure: ``model_path/ORIENTATION/checkpoints/checkpoint.pt``)
    or a file path directly pointing to the **.pt** file to load. If **model_path** is nor set, it is expected to be found
    on the parent folder path of **results_folder**.
    Optionally it is possible to run a post_processing step and save the resulting volume.
    The different orientation predictions can be combined averaging them or taking the maximum predicted value.
    The prediction file is saved in ``result_folder/volume_id/volume_id.nii.gz``, the optional logits output in
     ``result_folder/volume_id/volume_id_logits_suffix.nii.gz`` and the post_processing output in
     ``result_folder/volume_id/volume_id_post_processing_suffix.nii.gz``
    """  # noqa: E501

    def __init__(
        self,
        result_folder: Union[str, PathLike],
        config_dict: Dict[str, Any],
        pre_processing_3D_transform: Callable[[str, Dict], Transform],
        post_processing_2D_transform: Callable[[str, Dict], Transform],
        model_path: Union[str, PathLike] = None,
    ):
        """
        Initialize Engine for 2D to 3D evaluation.

        Parameters
        ----------
        result_folder : Union[str, PathLike]
            Folder path where the predictions are saved.
        config_dict : Dict[str, Any]
            Configuration settings for the 3D Evaluator Engines.
        pre_processing_3D_transform : Callable[[str, Dict], Transform]
            Callable which expects orientation string and configuration dictionary as input and returns the Transform
            to be used in the 3D pre processing step.
        post_processing_2D_transform : Callable[[str, Dict], Transform]
            Callable which expects orientation string and configuration dictionary as input and returns the Transform
            to be used in the 2D post processing step.
        model_path : Union[str, PathLike]
            Path where to find the trained models. Can be a folder path or a file path.
        """
        self.config_dict = config_dict
        self.orientations = list(self.config_dict["slice_size_2d"].keys())
        self.checkpoint_loader = None
        self.result_folder = result_folder
        self.pre_processing_3D_transform = pre_processing_3D_transform
        self.post_processing_2D_transform = post_processing_2D_transform

        if model_path is None:
            self.model_path = str(Path(self.result_folder).parent)
        else:
            self.model_path = model_path

        self.combine_orientations = "mean"
        if "combine_orientations" in config_dict:
            self.combine_orientations = config_dict["combine_orientations"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = self._load_net()

        self.prediction_saver = HiveNiftiSaver(output_dir=self.result_folder, output_postfix="")

        # SOFTMAX
        self.save_logits = False
        if "save_logits" in config_dict:
            self.save_logits = config_dict["save_logits"]
        self.logits_suffix = "softmax"
        if "logits_suffix" in config_dict:
            self.logits_suffix = config_dict["logits_suffix"]
        if self.save_logits:
            self.logits_saver = HiveNiftiSaver(output_dir=self.result_folder, output_postfix=self.logits_suffix)

        # POST-PROCESSING
        self.run_post_processing = True
        if "run_post_processing" in config_dict:
            self.run_post_processing = config_dict["run_post_processing"]
        self.save_post_processing = True
        if "save_post_processing" in config_dict:
            self.save_post_processing = config_dict["save_post_processing"]
        self.post_processing_suffix = "post"
        if "post_processing_suffix" in config_dict:
            self.post_processing_suffix = config_dict["post_processing_suffix"]

        if self.save_post_processing:
            self.post_saver = HiveNiftiSaver(output_dir=self.result_folder, output_postfix=self.post_processing_suffix)

        self.image_meta_dict = None
        self.inferer = SimpleInferer()

        super().__init__(self._process)

    def _load_net(self):
        if "class_import" in self.config_dict["net_config"]:
            import_class = importlib.import_module(self.config_dict["net_config"]["class_import"])
        else:
            import_class = monai.networks.nets

        if hasattr(import_class, self.config_dict["net_config"]["class_name"]):
            net = getattr(import_class, self.config_dict["net_config"]["class_name"])(
                **self.config_dict["net_config"]["class_params"]
            ).to(self.device)
            return net
        else:
            raise AttributeError("{} has no attribute {}".format(import_class, self.config_dict["net_config"]["class_name"]))

    def _process(self, engine, batch):
        batch = decollate_batch(batch)
        prediction_list = []
        for data in batch:

            logits_output = []

            for orientation in self.orientations:
                orientation_output = self._run_2d_process(orientation, data)
                logits_output.append(orientation_output)
            logits_output = torch.stack(logits_output)
            if self.combine_orientations == "mean":
                logits_output = torch.mean(logits_output, dim=0)
            elif self.combine_orientations == "max":
                logits_output = torch.max(logits_output, dim=0)
            else:
                raise ValueError(
                    "{} is not accepted as a combination of the orientation predictions".format(self.combine_orientations)
                )
            if self.save_logits:
                self.logits_saver.save(logits_output[0], self.image_meta_dict)

            if logits_output.shape[1] > 1:
                prediction = torch.argmax(logits_output, dim=1).type(torch.int8)
            else:
                prediction = (logits_output > 0.5).int()
                prediction = torch.squeeze(prediction, dim=1)

            self.prediction_saver.save(prediction, self.image_meta_dict)

            if self.run_post_processing:
                post_processing_output = self._post_processing(prediction)
                if self.save_post_processing:
                    self.post_saver.save(post_processing_output, self.image_meta_dict)
                prediction_list.append(post_processing_output)

            else:
                prediction_list.append(prediction)

        return prediction_list

    def _post_processing(self, prediction):
        foreground_labels = self.config_dict["label_dict"].copy()
        foreground_labels.pop("0", None)
        foreground_labels = [int(key) for key in foreground_labels]
        post_processing_transform = Compose(
            [
                AsDiscrete(to_onehot=True, n_classes=len(self.config_dict["label_dict"])),
                # ThresholdForegroundLabels(labels=foreground_labels, threshold_value=0.3),
                KeepLargestConnectedComponent(applied_labels=foreground_labels),
            ]
        )
        post_processing_output = post_processing_transform(prediction).type(torch.int8)
        post_processing_output = torch.argmax(post_processing_output, dim=0, keepdim=True)

        return post_processing_output

    def _run_2d_process(self, orientation, data):
        self._load_checkpoint(orientation)
        val_transforms = self.pre_processing_3D_transform(orientation, self.config_dict)
        val_ds = monai.data.Dataset(data=[data], transform=val_transforms)
        val_loader = monai.data.DataLoader(val_ds, batch_size=1, num_workers=1, collate_fn=list_data_collate)
        with torch.no_grad():
            for val_data in val_loader:
                val_data = decollate_batch(val_data)[0]

                val_images = val_data["image"].to("cpu")
                val_images = torch.transpose(val_images, 0, 1)

                n_workers = 0
                if "N_THREADS" in os.environ:
                    n_workers = os.environ["N_THREADS"]
                val_2d_ds = monai.data.ArrayDataset(img=val_images.numpy())
                val_2d_loader = monai.data.DataLoader(
                    val_2d_ds,
                    batch_size=self.config_dict["batch_size"],
                    num_workers=n_workers,
                    pin_memory=torch.cuda.is_available(),
                    collate_fn=list_data_collate,
                )
                output_2d_list = []
                post_trans = Compose(
                    [
                        self.post_processing_2D_transform(orientation, self.config_dict),
                        ToListMemorySaver(output_list=output_2d_list),
                    ]
                )

                for val_2d_data in tqdm(val_2d_loader, desc="{} inference".format(orientation.capitalize())):
                    val_2d_data = val_2d_data.to(self.device)
                    pred = self.inferer(inputs=val_2d_data, network=self.net)
                    _ = [post_trans(i) for i in decollate_batch(pred)]

                logits_output = torch.permute(torch.stack(output_2d_list), (1, 2, 0, 3, 4))
                orientation_index = ORIENTATION_MAP[orientation]
                axis_index = val_data["image_meta_dict"]["axis_orientation"].index(orientation_index)
                logits_output = torch.transpose(logits_output, 2, axis_index + 2)

                logits_output = torch.nn.functional.interpolate(
                    logits_output, size=tuple(val_data["image_meta_dict"]["spatial_shape"].tolist()), mode="trilinear"
                )

                self.image_meta_dict = val_data["image_meta_dict"]
                return logits_output

    def _load_checkpoint(self, orientation):
        if self.checkpoint_loader is not None:
            self.remove_event_handler(self.checkpoint_loader, Events.STARTED)

        if Path(self.model_path).is_dir():
            checkpoint_files = subfiles(str(Path(self.model_path).joinpath(orientation, "checkpoints")), suffix=".pt")
        elif Path(self.model_path).is_file():
            checkpoint_files = [self.model_path]
        else:
            raise FileNotFoundError("Checkpoint file does not exist at {}".format(str(Path(self.model_path))))
        if len(checkpoint_files) > 0:
            self.checkpoint_loader = CheckpointLoader(load_path=checkpoint_files[0], load_dict={"net": self.net})
            self.checkpoint_loader.attach(self)
            self.fire_event(Events.STARTED)
        else:
            raise FileNotFoundError(
                "Checkpoint file does not exist at {}".format(str(Path(self.model_path).joinpath(orientation, "checkpoints")))
            )


class HiveEnsembled2Dto3DEvaluator(Hive2DTo3DEvaluator):
    """
    Hive Engine for ensembled evaluation. For each case, the evaluation ( with .. py:class::Hive2DTo3DEvaluator ) is performed
    M times, one for each trained model specified in **model_path_list**.

    The ensembled evaluation can be performed by averaging or by majority voting.

    """  # TODO document output file path specs

    def __init__(
        self,
        result_folder: Union[str, PathLike],
        config_dict: Dict[str, Any],
        pre_processing_3D_transform: Callable[[str, Dict], Transform],
        post_processing_2D_transform: Callable[[str, Dict], Transform],
        model_path_list: List[Union[str, PathLike]],
        mean_ensemble: bool = False,
    ):
        """
        Initialize Ensembled 2D to 3D Evaluator.

        Parameters
        ----------
        result_folder : Union[str, PathLike]
            Folder path where the predictions are saved.
        config_dict : Dict[str, Any]
            Configuration settings for the 3D Evaluator Engines.
        pre_processing_3D_transform : Callable[[str, Dict], Transform]
            Callable which expects orientation string and configuration dictionary as input and returns the Transform
            to be used in the 3D pre processing step.
        post_processing_2D_transform : Callable[[str, Dict], Transform]
            Callable which expects orientation string and configuration dictionary as input and returns the Transform
            to be used in the 2D post processing step.
        model_path_list : List[Union[str, PathLike]]
            List of paths where to find the trained models for the ensembled evaluation.
        mean_ensemble : bool
            Flag to set the ensemble method. If False, Majority Voting is performed instead.
        """
        super().__init__(result_folder, config_dict, pre_processing_3D_transform, post_processing_2D_transform)
        self.model_path_list = model_path_list
        self.model_path = model_path_list[0]
        self.mean_ensemble = mean_ensemble
        if self.mean_ensemble:
            self.run_post_processing = False
        self.save_logits = self.mean_ensemble

        logging.getLogger("validator").info("Running model ensembled evaluation with models : %s" % self.model_path_list)

    def _process(self, engine, batch):
        ensembled_batch_predictions = []
        for data in batch:
            if not self.mean_ensemble:
                ensebled_predictions = []
                ensembled_batch_predictions.append(ensebled_predictions)

        for idx, model_path in enumerate(self.model_path_list):
            self.model_path = model_path
            if self.save_post_processing:
                self.post_saver.set_output_postfix(self.post_processing_suffix + "_" + str(idx))
            if self.mean_ensemble:
                self.logits_saver.set_output_postfix(self.logits_suffix + "_" + str(idx))

            self.prediction_saver.set_output_postfix(str(idx))
            prediction_list = super()._process(engine, batch)
            if not self.mean_ensemble:
                for index, prediction in enumerate(prediction_list):
                    ensembled_batch_predictions[index].append(prediction)

        if self.mean_ensemble:
            self.average_ensemble(batch)
        else:
            self.majority_vote_ensemble(ensembled_batch_predictions)

    def majority_vote_ensemble(self, ensembled_batch_predictions):
        for ensebled_predictions in ensembled_batch_predictions:
            ens_pred = VoteEnsemble(num_classes=len(self.config_dict["label_dict"]))(ensebled_predictions)
            if self.run_post_processing:
                self.prediction_saver.set_output_postfix(self.post_processing_suffix)
            else:
                self.prediction_saver.set_output_postfix("")
            self.prediction_saver.save(ens_pred, self.image_meta_dict)

    def average_ensemble(self, batch):
        batch_data = decollate_batch(batch)
        for data in batch_data:
            image_id = Path(data["image"]).name[: -len(self.config_dict["FileExtension"])]
            logits_ensemble = []
            for index, _ in enumerate(self.model_path_list):
                file_post_fix = ""
                if self.logits_suffix is not None and self.logits_suffix != "":
                    file_post_fix += "_" + self.logits_suffix
                file_post_fix += "_{}".format(index)
                logits_data = {
                    "image": str(
                        Path(self.result_folder).joinpath(image_id, image_id + file_post_fix + self.config_dict["FileExtension"])
                    )
                }
                logits_data = Compose([LoadImaged(keys=["image"]), AsChannelFirstD(keys=["image"]), ToTensord(keys=["image"])])(
                    logits_data
                )
                self.image_meta_dict = logits_data["image_meta_dict"]
                logits_ensemble.append(torch.clone(logits_data["image"]))

            logits_ensemble = MeanEnsemble()(logits_ensemble)
            output_logits_path = str(
                Path(self.result_folder).joinpath(
                    image_id, image_id + "_" + self.logits_suffix + self.config_dict["FileExtension"]
                )
            )
            self.logits_saver.save_with_path(logits_ensemble, path=output_logits_path, meta_data=self.image_meta_dict)
            prediction = torch.argmax(logits_ensemble, dim=0, keepdim=True)
            prediction_path = str(Path(self.result_folder).joinpath(image_id, image_id + self.config_dict["FileExtension"]))
            self.prediction_saver.save_with_path(prediction, path=prediction_path, meta_data=self.image_meta_dict)

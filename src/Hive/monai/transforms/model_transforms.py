from Hive.monai.transforms import OrientToRAId
from monai.transforms import (
    Activations,
    AddChanneld,
    AddChannel,
    NormalizeIntensityd,
    Resized,
    ToTensord,
    Compose,
    LoadImaged,
)


def pre_processing_25D_transform(orientation, config_dict):
    spatial_size = [-1]
    [spatial_size.append(size) for size in config_dict["slice_size_2d"][orientation]]
    transform = Compose(
        [
            LoadImaged(keys=["image"]),
            OrientToRAId(keys=["image"], slicing_axes=orientation),
            AddChanneld(keys=["image"]),
            NormalizeIntensityd(
                keys=["image"], subtrahend=config_dict["norm_stats"]["mean"], divisor=config_dict["norm_stats"]["sd"]
            ),
            Resized(keys=["image"], spatial_size=spatial_size, mode="area"),
            ToTensord(keys=["image"]),
        ]
    )
    return transform


def post_processing_25D_transform(orientation, config_dict):
    transform = Compose([Activations(softmax=True), AddChannel()])
    return transform

from typing import Any

from monai.config import KeysCollection
from monai.transforms import MapTransform


class ThresholdForegroundLabelsd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`Hive.monai.transforms.ThresholdForegroundLabels`,
    """

    def __init__(self, keys: KeysCollection, labels, threshold_value=0.5, allow_missing_keys: bool = False) -> None:
        """

        Parameters
        ----------
        keys: keys of the corresponding items to be thresholded.
        labels: channels to be considered as foreground.
        threshold_value: threshold value for foreground classes.
        allow_missing_keys: don't raise exception if key is missing.
        """
        self.foreground_labels = labels
        self.threshold = threshold_value
        MapTransform.__init__(self, keys, allow_missing_keys)

    def __call__(
            self,
            data: Any,
    ):
        for key in self.keys:
            labels = list(range(data[key].shape[0]))
            background_labels = [label for label in labels if label not in self.foreground_labels]
            data[key][:, background_labels, ...] = (
                        data[key][:, background_labels, ...] >= (1 - self.threshold)).float()
            data[key][:, self.foreground_labels, ...] = (
                        data[key][:, self.foreground_labels, ...] >= self.threshold).float()

        return data

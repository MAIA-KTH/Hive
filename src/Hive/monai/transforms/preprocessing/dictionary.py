import cupy as cp
from cupyx.scipy.ndimage import binary_dilation, morphological_gradient
from monai.config import KeysCollection
from monai.transforms import MapTransform


class LungLobeMapToFissureMaskd(MapTransform):
    """
    Dictionary-based Transform to convert a Lung Lobe map into a Fissure binary mask.
    """

    def __init__(self, keys: KeysCollection, n_labels: int, dilation_iterations: int = 1, allow_missing_keys: bool = False):
        """

        Parameters
        ----------
        keys : KeysCollection
            keys of the corresponding Numpy array items in the dictionary to be converted.
        n_labels : int
            Number of labels in the Lung Lobe map.
        dilation_iterations : int
            Number of iteration to run the binary dilation after creating the gradient mask. Default: 1.
        allow_missing_keys : bool
            don't raise exception if key is missing.
        """
        self.n_labels = n_labels
        self.dilation_iterations = dilation_iterations
        super().__init__(keys, allow_missing_keys)

    def __call__(
        self,
        data,
    ):
        for key in self.keys:
            data[key][data[key] == 0] = -(self.n_labels + 1)
            label_gradient = morphological_gradient(cp.asarray(data[key]), size=(3, 3, 3))
            label_gradient = cp.where((label_gradient > 0) & (label_gradient < self.n_labels), 1, 0).astype(cp.uint8)
            dilated_mask = binary_dilation(label_gradient, iterations=self.dilation_iterations, brute_force=True)
            data[key] = cp.asnumpy(dilated_mask)
        return data

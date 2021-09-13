import math
from typing import List

import numpy as np
from nptyping import NDArray


def get_axis_order_to_RAI(rotation_matrix: NDArray[(3, 3), float]) -> (List[int], List[bool]):
    """
    Given a rotation matrix, returns the axis orientations, following the convention 0 = R ->L , 1 = A -> P,
     2 = I -> S . The returned flip map gives the indication whether to flip or not the single axes direction
     Example: [2, 0, 1], [False, True, False] corresponds to a I -> S, L -> R, A -> P (ILA) orientation.
    Parameters
    ----------
    rotation_matrix: 3x3 float Numpy array, representing the rotation matrix of a transform

    Returns
    -------
    List of int, containing the indexes of the axis orientations
    List of bool , indicating whether to flip or not the single axes
    """
    axis_orientation = list(np.argmax(np.abs(rotation_matrix), axis=1))

    flip_map = [False, False, False]

    oriented_rotation_matrix = []
    for idx in range(3):
        oriented_rotation_matrix.append(rotation_matrix[axis_orientation.index(idx)])

    axis = np.identity(3)
    for dim in range(3):
        projection = np.dot(axis[dim], oriented_rotation_matrix[dim])
        if projection < 0:
            flip_map[dim] = True

    flip_axes = []
    for dim in range(3):
        flip_axes.append(flip_map[axis_orientation[dim]])

    return axis_orientation, flip_axes


def get_quatern_a(quatern_b: float, quatern_c: float, quatern_d: float) -> NDArray[(1,), float]:
    """
    When a Quaternion representation is used for the rotation matrix in a NIFTI volume, returns Quaternion A ,
    given Quaternion B,C and D
    Parameters
    ----------
    quatern_b: float
    quatern_c: float
    quatern_d: float

    Returns
    -------
    Quaternion A
    """
    quatern_a = math.sqrt(1 - math.pow(quatern_b, 2) - math.pow(quatern_c, 2) - math.pow(quatern_d, 2))
    return np.array(quatern_a, dtype=np.float32)

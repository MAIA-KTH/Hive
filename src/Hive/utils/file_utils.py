import os
from os import PathLike
from pathlib import Path
from typing import Union, List

from Hive.utils.log_utils import get_logger

logger = get_logger(__name__)


def subfiles(folder: Union[str, PathLike], join: bool = True, prefix: str = None, suffix: str = None,
             sort: bool = True) -> List[str]:
    """
    Given a folder path, returns a list with all the files in the folder.

    Parameters
    ----------
    folder : Union[str, PathLike]
        Folder path.
    join : bool
        Flag to return the complete file paths or only the relative file names.
    prefix : str
        Filter the files with the specified prefix.
    suffix : str
        Filter the files with the specified suffix.
    sort : bool
        Flag to sort the files in the list by alphabetical order.

    Returns
    -------
    Filename list.
    """
    if join:
        l = os.path.join  # noqa: E741
    else:
        l = lambda x, y: y  # noqa: E741, E731
    res = [
        l(folder, i.name)
        for i in Path(folder).iterdir()
        if i.is_file() and (prefix is None or i.name.startswith(prefix)) and (suffix is None or i.name.endswith(suffix))
    ]
    if sort:
        res.sort()
    return res


def subfolders(folder: Union[str, PathLike], join: bool = True, sort: bool = True) -> List[str]:
    """
     Given a folder path, returns a list with all the subfolders in the folder.

    Parameters
    ----------
    folder : Union[str, PathLike]
        Folder path.
    join : bool
        Flag to return the complete folder paths or only the relative folder names.
    sort : bool
        Flag to sort the sub folders in the list by alphabetical order.

    Returns
    -------
    Sub folder list.
    """
    if join:
        l = os.path.join  # noqa: E741
    else:
        l = lambda x, y: y  # noqa: E741, E731
    res = [l(folder, i.name) for i in Path(folder).iterdir() if i.is_dir()]
    if sort:
        res.sort()
    return res

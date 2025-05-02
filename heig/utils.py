import gzip
import bz2
import logging
from functools import reduce
import numpy as np
from scipy.linalg import cho_solve, cho_factor


def GetLogger(logpath):
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(logpath, mode="w")
    # fh.setLevel(logging.INFO)
    log.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    log.addHandler(sh)

    return log


def sec_to_str(t):
    """Convert seconds to days:hours:minutes:seconds"""
    [d, h, m, s, n] = reduce(
        lambda ll, b: divmod(ll[0], b) + ll[1:], [(t, 1), 60, 60, 24]
    )
    f = ""
    if d > 0:
        f += "{D}d:".format(D=d)
    if h > 0:
        f += "{H}h:".format(H=h)
    if m > 0:
        f += "{M}m:".format(M=m)

    f += "{S}s".format(S=s)
    return f


def check_compression(dir):
    """
    Checking which compression should use

    Parameters:
    ------------
    dir: diretory to the dataset

    Returns:
    ---------
    openfunc: function to open the file
    compression: type of compression

    """
    if dir.endswith("gz") or dir.endswith("bgz"):
        compression = "gzip"
        openfunc = gzip.open
    elif dir.endswith("bz2"):
        compression = "bz2"
        openfunc = bz2.BZ2File
    elif (
        dir.endswith("zip")
        or dir.endswith("tar")
        or dir.endswith("tar.gz")
        or dir.endswith("tar.bz2")
    ):
        raise ValueError(
            "files with suffix .zip, .tar, .tar.gz, .tar.bz2 are not supported"
        )
    else:
        openfunc = open
        compression = None

    return openfunc, compression


def find_loc(num_list, target):
    """
    Finding the target number from a sorted list of numbers by binary search

    Parameters:
    ------------
    num_list: a sorted list of numbers
    target: the target number

    Returns:
    ---------
    the exact index or -1

    """
    l = 0
    r = len(num_list) - 1
    while l <= r:
        mid = (l + r) // 2
        if num_list[mid] == target:
            return mid
        elif num_list[mid] > target:
            r = mid - 1
        else:
            l = mid + 1
    return r


def inv(A):
    """
    Computing inverse for a symmetric and positive-definite matrix

    """
    dtype = A.dtype
    cho_factors = cho_factor(A)
    A_inv = cho_solve(cho_factors, np.eye(A.shape[0]))

    return A_inv.astype(dtype)

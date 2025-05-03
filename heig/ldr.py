import numpy as np
import pandas as pd
import heig.input.dataset as ds
from numba import njit, prange
from heig.image import ImageManager
from heig.utils import inv


def projection_ldr(ldr, covar):
    """
    Computing S'(I - M)S/n = S'S - S'X(X'X)^{-1}X'S/n,
    where I is the identity matrix,
    M = X(X'X)^{-1}X' is the project matrix for X,
    S is the LDR matrix.

    Parameters:
    ------------
    ldr (n, r): low-dimension representaion of imaging data
    covar (n, p): covariates, including the intercept

    Returns:
    ---------
    ldr_cov: variance-covariance matrix of LDRs

    """
    n = ldr.shape[0]
    inner_ldr = np.dot(ldr.T, ldr)
    inner_covar = np.dot(covar.T, covar)
    inner_covar_inv = inv(inner_covar)
    ldr_covar = np.dot(ldr.T, covar)
    part2 = np.dot(np.dot(ldr_covar, inner_covar_inv), ldr_covar.T)
    ldr_cov = (inner_ldr - part2) / n
    ldr_cov = ldr_cov.astype(np.float32)

    return ldr_cov


@njit
def dot(A, B):
    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    return np.dot(A, B)


@njit(parallel=True, fastmath=True)
def normalize_images(images_):
    n_samples, n_features = images_.shape
    means = np.zeros(n_features, dtype=images_.dtype)
    stds = np.zeros(n_features, dtype=images_.dtype)

    # Compute mean for each column (feature)
    for j in prange(n_features):
        for i in range(n_samples):
            means[j] += images_[i, j]
        means[j] /= n_samples

    # Compute std for each column
    for j in prange(n_features):
        for i in range(n_samples):
            diff = images_[i, j] - means[j]
            stds[j] += diff * diff
        stds[j] = np.sqrt(stds[j] / n_samples)

    # Normalize
    out = np.empty_like(images_)
    for i in prange(n_samples):
        for j in range(n_features):
            out[i, j] = (images_[i, j] - means[j]) / stds[j]

    return out


@njit
def image_recovery_quality(images, ldrs, bases):
    """
    Computing correlation between raw images and reconstructed images

    Parameters:
    ------------
    images: a np.array of normalized raw images (N, r)
    ldrs: a np.array of constructed LDRs (n, r)
    bases: a np.array of corresponding bases (N, r)

    Returns:
    ---------
    corr: a np.array of correlation coefficients between raw and reconstructed images

    """
    rec_images = dot(bases, ldrs.T)
    rec_images = normalize_images(rec_images)

    corr = np.zeros(images.shape[1], dtype=np.float32)
    for i in range(images.shape[1]):
        total = 0.0
        for j in range(images.shape[0]):
            total += images[j, i] * rec_images[j, i]
        corr[i] = total / images.shape[0]

    return corr


@njit
def construct_ldr_batch(
    images_, start_idx, end_idx, bases, alt_n_ldrs_list, rec_corr, ldrs
):
    """
    Construting LDRs in batch

    Parameters:
    ------------
    images_: a np.array of raw images (n1, N)
    start_idx: start index
    end_idx: end index
    bases: a np.array of bases (N, r)
    alt_n_ldrs_list: a np.array of alternative number of LDRs
    rec_corr: a np.array of reconstruction correlation
    ldrs: a np.array of LDRs (n1, r)

    """
    ldrs_ = dot(images_, bases)
    ldrs[start_idx:end_idx] = ldrs_
    images_ = images_.T
    images_ = normalize_images(images_)

    for i in range(len(alt_n_ldrs_list)):
        alt_n_ldrs = alt_n_ldrs_list[i]
        image_rec_corr = image_recovery_quality(
            images_, ldrs_[:, :alt_n_ldrs], bases[:, :alt_n_ldrs]
        )
        rec_corr[i][start_idx:end_idx] = image_rec_corr


def print_alt_corr(rec_corr, log):
    """
    Printing a table of reconstruction correlation
    using varying numbers of LDRs

    """
    max_key_len = max(len(str(key)) for key in rec_corr.keys())
    max_val_len = max(len(str(value)) for value in rec_corr.values())
    max_len = max([max_key_len, max_val_len])
    keys_str = "  ".join(f"{str(key):<{max_len}}" for key in rec_corr.keys())
    values_str = "  ".join(f"{str(value):<{max_len}}" for value in rec_corr.values())

    log.info(
        "Mean correlation between reconstructed images and raw images using varying numbers of LDRs:"
    )
    log.info(keys_str)
    log.info(values_str)

    max_corr = max(rec_corr.values())
    max_n_ldrs = max(rec_corr.keys())
    if max_corr < 0.85:
        log.info(
            (
                f"Using {max_n_ldrs} LDRs can achieve a correlation coefficient of {max_corr}, "
                "which might be too low, consider increasing LDRs.\n"
            )
        )


def check_input(args):
    # required arguments
    if args.image is None:
        raise ValueError("--image is required")
    if args.covar is None:
        raise ValueError("--covar is required")
    if args.bases is None:
        raise ValueError("--bases is required")


def run(args, log):
    check_input(args)

    # read bases and extract top n_ldrs
    bases = np.load(args.bases)
    n_voxels, n_bases = bases.shape
    log.info(f"{n_bases} bases of {n_voxels} voxels (vertices) read from {args.bases}")

    if args.n_ldrs is not None:
        if args.n_ldrs <= n_bases:
            n_ldrs = args.n_ldrs
            bases = bases[:, :n_ldrs]
        else:
            raise ValueError("the number of bases is less than --n-ldrs")
    else:
        n_ldrs = n_bases

    try:
        # read images
        images = ImageManager(args.image, args.voxels)
        if n_voxels != images.n_voxels:
            raise ValueError("the images and bases have different resolution")

        # read covariates
        log.info(f"Read covariates from {args.covar}")
        covar = ds.Covar(args.covar, args.cat_covar_list)

        # keep common subjects
        common_idxs = ds.get_common_idxs(images.ids, covar.data.index, args.keep)
        common_idxs = ds.remove_idxs(common_idxs, args.remove)
        images.keep_and_remove(common_idxs)
        log.info(f"{len(common_idxs)} common subjects in these files.")

        # contruct ldrs
        ldrs = np.zeros((len(common_idxs), n_ldrs), dtype=np.float32)
        start_idx, end_idx = 0, 0
        rec_corr = np.zeros((10, len(common_idxs)), dtype=np.float32)
        alt_n_ldrs_list = np.array([
            int(n_ldrs * prop)
            for prop in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)
        ])

        log.info(f"Constructing {n_ldrs} LDRs ...")
        for images_, _ in images.image_reader():
            start_idx = end_idx
            end_idx += images_.shape[0]
            construct_ldr_batch(
                images_,
                start_idx,
                end_idx,
                bases,
                alt_n_ldrs_list,
                rec_corr,
                ldrs,
            )

        rec_corr_dict = dict()
        for i in range(len(alt_n_ldrs_list)):
            rec_corr_dict[alt_n_ldrs_list[i]] = round(np.mean(rec_corr[i]), 2)

        print_alt_corr(rec_corr_dict, log)

        # process covar
        covar.keep_and_remove(common_idxs)
        covar.cat_covar_intercept()
        log.info(
            f"{covar.data.shape[1]} fixed effects in the covariates (including the intercept)."
        )

        # var-cov matrix of projected LDRs
        ldr_cov = projection_ldr(ldrs, np.array(covar.data))
        log.info(
            f"Removed covariate effects from LDRs and computed variance-covariance matrix.\n"
        )

        # save the output
        ldr_df = pd.DataFrame(ldrs, index=images.extracted_ids)
        ldr_df.to_csv(f"{args.out}_ldr_top{n_ldrs}.txt", sep="\t")
        np.save(f"{args.out}_ldr_cov_top{n_ldrs}.npy", ldr_cov)

        log.info(f"Saved the raw LDRs to {args.out}_ldr_top{n_ldrs}.txt")
        log.info(
            (
                f"Saved the variance-covariance matrix of covariate-effect-removed LDRs "
                f"to {args.out}_ldr_cov_top{n_ldrs}.npy"
            )
        )

    finally:
        if "images" in locals():
            images.close()

import h5py
import logging
import numpy as np
import pandas as pd
from scipy.linalg import cho_solve, cho_factor
import heig.input.dataset as ds


def check_input(args):
    # required arguments
    if args.bases is None:
        raise ValueError("--bases is required")
    if args.ldrs is None:
        raise ValueError("--ldrs is required")
    if args.covar is None:
        raise ValueError("--covar is required")


class NullModel:
    """
    Reading and processing null model

    """

    def __init__(self, file_path):
        with h5py.File(file_path, "r") as file:
            self.covar = file["covar"][:]
            self.resid_ldr = file["resid_ldr"][:]
            self.bases = file["bases"][:]
            ids = file["id"][:]

        self.n_voxels, self.n_ldrs = self.bases.shape
        self.n_subs = self.covar.shape[0]
        self.ids = pd.MultiIndex.from_arrays(ids.astype(str).T, names=["FID", "IID"])
        self.id_idxs = np.arange(self.n_subs)
        self.voxel_idxs = np.arange(self.n_voxels)
        self.logger = logging.getLogger(__name__)

    def select_ldrs(self, n_ldrs=None):
        if n_ldrs is not None:
            if n_ldrs <= self.n_ldrs:
                self.n_ldrs = n_ldrs
                self.resid_ldr = self.resid_ldr[:, :n_ldrs]
                self.bases = self.bases[:, :n_ldrs]
                self.logger.info(f"Keeping the top {n_ldrs} LDRs and bases.")
            else:
                raise ValueError("--n-ldrs is greater than #LDRs in null model")

    def select_voxels(self, voxel_idxs=None):
        if voxel_idxs is not None:
            if np.max(voxel_idxs) < self.n_voxels:
                self.voxel_idxs = voxel_idxs
                self.bases = self.bases[voxel_idxs]
                self.logger.info(f"{len(voxel_idxs)} voxel(s) included.")
            else:
                raise ValueError("--voxels index (one-based) out of range")

    def keep(self, keep_idvs):
        """
        Keep subjects
        this method will only be invoked after extracting common subjects

        Parameters:
        ------------
        keep_idvs: a list or pd.MultiIndex of subject ids to keep

        """
        if isinstance(keep_idvs, list):
            keep_idvs = pd.MultiIndex.from_arrays(
                [keep_idvs, keep_idvs], names=["FID", "IID"]
            )
        common_ids = ds.get_common_idxs(keep_idvs, self.ids).get_level_values("IID")
        ids_df = pd.DataFrame(
            {"id": self.id_idxs}, index=self.ids.get_level_values("IID")
        )
        ids_df = ids_df.loc[common_ids]
        id_idxs = ids_df["id"].values
        self.resid_ldr = self.resid_ldr[id_idxs]
        self.covar = self.covar[id_idxs]
        if self.resid_ldr.shape[0] == 0 or self.covar.shape[0] == 0:
            raise ValueError("no subject remaining in the null model")

    def remove_dependent_columns(self):
        """
        Removing dependent columns from covariate matrix

        """
        rank = np.linalg.matrix_rank(self.covar)
        if rank < self.covar.shape[1]:
            _, R = np.linalg.qr(self.covar)
            independent_columns = np.where(np.abs(np.diag(R)) > 1e-10)[0]
            self.covar = self.covar[:, independent_columns]
            if self.covar.shape[1] == 0:
                raise ValueError("no covariate remaining in the null model")


def fit_null_model(covar, ldrs):
    """
    Fitting a null model

    Parameters:
    ------------
    covar (n, p): a np.array of covariates (including the intercept)
    ldrs (n, r): a np.array of LDRs

    Returns:
    ---------
    resid_ldr (n, r): LDR residuals

    """
    inner_covar = np.dot(covar.T, covar)  # (X'X) (p, p)
    covar_ldrs = np.dot(covar.T, ldrs)  # X'\Xi (p, r)
    c, lower = cho_factor(inner_covar)
    beta = cho_solve((c, lower), covar_ldrs)
    y_pred = np.dot(covar, beta)
    resid_ldr = ldrs - y_pred

    return resid_ldr


def run(args, log):
    check_input(args)

    # read ldrs, bases, and inner_ldr
    log.info(f"Read LDRs from {args.ldrs}")
    ldrs = ds.Dataset(args.ldrs)
    log.info(f"{ldrs.data.shape[1]} LDRs and {ldrs.data.shape[0]} subjects.")
    bases = np.load(args.bases)
    log.info(f"{bases.shape[1]} bases read from {args.bases}")

    # read covar
    log.info(f"Read covariates from {args.covar}")
    covar = ds.Covar(args.covar, args.cat_covar_list)
    if args.extract_covar_list is not None:
        extract_covar_list = ds.parse_input(args.extract_covar_list)
        covar.extract_covar(extract_covar_list)
    if args.exclude_covar_list is not None:
        exclude_covar_list = ds.parse_input(args.exclude_covar_list)
        covar.exclude_covar(exclude_covar_list)

    common_idxs = ds.get_common_idxs(ldrs.data.index, covar.data.index, args.keep)
    common_idxs = ds.remove_idxs(common_idxs, args.remove)
    log.info(f"{len(common_idxs)} subjects common in these files.")

    # keep common subjects
    covar.keep_and_remove(common_idxs)
    covar.cat_covar_intercept()
    log.info(
        f"{covar.data.shape[1]} fixed effects in the covariates (including the intercept)."
    )
    ldrs.keep_and_remove(common_idxs)
    # ldrs.to_single_index()
    ids = ldrs.get_ids()

    # keep selected LDRs
    if args.n_ldrs is not None:
        n_ldrs = args.n_ldrs
    else:
        n_ldrs = np.min((ldrs.data.shape[1], bases.shape[1]))
    bases, _, _, ldrs.data = ds.keep_ldrs(n_ldrs, bases, resid_ldrs=ldrs.data)
    log.info(f"Keeping the top {n_ldrs} LDRs and bases.")
    covar.data = np.array(covar.data)
    ldrs.data = np.array(ldrs.data)

    # fit the null model
    resid_ldr = fit_null_model(covar.data, ldrs.data)

    with h5py.File(f"{args.out}_null_model.h5", "w") as file:
        file.create_dataset("covar", data=covar.data, dtype="float32")
        file.create_dataset("resid_ldr", data=resid_ldr, dtype="float32")
        file.create_dataset("bases", data=bases, dtype="float32")
        file.create_dataset("id", data=np.array(ids.tolist(), dtype="S10"))

    log.info(f"\nSaved the null model to {args.out}_null_model.h5")

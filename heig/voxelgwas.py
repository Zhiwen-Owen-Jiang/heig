import numpy as np
import threading
import concurrent.futures
from scipy.stats import chi2
from numba import njit
from tqdm import tqdm
from heig import sumstats
import heig.input.dataset as ds


@njit()
def compute_ztz_inv_batch(
    ldr_beta_batch, ldr_z_batch, ldr_var_batch, n, n_ldrs
):
    """
    Computing (Z'Z)^{-1} from summary statistics in batch

    """
    ldr_se_batch = ldr_beta_batch / ldr_z_batch
    ztz_inv_batch = np.sum(
        (ldr_se_batch * ldr_se_batch + ldr_beta_batch * ldr_beta_batch / n)
        / ldr_var_batch,
        axis=1,
    )
    return ztz_inv_batch / n_ldrs


@njit()
def recover_beta_batch(ldr_beta_batch, bases_batch):
    """
    Computing voxel beta in batch

    """
    ldr_beta_batch = np.ascontiguousarray(ldr_beta_batch)
    bases_batch = np.ascontiguousarray(bases_batch)
    voxel_beta_batch = np.dot(ldr_beta_batch, bases_batch)
    return voxel_beta_batch


@njit()
def recover_se_numba(voxel_idxs, voxel_beta, bases, ldr_cov, ztz_inv, n):
    base = bases[voxel_idxs]  # (q, r)
    if base.ndim == 1:
        base = base.reshape(-1, 1)
    base = np.ascontiguousarray(base)
    ldr_cov = np.ascontiguousarray(ldr_cov)
    part1 = np.sum(np.dot(base, ldr_cov) * base, axis=1)  # (q, )
    voxel_beta_squared = voxel_beta * voxel_beta
    voxel_beta_squared /= n
    voxel_se = part1 * ztz_inv
    voxel_se -= voxel_beta_squared
    voxel_se = np.sqrt(voxel_se)
    return voxel_se


class VGWAS:
    def __init__(self, bases, ldr_cov, ldr_gwas, snp_idxs, n, threads):
        """
        Parameters:
        ------------
        bases: a np.array of bases (N, r)
        ldr_cov: a np.array of variance-covariance matrix of LDRs (r, r)
        ldr_gwas: a GWAS instance
        snp_idxs: numerical indices of SNPs to extract (d, )
        n: sample sizes of SNPs (d, 1)
        threads: number of threads

        """
        self.bases = bases
        self.ldr_cov = ldr_cov
        self.ldr_gwas = ldr_gwas
        self.ldr_idxs = list(range(ldr_gwas.n_gwas))
        self.snp_idxs = snp_idxs
        self.n = n
        self.ztz_inv = self._compute_ztz_inv(threads)  # (d, 1)
        # self.ztz_inv = self._compute_ztz_inv_numba(threads)  # (d, 1)

    def _compute_ztz_inv_numba(self, threads):
        """
        Computing (Z'Z)^{-1} from summary statistics

        Parameters:
        ------------
        threads: number of threads

        Returns:
        ---------
        ztz_inv: a np.array of (Z'Z)^{-1} (d, 1)

        """
        ldr_var = np.diag(self.ldr_cov)
        n_ldrs = self.bases.shape[1]
        ztz_inv = np.zeros(np.sum(self.snp_idxs), dtype=np.float32)
        i = 0
        data_reader = self.ldr_gwas.data_reader(
            "both", self.ldr_idxs, self.snp_idxs, all_gwas=False
        )

        for ldr_beta_batch, ldr_z_batch in data_reader:
            batch_size = ldr_beta_batch.shape[1]
            ldr_var_batch = ldr_var[i : i + batch_size]
            ztz_inv_batch = compute_ztz_inv_batch(
                ldr_beta_batch, ldr_z_batch, ldr_var_batch, self.n, n_ldrs
            )
            i += batch_size
            ztz_inv += ztz_inv_batch
        ztz_inv = ztz_inv.reshape(-1, 1)

        return ztz_inv

    def _compute_ztz_inv(self, threads):
        """
        Computing (Z'Z)^{-1} from summary statistics

        Parameters:
        ------------
        threads: number of threads

        Returns:
        ---------
        ztz_inv: a np.array of (Z'Z)^{-1} (d, 1)

        """
        ldr_var = np.diag(self.ldr_cov)
        ztz_inv = np.zeros(np.sum(self.snp_idxs), dtype=np.float32)
        n_ldrs = self.bases.shape[1]

        futures = []
        i = 0
        data_reader = self.ldr_gwas.data_reader(
            "both", self.ldr_idxs, self.snp_idxs, all_gwas=False
        )
        lock = threading.Lock()

        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            for ldr_beta_batch, ldr_z_batch in data_reader:
                futures.append(
                    executor.submit(
                        self._compute_ztz_inv_batch,
                        ztz_inv,
                        ldr_beta_batch,
                        ldr_z_batch,
                        ldr_var,
                        i,
                        lock,
                    )
                )
                batch_size = ldr_beta_batch.shape[1]
                i += batch_size

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    executor.shutdown(wait=False)
                    raise RuntimeError(f"Computation terminated due to error: {exc}")

        ztz_inv /= n_ldrs
        ztz_inv = ztz_inv.reshape(-1, 1)

        return ztz_inv

    def _compute_ztz_inv_batch(
        self, ztz_inv, ldr_beta_batch, ldr_z_batch, ldr_var, i, lock
    ):
        """
        Computing (Z'Z)^{-1} from summary statistics in batch

        """
        ldr_se_batch = ldr_beta_batch / ldr_z_batch
        batch_size = ldr_beta_batch.shape[1]
        ztz_inv_batch = np.sum(
            (ldr_se_batch * ldr_se_batch + ldr_beta_batch * ldr_beta_batch / self.n)
            / ldr_var[i : i + batch_size],
            axis=1,
        )
        with lock:
            ztz_inv += ztz_inv_batch

    def recover_beta_numba(self, voxel_idxs, threads):
        """
        Recovering voxel beta

        Parameters:
        ------------
        voxel_idxs: a list of voxel idxs (q)
        threads: number of threads

        Returns:
        ---------
        voxel_beta: a np.array of voxel beta (d, q)

        """
        voxel_beta = np.zeros(
            (np.sum(self.snp_idxs), len(voxel_idxs)), dtype=np.float32
        )
        data_reader = self.ldr_gwas.data_reader(
            "beta", self.ldr_idxs, self.snp_idxs, all_gwas=False
        )
        bases = self.bases[voxel_idxs]  # (q, r)

        i = 0
        for ldr_beta_batch in data_reader:
            batch_size = ldr_beta_batch.shape[1]
            bases_batch = bases[:, i : i + batch_size].T
            voxel_beta_batch = recover_beta_batch(ldr_beta_batch, bases_batch)
            i += batch_size
            voxel_beta += voxel_beta_batch

        return voxel_beta

    def recover_beta(self, voxel_idxs, threads):
        """
        Recovering voxel beta

        Parameters:
        ------------
        voxel_idxs: a list of voxel idxs (q)
        threads: number of threads

        Returns:
        ---------
        voxel_beta: a np.array of voxel beta (d, q)

        """
        voxel_beta = np.zeros(
            (np.sum(self.snp_idxs), len(voxel_idxs)), dtype=np.float32
        )
        data_reader = self.ldr_gwas.data_reader(
            "beta", self.ldr_idxs, self.snp_idxs, all_gwas=False
        )
        base = self.bases[voxel_idxs]  # (q, r)

        lock = threading.Lock()
        i = 0
        futures = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            for ldr_beta_batch in data_reader:
                futures.append(
                    executor.submit(
                        self._recover_beta_batch,
                        voxel_beta,
                        ldr_beta_batch,
                        base,
                        i,
                        lock,
                    )
                )
                batch_size = ldr_beta_batch.shape[1]
                i += batch_size

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    executor.shutdown(wait=False)
                    raise RuntimeError(f"Computation terminated due to error: {exc}")

        return voxel_beta

    def _recover_beta_batch(self, voxel_beta, ldr_beta_batch, base, i, lock):
        """
        Computing voxel beta in batch

        """
        batch_size = ldr_beta_batch.shape[1]
        voxel_beta_batch = np.dot(ldr_beta_batch, base[:, i : i + batch_size].T)
        with lock:
            voxel_beta += voxel_beta_batch

    def recover_se(self, voxel_idxs, voxel_beta):
        """
        Recovering standard errors for voxel-level genetic effects

        Parameters:
        ------------
        voxel_idx: a list of voxel idxs (q)
        voxel_beta: a np.array of voxel beta (d, q)

        Returns:
        ---------
        voxel_se: a np.array of standard errors for voxel-level genetic effects (d, )

        """
        base = np.atleast_2d(self.bases[voxel_idxs])  # (q, r)
        part1 = np.sum(np.dot(base, self.ldr_cov) * base, axis=1)  # (q, )
        voxel_beta_squared = voxel_beta * voxel_beta
        voxel_beta_squared /= self.n
        voxel_se = part1 * self.ztz_inv
        voxel_se -= voxel_beta_squared
        np.sqrt(voxel_se, out=voxel_se)
        # voxel_se = np.sqrt(part1 * self.ztz_inv - voxel_beta * voxel_beta / self.n) # slow

        return voxel_se


def voxel_reader(n_snps, voxel_list):
    """
    Doing voxel GWAS in batch, each block less than 3 GB

    """
    n_voxels = len(voxel_list)
    memory_use = n_snps * n_voxels * np.dtype(np.float32).itemsize / (1024**3)
    if memory_use <= 3:
        batch_size = n_voxels
    else:
        batch_size = int(n_voxels / memory_use * 3)

    for i in range(0, n_voxels, batch_size):
        yield voxel_list[i : i + batch_size]


def write_header(snp_info, outpath):
    """
    Writing output header

    """
    output_header = snp_info.head(0).copy()
    output_header.insert(0, "INDEX", None)
    output_header["BETA"] = None
    output_header["SE"] = None
    output_header["Z"] = None
    output_header["P"] = None
    output_header = output_header.to_csv(sep="\t", header=True, index=None)
    with open(outpath, "w") as file:
        file.write(output_header)


def _process_voxels_batch(
    i,
    voxel_idx,
    all_sig_idxs,
    snp_info,
    voxel_beta,
    voxel_se,
    voxel_z,
    all_sig_idxs_voxel,
):
    """
    Processing each voxel

    """
    if all_sig_idxs_voxel[i]:
        sig_idxs = all_sig_idxs[:, i]
        sig_snps = snp_info.loc[sig_idxs].copy()
        sig_snps["BETA"] = voxel_beta[sig_idxs, i]
        sig_snps["SE"] = voxel_se[sig_idxs, i]
        sig_snps["Z"] = voxel_z[sig_idxs, i]
        sig_snps["P"] = chi2.sf(sig_snps["Z"] ** 2, 1)
        sig_snps.insert(0, "INDEX", [voxel_idx + 1] * np.sum(sig_idxs))
        sig_snps_output = sig_snps.to_csv(
            sep="\t", header=False, na_rep="NA", index=None, float_format="%.5e"
        )
        return i, sig_snps_output
    return i, None


def process_voxels(
    voxel_idxs,
    all_sig_idxs,
    snp_info,
    voxel_beta,
    voxel_se,
    voxel_z,
    all_sig_idxs_voxel,
    outpath,
    threads,
):
    """
    Processing voxels in parallel

    Parameters:
    ------------
    voxel_idxs: a list of voxel idxs (q)
    all_sig_idxs: a np.array of boolean significant indices (d, q)
    snp_info: a pd.DataFrame of all SNPs (d, x)
    voxel_beta: a np.array of voxel beta (d, q)
    voxel_se: a np.array of voxel se (d, q)
    voxel_z: a np.array of voxel z-score (d, q)
    all_sig_idxs_voxel: a np.array of boolean indices of any significant SNPs (q, )
    outpath: a directory of output
    threads: number of threads

    """
    results_dict = {}
    future_to_idx = {}
    next_write_i = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        for i, voxel_idx in enumerate(voxel_idxs):
            result = executor.submit(
                _process_voxels_batch,
                i,
                voxel_idx,
                all_sig_idxs,
                snp_info,
                voxel_beta,
                voxel_se,
                voxel_z,
                all_sig_idxs_voxel,
            )
            future_to_idx[result] = i

        for future in concurrent.futures.as_completed(future_to_idx):
            i, sig_snps_output = future.result()
            results_dict[i] = sig_snps_output
            while next_write_i in results_dict:
                if results_dict[next_write_i] is not None:
                    with open(outpath, "a") as file:
                        file.write(results_dict.pop(next_write_i))
                next_write_i += 1


def check_input(args, log):
    # required arguments
    if args.ldr_sumstats is None:
        raise ValueError("--ldr-sumstats is required")
    if args.bases is None:
        raise ValueError("--bases is required")
    if args.ldr_cov is None:
        raise ValueError("--ldr-cov is required")

    # optional arguments
    if (
        args.chr_interval is None
        and args.voxels is None
        and args.sig_thresh is None
        and args.extract is None
    ):
        log.info(
            (
                "WARNING: generating all voxelwise summary statistics will require large disk space. "
                "Specify a p-value threshold by --sig-thresh to screen out insignificant results."
            )
        )

    # process some arguments
    if args.chr_interval is not None:
        try:
            start, end = args.chr_interval.split("-")
            start_chr, start_pos = [int(x) for x in start.split(":")]
            end_pos = int(end)
        except:
            raise ValueError(
                "--chr-interval (--range) should be in this format: <CHR>:<POS1>-<POS2>"
            )
        if start_pos > end_pos:
            raise ValueError(
                (
                    f"starting with {start_pos} "
                    f"while ending with {end_pos} "
                    "is not allowed"
                )
            )
    else:
        start_chr, start_pos, end_pos = None, None, None

    return start_chr, start_pos, end_pos


def run(args, log):
    # checking input
    target_chr, start_pos, end_pos = check_input(args, log)

    # reading data
    ldr_cov = np.load(args.ldr_cov)
    log.info(f"Read variance-covariance matrix of LDRs from {args.ldr_cov}")
    bases = np.load(args.bases)
    log.info(f"{bases.shape[1]} bases read from {args.bases}")

    try:
        ldr_gwas = sumstats.read_sumstats(args.ldr_sumstats)
        log.info(
            f"{ldr_gwas.n_snps} SNPs read from LDR summary statistics {args.ldr_sumstats}"
        )

        # keep selected LDRs
        if args.n_ldrs is not None:
            bases, ldr_cov, ldr_gwas, _ = ds.keep_ldrs(
                args.n_ldrs, bases, ldr_cov, ldr_gwas
            )
            log.info(f"Keeping the top {args.n_ldrs} LDRs.")

        if bases.shape[1] != ldr_cov.shape[0] or bases.shape[1] != ldr_gwas.n_gwas:
            raise ValueError(
                (
                    "inconsistent dimension for bases, variance-covariance matrix of LDRs, "
                    "and LDR summary statistics. "
                    "Try to use --n-ldrs"
                )
            )

        # getting the outpath and SNP list
        outpath = args.out
        if args.voxels is not None:
            if np.max(args.voxels) + 1 <= bases.shape[0] and np.min(args.voxels) >= 0:
                log.info(f"{len(args.voxels)} voxel(s) included.")
            else:
                raise ValueError("--voxels index (one-based) out of range")
        else:
            args.voxels = np.arange(bases.shape[0])

        if target_chr:
            snp_idxs = (
                (ldr_gwas.snpinfo["POS"] > start_pos)
                & (ldr_gwas.snpinfo["POS"] < end_pos)
                & (ldr_gwas.snpinfo["CHR"] == target_chr)
            ).to_numpy()
            outpath += f"_chr{target_chr}_start{start_pos}_end{end_pos}.txt"
            log.info(
                f"{np.sum(snp_idxs)} SNP(s) on chromosome {target_chr} from {start_pos} to {end_pos}."
            )
        else:
            snp_idxs = ~ldr_gwas.snpinfo["SNP"].isna().to_numpy()
            outpath += ".txt"

        if args.extract is not None:
            idx_extract_snps = (
                ldr_gwas.snpinfo["SNP"].isin(args.extract["SNP"])
            ).to_numpy()
            snp_idxs = snp_idxs & idx_extract_snps

        if args.exclude is not None:
            idx_exclude_snps = (
                ~(ldr_gwas.snpinfo["SNP"].isin(args.exclude["SNP"]))
            ).to_numpy()
            snp_idxs = snp_idxs & idx_exclude_snps

        # extracting SNPs
        ldr_n = np.array(ldr_gwas.snpinfo["N"]).reshape(-1, 1)
        ldr_n = ldr_n[snp_idxs]
        snp_info = ldr_gwas.snpinfo.loc[snp_idxs]

        # getting threshold
        if args.sig_thresh:
            thresh_chisq = chi2.ppf(1 - args.sig_thresh, 1)
        else:
            thresh_chisq = 0

        # doing analysis
        log.info(
            f"Recovering voxel-level GWAS results for {np.sum(snp_idxs)} SNP(s) ..."
        )
        write_header(snp_info, outpath)
        vgwas = VGWAS(bases, ldr_cov, ldr_gwas, snp_idxs, ldr_n, args.threads)

        for voxel_idxs in tqdm(
            voxel_reader(np.sum(snp_idxs), args.voxels),
            desc=f"Doing GWAS for {len(args.voxels)} voxel(s) in batch",
        ):
            voxel_beta = vgwas.recover_beta(voxel_idxs, args.threads)
            # voxel_beta = vgwas.recover_beta_numba(voxel_idxs, args.threads)
            # voxel_se = vgwas.recover_se(voxel_idxs, voxel_beta)
            voxel_se = recover_se_numba(
                voxel_idxs, voxel_beta, vgwas.bases, vgwas.ldr_cov, vgwas.ztz_inv, vgwas.n
            )
            voxel_z = voxel_beta / voxel_se
            all_sig_idxs = voxel_z * voxel_z >= thresh_chisq
            all_sig_idxs_voxel = all_sig_idxs.any(axis=0)

            process_voxels(
                voxel_idxs,
                all_sig_idxs,
                snp_info,
                voxel_beta,
                voxel_se,
                voxel_z,
                all_sig_idxs_voxel,
                outpath,
                args.threads,
            )

        log.info(f"\nSaved the output to {outpath}")

    finally:
        if "ldr_gwas" in locals():
            ldr_gwas.close()

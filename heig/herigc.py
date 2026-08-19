import os
import h5py
import numpy as np
import pandas as pd
import threading
import concurrent.futures
from filelock import FileLock
from abc import ABC, abstractmethod
from scipy.stats import chi2
from heig import sumstats
import heig.input.dataset as ds
from heig.ldmatrix import LDmatrix
from heig.ldsc import LDSC


def check_input(args, log):
    # required arguments
    if args.ldr_sumstats is None:
        raise ValueError("--ldr-sumstats is required")
    if args.bases is None:
        raise ValueError("--bases is required")
    if args.ldr_cov is None:
        raise ValueError("--ldr-cov is required")
    if args.ld_inv is None:
        raise ValueError("--ld-inv is required")
    if args.ld is None:
        raise ValueError("--ld is required")
    if args.overlap and args.y2_sumstats is None:
        log.info("WARNING: ignoring --overlap as --y2-sumstats has not been specified.")

    ds.check_existence(args.y2_sumstats, ".snpinfo")
    ds.check_existence(args.y2_sumstats, ".sumstats")


class CommonSNPs:
    """
    Extracting common snps from multiple snp lists in parallel

    """

    def __init__(self, *snp_list, exclude_snps, threads, match_alleles=True):
        """
        Parameters:
        ------------
        snp_list: a list of snp lists
        exclude_snps: a pd.DataFrame of SNPs to exclude
        threads: number of threads

        Returns:
        ---------
        common_snps: a pd.Series of common snps

        """
        self.snp_list = snp_list
        self.common_snps = self._merge_snp_list()
        if exclude_snps is not None:
            self.common_snps = self.common_snps[~(self.common_snps["SNP"].isin(exclude_snps["SNP"]))]
        if match_alleles:
            matched_alleles_set = self._match_alleles(self.common_snps, threads)
            self.common_snps = self.common_snps.loc[matched_alleles_set, "SNP"]
        else:
            self.common_snps = self.common_snps[["SNP"]]

    def _merge_snp_list(self):
        """
        Merging multiple SNP files

        """
        n_snp_list = len(self.snp_list)
        if n_snp_list == 0:
            raise ValueError("no SNP list provided")

        common_snps = None
        for i in range(len(self.snp_list)):
            if hasattr(self.snp_list[i], "ldinfo"):
                snp = self.snp_list[i].ldinfo[["SNP", "A1", "A2"]]
                snp = snp.rename({"A1": f"A1_{i}", "A2": f"A2_{i}"}, axis=1)
            elif hasattr(self.snp_list[i], "snpinfo"):
                snp = self.snp_list[i].snpinfo[["SNP", "A1", "A2"]]
                snp = snp.rename({"A1": f"A1_{i}", "A2": f"A2_{i}"}, axis=1)
            elif hasattr(self.snp_list[i], "SNP"):
                snp = self.snp_list[i]["SNP"]
            if not isinstance(common_snps, pd.DataFrame):
                common_snps = snp.copy()
            else:
                common_snps = common_snps.merge(snp, on="SNP")

        if common_snps is None:
            raise ValueError(
                "all the input snp lists are None or do not have a SNP column"
            )

        common_snps.drop_duplicates(subset=["SNP"], keep=False, inplace=True)
        if len(common_snps) == 0:
            raise ValueError("no common SNPs exist")

        return common_snps

    def _match_alleles(self, common_snps, threads):
        """
        Ensuring each common SNP has identical alleles

        """
        allele_columns = [col for col in common_snps.columns if col.startswith("A")]

        indices = common_snps.index.values
        chunk_size = common_snps.shape[0] // threads
        splits = [
            indices[i : i + chunk_size] for i in range(0, len(indices), chunk_size)
        ]
        chunks = [common_snps.loc[split, allele_columns] for split in splits]

        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            results = executor.map(self._process_chunk, chunks)

        return pd.concat(results)

    @staticmethod
    def _process_chunk(df_chunk):
        return df_chunk.apply(lambda x: len(set(x)) == 2, axis=1)


class Estimation(ABC):
    def __init__(self, ldr_gwas, ld, ld_inv, bases, ldr_cov):
        """
        An abstract class for estimating heritability and genetic correlation

        Parameters:
        ------------
        ldr_gwas: a GWAS instance
        ld: a LDmatrix instance
        ld_inv: a LDmatrix instance
        bases: a np.array of bases (N, r)
        ldr_cov: a np.array of variance-covariance matrix of LDRs (r, r)

        """
        self.ldr_gwas = ldr_gwas
        self.n = ldr_gwas.snpinfo["N"].values.reshape(-1, 1)
        self.N = bases.shape[0]
        self.nbar = np.mean(self.n)
        self.ld = ld
        self.ld_inv = ld_inv
        self.r = ldr_gwas.n_gwas
        self.bases = bases
        self.ldr_cov = ldr_cov
        self.block_ranges = ld.block_ranges
        self.merged_blocks = ld.merge_blocks()
        self.sigmaX_var = np.sum(np.dot(bases, ldr_cov) * bases, axis=1)
        self.ldr_z = self._ldr_sumstats_reader()

    def _ldr_sumstats_reader(self):
        """
        Reading LDRs sumstats from HDF5 file and preprocessing

        Returns:
        ---------
        preprocessed z scores (n_snps, n_ldrs)

        """
        ldr_idxs = list(range(self.ldr_gwas.n_gwas))
        z = self.ldr_gwas.data_reader(
            "z", ldr_idxs, self.ldr_gwas.snp_idxs, all_gwas=True
        )
        ldr_se = np.sqrt(np.diag(self.ldr_cov))
        z[self.ldr_gwas.change_sign] = -1 * z[self.ldr_gwas.change_sign]
        z *= ldr_se
        z /= np.sqrt(self.n)

        return z

    def _get_heri_se(self, heri, d, n):
        """
        Estimating standard error of heritability estimates

        Parameters:
        ------------
        heri: an N by 1 vector of heritability estimates
        d: number of SNPs, or Tr(R\Omega)
        n: sample size

        Returns:
        ---------
        An N by 1 vector of standard error estimates

        """
        part1 = 2 * d / n**2
        part2 = 2 * heri / n
        part3 = 2 * heri * (1 - heri) / n
        res = part1 + part2 + part3
        res[np.abs(res) < 10**-10] = 0
        return np.sqrt(res)

    @staticmethod
    def _qc(est, est_se, est_min, est_max, est_se_min, est_se_max):
        invalid_est = (
            (est > est_max)
            | (est < est_min)
            | (est_se > est_se_max)
            | (est_se < est_se_min)
        )
        est[invalid_est] = np.nan
        est_se[invalid_est] = np.nan
        return est, est_se

    @abstractmethod
    def _block_wise_estimate_parallel(self):
        """
        Computing by (merged) LD blocks

        """
        pass


class OneSample(Estimation):
    """
    Estimating heritability and genetic correlation within images

    """

    def __init__(self, ldr_gwas, ld, ld_inv, bases, ldr_cov, threads):
        super().__init__(ldr_gwas, ld, ld_inv, bases, ldr_cov)

        self.ld_rank, self.ldr_gene_cov = self._block_wise_estimate_parallel(threads)
        del self.ldr_z

        self.gene_var = np.sum(
            np.dot(self.bases, self.ldr_gene_cov) * self.bases, axis=1
        )
        self.heri = self.gene_var / self.sigmaX_var
        self.heri[self.heri <= 0] = 1e-5
        self.heri[self.heri >= 1] = 1 - 1e-5
        self.heri_se = self._get_heri_se(self.heri, self.ld_rank, self.nbar)
        self.heri, self.heri_se = self._qc(self.heri, self.heri_se, 0, 1, 0, 1)

    def _block_wise_estimate_parallel(self, threads):
        """
        Computing Tr(R\Omega) and LDR genetic covariance by LD block in parallel

        """
        ld_rank = np.zeros(1, dtype=np.float32)
        ldr_gene_cov = np.zeros((self.r, self.r), dtype=np.float32)
        lock = threading.Lock()
        futures = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            merged_block_reader = (
                self.block_ranges[block]
                for merged_block in self.merged_blocks
                for block in merged_block
            )
            for (start, end), ld_block, ld_inv_block in zip(
                merged_block_reader, self.ld.data, self.ld_inv.data
            ):
                futures.append(
                    executor.submit(
                        self._block_wise_estimate,
                        ld_rank,
                        ldr_gene_cov,
                        start,
                        end,
                        ld_block,
                        ld_inv_block,
                        lock,
                    )
                )

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    executor.shutdown(wait=False)
                    raise RuntimeError(f"Computation terminated due to error: {exc}")

        return ld_rank, ldr_gene_cov

    def _block_wise_estimate(
        self, ld_rank, ldr_gene_cov, start, end, ld_block, ld_inv_block, lock
    ):
        """
        block_ld: eigenvectors * sqrt(eigenvalues)
        block_ld_inv: eigenvectors * sqrt(eigenvalues ** -1)

        """
        block_ld_ld_inv = np.dot(ld_block.T, ld_inv_block)
        ld_block_rank = np.sum(block_ld_ld_inv * block_ld_ld_inv)
        block_z = self.ldr_z[start:end]
        z_mat_block_ld_inv = np.dot(block_z.T, ld_inv_block)  # (r, ld_size)
        block_gene_cov = (
            np.dot(z_mat_block_ld_inv, z_mat_block_ld_inv.T)
            - ld_block_rank * self.ldr_cov / self.nbar
        )  # (r, r)
        with lock:
            ld_rank += ld_block_rank
            ldr_gene_cov += block_gene_cov

    def get_gene_cor_se(self, out_dir, threads):
        """
        Computing genetic correlation and its se by block
        Saving to a HDF5 file

        """
        block_size = 100
        mean_gene_cor = 0
        min_gene_cor = 1
        mean_gene_cor_se = 0
        with h5py.File(f"{out_dir}_gc.h5", "w") as file:
            gc = file.create_dataset("gc", shape=(self.N, self.N), dtype="float32")
            se = file.create_dataset("se", shape=(self.N, self.N), dtype="float32")

            futures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
                for start in range(0, self.N, block_size):
                    futures.append(
                        executor.submit(
                            self._get_gene_cor_se_block, start, gc, se, out_dir
                        )
                    )

                for future in concurrent.futures.as_completed(futures):
                    try:
                        gene_cor_sum, gene_cor_min, gene_cor_se_sum = future.result()
                        mean_gene_cor += gene_cor_sum
                        mean_gene_cor_se += gene_cor_se_sum
                        min_gene_cor = np.min((min_gene_cor, gene_cor_min))
                    except Exception as exc:
                        executor.shutdown(wait=False)
                        raise RuntimeError(
                            f"Computation terminated due to error: {exc}"
                        )

        mean_gene_cor /= self.N**2
        mean_gene_cor_se /= self.N**2

        if os.path.exists(f"{out_dir}.gc.h5.lock"):
            os.remove(f"{out_dir}.gc.h5.lock")

        return mean_gene_cor, min_gene_cor, mean_gene_cor_se

    def _get_gene_cor_se_block(self, start, gc, se, out_dir):
        """
        Computing genetic correlation and se for a block

        """
        end = start + 100
        gene_cov = np.dot(
            np.dot(self.bases[start:end], self.ldr_gene_cov), self.bases.T
        )
        gene_cov[gene_cov == 0] = 0.01
        gene_cor = gene_cov / np.sqrt(np.outer(self.gene_var[start:end], self.gene_var))
        gene_cor2 = gene_cor * gene_cor
        part1 = 1 - gene_cor2

        part2 = (
            np.dot(np.dot(self.bases[start:end], self.ldr_cov), self.bases.T) / gene_cov
        )
        part2 -= 1
        del gene_cov

        inv_heri1 = np.repeat(1 / self.heri[start:end] - 1, self.N).reshape(-1, self.N)
        inv_heri2 = (
            np.repeat(1 / self.heri - 1, gene_cor.shape[0]).reshape(self.N, -1).T
        )
        part3 = inv_heri1 + inv_heri2
        temp1 = inv_heri1 - part2
        temp2 = inv_heri2 - part2
        temp3 = inv_heri1 * inv_heri2
        del inv_heri1, inv_heri2

        gene_cor_se = np.zeros((gene_cor.shape[0], self.N), dtype=np.float32)
        n = self.nbar
        d = self.ld_rank
        gene_cor_se += (4 / n + d / n**2) * part1 * part1
        gene_cor_se += (1 / n + d / n**2) * part1 * part3
        gene_cor_se -= (1 / n + d / n**2) * part1 * 2 * gene_cor2 * part2
        gene_cor_se += d / n**2 / 2 * gene_cor2 * temp1 * temp1
        gene_cor_se += d / n**2 / 2 * gene_cor2 * temp2 * temp2
        gene_cor_se += d / n**2 * gene_cor2 * gene_cor2 * part2 * part2
        gene_cor_se += d / n**2 * temp3
        gene_cor_se -= d / n**2 * gene_cor2 * part2 * part3

        # the diagonal of this block is at global columns start:end
        block_idx = np.arange(gene_cor.shape[0])
        gene_cor[block_idx, start + block_idx] = 1
        gene_cor_se[block_idx, start + block_idx] = 0
        gene_cor, gene_cor_se = self._qc(gene_cor, gene_cor_se, -1, 1, 0, 1)
        np.sqrt(gene_cor_se, out=gene_cor_se)

        with FileLock(f"{out_dir}.gc.h5.lock"):
            gc[start:end] = gene_cor
            se[start:end] = gene_cor_se

        return np.nansum(gene_cor), np.nanmin(gene_cor), np.nansum(gene_cor_se)


class TwoSample(Estimation):
    """
    Estimating genetic correlation between images and non-imaging phenotypes

    """

    def __init__(
        self, ldr_gwas, ld, ld_inv, bases, ldr_cov, y2_gwas, threads, overlap=False
    ):
        super().__init__(ldr_gwas, ld, ld_inv, bases, ldr_cov)
        self.y2_gwas = y2_gwas
        self.n2 = y2_gwas.snpinfo["N"].values.reshape(-1, 1)
        self.n2bar = np.mean(self.n2)
        self.y2_z = self._y2_sumstats_reader()

        (
            self.ld_block_rank,
            self.ldr_block_gene_cov,
            y2_block_gene_cov,
            ldr_y2_block_gene_cov_part1,
        ) = self._block_wise_estimate_parallel(threads)

        self.ld_rank = np.sum(self.ld_block_rank)
        self.ldr_gene_cov = np.sum(self.ldr_block_gene_cov, axis=0)
        self.y2_heri = np.atleast_1d(
            np.sum(y2_block_gene_cov)
        )  # since the sum stats have been normalized
        ldr_y2_gene_cov_part1 = np.sum(ldr_y2_block_gene_cov_part1, axis=0)

        # image heritability
        self.heri = (
            np.sum(np.dot(self.bases, self.ldr_gene_cov) * self.bases, axis=1)
            / self.sigmaX_var
        )
        self.heri_se = self._get_heri_se(self.heri, self.ld_rank, self.nbar)
        self.heri, self.heri_se = self._qc(self.heri, self.heri_se, 0, 1, 0, 1)

        if not overlap:
            self.gene_cor_y2 = np.squeeze(
                self._get_gene_cor_y2(ldr_y2_gene_cov_part1, self.heri, self.y2_heri)
            )
            self.gene_cor_y2_se = self._get_gene_cor_se(
                self.heri,
                self.y2_heri,
                self.gene_cor_y2,
                self.ld_rank,
                self.nbar,
                self.n2bar,
            )
        else:
            ldscore = ld.ldinfo["ldscore"].values
            n_merged_blocks = len(self.merged_blocks)

            # compute left-one-block-out cross-trait LDSC intercept
            self.ldr_heri = np.diag(self.ldr_gene_cov) / np.diag(self.ldr_cov)
            self.ldr_z /= np.sqrt(np.diag(self.ldr_cov))
            self.ldr_z *= np.sqrt(self.n)
            self.y2_z *= np.sqrt(self.n2)

            ldsc_intercept = LDSC(
                self.ldr_z,
                self.y2_z,
                ldscore,
                self.ldr_heri,
                self.y2_heri,
                self.n,
                self.n2,
                self.ld_rank,
                self.block_ranges,
                self.merged_blocks,
                threads,
            )
            del self.ldr_z

            # compute left-one-block-out heritability
            ldr_lobo_gene_cov = self._lobo_estimate(
                self.ldr_gene_cov, self.ldr_block_gene_cov
            )
            del self.ldr_block_gene_cov
            y2_lobo_heri = self._lobo_estimate(self.y2_heri, y2_block_gene_cov)
            image_lobo_heri = self._image_lobo_heri(ldr_lobo_gene_cov, threads)
            image_lobo_heri /= self.sigmaX_var
            image_lobo_heri[image_lobo_heri <= 0] = np.nan
            del ldr_lobo_gene_cov

            # compute left-one-block-out genetic correlation
            ldr_y2_lobo_gene_cov_part1 = self._lobo_estimate(
                ldr_y2_gene_cov_part1, ldr_y2_block_gene_cov_part1
            )
            ld_rank_lobo = self._lobo_estimate(self.ld_rank, self.ld_block_rank)
            lobo_gene_cor = self._get_gene_cor_ldsc(
                ldr_y2_lobo_gene_cov_part1,
                ld_rank_lobo,
                ldsc_intercept.lobo_ldsc,
                image_lobo_heri,
                y2_lobo_heri,
            )  # n_blocks * N

            # compute genetic correlation using all blocks
            image_y2_gene_cor = self._get_gene_cor_ldsc(
                ldr_y2_gene_cov_part1,
                self.ld_rank,
                ldsc_intercept.total_ldsc,
                self.heri,
                self.y2_heri,
            )  # 1 * N

            # compute jackknite estimate of genetic correlation and se
            self.gene_cor_y2, self.gene_cor_y2_se = self._jackknife(
                image_y2_gene_cor, lobo_gene_cor, n_merged_blocks
            )

        self.y2_heri_se = self._get_heri_se(self.y2_heri, self.ld_rank, self.n2bar)
        self.gene_cor_y2, self.gene_cor_y2_se = self._qc(
            self.gene_cor_y2, self.gene_cor_y2_se, -1, 1, 0, 1
        )
        self.y2_heri, self.y2_heri_se = self._qc(
            self.y2_heri, self.y2_heri_se, 0, 1, 0, 1
        )

    def _y2_sumstats_reader(self):
        """
        Reading y2 summstats from HDF5 file and preprocessing:

        Returns:
        ---------
        z: preprocessed z scores

        """
        y2_idxs = list(range(self.y2_gwas.n_gwas))
        z = self.y2_gwas.data_reader("z", y2_idxs, self.y2_gwas.snp_idxs, all_gwas=True)

        z[self.y2_gwas.change_sign] = -1 * z[self.y2_gwas.change_sign]
        z /= np.sqrt(self.n2)

        return z

    def _block_wise_estimate_parallel(self, threads):
        """
        Computing Tr(R\Omega), y2 genetic variance,
        LDR-y2 genetic covariance, LDR genetic covariance by merged block in parallel

        """
        n_blocks = len(self.merged_blocks)
        y2_block_gene_cov = np.zeros(n_blocks, dtype=np.float32)
        ldr_y2_block_gene_cov_part1 = np.zeros((n_blocks, self.r), dtype=np.float32)
        ld_block_rank = np.zeros(n_blocks, dtype=np.float32)
        ldr_block_gene_cov = np.zeros((n_blocks, self.r, self.r), dtype=np.float32)

        lock = threading.Lock()
        futures = []
        ld_merged_block_reader = (
            [next(self.ld.data) for _ in merged_block]
            for merged_block in self.merged_blocks
        )
        ld_inv_merged_block_reader = (
            [next(self.ld_inv.data) for _ in merged_block]
            for merged_block in self.merged_blocks
        )
        block_range_reader = (
            [self.block_ranges[i] for i in merged_block]
            for merged_block in self.merged_blocks
        )
        merged_block_reader = enumerate(
            zip(block_range_reader, ld_merged_block_reader, ld_inv_merged_block_reader)
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            for i, (
                block_range,
                ld_merged_block,
                ld_inv_merged_block,
            ) in merged_block_reader:
                futures.append(
                    executor.submit(
                        self._block_wise_estimate,
                        ld_block_rank,
                        ldr_block_gene_cov,
                        y2_block_gene_cov,
                        ldr_y2_block_gene_cov_part1,
                        i,
                        block_range,
                        ld_merged_block,
                        ld_inv_merged_block,
                        lock,
                    )
                )

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    executor.shutdown(wait=False)
                    raise RuntimeError(f"Computation terminated due to error: {exc}")

        return (
            ld_block_rank,
            ldr_block_gene_cov,
            y2_block_gene_cov,
            ldr_y2_block_gene_cov_part1,
        )

    def _block_wise_estimate(
        self,
        ld_block_rank,
        ldr_block_gene_cov,
        y2_block_gene_cov,
        ldr_y2_block_gene_cov_part1,
        i,
        block_range,
        ld_merged_block,
        ld_inv_merged_block,
        lock,
    ):
        """
        ld_block: eigenvectors * sqrt(eigenvalues)
        ld_block_inv: eigenvectors * sqrt(eigenvalues ** -1)

        """
        block_rank = np.zeros(1, dtype=np.float32)
        block_gene_cov = np.zeros((self.r, self.r), dtype=np.float32)
        block_gene_var_y2 = np.zeros(1, dtype=np.float32)
        block_gene_cov_y2 = np.zeros(self.r, dtype=np.float32)

        for (start, end), ld_block, ld_block_inv in zip(
            block_range, ld_merged_block, ld_inv_merged_block
        ):
            ld_block_ld_inv = np.dot(ld_block.T, ld_block_inv)
            block_rank_ = np.sum(ld_block_ld_inv * ld_block_ld_inv)
            block_rank += block_rank_

            block_z = self.ldr_z[start:end]
            z_mat_ld_block_inv = np.dot(block_z.T, ld_block_inv)
            block_y2z = self.y2_z[start:end]
            y2_ld_block_inv = np.dot(block_y2z.T, ld_block_inv)

            block_gene_var_y2 += np.squeeze(
                np.dot(y2_ld_block_inv, y2_ld_block_inv.T) - block_rank_ / self.n2bar
            )
            block_gene_cov_y2 += np.squeeze(
                np.dot(z_mat_ld_block_inv, y2_ld_block_inv.T)
            )
            block_gene_cov += (
                np.dot(z_mat_ld_block_inv, z_mat_ld_block_inv.T)
                - block_rank_ * self.ldr_cov / self.nbar
            )

        with lock:
            ld_block_rank[i] = block_rank
            ldr_block_gene_cov[i, :, :] = block_gene_cov
            y2_block_gene_cov[i] = block_gene_var_y2
            ldr_y2_block_gene_cov_part1[i, :] = block_gene_cov_y2

    def _image_lobo_heri(self, ldr_lobo_gene_cov, threads):
        """
        Computing left-one-block-out image heritability estimates in parallel

        """
        image_lobo_heri = np.zeros((len(self.merged_blocks), self.N), dtype=np.float32)
        futures = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            for i in range(0, self.N, 1000):
                futures.append(
                    executor.submit(
                        self._image_lobo_heri_batch,
                        i,
                        ldr_lobo_gene_cov,
                        image_lobo_heri,
                    )
                )

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    executor.shutdown(wait=False)
                    raise RuntimeError(f"Computation terminated due to error: {exc}")

        return image_lobo_heri

    def _image_lobo_heri_batch(self, i, ldr_lobo_gene_cov, image_lobo_heri):
        bases = self.bases[i : i + 1000]
        image_lobo_heri_batch = np.matmul(
            np.swapaxes(ldr_lobo_gene_cov, 1, 2), bases.T
        ).swapaxes(1, 2)
        image_lobo_heri_batch = np.sum(
            image_lobo_heri_batch * np.expand_dims(bases, 0), axis=2
        )
        image_lobo_heri[:, i : i + 1000] = image_lobo_heri_batch

    def _get_gene_cor_ldsc(self, part1, ld_rank, ldsc, heri1, heri2):
        """
        Computing LDSC intercept-involved genetic correlation

        """
        ldr_gene_cov = part1 - ld_rank.reshape(-1, 1) * ldsc * np.sqrt(
            np.diagonal(self.ldr_cov)
        ) / np.sqrt(self.nbar) / np.sqrt(self.n2bar)
        gene_cor = self._get_gene_cor_y2(ldr_gene_cov.T, heri1, heri2)

        return gene_cor

    def _get_gene_cor_y2(self, inner_part, heri1, heri2):
        bases_inner_part = np.dot(self.bases, inner_part).reshape(
            self.bases.shape[0], -1
        )
        gene_cov_y2 = bases_inner_part / np.sqrt(self.sigmaX_var).reshape(-1, 1)
        gene_cor_y2 = gene_cov_y2.T / np.sqrt(heri1 * heri2)

        return gene_cor_y2

    def _jackknife(self, total, lobo, n_blocks):
        """
        Jackknite estimator

        Parameters:
        ------------
        total: the estimate using all blocks
        lobo: an n_blocks by N matrix of lobo estimates
        n_blocks: the number of blocks

        Returns:
        ---------
        estimate: an N by 1 vector of jackknife estimates
        se: an N by 1 vector of jackknife se estimates

        """
        mean_lobo = np.mean(lobo, axis=0)
        estimate = np.squeeze(n_blocks * total - (n_blocks - 1) * mean_lobo)
        se = np.squeeze(
            np.sqrt((n_blocks - 1) / n_blocks * np.sum((lobo - mean_lobo) ** 2, axis=0))
        )

        return estimate, se

    def _lobo_estimate(self, total_est, block_ests):
        """
        Computing left-one-block-out estimates

        """
        lobo_est = []

        for block_est in block_ests:
            lobo_est_i = total_est.copy()
            lobo_est_i -= block_est
            lobo_est.append(lobo_est_i)

        return np.array(lobo_est)

    def _get_gene_cor_se(self, heri, heri_y2, gene_cor_y2, d, n1, n2):
        """
        Estimating standard error of two-sample genetic correlation estimates
        without sample overlap

        Parameters:
        ------------
        heri: an N by 1 vector of heritability estimates of images
        heri_y2: an 1 by 1 number of heritability estimate of a single trait
        gene_cor_y2: an N by 1 vector of genetic correlation estimates
                    between images and a single trait
        d: number of SNPs, or Tr(R\Omega)
        n1: sample size of images
        n2: sample size of the single trait

        Returns:
        ---------
        An N by N matrix of standard error estimates
        This estimator assumes no sample overlap

        """
        gene_cor_y2sq = gene_cor_y2 * gene_cor_y2
        gene_cor_y2sq1 = 1 - gene_cor_y2sq

        var = np.zeros(self.N, dtype=np.float32)
        var += gene_cor_y2sq / (2 * heri * heri) * d / n1**2
        var += gene_cor_y2sq / (2 * heri_y2 * heri_y2) * d / n2**2
        var += 1 / (heri * heri_y2) * d / (n1 * n2)
        var += gene_cor_y2sq1 / (heri * n1)
        var += gene_cor_y2sq1 / (heri_y2 * n2)

        return np.sqrt(var)


def format_heri(heri, heri_se, log):
    log.info("Removed out-of-bound results (if any)\n")
    chisq = (heri / heri_se) ** 2
    pv = chi2.sf(chisq, 1)
    data = {
        "INDEX": range(1, len(heri) + 1),
        "H2": heri,
        "SE": heri_se,
        "CHISQ": chisq,
        "P": pv,
    }
    output = pd.DataFrame(data)

    return output


def format_gene_cor_y2(heri, heri_se, gene_cor, gene_cor_se, log):
    log.info("Removed out-of-bound results (if any)\n")
    heri_chisq = (heri / heri_se) ** 2
    heri_pv = chi2.sf(heri_chisq, 1)
    gene_cor_chisq = (gene_cor / gene_cor_se) ** 2
    gene_cor_pv = chi2.sf(gene_cor_chisq, 1)
    data = {
        "INDEX": range(1, len(gene_cor) + 1),
        "H2": heri,
        "H2_SE": heri_se,
        "H2_CHISQ": heri_chisq,
        "H2_P": heri_pv,
        "GC": gene_cor,
        "GC_SE": gene_cor_se,
        "GC_CHISQ": gene_cor_chisq,
        "GC_P": gene_cor_pv,
    }
    output = pd.DataFrame(data)

    return output


def print_results_two(heri_gc, output, overlap):
    msg = "Heritability of the image\n"
    msg += "-------------------------\n"
    msg += (
        f"Mean heritability: {np.nanmean(output['H2']):.4f} "
        f"({np.nanmean(output['H2_SE']):.4f})\n"
    )
    msg += f"Median heritability: {np.nanmedian(output['H2']):.4f}\n"
    msg += f"Max heritability: {np.nanmax(output['H2']):.4f}\n"
    msg += f"Min heritability: {np.nanmin(output['H2']):.4f}\n"
    msg += "\n"

    chisq_y2_heri = (heri_gc.y2_heri[0] / heri_gc.y2_heri_se[0]) ** 2
    pv_y2_heri = chi2.sf(chisq_y2_heri, 1)
    msg += "Heritability of the non-imaging trait\n"
    msg += "-------------------------------------\n"
    msg += (
        f"Total observed scale heritability: {heri_gc.y2_heri[0]:.4f} "
        f"({heri_gc.y2_heri_se[0]:.4f})\n"
    )
    msg += f"Chi^2: {chisq_y2_heri:.4f}\n"
    msg += f"P: {pv_y2_heri:.4f}\n"
    msg += "\n"

    if overlap:
        msg += "Genetic correlation (with sample overlap)\n"
        msg += "-----------------------------------------\n"
    else:
        msg += "Genetic correlation (without sample overlap)\n"
        msg += "--------------------------------------------\n"
    msg += (
        f"Mean genetic correlation: {np.nanmean(output['GC']):.4f} "
        f"({np.nanmean(output['GC_SE']):.4f})\n"
    )
    msg += f"Median genetic correlation: {np.nanmedian(output['GC']):.4f}\n"
    msg += f"Max genetic correlation: {np.nanmax(output['GC']):.4f}\n"
    msg += f"Min genetic correlation: {np.nanmin(output['GC']):.4f}\n"

    return msg


def print_results_heri(heri_output):
    msg = "Heritability of the image\n"
    msg += "-------------------------\n"
    msg += (
        f"Mean heritability: {np.nanmean(heri_output['H2']):.4f} "
        f"({np.nanmean(heri_output['SE']):.4f})\n"
    )
    msg += f"Median heritability: {np.nanmedian(heri_output['H2']):.4f}\n"
    msg += f"Max heritability: {np.nanmax(heri_output['H2']):.4f}\n"
    msg += f"Min heritability: {np.nanmin(heri_output['H2']):.4f}\n"

    return msg


def print_results_gc(mean_gene_cor, min_gene_cor, mean_gene_cor_se):
    msg = "\n"
    msg += "Genetic correlation of the image\n"
    msg += "--------------------------------\n"
    msg += (
        f"Mean genetic correlation: {mean_gene_cor:.4f} " f"({mean_gene_cor_se:.4f})\n"
    )
    msg += f"Min genetic correlation: {min_gene_cor:.4f}\n"

    return msg


def run(args, log):
    check_input(args, log)

    # read LD matrices
    ld = LDmatrix(args.ld)
    log.info(f"Read LD matrix from {args.ld}")
    ld_inv = LDmatrix(args.ld_inv)
    log.info(f"Read LD inverse matrix from {args.ld_inv}")

    if ld.ldinfo.shape[0] != ld_inv.ldinfo.shape[0]:
        raise ValueError(
            (
                "the LD matrix and LD inverse matrix have different number of SNPs. "
                "It is highly likely that the files were misspecified or modified"
            )
        )
    if not np.equal(
        ld.ldinfo[["A1", "A2"]].values, ld_inv.ldinfo[["A1", "A2"]].values
    ).all():
        raise ValueError(
            "LD matrix and LD inverse matrix have different alleles for some SNPs"
        )
    log.info(f"{ld.ldinfo.shape[0]} SNPs read from LD matrix (and its inverse).")

    # read bases and ldr_cov
    bases = np.load(args.bases)
    log.info(f"{bases.shape[1]} bases read from {args.bases}")
    ldr_cov = np.load(args.ldr_cov)
    log.info(f"Read variance-covariance matrix of LDRs from {args.ldr_cov}")

    try:
        # read LDR gwas
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

        # check numbers of LDRs are the same
        if bases.shape[1] != ldr_cov.shape[0] or bases.shape[1] != ldr_gwas.n_gwas:
            raise ValueError(
                (
                    "inconsistent dimension in bases, variance-covariance matrix of LDRs, "
                    "and LDR summary statistics. "
                    "Try to use --n-ldrs"
                )
            )

        # read y2 gwas
        if args.y2_sumstats:
            y2_gwas = sumstats.read_sumstats(args.y2_sumstats)
            log.info(
                f"{y2_gwas.n_snps} SNPs read from non-imaging summary statistics {args.y2_sumstats}"
            )
        else:
            y2_gwas = None

        # get common snps from gwas, LD matrices, and keep_snps
        common_snps = CommonSNPs(
            ld,
            ld_inv,
            ldr_gwas,
            y2_gwas,
            args.extract,
            exclude_snps=args.exclude,
            threads=args.threads,
        )
        log.info(
            (
                f"{len(common_snps.common_snps)} SNPs common in these files with identical alleles. "
                "Extracting them from each file ..."
            )
        )

        # extract common snps in LD matrix
        ld.extract(common_snps.common_snps)
        ld_inv.extract(common_snps.common_snps)

        # extract common snps in summary statistics and do alignment
        ldr_gwas.extract_snps(ld.ldinfo["SNP"])  # extract snp id
        ldr_gwas.align_alleles(ld.ldinfo)  # get +/-

        if args.y2_sumstats:
            y2_gwas.extract_snps(ld.ldinfo["SNP"])
            y2_gwas.align_alleles(ld.ldinfo)
        log.info(f"Aligned genetic effects of summary statistics to the same allele.\n")

        log.info("Computing heritability and/or genetic correlation ...")
        if not args.y2_sumstats:
            heri_gc = OneSample(ldr_gwas, ld, ld_inv, bases, ldr_cov, args.threads)
            heri_output = format_heri(heri_gc.heri, heri_gc.heri_se, log)
            msg = print_results_heri(heri_output)
            log.info(f"{msg}")
            heri_output.to_csv(
                f"{args.out}_heri.txt",
                sep="\t",
                index=None,
                float_format="%.5e",
                na_rep="NA",
            )
            log.info(f"Saved the heritability results to {args.out}_heri.txt")

            if not args.heri_only:
                mean_gene_cor, min_gene_cor, mean_gene_cor_se = heri_gc.get_gene_cor_se(
                    args.out, args.threads
                )
                msg = print_results_gc(mean_gene_cor, min_gene_cor, mean_gene_cor_se)
                log.info(f"{msg}")
                log.info(f"Saved the genetic correlation results to {args.out}_gc.h5")
        else:
            heri_gc = TwoSample(
                ldr_gwas,
                ld,
                ld_inv,
                bases,
                ldr_cov,
                y2_gwas,
                args.threads,
                args.overlap,
            )
            gene_cor_y2_output = format_gene_cor_y2(
                heri_gc.heri,
                heri_gc.heri_se,
                heri_gc.gene_cor_y2,
                heri_gc.gene_cor_y2_se,
                log,
            )
            msg = print_results_two(heri_gc, gene_cor_y2_output, args.overlap)
            log.info(f"{msg}")
            gene_cor_y2_output.to_csv(
                f"{args.out}_gc.txt",
                sep="\t",
                index=None,
                float_format="%.5e",
                na_rep="NA",
            )
            log.info(f"Saved the genetic correlation results to {args.out}_gc.txt")
    finally:
        if 'ldr_gwas' in locals(): 
            ldr_gwas.close()
        if args.y2_sumstats and 'y2_gwas' in locals():
            y2_gwas.close()

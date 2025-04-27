import numpy as np
import pandas as pd
from scipy.stats import t
from tqdm import tqdm
from heig.sumstats import read_sumstats
from heig.voxelgwas import VGWAS, voxel_reader, recover_se_numba
from heig.herigc import CommonSNPs
import heig.input.dataset as ds

from numba import float64, int64
from numba.experimental import jitclass
from numba import prange

"""
Partition h2 for each voxel in an image
Only binary annotations are allowed

1. merge ldr summary statistics with annotations
2. split into 200 blocks, calculate x'x, (200, n_annot, n_annot)
3. reconstruct summary statistics for each voxel, calculate x'y, (200, n_annot, n_voxels)
4. calculate tau, prop_h2, and enrichment 

"""

spec = [
    ("ref_ld", float64[:, :]),
    ("w_ld", float64[:, :]),
    ("ldr_n", int64[:, :]),
    ("Nbar", float64),
    ("blocks", int64[:, :]),
    ("n_blocks", int64),
    ("n_annot", int64),
    ("n_coef", int64),
    ("overlap_matrix", int64[:, :]),
    ("M_annot", int64[:]),
    ("M_tot", float64),
    ("overlap_matrix_prop", float64[:, :]),
    ("overlap_matrix_diff", float64[:, :]),
    ("init_w", float64[:, :]),
    ("XwX_blocks", float64[:, :, :]),
    ("Xwy_blocks", float64[:, :]),
    ("XwX", float64[:, :]),
    ("Xwy", float64[:]),
    ("coef", float64[:]),
    ("coef_cov", float64[:, :]),
    ("coef_se", float64[:]),
    ("cat", float64[:]),
    ("cat_cov", float64[:, :]),
    ("cat_se", float64[:]),
    ("tot", float64),
    ("tot_cov", float64),
    ("tot_se", float64),
    ("prop", float64[:]),
    ("prop_cov", float64[:, :]),
    ("prop_se", float64[:]),
    ("enrichment", float64[:]),
    ("M_prop", float64[:]),
    ("intercept", float64),
    ("intercept_se", float64),
]


@jitclass(spec)
class PartitionHeritability:
    def __init__(self, ref_ld, w_ld, ldr_n, overlap_matrix, M_annot):
        """
        Parameters:
        ------------
        ref_ld (n_snp, n_annot): np.array of reference LD
        w_ld (n_snp, 1): np.array of regression weight
        ldr_n (n_snp, 1): np.array of sample size for each SNP
        overlap_matrix (n_annot, n_annot): np.array of overlap matrix
        M_annot (n_annot,): np.array of number of SNPs in each annotation

        """
        self.ref_ld = ref_ld
        self.w_ld = w_ld
        self.ldr_n = ldr_n
        self.Nbar = np.mean(self.ldr_n)
        self.blocks = self._split_blocks(ref_ld.shape[0])
        self.n_blocks = len(self.blocks)
        self.n_annot = ref_ld.shape[1]
        self.n_coef = self.n_annot + 1

        self.overlap_matrix = overlap_matrix
        self.M_annot = M_annot
        self.M_tot = np.sum(M_annot)
        self.overlap_matrix_prop = overlap_matrix / M_annot

    def setup(self):
        self.overlap_matrix_diff = self._get_overlap_matrix_diff()
        self.init_w = self._init_weights()
        self.XwX_blocks = self._block_XwX()

    @staticmethod
    def _split_blocks(n_snps, n_blocks=200):
        """
        Split SNPs into 200 blocks

        """
        block_size = n_snps // n_blocks
        blocks = np.zeros((n_blocks, 2), dtype=np.int64) 
        for i in range(n_blocks):
            start = i * block_size
            end = (i + 1) * block_size if i < n_blocks - 1 else n_snps
            blocks[i] = start, end
        return blocks
    
    # @njit(parallel=True)
    def _get_overlap_matrix_diff(self):
        M_tot = self.M_annot[0]
        overlap_matrix_diff = np.zeros((self.n_annot, self.n_annot))
        for i in prange(self.n_annot):
            if M_tot != self.M_annot[i]: 
                overlap_matrix_diff[i] = (
                    self.overlap_matrix[i] / self.M_annot[i] -
                    (self.M_annot - self.overlap_matrix[i]) / (M_tot - self.M_annot[i])
                )
        return overlap_matrix_diff
        
    def _init_weights(self):
        """
        Initialize weights for each annotation

        """
        x_tot = np.sum(self.ref_ld, axis=1).reshape(-1, 1)
        x_tot = np.maximum(x_tot, 1)
        hsq = 0.15 # 0.18884230401681382
        self.w_ld = np.maximum(self.w_ld, 1)
        c = hsq * self.ldr_n / self.M_tot
        het_w = 1 / (2 * (1 + c * x_tot) ** 2)
        oc_w = 1 / self.w_ld
        w = np.sqrt(het_w * oc_w)
        w = w / np.sum(w)
        return w
    
    # @njit(parallel=True, fastmath=True)
    def _block_XwX(self):
        """
        get XwX for each block

        """
        self.ref_ld = (self.ref_ld * self.ldr_n) / self.Nbar
        self.ref_ld = np.append(self.ref_ld, np.ones((self.ref_ld.shape[0], 1)), axis=1)
        self.ref_ld = self.ref_ld * self.init_w

        XwX_blocks = np.zeros((self.n_blocks, self.n_coef, self.n_coef))
        for i in prange(self.n_blocks):
            start, end = self.blocks[i]
            XwX_blocks[i] = self.dot(self.ref_ld[start:end].T, self.ref_ld[start:end])

        self.ref_ld = self.ref_ld * self.init_w  # Xw

        return XwX_blocks
    
    def run(self, y):
        """
        Run the partition heritability analysis

        Parameters:
        ------------
        y (n_snp,): np.array of chisq statistics

        """
        self.Xwy_blocks = self._block_Xwy(y)
        self.XwX, self.Xwy, total_coef = self._total_coef()
        lobo_coef = self._lobo_coef()
        pseudovalues = self.n_blocks * total_coef - (self.n_blocks - 1) * lobo_coef
        jknife_se, jknife_cov = self._jackknife(pseudovalues)

        self.coef, self.coef_cov, self.coef_se = self._coef(total_coef, jknife_cov)
        self.cat, self.cat_cov, self.cat_se = self._cat()
        self.tot, self.tot_cov, self.tot_se = self._tot()
        self.prop, self.prop_cov, self.prop_se = self._prop(lobo_coef)
        self.enrichment, self.M_prop = self._enrichment()
        self.intercept, self.intercept_se = self._intercept(total_coef, jknife_se)
        output = self._overlap_output()       
        
        return output

    @staticmethod
    def inv(A):
        L = np.linalg.cholesky(A)
        inv_L = np.linalg.inv(L)
        inv_A = inv_L.T @ inv_L
        return inv_A
    
    @staticmethod
    def dot(A, B):
        A = np.ascontiguousarray(A)
        B = np.ascontiguousarray(B)
        return np.dot(A, B)

    # @njit(parallel=True, fastmath=True)
    def _block_Xwy(self, y):
        """
        get Xwy for each block

        """
        Xwy_blocks = np.zeros((self.n_blocks, self.n_coef))
        for i in prange(self.n_blocks):
            start, end = self.blocks[i]
            Xwy_blocks[i] = self.dot(self.ref_ld[start:end].T, y[start:end])
        return Xwy_blocks

    def _total_coef(self):
        XwX = np.sum(self.XwX_blocks, axis=0)
        Xwy = np.sum(self.Xwy_blocks, axis=0)
        return XwX, Xwy, self.dot(self.inv(XwX), Xwy)
    
    # @njit(parallel=True, fastmath=True)
    def _lobo_coef(self):
        lobo_coef = np.zeros((self.n_blocks, self.n_coef))
        for i in prange(self.n_blocks):
            XwX_i = self.XwX - self.XwX_blocks[i]
            Xwy_i = self.Xwy - self.Xwy_blocks[i]
            lobo_coef[i] = self.dot(self.inv(XwX_i), Xwy_i)
        return lobo_coef
    
    def _jackknife(self, pseudovalues):
        """
        Jackknite estimator

        """
        jknife_cov = np.cov(pseudovalues.T, ddof=1) / self.n_blocks
        jknife_se = np.sqrt(np.diag(jknife_cov))
        # jknife_est = np.mean(pseudovalues, axis=0)
        return jknife_se, jknife_cov
    
    def _coef(self, total_coef, jknife_cov):
        coef = total_coef[0:-1]
        coef_cov = jknife_cov[0:-1, 0:-1]
        coef_se = np.sqrt(np.diag(coef_cov)) / self.Nbar 
        return coef, coef_cov, coef_se
    
    def _cat(self):
        cat = self.M_annot / self.Nbar * self.coef
        M_annot = (self.M_annot / self.Nbar).reshape(1, -1)
        cat_cov = self.dot(M_annot.T, M_annot) * self.coef_cov
        cat_se = np.sqrt(np.diag(cat_cov))
        return cat, cat_cov, cat_se
    
    def _tot(self):
        tot = np.sum(self.cat)
        tot_cov = np.sum(self.cat_cov)
        tot_se = np.sqrt(tot_cov)
        return tot, tot_cov, tot_se
    
    def _prop(self, lobo):
        lobo = lobo[:, 0:-1]
        numer_delete_vals = self.M_annot * lobo / self.Nbar
        denom_delete_vals = np.sum(numer_delete_vals, axis=1).reshape(-1, 1)
        denom_delete_vals = self.dot(denom_delete_vals, np.ones((1, self.n_annot)))
        prop = self.cat / self.tot

        pseudovalues = self.n_blocks * prop - (self.n_blocks - 1) * numer_delete_vals / denom_delete_vals
        jknife_se, jknife_cov = self._jackknife(pseudovalues)
        return prop, jknife_cov, jknife_se

    def _enrichment(self):
        M_prop = self.M_annot / self.M_tot
        enrichment = self.cat / self.M_annot / (self.tot / self.M_tot)
        return enrichment, M_prop
    
    def _intercept(self, total_coef, jknife_se):
        intercept = total_coef[-1]
        intercept_se = jknife_se[-1]
        return intercept, intercept_se

    def _overlap_output(self):
        M_tot = self.M_annot[0]
        prop_hsq_overlap = self.dot(self.overlap_matrix_prop, self.prop.T)
        prop_hsq_overlap_var = np.sum(self.dot(self.overlap_matrix_prop, self.prop_cov) * self.overlap_matrix_prop, axis=1)
        prop_hsq_overlap_var = np.maximum(prop_hsq_overlap_var, 1e-10)
        prop_hsq_overlap_se = np.sqrt(prop_hsq_overlap_var)

        prop_M_overlap = self.M_annot / M_tot
        enrichment = prop_hsq_overlap / prop_M_overlap
        enrichment_se = prop_hsq_overlap_se / prop_M_overlap

        diff_est = self.dot(self.overlap_matrix_diff, self.coef)
        diff_var = np.sum(self.dot(self.overlap_matrix_diff, self.coef_cov) * self.overlap_matrix_diff, axis=1)
        diff_se = np.sqrt(diff_var)
        diff_se = np.maximum(diff_se, 1e-10)

        self.coef = self.coef / self.Nbar

        return (
            prop_M_overlap, prop_hsq_overlap, prop_hsq_overlap_se, 
            enrichment, enrichment_se, diff_est, diff_se, self.coef, self.coef_se
        )


def organized_output(
        prop_M_overlap, prop_hsq_overlap, prop_hsq_overlap_se, 
        enrichment, enrichment_se, diff_est, diff_se, coef, coef_se, 
        annot_names, voxel_idx, n_blocks=200
):
    diff_p = 2 * t.sf(abs(diff_est / diff_se), n_blocks)
    df = pd.DataFrame({
        "Index": voxel_idx,
        "Category": annot_names,
        "Prop_SNPs": prop_M_overlap,
        "Prop_h2": prop_hsq_overlap,
        "Prop_h2_se": prop_hsq_overlap_se,
        "Enrichment": enrichment,
        "Enrichment_se": enrichment_se,
        "Enrichment_p": diff_p,
        "Coefficient": coef,
        "Coefficient_se": coef_se, 
    })

    return df


def check_input(args, log):
    # required arguments
    if args.ldr_sumstats is None:
        raise ValueError("--ldr-sumstats is required")
    if args.bases is None:
        raise ValueError("--bases is required")
    if args.ldr_cov is None:
        raise ValueError("--ldr-cov is required")
    if args.ref_ld_chr is None:
        raise ValueError("--ref-ld-chr is required")
    if args.w_ld_chr is None:
        raise ValueError("--w-ld-chr is required")
    if args.frqfile_chr is None:
        raise ValueError("--frqfile-chr is required")


def run(args, log):
    # checking input
    check_input(args, log)

    # reading data
    ldr_cov = np.load(args.ldr_cov)
    log.info(f"Read variance-covariance matrix of LDRs from {args.ldr_cov}")
    bases = np.load(args.bases)
    log.info(f"{bases.shape[1]} bases read from {args.bases}")

    try:
        ldr_sumstats = read_sumstats(args.ldr_sumstats)
        log.info(
            f"{ldr_sumstats.n_snps} SNPs read from LDR summary statistics {args.ldr_sumstats}"
        )
        
        ref_ld, annot_names = ds.read_ld(args.ref_ld_chr, read_name=True)
        log.info(
            (
                f"{ref_ld.shape[0]} SNPs and {len(annot_names)} annotations "
                f"read from reference panel {args.ref_ld_chr}"
            )
        )

        w_ld, _ = ds.read_ld(args.w_ld_chr, read_name=False)
        log.info(
            f"{w_ld.shape[0]} SNPs read from regression weight {args.w_ld_chr}"
        )
        
        # extract common SNPs
        common_snps = CommonSNPs(
            ldr_sumstats, 
            ref_ld, 
            w_ld, 
            args.extract, 
            exclude_snps=args.exclude, 
            threads=args.threads,
            match_alleles=False,
        )
        log.info(
            (
                f"{len(common_snps.common_snps)} SNPs common in these files. "
                "Extracting them from each file ..."
            )
        )

        # ldr_sumstats.extract_snps(common_snps.common_snps)
        ref_ld = common_snps.common_snps.merge(ref_ld, on="SNP").iloc[:, 1:].values
        w_ld = common_snps.common_snps.merge(w_ld, on="SNP").iloc[:, 1:].values

        # read overlap matrix
        overlap_matrix = ds.read_ld_annot(args.ref_ld_chr, args.frqfile_chr)
        M_annot = overlap_matrix[0]
        log.info(f"Read overlap matrix from {args.ref_ld_chr}")
        # overlap_matrix = np.load('/work/users/o/w/owenjf/image_genetics/methods/package_pub/test_output/partition_h2/overlap_matrix.npy')
        # M_annot = overlap_matrix[0]

        # keep selected LDRs
        if args.n_ldrs is not None:
            bases, ldr_cov, ldr_sumstats, _ = ds.keep_ldrs(
                args.n_ldrs, bases, ldr_cov, ldr_sumstats
            )
            log.info(f"Keeping the top {args.n_ldrs} LDRs.")

        if bases.shape[1] != ldr_cov.shape[0] or bases.shape[1] != ldr_sumstats.n_gwas:
            raise ValueError(
                (
                    "inconsistent dimension for bases, variance-covariance matrix of LDRs, "
                    "and LDR summary statistics. "
                    "Try to use --n-ldrs"
                )
            )

        # select voxels
        if args.voxels is not None:
            if np.max(args.voxels) + 1 <= bases.shape[0] and np.min(args.voxels) >= 0:
                log.info(f"{len(args.voxels)} voxel(s) included.")
            else:
                raise ValueError("--voxels index (one-based) out of range")
        else:
            args.voxels = np.arange(bases.shape[0])

        # doing analysis
        log.info(f"\nPartitioning heritability ...")
        snp_idxs = ldr_sumstats.snpinfo["SNP"].isin(common_snps.common_snps["SNP"]).to_numpy()
        ldr_n = np.array(ldr_sumstats.snpinfo["N"][snp_idxs]).reshape(-1, 1)
        partition_h2 = PartitionHeritability(
            ref_ld, w_ld, ldr_n, overlap_matrix, M_annot
        )
        partition_h2.setup()
        vgwas = VGWAS(bases, ldr_cov, ldr_sumstats, snp_idxs, ldr_n, args.threads)

        all_df = []
        for voxel_idxs in tqdm(
            voxel_reader(np.sum(snp_idxs), args.voxels),
            desc=f"Reconstructing GWAS for {len(args.voxels)} voxel(s) in batch",
        ):
            voxel_beta = vgwas.recover_beta(voxel_idxs, args.threads)
            # voxel_se = vgwas.recover_se(voxel_idxs, voxel_beta)
            voxel_se = recover_se_numba(
                voxel_idxs, voxel_beta, vgwas.bases, vgwas.ldr_cov, vgwas.ztz_inv, vgwas.n
            )
            voxel_chisq = ((voxel_beta / voxel_se) ** 2).astype(np.float64)
            for i, voxel_idx in enumerate(voxel_idxs):
                output = partition_h2.run(voxel_chisq[:, i])
                df = organized_output(
                    *output, 
                    annot_names=annot_names, 
                    voxel_idx=voxel_idx+1,
                    n_blocks=partition_h2.n_blocks)
                all_df.append(df)

        all_df = pd.concat(all_df, ignore_index=True)
        for _, annot_df in all_df.groupby("Category"):
            category = annot_df.iloc[0]["Category"]
            annot_df = annot_df.drop(columns=["Category"])
            annot_df.to_csv(
                f"{args.out}_partition_h2_{category}.txt",
                index=False,
                sep="\t",
                header=True,
                float_format="%.5e",
            )
        log.info(
            f"Saved partition heritability for to {args.out}_partition_h2_*.txt"
        ) 

    finally:
        if "ldr_sumstats" in locals():
            ldr_sumstats.close()
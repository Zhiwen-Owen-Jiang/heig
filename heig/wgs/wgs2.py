import h5py
import logging
import numpy as np
import hail as hl
import heig.input.dataset as ds
from concurrent.futures import ThreadPoolExecutor
from heig.wgs.relatedness import LOCOpreds
from heig.wgs.null import NullModel
from heig.wgs.utils import *
from heig.wgs.mt import SparseGenotype
from heig.utils import log_execution_time
from scipy.sparse import csr_matrix


"""
Generate rare variant summary statistics:
1. LDR score statistics: U = Z'(I-M)\Xi, where relatedness has been removed from \Xi
2. Spark matrix of inner product of Z: Z'Z
3. low-dimensional format of Z'X(X'X)^{-1/2} = Z'UV', where X = UDV'
4. variance of each voxel under the null model: sigma^2

Allow to input a chromosome and save the summary statistics

"""

def get_rv_sumstats_part1(bases, resid_ldr, locus, vset, out_dir, loco_preds=None):
    """
    Generating rare variants summary statistics (part1) specific to images

    Parameters:
    ------------
    bases: (N, r) np.array, functional bases
    resid_ldr: (n, r) np.array, LDR residuals
    locus: a hail.Table including locus, grch37, variant_type
    vset: (m, n) csr_matrix of genotype
    out_dir: prefix of output directory
    loco_preds: a LOCOpreds instance of loco predictions
        loco_preds.data_reader(j) returns loco preds for chrj with matched subjects

    """
    # bases
    bases = bases.astype(np.float32)

    # adjust for relatedness
    if loco_preds is not None:
        # extract chr
        chr = locus.aggregate(hl.agg.take(locus.locus.contig, 1)[0])
        if locus.locus.dtype.reference_genome.name == "GRCh38":
            chr = int(chr.replace("chr", ""))
        else:
            chr = int(chr)
        resid_ldr = resid_ldr - loco_preds.data_reader(chr) # (I-M)\Xi, (n, r)
    resid_ldr = resid_ldr.astype(np.float32)

    # inner product of LDR residuals
    inner_ldr = np.dot(resid_ldr.T, resid_ldr).astype(np.float32) # \Xi'(I-M)\Xi, (r, r)

    # half ldr score
    half_ldr_score = vset @ resid_ldr  # Z'(I-M)\Xi, (m, r)
    n_variants, n_subs = vset.shape

    # save sumstats
    with h5py.File(f"{out_dir}_part1_rv_sumstats.h5", "w") as file:
        file.create_dataset("bases", data=bases, dtype="float32")
        file.create_dataset("inner_ldr", data=inner_ldr, dtype="float32")
        file.create_dataset("half_ldr_score", data=half_ldr_score, dtype="float32")
        file.attrs["n_subs"] = n_subs
        file.attrs["n_variants"] = n_variants


def get_rv_sumstats_part2(locus, maf, mac, covar, vset, bandwidth, threads, out_dir):
    """
    Generating rare variants summary statistics (part2) not specific to images

    Parameters:
    ------------
    locus: a hail.Table including locus, grch37, variant_type
    maf: (m, ) np.array, minor allele frequency
    mac: (N, r) np.array, minor allele count
    covar: (n, p) np.array, covariates
    vset: (m, n) csr_matrix of genotype
    bandwidth: bandwidth of the banded LD matrix
    threads: number of threads
    out_dir: prefix of output directory
    
    """
    # X = UDV', X'(X'X)^{-1/2} = UV'
    covar_U, _, covar_Vt = np.linalg.svd(covar, full_matrices=False)
    half_covar_proj = np.dot(covar_U, covar_Vt).astype(np.float32)
    n_subs, n_covars = covar.shape

    # Z'X(X'X)^{-1/2} = Z'UV', where X = UDV'
    vset_half_covar_proj = vset @ half_covar_proj  # Z'UV', (m, p)

    # Z'Z (m, m)
    positions = np.array(locus.locus.position.collect())
    ld = SparseBandedLD(vset, positions, bandwidth, threads)
    n_variants = vset.shape[0]

    with h5py.File(f"{out_dir}_part2_data.h5", "w") as file:
        file.create_dataset("maf", data=maf, dtype="float32")
        file.create_dataset("mac", data=mac, dtype="uint16")
        # file.create_dataset("is_rare", data=is_rare)
        file.create_dataset(
            "vset_half_covar_proj", data=vset_half_covar_proj, dtype="float32"
        )
        file.create_dataset(
            "vset_ld_diag", data=ld.data[0], dtype="uint16"
        )
        file.create_dataset(
            "vset_ld_data", data=ld.data[1], dtype="uint16"
        )
        file.create_dataset("vset_ld_row", data=ld.data[2], dtype="int32")
        file.create_dataset("vset_ld_col", data=ld.data[3], dtype="int32")
        file.create_dataset("vset_ld_shape", data=ld.data[4], dtype="int32")
        file.create_dataset("block_size", data=ld.block_size, dtype="int32")
        file.attrs["n_subs"] = n_subs
        file.attrs["n_covars"] = n_covars
        file.attrs["n_variants"] = n_variants
        file.attrs["bandwidth"] = ld.bandwidth

    locus.write(f"{out_dir}_part2_locus_info.ht", overwrite=True)


class SparseBandedLD:
    def __init__(self, vset, positions, bandwidth, threads=1):
        self.vset = vset
        self.positions = positions
        self.bandwidth = bandwidth
        self.threads = threads
        self.block_size = self._get_block_size()
        self.blocks = self._get_blocks()
        self.data = self._sparse_banded()

    @staticmethod
    def _find_loc(positions, target):
        """
        Find index for a specific position, considering potential duplicates in positions 
        
        """
        l = 0
        r = len(positions) - 1
        while l <= r:
            mid = (l + r) // 2
            if positions[mid] > target:
                r = mid - 1
            else:
                l = mid + 1
        return l

    def _get_block_size(self):
        """
        For each variant, get the number of variants in the block starting with the variant
        
        """
        block_size = list()
        for start_idx, start in enumerate(self.positions):
            end_idx = self._find_loc(self.positions, start + self.bandwidth)
            block_size.append(end_idx - start_idx)
        block_size.append(0)
        return block_size

    def _get_blocks(self):
        """
        Get the shape of data (nrows, ncols) needed to calculate each block 
        
        """
        blocks = list()
        start = 0
        for i in range(len(self.block_size) - 1):
            if i == len(self.block_size) - 2 or self.block_size[i] <= self.block_size[i+1]:
                # blocks.append((i - start + 1, self.block_size[start]))
                blocks.append((start, i - start + 1, self.block_size[start]))
                start = i + 1
        return blocks
    
    def _sparse_banded(self):
        """
        Create a sparse banded LD matrix by blocks

        """
        diagonal_data = list()
        banded_data = list()
        banded_row = list()
        banded_col = list()
        n_variants = self.vset.shape[0]
        # start = 0

        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            results = executor.map(lambda block: self._process_block(block), self.blocks)
            
        for diagonal, row, col, banded in results:
            diagonal_data.append(diagonal)
            banded_row.append(row)
            banded_col.append(col)
            banded_data.append(banded)

        # for block in self.blocks:
        #     start = block[0]
        #     end1 = start + block[1]
        #     end2 = start + block[2]
        #     vset_block1 = self.vset[start:end1].astype(np.uint16)
        #     vset_block2 = self.vset[start:end2].astype(np.uint16)
        #     ld_rec = vset_block1 @ vset_block2.T
        #     ld_rec_row, ld_rec_col = ld_rec.nonzero()
        #     ld_rec_row += start
        #     ld_rec_col += start
        #     ld_rec_data = ld_rec.data
        #     # start = end1

        #     diagonal_data.append(ld_rec_data[ld_rec_row == ld_rec_col])
        #     mask = ld_rec_col > ld_rec_row
            
        #     banded_row.append(ld_rec_row[mask])
        #     banded_col.append(ld_rec_col[mask])
        #     banded_data.append(ld_rec_data[mask])

        diagonal_data = np.concatenate(diagonal_data)
        banded_row = np.concatenate(banded_row)
        banded_col = np.concatenate(banded_col)
        banded_data = np.concatenate(banded_data)
        shape = np.array([n_variants, n_variants])
        
        if len(diagonal_data) != n_variants:
            raise ValueError('0 is not allowed in the diagonal of LD matrix')

        return diagonal_data, banded_data, banded_row, banded_col, shape
    
    def _process_block(self, block):
        start = block[0]
        end1 = start + block[1]
        end2 = start + block[2]
        vset_block1 = self.vset[start:end1].astype(np.uint16)
        vset_block2 = self.vset[start:end2].astype(np.uint16)
        ld_rec = vset_block1 @ vset_block2.T
        ld_rec_row, ld_rec_col = ld_rec.nonzero()
        ld_rec_row += start
        ld_rec_col += start
        ld_rec_data = ld_rec.data

        diagonal_data = ld_rec_data[ld_rec_row == ld_rec_col]
        mask = ld_rec_col > ld_rec_row
        banded_row = ld_rec_row[mask]
        banded_col = ld_rec_col[mask]
        banded_data = ld_rec_data[mask]

        return diagonal_data, banded_row, banded_col, banded_data
    

class RVsumstats:
    """
    Reading and processing rare variant summary statistics for analysis

    """

    def __init__(self, part1_prefix, part2_prefix):
        self.locus = hl.read_table(f"{part2_prefix}_locus_info.ht").key_by("locus", "alleles")
        self.locus = self.locus.add_index("idx")
        self.geno_ref = self.locus.reference_genome.collect()[0]
        self.variant_type = self.locus.variant_type.collect()[0]

        with h5py.File(f"{part1_prefix}_rv_sumstats.h5", "r") as file:
            self.bases = file["bases"][:]  # (N, r)
            self.inner_ldr = file["inner_ldr"][:]  # (r, r)
            self.half_ldr_score = file["half_ldr_score"][:]
            
        with h5py.File(f"{part2_prefix}_data.h5", "r") as file:
            self.maf = file["maf"][:]
            self.mac = file["mac"][:]
            self.vset_half_covar_proj = file["vset_half_covar_proj"][:]
            self.block_size = file["block_size"][:]
            self.bandwidth = file.attrs["bandwidth"]
            self.n_variants = file.attrs["n_variants"]
            self.n_subs = file.attrs["n_subs"]
            self.n_covars = file.attrs["n_covars"]

            vset_ld_diag = file["vset_ld_diag"][:]
            vset_ld_data = file["vset_ld_data"][:]
            vset_ld_row = file["vset_ld_row"][:]
            vset_ld_col = file["vset_ld_col"][:]
            vset_ld_shape = file["vset_ld_shape"][:]
            
        if self.half_ldr_score.shape[0] != len(self.maf):
            raise ValueError('part1 and part2 data has different number of variants')

        self.vset_ld = self._reconstruct_vset_ld(
            vset_ld_diag, vset_ld_data, vset_ld_row, vset_ld_col, vset_ld_shape
        )

        self.voxel_idxs = np.arange(self.bases.shape[0])
        self.logger = logging.getLogger(__name__)

    def calculate_var(self):
        self.var = np.sum(np.dot(self.bases, self.inner_ldr) * self.bases, axis=1)
        self.var /= self.n_subs - self.n_covars  # (N, )

    @staticmethod
    def _reconstruct_vset_ld(diag, data, row, col, shape):
        """
        Reconstructing a sparse matrix.

        Parameters:
        ------------
        diag: diagonal data
        data: banded data
        row: row index of upper band
        col: col index of lower band
        shape: sparse matrix shape

        Returns:
        ---------
        full_matrix: sparse banded matrix.

        """
        lower_row = col
        lower_col = row
        diag_row_col = np.arange(shape[0])

        full_row = np.concatenate([row, lower_row, diag_row_col])
        full_col = np.concatenate([col, lower_col, diag_row_col])
        full_data = np.concatenate([data, data, diag])

        full_matrix = csr_matrix((full_data, (full_row, full_col)), shape=shape)

        return full_matrix

    def select_ldrs(self, n_ldrs=None):
        if n_ldrs is not None:
            if n_ldrs <= self.bases.shape[1] and n_ldrs <= self.half_ldr_score.shape[1]:
                self.bases = self.bases[:, :n_ldrs]
                self.half_ldr_score = self.half_ldr_score[:, :n_ldrs]
                self.inner_ldr = self.inner_ldr[:n_ldrs, :n_ldrs]
                self.logger.info(f"Keeping the top {n_ldrs} LDRs.")
            else:
                raise ValueError("--n-ldrs is greater than #LDRs")

    def select_voxels(self, voxel_idxs=None):
        if voxel_idxs is not None:
            if np.max(voxel_idxs) < self.bases.shape[0]:
                self.voxel_idxs = voxel_idxs
                self.bases = self.bases[voxel_idxs]
                self.logger.info(f"{len(voxel_idxs)} voxel(s) included.")
            else:
                raise ValueError("--voxels index (one-based) out of range")

    def extract_exclude_locus(self, extract_locus, exclude_locus, extract_chrs):
        """
        Extracting and excluding variants by locus

        Parameters:
        ------------
        extract_locus: a hail.set of locus
        exclude_locus: a hail.set of locus
        extract_chrs: a set of unique chromosomes

        """
        if extract_locus is not None and extract_chrs is not None:
            # self.locus = self.locus.filter(hl.is_defined(extract_locus[self.locus.locus]))
            filter_chrs = hl.any(lambda c: self.locus.locus.contig == c, hl.set(extract_chrs))
            self.locus = self.locus.filter(filter_chrs)
            self.locus = self.locus.filter(extract_locus.contains(self.locus.locus))
        if exclude_locus is not None:
            # self.locus = self.locus.filter(~hl.is_defined(exclude_locus.contains(self.locus.locus)))
            self.locus = self.locus.filter(~exclude_locus.contains(self.locus.locus))

    def extract_chr_interval(self, chr_interval=None):
        """
        Extacting a chr interval

        Parameters:
        ------------
        chr_interval: chr interval to extract

        """
        if chr_interval is not None:
            chr, start, end = parse_interval(chr_interval, self.geno_ref)
            interval = hl.locus_interval(
                chr, start, end, reference_genome=self.geno_ref, includes_end=True
            )
            self.locus = self.locus.filter(interval.contains(self.locus.locus))

    def extract_maf(self, maf_min=None, maf_max=None):
        """
        Extracting variants by MAF

        """
        if maf_min is not None:
            variant_idx = np.arange(len(self.maf))
            keep_variant_idx = set(variant_idx[self.maf > maf_min])
            self.locus = self.locus.filter(
                hl.literal(keep_variant_idx).contains(self.locus.idx)
            )
        if maf_max is not None:
            variant_idx = np.arange(len(self.maf))
            keep_variant_idx = set(variant_idx[self.maf <= maf_max])
            self.locus = self.locus.filter(
                hl.literal(keep_variant_idx).contains(self.locus.idx)
            )
            
    def extract_mac(self, mac_min=None, mac_max=None):
        """
        Extracting variants by MAC

        """
        if mac_min is not None:
            variant_idx = np.arange(len(self.mac))
            keep_variant_idx = set(variant_idx[self.mac > mac_min])
            self.locus = self.locus.filter(
                hl.literal(keep_variant_idx).contains(self.locus.idx)
            )
        if mac_max is not None:
            variant_idx = np.arange(len(self.mac))
            keep_variant_idx = set(variant_idx[self.mac <= mac_max])
            self.locus = self.locus.filter(
                hl.literal(keep_variant_idx).contains(self.locus.idx)
            )

    def annotate(self, annot):
        """
        Annotating functional annotations to locus
        ensuring no NA in annotations

        """
        if annot is not None:
            self.locus = self.locus.annotate(annot=annot[self.locus.key])
            self.locus = self.locus.filter(hl.is_defined(self.locus.annot))
        self.locus = self.locus.cache()
        return self.locus

    def parse_data(self, numeric_idx):
        """
        Parsing data for analysis

        Parameters:
        ------------
        numeric_idx: a list of numeric indices,
        which are collected from `idx` in self.locus after filtering by annotations.

        Returns:
        ---------
        half_ldr_score: Z'(I-M)\Xi
        cov_mat: Z'(I-M)Z
        maf: a np.array of MAF
        mac: a np.array of MAC

        """
        # if max(numeric_idx) - min(numeric_idx) > self.bandwidth:
        if numeric_idx[-1] - numeric_idx[0] >= self.block_size[numeric_idx[0]]:
            self.logger.info("WARNING: the variant set has exceeded the bandwidth of LD matrix.")
            return None, None, None, None
        half_ldr_score = np.array(self.half_ldr_score[numeric_idx])
        vset_half_covar_proj = np.array(self.vset_half_covar_proj[numeric_idx])
        vset_ld = self.vset_ld[numeric_idx][:, numeric_idx]
        cov_mat = np.array((vset_ld - vset_half_covar_proj @ vset_half_covar_proj.T))
        maf = self.maf[numeric_idx]
        mac = self.mac[numeric_idx]
        # is_rare = mac < mac_thresh

        return half_ldr_score, cov_mat, maf, mac


def get_interval(locus):
    """
    Extracting chr, start position, end position, and #variants
    locus has >= 2 variants

    Parameters:
    ------------
    locus: a hail.Table of annotated locus info

    """
    all_locus = locus.locus.collect()
    chr = all_locus[0].contig
    start = all_locus[0].position
    end = all_locus[-1].position

    return chr, start, end


def extract_chr_interval(locus, gene_name, chr_interval, geno_ref, log):
    """
    Extacting a chr interval

    Parameters:
    ------------
    locus: a hail.Table of annotated locus info
    chr_interval: chr interval to extract

    """
    chr, start, end = parse_interval(chr_interval, geno_ref)
    interval = hl.locus_interval(chr, start, end, reference_genome=geno_ref, includes_end=True)
    locus = locus.filter(interval.contains(locus.locus))
    locus = locus.cache()
    n_variants = locus.count()

    if n_variants <= 1:
        log.info(f"\nSkipping gene {gene_name} (< 2 variants).")
        return None
    else:
        log.info(
            f"\n{n_variants} variants in gene {gene_name} overlapping in the summary statistics and annotations."
        )
    return locus


def check_input(args, log):
    # required arguments
    if args.sparse_genotype is None:
        raise ValueError("--sparse-genotype is required")
    if args.spark_conf is None:
        raise ValueError("--spark-conf is required")
    if args.null_model is None:
        raise ValueError("--null-model is required")

    if args.variant_type is None:
        args.variant_type = "variant"
        log.info(f"Set --variant-type as default 'variant'.")

    if args.maf_max is None:
        if args.maf_min is not None and args.maf_min < 0.01 or args.maf_min is None:
            args.maf_max = 0.01
            log.info(f"Set --maf-max as default 0.01")

    if args.bandwidth is None:
        args.bandwidth = 3000000
        log.info("Set --bandwidth as default 3000000 (3 MB)")
    elif args.bandwidth <= 1:
        raise ValueError("--bandwidth must be greater than 1")


def run(args, log):
    # checking if input is valid
    check_input(args, log)
    try:
        init_hail(args.spark_conf, args.grch37, args.out, log)

        # reading data and selecting LDRs
        log.info(f"Read null model from {args.null_model}")
        null_model = NullModel(args.null_model)
        null_model.select_ldrs(args.n_ldrs)

        # reading sparse genotype data
        sparse_genotype = SparseGenotype(args.sparse_genotype)
        log.info(f"Read sparse genotype data from {args.sparse_genotype}")
        log.info(f"{sparse_genotype.vset.shape[1]} subjects and {sparse_genotype.vset.shape[0]} variants.")

        # read loco preds
        if args.loco_preds is not None:
            log.info(f"Read LOCO predictions from {args.loco_preds}")
            loco_preds = LOCOpreds(args.loco_preds)
            if args.n_ldrs is not None:
                loco_preds.select_ldrs((0, args.n_ldrs))
            if loco_preds.ldr_col[1] - loco_preds.ldr_col[0] != null_model.n_ldrs:
                raise ValueError(
                    (
                        "inconsistent dimension in LDRs and LDR LOCO predictions. "
                        "Try to use --n-ldrs"
                    )
                )
            common_ids = ds.get_common_idxs(
                sparse_genotype.ids.index,
                null_model.ids,
                loco_preds.ids,
                args.keep,
            )
        else:
            common_ids = ds.get_common_idxs(
                sparse_genotype.ids.index, null_model.ids, args.keep
            )
        common_ids = ds.remove_idxs(common_ids, args.remove)

        # log.info(f"Processing sparse genetic data ...")
        if args.extract_locus is not None:
            args.extract_locus, unique_chrs = read_extract_locus(args.extract_locus, args.grch37, log)
        else:
            unique_chrs = None
        if args.exclude_locus is not None:
            args.exclude_locus = read_exclude_locus(args.exclude_locus, args.grch37, log)

        sparse_genotype.keep(common_ids)
        sparse_genotype.extract_variant_type(args.variant_type)
        sparse_genotype.extract_exclude_locus(args.extract_locus, args.exclude_locus, unique_chrs)
        sparse_genotype.extract_chr_interval(args.chr_interval)
        sparse_genotype.extract_maf(args.maf_min, args.maf_max)
        sparse_genotype.extract_mac(args.mac_min, args.mac_max)

        # extract and align subjects with the genotype data
        null_model.keep(common_ids)
        null_model.remove_dependent_columns()
        log.info(f"{len(common_ids)} common subjects in the data.")
        log.info(
            (
                f"{null_model.covar.shape[1]} fixed effects in the covariates (including the intercept) "
                "after removing redundant effects.\n"
            )
        )

        if args.loco_preds is not None:
            loco_preds.keep(common_ids)
        else:
            loco_preds = None

        vset, locus, maf, mac = sparse_genotype.parse_data()
        log.info(f"{vset.shape[0]} variants after processing.")
        log.info("Computing summary statistics (part1) specific to images ...")
        get_rv_sumstats_part1(
            null_model.bases, null_model.resid_ldr, locus, vset, args.out, loco_preds
        )

        log.info(
            (
                f"\nSaved summary statistics (part1) to {args.out}_part1_rv_sumstats.h5"
            )
        )

        if args.make_part2:
            log.info(f"\nComputing summary statitics (part2) not specific to images ...")
            get_rv_sumstats_part2(
                locus, maf, mac, null_model.covar, vset, args.bandwidth, args.threads, args.out
            )
            log.info(
                (
                    f"\nSaved summary statitics (part2) to\n"
                    f"{args.out}_part2_data.h5\n"
                    f"{args.out}_part2_locus_info.ht"
                )
            )

    finally:
        if "loco_preds" in locals() and args.loco_preds is not None:
            loco_preds.close()

        clean(args.out)

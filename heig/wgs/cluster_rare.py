import time
import logging
import numpy as np
import hail as hl
# from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from scipy.sparse import csr_matrix
import heig.input.dataset as ds
from heig.wgs.null import NullModel
from heig.wgs.relatedness import LOCOpreds
from heig.wgs.vsettest import VariantSetTest
from heig.wgs.mt import SparseGenotype
from heig.wgs.coding import Coding
from heig.wgs.utils import PermDistribution
from heig.utils import find_loc
from heig.wgs.utils import *


"""
cluster inference

1. generate a permutation sample from null model
2. compute rv summary statistics
3. split summary statistics into random sets, permute sets?
4. do analysis using STAAR
5. count cluster size

input:
1. null model
2. sparse genotype
3. sig threshold
4. number of permutation samples

output:
1. null distribution of cluster size 

TODO:
1. as of (v1.3.0), cluster analysis only supports FAVOR annot and snv
    
"""


class RVcluster:
    """
    Computing cluster size (i.e., number of associated voxels)
    for variant sets in a permutation sample

    1. compute rare variant summary statistics
    2. sliding window analysis
    3. count cluster size for each sample-variant-set pair for multiple tests

    """

    def __init__(
            self, 
            null_model, 
            vset,
            chr, 
            gene_numeric_idxs,
            phred_cate,
            annot_name,
            maf,
            mac,
            tests,
            perm,
            cmac_bins_count,
            sig_thresh=2.5e-6, 
            mac_thresh=10,
            threads=1,
            loco_preds=None,
            voxels=None
        ):
        """
        Parameters:
        ------------
        null_model: a NullModel instance
        vset: (m, n) csr_matrix of genotype
        chr: chromosome
        gene_numeric_idxs: a dict of variant idxs for genes
        phred_cate: functional annotations
        annot_name: names of annotations
        maf: a np.array of MAF
        mac: a np.array of MAC
        tests: a list of rv tests
        perm: an instance of PermDistribution
        cmac_bins_count: a dict of number of genes for each cMAC bin
        sig_thresh: significant threshold
        mac_thresh: a MAC threshold to denote ultrarare variants for ACAT-V
        threads: number of threads
        loco_preds: a LOCOpreds instance of loco predictions
            loco_preds.data_reader(j) returns loco preds for chrj with matched subjects
        voxels: a np.array of voxel idxs (0-based), may be None

        """
        self.bases = null_model.bases.astype(np.float32)
        self.vset = vset
        self.numeric_idx_list = gene_numeric_idxs
        self.phred_cate = phred_cate
        self.annot_name = annot_name
        self.maf = maf
        self.mac = mac
        self.tests = tests
        self.perm = perm
        self.cmac_bins_count = cmac_bins_count
        self.all_bins = list(self.cmac_bins_count.keys())
        self.sig_thresh = sig_thresh
        self.mac_thresh = mac_thresh
        self.threads = threads
        self.n_variants, self.n_subs = self.vset.shape
        self.n_covars = null_model.covar.shape[1]
        self.logger = logging.getLogger(__name__)
        
        covar_U, _, covar_Vt = np.linalg.svd(null_model.covar, full_matrices=False)
        half_covar_proj = np.dot(covar_U, covar_Vt).astype(np.float32)
        self.vset_half_covar_proj = self.vset @ half_covar_proj

        if loco_preds is not None:
            self.resid_ldr = null_model.resid_ldr - loco_preds.data_reader(chr)
        else:
            self.resid_ldr = null_model.resid_ldr
        inner_ldr = np.dot(self.resid_ldr.T, self.resid_ldr).astype(np.float32)
        self.var = np.sum(np.dot(self.bases, inner_ldr) * self.bases, axis=1)
        self.var /= self.n_subs - self.n_covars  # (N, )

        if voxels is None:
            self.voxels = np.arange(self.bases.shape[0])
        else:
            self.voxels = voxels

        self.vset_ld = self._get_band_ld_matrix()
        self.vset_test = VariantSetTest(self.bases, self.var, self.perm, self.voxels)
        
    def _get_band_ld_matrix(self):
        """
        Creating a banded sparse LD matrix with the bandwidth being
        the length of the largest gene
        
        """
        bandwidth = max([len(v) for _, v in self.numeric_idx_list.items()])
        diagonal_data = list()
        banded_data = list()
        banded_row = list()
        banded_col = list()

        for start in range(0, self.n_variants, bandwidth):
            end1 = start + bandwidth
            end2 = end1 + bandwidth
            vset_block1 = self.vset[start:end1].astype(np.uint16)
            vset_block2 = self.vset[start:end2].astype(np.uint16)
            ld_rec = vset_block1 @ vset_block2.T
            ld_rec_row, ld_rec_col = ld_rec.nonzero()
            ld_rec_row += start
            ld_rec_col += start
            ld_rec_data = ld_rec.data

            diagonal_data.append(ld_rec_data[ld_rec_row == ld_rec_col])
            mask = (np.abs(ld_rec_row - ld_rec_col) <= bandwidth) & (
                ld_rec_col > ld_rec_row
            )
            banded_row.append(ld_rec_row[mask])
            banded_col.append(ld_rec_col[mask])
            banded_data.append(ld_rec_data[mask])

        diagonal_data = np.concatenate(diagonal_data)
        banded_row = np.concatenate(banded_row)
        banded_col = np.concatenate(banded_col)
        banded_data = np.concatenate(banded_data)
        shape = np.array([self.n_variants, self.n_variants])

        lower_row = banded_col
        lower_col = banded_row
        diag_row_col = np.arange(shape[0])

        full_row = np.concatenate([banded_row, lower_row, diag_row_col])
        full_col = np.concatenate([banded_col, lower_col, diag_row_col])
        full_data = np.concatenate([banded_data, banded_data, diagonal_data])

        vset_ld = csr_matrix((full_data, (full_row, full_col)), shape=shape)
        return vset_ld

    def _compute_sumstats(self, sample_id):
        """
        A permutation sample is generated by v_i*\\xi_{ij} for j = 1...r
        Each time partition new variant sets

        """
        # np.random.seed(sample_id)
        resid_ldr_rand = self.resid_ldr[np.random.permutation(self.n_subs)]
        # inner_ldr = np.dot(resid_ldr_rand.T, resid_ldr_rand).astype(np.float32)
        # self.var = np.sum(np.dot(self.bases, inner_ldr) * self.bases, axis=1)
        # self.var /= self.n_subs - self.n_covars  # (N, )
        self.half_ldr_score = self.vset @ resid_ldr_rand 

    # def _variant_set_test(self, sample_id):
    #     """
    #     A wrapper function of variant set test for multiple sets

    #     """
    #     # sig_pvalues_list = []
    
    #     with ThreadPoolExecutor(max_workers=self.threads) as executor:
    #         futures = [
    #             executor.submit(
    #                 self._variant_set_test_, sample_id, gene_id, gene_name, numeric_idx
    #             )
    #             for gene_id, (gene_name, numeric_idx) in enumerate(self.numeric_idx_list.items())
    #         ]
                    
    #         sig_pvalues_list = [
    #             future.result() for future in as_completed(futures) if future.result() is not None
    #         ]
        
    #     return sig_pvalues_list
    
    def _variant_set_test(self, sample_id):
        """
        A wrapper function of variant set test for multiple sets
        
        """
        sig_pvalues_list = []
        for gene_id, (gene_name, numeric_idx) in enumerate(self.numeric_idx_list.items()):
            sig_pvalues = self._variant_set_test_(
                sample_id, gene_id, gene_name, numeric_idx
            )
            if sig_pvalues is not None:
                sig_pvalues_list.append(sig_pvalues)

        return sig_pvalues_list

    def _variant_set_test_(self, sample_id, gene_id, gene_name, numeric_idx):
        """
        Testing a single variant set
        
        """
        # vset_test = VariantSetTest(self.bases, self.var, self.perm, self.voxels)
        half_ldr_score, cov_mat, maf, mac = self._parse_data(numeric_idx)
        if self.phred_cate is not None:
            annot = self.phred_cate[numeric_idx]
        else:
            annot = None
        is_rare = mac < self.mac_thresh
        cmac = np.sum(mac)
        self.vset_test.input_vset(half_ldr_score, cov_mat, maf, cmac, is_rare, annot)
        pvalues, _ = self.vset_test.do_inference_tests(self.tests, self.annot_name)
        pvalues.insert(0, "INDEX", self.voxels+1)
        sig_pvalues = pvalues.loc[pvalues.iloc[:, 1] < self.sig_thresh]
        if len(sig_pvalues) > 0:
            sig_pvalues = sig_pvalues.iloc[:, :2]
            sig_pvalues.columns = ["INDEX", "P"]
            bin_idx = find_loc([x[0] for x in self.all_bins], cmac)
            cmac_bin_count = self.cmac_bins_count[self.all_bins[bin_idx]]
            sig_pvalues.insert(0, "CMAC_BIN_COUNT", cmac_bin_count)
            sig_pvalues.insert(0, "CMAC", cmac)
            sig_pvalues.insert(0, "N_VARIANTS", len(mac))
            sig_pvalues.insert(0, "GENE", gene_name)
            sig_pvalues.insert(0, "GENE_ID", gene_id+1)
            sig_pvalues.insert(0, "SAMPLE_ID", sample_id+1)
        else:
            sig_pvalues = None

        return sig_pvalues
    
    def _parse_data(self, numeric_idx):
        """
        Extracting data for a variant set to test
        
        """
        half_ldr_score = np.array(self.half_ldr_score[numeric_idx])
        vset_half_covar_proj = np.array(self.vset_half_covar_proj[numeric_idx])
        vset_ld = self.vset_ld[numeric_idx][:, numeric_idx]
        cov_mat = np.array((vset_ld - vset_half_covar_proj @ vset_half_covar_proj.T)) # TODO: compute once
        maf = self.maf[numeric_idx]
        mac = self.mac[numeric_idx]
        
        return half_ldr_score, cov_mat, maf, mac
        
    def cluster_analysis(self, sample_id):
        """
        The main function for computing cluster size for a permutation sample

        Parameters:
        ------------
        sample_id: int, permutation sample id (0-based)

        """
        self._compute_sumstats(sample_id)
        sig_pvalues_list = self._variant_set_test(sample_id)

        return sig_pvalues_list


def creating_mask(
        locus, 
        variant_sets, 
        variant_category, 
        vset, 
        maf, 
        mac, 
        cmac_min, 
        cmac_max=np.inf, 
        use_annot_weights=False
    ):
    """
    Creating masks for a variant category and split into genes

    Parameters:
    ------------
    locus: a hail.Table of locus info
    variant_sets: a pd.DataFrame of genes 
    variant_category: variant category
    vset: (m, n) csr_matrix of genotype
    maf: a np.array of MAF
    mac: a np.array of MAC
    cmac_min: min of cMAC
    cmac_max: max of cMAC
    use_annot_weights: boolean, using annotation weights

    Returns:
    ---------
    chr: chromosome of the genotype
    gene_numeric_idxs: a list of list of variant idxs for each gene
    phred_cate: a np.array of functional annotations
    annot_name: annotation names
    vset: genotype of the extracted variants 
    maf: MAF of the extracted variants
    mac: MAC of the extracted variants
    
    """
    locus = locus.add_index("idx")
    variant_type = locus.variant_type.collect()[0]
    geno_ref = locus.reference_genome.collect()[0]
    mask = Coding(locus, variant_type)
    mask_idx = mask.category_dict[variant_category]
    numeric_idx, phred_cate = mask.parse_annot(mask_idx, use_annot_weights)
    annot_name = mask.annot_name

    chr = locus.aggregate(hl.agg.take(locus.locus.contig, 1)[0])
    if locus.locus.dtype.reference_genome.name == "GRCh38":
        chr = int(chr.replace("chr", ""))
    else:
        chr = int(chr)

    vset = vset[numeric_idx]
    maf = maf[numeric_idx]
    mac = mac[numeric_idx]

    gene_numeric_idxs = dict()
    positions = np.array(locus.locus.position.collect())[numeric_idx]
    cmac_bins_count = {
        (2,2): 0, (3,3): 0, (4,4): 0, (5,5): 0, 
        (6,7): 0, (8,9): 0, (10,11): 0, (12,14): 0, 
        (15,20): 0, (21,30): 0, (31,60): 0, (61,100): 0,
        (101, 500): 0, (501,): 0
    }
    all_bins = list(cmac_bins_count.keys())

    for _, gene in variant_sets.iterrows():
        _, start, end = parse_interval(gene[1], geno_ref)
        start_idx = find_loc(positions, start)
        end_idx = find_loc(positions, end) + 1
        if start_idx == -1 or positions[start_idx] != start:
            start_idx += 1
        if end_idx > start_idx + 1:
            numeric_idxs = list(range(start_idx, end_idx))
            cmac = np.sum(mac[numeric_idxs])
            if cmac >= cmac_min and cmac <= cmac_max:
                gene_numeric_idxs[gene[0]] = numeric_idxs
                bin_idx = find_loc([x[0] for x in all_bins], cmac)
                cmac_bins_count[all_bins[bin_idx]] += 1

    return chr, gene_numeric_idxs, cmac_bins_count, phred_cate, annot_name, vset, maf, mac


def check_input(args, log):
    if args.sparse_genotype is None:
        raise ValueError("--sparse-genotype is required")
    if args.spark_conf is None:
        raise ValueError("--spark-conf is required")
    if args.annot_ht is None:
        raise ValueError("--annot-ht (FAVOR annotation) is required")
    if args.null_model is None:
        raise ValueError("--null-model is required")
    if args.variant_category is None:
        raise ValueError("--variant-category is required")
    if args.perm is None:
        raise ValueError("--perm is required")
    
    if args.rv_tests is None:
        args.rv_tests = ["staar"]
    args.variant_category = args.variant_category.lower()
    if args.variant_category not in {
        "plof",
        "plof_ds",
        "missense",
        "disruptive_missense",
        "synonymous",
        "ptv",
        "ptv_ds",
        }:
        raise ValueError(f"invalid variant category: {args.variant_category}")
    if args.n_bootstrap is None:
        args.n_bootstrap = 100
        log.info("Set #permutation as 100")
    if args.mac_thresh is None:
        args.mac_thresh = 30
        log.info(f"Set --mac-thresh as default 30")
    elif args.mac_thresh < 0:
        raise ValueError("--mac-thresh must be greater than 0")
    if args.sig_thresh is None:
        args.sig_thresh = 2.5e-6
        log.info("Set significance threshold as 2.5e-6")
    if args.cmac_min is None:
        args.cmac_min = 2
        log.info(f"Set --cmac-min as default 2")
    if args.cmac_max is None:
        args.cmac_max = np.inf
    if args.cmac_min <= 100 and ("staar" in args.rv_tests or "skat" in args.rv_tests):
        log.info(
            ("WARNING: Burden/SKAT/STAAR cannot be used for genes with cMAC <= 100. "
             "Only permutation test will be used.")
        )
    if args.cmac_min <= 500 and args.use_annot_weights:
        log.info("WARNING: annotation weights cannot be used for genes with cMAC <= 500.")


def run(args, log):
    # checking if input is valid
    check_input(args, log)
    try:
        init_hail(args.spark_conf, args.grch37, args.out, log)

        # reading data and selecting LDRs
        log.info(f"Read null model from {args.null_model}")
        null_model = NullModel(args.null_model)
        null_model.select_ldrs(args.n_ldrs)
        null_model.select_voxels(args.voxels)

        # reading sparse genotype data
        sparse_genotype = SparseGenotype(args.sparse_genotype)
        log.info(f"Read sparse genotype data from {args.sparse_genotype}")
        log.info((f"{sparse_genotype.vset.shape[1]} subjects and "
                  f"{sparse_genotype.vset.shape[0]} variants."))

        # reading loco preds
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

        # extract and align subjects with the genotype data
        null_model.keep(common_ids)
        null_model.remove_dependent_columns()
        log.info(f"{len(common_ids)} common subjects in the data.")
        log.info(
            (
                f"{null_model.covar.shape[1]} fixed effects in the covariates "
                "(including the intercept) after removing redundant effects.\n"
            )
        )

        if args.loco_preds is not None:
            loco_preds.keep(common_ids)
        else:
            loco_preds = None

        # log.info(f"Processing sparse genetic data ...")
        if args.extract_locus is not None:
            args.extract_locus, unique_chrs = read_extract_locus(args.extract_locus, args.grch37, log)
        if args.exclude_locus is not None:
            args.exclude_locus = read_exclude_locus(args.exclude_locus, args.grch37, log)
        
        sparse_genotype.keep(common_ids)
        sparse_genotype.extract_exclude_locus(args.extract_locus, args.exclude_locus, unique_chrs)
        sparse_genotype.extract_chr_interval(args.chr_interval)
        sparse_genotype.extract_maf(args.maf_min, args.maf_max)
        sparse_genotype.extract_mac(args.mac_min, args.mac_max)

        # reading annotation
        log.info(f"Read functional annotations from {args.annot_ht}")
        annot = hl.read_table(args.annot_ht)
        sparse_genotype.annotate(annot)
        vset, locus, maf, mac = sparse_genotype.parse_data()

        # reading permutation
        log.info(f"Read permutation from {args.perm}")
        perm = PermDistribution(args.perm)

        # creating mask
        (
            chr, gene_numeric_idxs, cmac_bins_count, phred_cate, annot_name, vset, maf, mac
        ) = creating_mask(
            locus, 
            args.variant_sets, 
            args.variant_category,
            vset, 
            maf, 
            mac, 
            args.cmac_min, 
            args.cmac_max,
            args.use_annot_weights
        )
        if args.cmac_max == np.inf:
            log.info((f"Using {len(gene_numeric_idxs)} genes (cMAC >= {args.cmac_min}) of "
                      f"{args.variant_category} variants in permutation."
            ))
        else:
            log.info((f"Using {len(gene_numeric_idxs)} genes "
                      f"({args.cmac_min} <= cMAC <= {args.cmac_max}) of "
                      f"{args.variant_category} variants in permutation."
            ))

        # permutation
        cluster = RVcluster(
            null_model, 
            vset, 
            chr, 
            gene_numeric_idxs, 
            phred_cate,
            annot_name,
            maf, 
            mac, 
            args.rv_tests,
            perm,
            cmac_bins_count,
            args.sig_thresh, 
            args.mac_thresh,
            args.threads, 
            loco_preds,
            args.voxels,
        )

        print_head = True
        for i in tqdm(
            range(args.n_bootstrap), desc=f"{args.n_bootstrap} permutation samples"
        ):
            log.info(f"Doing permutation sample {i+1} ...")
            start_time = time.time()
            sig_pvalues_list = cluster.cluster_analysis(i)
            with open(args.out + ".txt", "a") as file:
                for sig_pvalues in sig_pvalues_list:
                    sig_pvalues['CMAC_BIN_COUNT'] *= args.n_bootstrap
                    if print_head:
                        sig_pvalues = sig_pvalues.to_csv(
                            sep="\t", header=True, na_rep="NA", index=None, float_format="%.5e"
                        )
                        print_head = False
                    else:
                        sig_pvalues = sig_pvalues.to_csv(
                            sep="\t", header=False, na_rep="NA", index=None, float_format="%.5e"
                        )
                    file.write(sig_pvalues)
            elapsed_time = int((time.time() - start_time))
            log.info(f"done ({elapsed_time}s)")

        # save results
        log.info(f"\nSaved significant associations to {args.out}.txt")

    finally:
        if "loco_preds" in locals() and args.loco_preds is not None:
            loco_preds.close()

        clean(args.out)

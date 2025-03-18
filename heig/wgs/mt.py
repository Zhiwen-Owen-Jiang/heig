import shutil
import numpy as np
import pandas as pd
import hail as hl
from hail.linalg import BlockMatrix
from scipy.sparse import lil_matrix, save_npz, load_npz
from heig.wgs.utils import *

"""
TODO:
1. Merge multiple sparse genotype datasets
2. Add an option of MAC filtering

"""


def check_input(args, log):
    if args.bfile is None and args.vcf is None and args.geno_mt is None:
        raise ValueError("--bfile or --vcf or --geno-mt is required")
    if args.spark_conf is None:
        raise ValueError("--spark-conf is required")

    if args.geno_mt is not None:
        args.vcf, args.bfile = None, None
    if args.bfile is not None:
        args.vcf = None
    if args.variant_type is None:
        args.variant_type = "snv"
        log.info(f"Set --variant-type as default 'snv'.")

    if args.qc_mode is None:
        args.qc_mode = "gwas"
    if not args.skip_qc:
        log.info(f"Set QC mode as {args.qc_mode}.")
        if args.qc_mode == "gwas" and args.save_sparse_genotype:
            raise ValueError("GWAS data cannot be saved as sparse genotype")
    if (args.bfile is not None or args.vcf is not None) and args.save_sparse_genotype:
        log.info(
            (
                "WARNING: directly saving a bfile or vcf as a sparse genotype can be "
                "very slow. Convert the bfile or vcf into mt first."
            )
        )
    if (
        (args.extract is not None or args.exclude is not None) and 
        (args.extract_locus is not None or args.exclude_locus is not None)
    ):
        raise ValueError("--extract/--exclude cannot be used with --extract-locus/--exclude-locus")
    
    if args.lift_over is not None:
        if args.lift_over not in {"GRCh37", "GRCh38"}:
            raise ValueError("--lift-over must be GRCh37 or GRCh38")
        if args.lift_over == "GRCh37" and args.grch37:
            raise TypeError("Cannot lift over from GRCh37 to GRCh37")
        if args.lift_over == "GRCh38" and not args.grch37:
            raise TypeError("Cannot lift over from GRCh38 to GRCh38")


def prepare_vset(snps_mt, variant_type):
    """
    Extracting data from MatrixTable

    Parameters:
    ------------
    snps_mt: a MatrixTable of genotype data
    variant_type: variant type

    Returns:
    ---------
    vset: (m, n) csr_matrix of genotype
    locus: a hail.Table of locus info

    """
    locus = snps_mt.rows().key_by().select("locus", "alleles")
    locus = locus.annotate_globals(
        reference_genome=locus.locus.dtype.reference_genome.name
    )
    locus = locus.annotate_globals(variant_type=variant_type)
    # bm = BlockMatrix.from_entry_expr(snps_mt.GT.n_alt_alleles(), mean_impute=True)
    bm = BlockMatrix.from_entry_expr(snps_mt.imputed_n_alt_alleles)
    if bm.shape[0] == 0 or bm.shape[1] == 0:
        raise ValueError("no variant in the genotype data")

    entries = bm.entries()
    non_zero_entries = entries.filter(entries.entry > 0)
    non_zero_entries = non_zero_entries.collect()
    rows, cols, values = zip(*map(lambda e: (e["i"], e["j"], e["entry"]), non_zero_entries))
    values = np.array(values, dtype=np.int8)

    vset = lil_matrix(bm.shape, dtype=np.int8)
    vset[rows, cols] = values
    vset = flip_variants(vset)
    vset = vset.tocsr()

    return vset, locus


def flip_variants(vset):
    """
    vset: a lil_matrix of sparse genotype
    
    """
    to_flip = np.squeeze(np.array(vset.mean(axis=1) / 2 > 0.5))
    vset[to_flip] = 2 - vset[to_flip].toarray().astype(np.int8)

    return vset


class SparseGenotype:
    """
    This module is used in --rv-sumstats
    order of steps:
    1. keep(), update maf
    2. extract_exclude_locus() and extract_chr_interval()
    3. extract_maf()

    """

    def __init__(self, prefix):
        """ "
        vset (m, n): csr_matrix
        locus: a hail.Table of locus info
        ids: a pd.DataFrame of ids with index FID and IID

        """
        self.vset = load_npz(f"{prefix}_genotype.npz")
        self.locus = hl.read_table(f"{prefix}_locus_info.ht").key_by("locus", "alleles")
        self.ids = pd.read_csv(
            f"{prefix}_id.txt", sep="\t", header=None, dtype={0: object, 1: object}
        )
        self.ids = self.ids.rename({0: "FID", 1: "IID"}, axis=1)
        # self.mac_thresh = mac_thresh

        self.locus = self.locus.add_index("idx")
        self.geno_ref = self.locus.reference_genome.collect()[0]
        self.ids["idx"] = list(range(self.ids.shape[0]))
        self.ids = self.ids.set_index(["FID", "IID"])
        self.variant_idxs = np.arange(self.vset.shape[0])
        self.maf_idx = np.full(self.vset.shape[0], True)
        self.mac_idx = np.full(self.vset.shape[0], True)
        self.maf, self.mac = self._update_maf()

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
        this method will only be invoked after keep()

        """
        if maf_min is None:
            # maf_min = 1 / len(self.ids) / 2
            maf_min = 0
        if maf_max is None:
            maf_max = 0.5
        # self.variant_idxs = self.variant_idxs[
        #     (self.maf > maf_min) & (self.maf <= maf_max)
        # ]
        self.maf_idx = (self.maf > maf_min) & (self.maf <= maf_max)
        if np.sum(self.maf_idx) == 0:
            raise ValueError("no variant in genotype data")
        
    def extract_mac(self, mac_min=None, mac_max=None):
        """
        Extracting variants by MAC
        this method will only be invoked after keep()
        
        """
        if mac_min is None:
            mac_min = 0
        if mac_max is None:
            mac_max = self.vset.shape[1]
        # self.variant_idxs = self.variant_idxs[
        #     (self.mac > mac_min) & (self.mac <= mac_max)
        # ]
        self.mac_idx = (self.mac > mac_min) & (self.mac <= mac_max)
        if np.sum(self.mac_idx) == 0:
            raise ValueError("no variant in genotype data")

    def keep(self, keep_idvs):
        """
        Keep subjects
        this method will only be invoked after extracting common subjects

        Parameters:
        ------------
        keep_idvs: a pd.MultiIndex of FID and IID

        Returns:
        ---------
        self.id_idxs: numeric indices of subjects

        """
        if not isinstance(keep_idvs, pd.MultiIndex):
            raise TypeError("keep_idvs must be a pd.MultiIndex instance")
        # self.ids = self.ids[self.ids.index.isin(keep_idvs)]
        self.ids = self.ids.loc[keep_idvs]
        if len(self.ids) == 0:
            raise ValueError("no subject in genotype data")
        self.vset = self.vset[:, self.ids["idx"].values]
        self.maf, self.mac = self._update_maf()

    def _update_maf(self):
        mac = np.squeeze(np.array(self.vset.sum(axis=1)))
        maf = mac / (self.vset.shape[1] * 2)

        return maf, mac
    
    def annotate(self, annot, cache=True):
        """
        Annotating functional annotations to locus
        ensuring no NA in annotations

        """
        if annot is not None:
            self.locus = self.locus.annotate(annot=annot[self.locus.key])
            self.locus = self.locus.filter(hl.is_defined(self.locus.annot))
        if cache:
            self.locus = self.locus.cache()
    
    def parse_data(self):
        """
        Parsing genotype data as a result of filtering

        """
        self.variant_idxs = self.variant_idxs[self.maf_idx & self.mac_idx]
        locus_idxs = set(self.locus.idx.collect())
        common_variant_idxs_set = locus_idxs.intersection(self.variant_idxs)
        locus = self.locus.filter(
            hl.literal(common_variant_idxs_set).contains(self.locus.idx)
        )
        locus = locus.drop("idx")
        common_variant_idxs = sorted(list(common_variant_idxs_set))
        common_variant_idxs = np.array(common_variant_idxs)
        vset = self.vset[common_variant_idxs]
        maf = self.maf[common_variant_idxs]
        # is_rare = self.mac[common_variant_idxs] < self.mac_thresh
        mac = self.mac[common_variant_idxs]

        return vset, locus, maf, mac


def run(args, log):
    check_input(args, log)
    try:
        init_hail(args.spark_conf, args.grch37, args.out, log)
        
        if args.extract_locus is not None:
            args.extract_locus, unique_chrs = read_extract_locus(args.extract_locus, args.grch37, log)
        else:
            unique_chrs = None
        if args.exclude_locus is not None:
            args.exclude_locus = read_exclude_locus(args.exclude_locus, args.grch37, log)

        # read genotype data
        gprocessor = read_genotype_data(args, log)

        # do preprocessing
        gprocessor.extract_exclude_locus(args.extract_locus, args.exclude_locus, unique_chrs)
        gprocessor.extract_exclude_snps(args.extract, args.exclude)
        gprocessor.extract_chr_interval(args.chr_interval)
        gprocessor.keep_remove_idvs(args.keep, args.remove)
        if not args.skip_qc:
            log.info(f"Processing genotype data ...")
            gprocessor.do_processing(mode=args.qc_mode)

        # liftover
        if args.lift_over:
            log.info(f"Lifting over to {args.lift_over}")
            gprocessor.lift_over(args.lift_over)

        # save
        if args.save_sparse_genotype:
            log.info("Constructing sparse genotype ...")
            vset, locus = prepare_vset(gprocessor.snps_mt, args.variant_type)
            log.info(
                f"{vset.shape[1]} subjects and {vset.shape[0]} variants in the sparse genotype"
            )
            snps_mt_ids = gprocessor.subject_id()
            save_npz(f"{args.out}_genotype.npz", vset)
            locus.write(f"{args.out}_locus_info.ht", overwrite=True)
            snps_mt_ids = pd.DataFrame({"FID": snps_mt_ids, "IID": snps_mt_ids})
            snps_mt_ids.to_csv(f"{args.out}_id.txt", sep="\t", header=None, index=None)
            log.info(
                (
                    f"Saved sparse genotype data at\n"
                    f"{args.out}_genotype.npz\n"
                    f"{args.out}_locus_info.ht\n"
                    f"{args.out}_id.txt"
                )
            )
        else:
            gprocessor.snps_mt.write(f"{args.out}.mt", overwrite=True)
            # post check
            gprocessor = GProcessor.read_matrix_table(f"{args.out}.mt")
            try:
                gprocessor.check_valid()
            except:
                shutil.rmtree(f"{args.out}.mt")
                raise
            log.info(f"Saved genotype data at {args.out}.mt")
    finally:
        clean(args.out)

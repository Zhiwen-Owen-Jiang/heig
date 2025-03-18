import os
import h5py
import time
import shutil
import json
import logging
import hail as hl
import numpy as np
import pandas as pd
from functools import reduce
from hail.linalg import BlockMatrix


__all__ = [
    "Annotation_name_catalog",
    "Annotation_catalog_name",
    "Annotation_name",
    "GProcessor",
    "init_hail",
    "parse_interval",
    "get_temp_path",
    "parse_locus",
    "read_genotype_data",
    "format_output",
    "clean",
    "IndexFile",
    "read_extract_locus",
    "read_exclude_locus"
]


Annotation_name_catalog = {
    "rs_num": "rsid",
    "GENCODE.Category": "genecode_comprehensive_category",
    "GENCODE.Info": "genecode_comprehensive_info",
    "GENCODE.EXONIC.Category": "genecode_comprehensive_exonic_category",
    "MetaSVM": "metasvm_pred",
    "GeneHancer": "genehancer",
    "CAGE": "cage_tc",
    "DHS": "rdhs",
    "CADD": "cadd_phred",
    "LINSIGHT": "linsight",
    "FATHMM.XF": "fathmm_xf",
    "aPC.EpigeneticActive": "apc_epigenetics_active",
    "aPC.EpigeneticRepressed": "apc_epigenetics_repressed",
    "aPC.EpigeneticTranscription": "apc_epigenetics_transcription",
    "aPC.Conservation": "apc_conservation",
    "aPC.LocalDiversity": "apc_local_nucleotide_diversity",
    "aPC.LocalDiversity(-)": "apc_local_nucleotide_diversity2",
    "aPC.Mappability": "apc_mappability",
    "aPC.TF": "apc_transcription_factor",
    "aPC.Protein": "apc_protein_function",
}


Annotation_catalog_name = dict()
for k, v in Annotation_name_catalog.items():
    Annotation_catalog_name[v] = k


Annotation_name = [
    "CADD",
    "LINSIGHT",
    "FATHMM.XF",
    "aPC.EpigeneticActive",
    "aPC.EpigeneticRepressed",
    "aPC.EpigeneticTranscription",
    "aPC.Conservation",
    "aPC.LocalDiversity",
    "aPC.LocalDiversity(-)",
    "aPC.Mappability",
    "aPC.TF",
    "aPC.Protein",
]


def init_hail(spark_conf_file, grch37, out, log):
    """
    Initializing hail

    Parameters:
    ------------
    spark_conf_file: spark configuration in json format
    grch37: if the reference genome is GRCh37
    out: output directory
    log: a logger

    """
    with open(spark_conf_file, "r") as file:
        spark_conf = json.load(file)

    if "spark.local.dir" not in spark_conf:
        spark_conf["spark.local.dir"] = out + "_spark"

    if grch37:
        geno_ref = "GRCh37"
    else:
        geno_ref = "GRCh38"
    log.info(f"Set {geno_ref} as the reference genome.")

    tmpdir = out + "_tmp"
    logdir = out + "_hail.log"
    hl.init(
        quiet=True,
        spark_conf=spark_conf,
        local_tmpdir=tmpdir,
        log=logdir,
        tmp_dir=tmpdir,
    )
    hl.default_reference = geno_ref


def clean(out):
    if os.path.exists(out + "_spark"):
        shutil.rmtree(out + "_spark")
    if os.path.exists(out + "_tmp"):
        for _ in range(100):  # Retry up to 100 times
            try:
                shutil.rmtree(out + "_tmp")
                break  # Break if successful
            except OSError as e:
                time.sleep(1)  # Wait and retry


class GProcessor:
    MODE = {
        "gwas": {
            # "defaults": {"maf_min": 0, "maf_max": 0.5},
            "defaults": {},
            "methods": [
                "_extract_variant_type",
                "_extract_maf",
                "_extract_mac",
                "_extract_call_rate",
                "_filter_hwe",
                # "_extract_chr_interval"
            ],
            "conditions": {
                "_extract_variant_type": ["variant_type"],
                "_extract_maf": ["maf_min", "maf_max"],
                "_extract_mac": ["mac_min", "mac_max"],
                "_extract_call_rate": ["call_rate"],
                "_filter_hwe": ["hwe"],
                # "_extract_chr_interval": ["chr", "start", "end"]
            },
        },
        "wgs": {
            "defaults": {
                "geno_ref": "GRCh38",
                "variant_type": "snv",
                "maf_max": 0.01,
                "maf_min": 0,
                # "mac_thresh": 10,
            },
            "methods": [
                "_vcf_filter",
                # "_flip_snps",
                "_extract_variant_type",
                "_extract_maf",
                "_extract_mac",
                "_extract_call_rate",
                "_filter_hwe",
                # "_annotate_rare_variants",
                "_filter_missing_alt",
                # "_impute_missing_snps",
                # "_extract_chr_interval"
            ],
            "conditions": {
                "_extract_variant_type": ["variant_type"],
                "_extract_maf": ["maf_min", "maf_max"],
                "_extract_mac": ["mac_min", "mac_max"],
                "_extract_call_rate": ["call_rate"],
                "_filter_hwe": ["hwe"],
                # "_extract_chr_interval": ["chr", "start", "end"]
            },
        },
    }

    PARAMETERS = {
        "variant_type": "Variant type",
        "geno_ref": "Reference genome",
        "maf_min": "Minimum MAF (>)",
        "maf_max": "Maximum MAF (<=)",
        "mac_min": "Minimum MAC (>)",
        "mac_max": "Maximum MAC (<=)",
        # "mac_thresh": "MAC threshold to annotate very rare variants (<=)",
        "call_rate": "Call rate (>=)",
        "hwe": "HWE p-value (>=)",
    }

    def __init__(
        self,
        snps_mt,
        grch37=None,
        variant_type=None,
        hwe=None,
        maf_min=None,
        maf_max=None,
        mac_min=None,
        mac_max=None,
        call_rate=None,
    ):
        """
        Genetic data processor

        Parameters:
        ------------
        snps_mt: a hl.MatrixTable genotype data
        variant_type: one of ('variant', 'snv', 'indel')
        grch37: if the reference genome is GRCh37
        maf_max: a float number between 0 and 0.5
        maf_min: a float number between 0 and 0.5, must be smaller than maf_max
        mac_max: an int number > 0
        mac_min: a int number smaller than mac_max
        call_rate: a float number between 0 and 1, 1 - genotype missingness
        hwe: a float number between 0 and 1, variants with a HWE pvalue less than
            this will be removed

        """
        self.snps_mt = snps_mt
        self.variant_type = variant_type
        self.geno_ref = "GRCh37" if grch37 else "GRCh38"
        self.maf_min = maf_min
        self.maf_max = maf_max
        self.mac_min = mac_min
        self.mac_max = np.min(mac_max, snps_mt.count_rows()) if mac_max is not None else None
        # self.mac_thresh = mac_thresh
        self.call_rate = call_rate
        self.hwe = hwe
        self.chr, self.start, self.end = None, None, None
        self.n_sub = snps_mt.count_cols()
        self.n_variants = snps_mt.count_rows()
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            (
                f"{self.n_sub} subjects and "
                f"{self.n_variants} variants in the genotype data.\n"
            )
        )

    def do_processing(self, mode):
        """
        Processing genotype data in a dynamic way
        extract_idvs() should be done before it

        Parameters:
        ------------
        mode: the analysis mode which affects preprocessing pipelines
            should be one of ('gwas', 'wgs'). For a given mode, there
            are default filtering and optional filtering.
        """
        self.snps_mt = hl.variant_qc(self.snps_mt, name="info")
        self.snps_mt = self.snps_mt.annotate_rows(
            minor_allele_index=hl.argmin(
                self.snps_mt.info.AF
            )  # Index of the minor allele
        )
        self.snps_mt = self.snps_mt.annotate_rows(
            maf=self.snps_mt.info.AF[self.snps_mt.minor_allele_index],
            mac=self.snps_mt.info.AC[self.snps_mt.minor_allele_index]
        )

        config = self.MODE.get(mode, {})
        defaults = config.get("defaults", {})
        methods = config.get("methods", [])
        conditions = config.get("conditions", {})
        attributes = self.__dict__.keys()

        for attr in attributes:
            if attr in defaults:
                setattr(self, attr, getattr(self, attr) or defaults.get(attr))

        self.logger.info("Variant QC parameters")
        self.logger.info("---------------------")
        for para_k, para_v in self.PARAMETERS.items():
            if getattr(self, para_k) is not None:
                self.logger.info(f"{para_v}: {getattr(self, para_k)}")
        if mode == "wgs":
            # self.logger.info("Flipped alleles for those with a MAF > 0.5")
            self.logger.info("Removed variants with missing alternative alleles.")
            self.logger.info("Extracted variants with PASS in FILTER.")
            self.logger.info("Imputed missing genotypes by 0.")
        self.logger.info("---------------------\n")

        variant_idx_list = list()
        for method in methods:
            method_conditions = conditions.get(method, [])
            if all(getattr(self, attr) is not None for attr in method_conditions):
                filtering_idx = getattr(self, method)()
                if filtering_idx is not None:
                    variant_idx_list.append(filtering_idx)
        
        if len(variant_idx_list) > 0:
            variant_idx = reduce(lambda x, y: x & y, variant_idx_list)
            self.snps_mt = self.snps_mt.filter_rows(variant_idx)

        if mode == "wgs":
            self._impute_missing_snps()
            # self._flip_snps()
            # self._annotate_rare_variants()

    @classmethod
    def read_matrix_table(cls, dir, *args, **kwargs):
        """
        Reading MatrixTable from a directory

        Parameters:
        ------------
        dir: directory to annotated VCF in MatrixTable

        """
        snps_mt = hl.read_matrix_table(dir)
        return cls(snps_mt, *args, **kwargs)

    @classmethod
    def import_plink(cls, bfile, grch37, *args, **kwargs):
        """
        Importing genotype data from PLINK triplets

        Parameters:
        ------------
        dir: directory to PLINK triplets (prefix only)
        grch37: if the reference genome is GRCh37

        """
        geno_ref, recode = cls._recode(grch37)
        snps_mt = hl.import_plink(
            bed=bfile + ".bed",
            bim=bfile + ".bim",
            fam=bfile + ".fam",
            reference_genome=geno_ref,
            contig_recoding=recode,
        )
        return cls(snps_mt, grch37, *args, **kwargs)

    @classmethod
    def import_vcf(cls, dir, grch37, *args, **kwargs):
        """
        Importing a VCF file as MatrixTable

        Parameters:
        ------------
        dir: directory to VCF file
        grch37: if the reference genome is GRCh37
        geno_ref: reference genome

        """
        if dir.endswith("vcf"):
            force_bgz = False
        elif dir.endswith("vcf.gz") or dir.endswith("vcf.bgz"):
            force_bgz = True
        else:
            raise ValueError("VCF suffix is incorrect")

        geno_ref, recode = cls._recode(grch37)
        vcf_mt = hl.import_vcf(
            dir, force_bgz=force_bgz, reference_genome=geno_ref, contig_recoding=recode
        )
        return cls(vcf_mt, grch37, *args, **kwargs)

    @staticmethod
    def _recode(grch37):
        if not grch37:
            geno_ref = "GRCh38"
            recode = {f"{i}": f"chr{i}" for i in (list(range(1, 23)) + ["X", "Y"])}
        else:
            geno_ref = "GRCh37"
            recode = {f"chr{i}": f"{i}" for i in (list(range(1, 23)) + ["X", "Y"])}
        return geno_ref, recode

    def save_interim_data(self, temp_dir):
        """
        Saving interim MatrixTable,
        which is useful for wgs where I/O is expensive

        Parameters:
        ------------
        temp_dir: directory to temporarily save the MatrixTable

        """
        self.logger.info(f"Saved preprocessed genotype data to {temp_dir}")
        self.snps_mt.write(temp_dir, overwrite=True)  # slow but fair
        self.snps_mt = hl.read_matrix_table(temp_dir)

    def check_valid(self):
        """
        Checking non-zero #variants

        """
        self.n_variants = self.snps_mt.count_rows()
        self.n_sub = self.snps_mt.count_cols()
        if self.n_variants == 0:
            raise ValueError("no variant remaining after preprocessing")
        if self.n_sub == 0:
            raise ValueError("no subject remaining after preprocessing")

    def subject_id(self):
        """
        Extracting subject ids

        Returns:
        ---------
        snps_mt_ids: a list of subject ids

        """
        snps_mt_ids = self.snps_mt.s.collect()
        if len(snps_mt_ids) == 0:
            raise ValueError("no subject remaining in the genotype data")
        return snps_mt_ids

    def annotate_cols(self, table, annot_name):
        """
        Annotating columns with values from a table
        the table is supposed to have the key 'IID'

        Parameters:
        ------------
        table: a hl.Table
        annot_name: annotation name

        """
        table = table.key_by("IID")
        annot_expr = {annot_name: table[self.snps_mt.s]}
        self.snps_mt = self.snps_mt.annotate_cols(**annot_expr)

    def _vcf_filter(self):
        """
        Extracting variants with a "PASS" in VCF FILTER

        """
        if "filters" in self.snps_mt.row:
            return (hl.len(self.snps_mt.filters) == 0) | hl.is_missing(self.snps_mt.filters)

    def _extract_variant_type(self):
        """
        Extracting variants with specified type

        """
        if self.variant_type == "variant":
            return
        elif self.variant_type == "snv":
            func = hl.is_snp  # the same as isSNV()
        elif self.variant_type == "indel":
            func = hl.is_indel
        else:
            raise ValueError("variant_type must be snv, indel or variant")
        return func(self.snps_mt.alleles[0], self.snps_mt.alleles[1])

    def _extract_maf(self):
        """
        Extracting variants with a maf_min < MAF <= maf_max

        """
        if self.maf_min is None:
            self.maf_min = 0
        if self.maf_min > self.maf_max:
            raise ValueError("maf_min is greater than maf_max")
        return (self.snps_mt.maf > self.maf_min) & (self.snps_mt.maf <= self.maf_max)
    
    def _extract_mac(self):
        """
        Extracting variants with a mac_min < MAC <= mac_max
        
        """
        if self.mac_min is None:
            self.mac_min = 0
        if self.mac_min > self.mac_max:
            raise ValueError("mac_min is greater than mac_max")
        return (self.snps_mt.mac > self.mac_min) & (self.snps_mt.mac <= self.mac_max)

    def _extract_call_rate(self):
        """
        Extracting variants with a call rate >= call_rate

        """
        return self.snps_mt.info.call_rate >= self.call_rate

    def _filter_hwe(self):
        """
        Filtering variants with a HWE pvalues < hwe

        """
        return self.snps_mt.info.p_value_hwe >= self.hwe

    def _filter_missing_alt(self):
        """
        Filtering variants with missing alternative allele

        """
        return ~hl.is_star(self.snps_mt.alleles[0], self.snps_mt.alleles[1])

    def _flip_snps(self):
        """
        Flipping variants with MAF > 0.5, and creating an annotation for maf

        """
        self.snps_mt = self.snps_mt.annotate_entries(
            flipped_n_alt_alleles=hl.if_else(
                self.snps_mt.minor_allele_index == 0,
                self.snps_mt.GT.ploidy - self.snps_mt.GT.n_alt_alleles(),
                self.snps_mt.GT.n_alt_alleles(),
            )
        )

    def _impute_missing_snps(self):
        """
        Imputing missing SNPs by 0

        """
        self.snps_mt = self.snps_mt.annotate_entries(
            imputed_n_alt_alleles=hl.or_else(self.snps_mt.GT.n_alt_alleles(), 0)
        )

    def _annotate_rare_variants(self):
        """
        Annotating if variants have a MAC <= mac_thresh

        """
        self.snps_mt = self.snps_mt.annotate_rows(
            is_rare=self.snps_mt.info.AC[self.snps_mt.minor_allele_index]
            <= self.mac_thresh
        )

    def extract_unique_chrs(self):
        """
        Extracting unique chromosomes

        """
        unique_chrs = set(
            self.snps_mt.aggregate_rows(hl.agg.collect(self.snps_mt.locus.contig))
        )
        if self.geno_ref == "GRCh38":
            unique_chrs = [int(chr.replace("chr", "")) for chr in unique_chrs]
        else:
            unique_chrs = [int(chr) for chr in unique_chrs]

        return unique_chrs

    def extract_exclude_snps(self, extract_variants, exclude_variants):
        """
        Extracting and excluding variants

        Parameters:
        ------------
        extract_variants: a pd.DataFrame of SNPs
        exclude_variants: a pd.DataFrame of SNPs

        """
        if 'rsid' not in self.snps_mt.row:
            raise AttributeError(
                ('rsid does not exist in the genotype data, '
                 'try to use --extract-locus/--exclude-locus')
            )

        if extract_variants is not None:
            extract_variants = hl.literal(set(extract_variants["SNP"]))
            self.snps_mt = self.snps_mt.filter_rows(
                extract_variants.contains(self.snps_mt.rsid)
            )

        if exclude_variants is not None:
            exclude_variants = hl.literal(set(exclude_variants["SNP"]))
            self.snps_mt = self.snps_mt.filter_rows(
                ~exclude_variants.contains(self.snps_mt.rsid)
            )

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
            # self.snps_mt = self.snps_mt.filter_rows(hl.is_defined(extract_locus[self.snps_mt.locus]))
            filter_chrs = hl.any(lambda c: self.snps_mt.locus.contig == c, hl.set(extract_chrs))
            self.snps_mt = self.snps_mt.filter_rows(filter_chrs)
            self.snps_mt = self.snps_mt.filter_rows(extract_locus.contains(self.snps_mt.locus))
        if exclude_locus is not None:
            # self.snps_mt = self.snps_mt.filter_rows(~hl.is_defined(exclude_locus[self.snps_mt.locus]))
            self.snps_mt = self.snps_mt.filter_rows(~exclude_locus.contains(self.snps_mt.locus))

    def extract_chr_interval(self, chr_interval=None):
        """
        Extracting a chr interval

        """
        if chr_interval is not None:
            self.chr, self.start, self.end = parse_interval(chr_interval, self.geno_ref)
            self.logger.info(
                f"Extracted variants in {self.chr} from {self.start} to {self.end}"
            )
            interval = hl.locus_interval(
                self.chr, self.start, self.end, reference_genome=self.geno_ref, includes_end=True
            )
            self.snps_mt = self.snps_mt.filter_rows(
                interval.contains(self.snps_mt.locus)
            )

    def keep_remove_idvs(self, keep_idvs, remove_idvs=None):
        """
        Keeping and removing subjects

        Parameters:
        ------------
        keep_idvs: a pd.MultiIndex/list/tuple/set of subject ids
        remove_idvs: a pd.MultiIndex/list/tuple/set of subject ids

        """
        if keep_idvs is not None:
            if isinstance(keep_idvs, pd.MultiIndex):
                keep_idvs = keep_idvs.get_level_values("IID").tolist()
            keep_idvs = hl.literal(set(keep_idvs))
            self.snps_mt = self.snps_mt.filter_cols(keep_idvs.contains(self.snps_mt.s))

        if remove_idvs is not None:
            if isinstance(remove_idvs, pd.MultiIndex):
                remove_idvs = remove_idvs.get_level_values("IID").tolist()
            remove_idvs = hl.literal(set(remove_idvs))
            self.snps_mt = self.snps_mt.filter_cols(
                ~remove_idvs.contains(self.snps_mt.s)
            )

    def extract_range(self):
        """
        Obtaining the chr and the max and min position
        only containing a single chr is valid

        """
        result = self.snps_mt.aggregate_rows(
            hl.struct(
                chr=hl.agg.take(self.snps_mt.locus.contig, 1)[0],
                min_pos=hl.agg.min(self.snps_mt.locus.position),
                max_pos=hl.agg.max(self.snps_mt.locus.position),
            )
        )

        # Save the results into variables
        chr = result.chr
        min_pos = result.min_pos
        max_pos = result.max_pos

        if self.geno_ref == "GRCh38":
            chr = int(chr.replace("chr", ""))
        else:
            chr = int(chr)

        return chr, min_pos, max_pos
    
    def ld_prune(self, r2=0.3, bp_window_size=500000):
        """
        LD pruning
        
        """
        self.snps_mt = hl.variant_qc(self.snps_mt, name="info")
        self.snps_mt = self.snps_mt.annotate_rows(
            minor_allele_index=hl.argmin(
                self.snps_mt.info.AF
            )  # Index of the minor allele
        )
        self.snps_mt = self.snps_mt.annotate_rows(
            maf=self.snps_mt.info.AF[self.snps_mt.minor_allele_index],
            mac=self.snps_mt.info.AC[self.snps_mt.minor_allele_index]
        )
        self.snps_mt = self.snps_mt.filter_rows(self.snps_mt.maf > 0)
        if self.snps_mt.count_rows() == 0:
            raise ValueError('no variant with a MAF > 0')
        self.snps_mt = self.snps_mt.filter_rows(hl.len(self.snps_mt.alleles) == 2)
        pruned_variant_table = hl.ld_prune(self.snps_mt.GT, r2=r2, bp_window_size=bp_window_size)
        self.snps_mt = self.snps_mt.filter_rows(hl.is_defined(pruned_variant_table[self.snps_mt.row_key]))
        
    def get_bm(self):
        bm = BlockMatrix.from_entry_expr(self.snps_mt.GT.n_alt_alleles(), mean_impute=True)
        return bm
    
    def get_locus(self):
        locus_info = self.snps_mt.row.select('locus').collect()
        locus_df = [f"{x.locus.contig}:{x.locus.position}" for x in locus_info]
        locus_df = pd.DataFrame({0: locus_df})
        return locus_df

    def cache(self):
        self.snps_mt = self.snps_mt.cache()
        self.logger.info("Caching the genotype data in memory.")

    def lift_over(self, to):
        self.snps_mt.count_rows()
        rg37 = hl.get_reference('GRCh37') 
        rg38 = hl.get_reference('GRCh38')

        base_dir = os.path.dirname(os.path.abspath(__file__))
        main_dir = os.path.dirname(os.path.dirname(base_dir))
        rg37.add_liftover(os.path.join(main_dir, f"misc/lift_over/references-grch37_to_grch38.over.chain.gz"), rg38) 
        rg38.add_liftover(os.path.join(main_dir, f"misc/lift_over/references-grch38_to_grch37.over.chain.gz"), rg37)

        self.snps_mt = self.snps_mt.annotate_rows(new_locus=hl.liftover(self.snps_mt.locus, to))
        self.snps_mt = self.snps_mt.filter_rows(hl.is_defined(self.snps_mt.new_locus))
        self.snps_mt = self.snps_mt.key_rows_by(locus=self.snps_mt.new_locus, alleles=self.snps_mt.alleles)
        self.geno_ref = to


def read_genotype_data(args, log):
    if args.geno_mt is not None:
        log.info(f"Read MatrixTable from {args.geno_mt}")
        read_func = GProcessor.read_matrix_table
        data_path = args.geno_mt
    elif args.bfile is not None:
        log.info(f"Read bfile from {args.bfile}")
        read_func = GProcessor.import_plink
        data_path = args.bfile
    elif args.vcf is not None:
        log.info(f"Read VCF from {args.vcf}")
        read_func = GProcessor.import_vcf
        data_path = args.vcf

    gprocessor = read_func(
        data_path,
        grch37=args.grch37,
        hwe=args.hwe,
        variant_type=args.variant_type,
        maf_min=args.maf_min,
        maf_max=args.maf_max,
        mac_min=args.mac_min,
        mac_max=args.mac_max,
        call_rate=args.call_rate,
    )

    return gprocessor


def parse_interval(chr_interval, geno_ref=None):
    """
    Converting chr_interval from string to readable format

    """
    if chr_interval is not None:
        try:
            start, end = chr_interval.split("-")
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
                    f"while ending with position is {end_pos} "
                    "is not allowed"
                )
            )
        if geno_ref == "GRCh37":
            start_chr = str(start_chr)
        elif geno_ref == "GRCh38":
            start_chr = "chr" + str(start_chr)
    else:
        start_chr, start_pos, end_pos = None, None, None

    return start_chr, start_pos, end_pos


def get_temp_path(outpath):
    """
    Generating a path for saving temporary files

    """
    temp_path = outpath + "_temp"
    i = np.random.choice(1000000, 1)[0]  # randomly select a large number
    temp_path += str(i)

    return temp_path


# def parse_locus(variant_list, geno_ref):
#     """
#     Parsing locus from a list of string

#     """
#     variant_list = list(variant_list)
#     if variant_list[0].count(":") != 1:
#         raise ValueError("variant must be in `chr:pos` format")

#     parsed_variants = [
#         hl.parse_locus(v, reference_genome=geno_ref) for v in variant_list
#     ]

#     variant_set = hl.literal(set(parsed_variants))

#     return variant_set


def parse_locus(extract_locus, temp_dir, geno_ref):
    """
    Parsing locus from a pd.DataFrame of string
    
    """
    extract_locus.to_csv(temp_dir + '_locus.txt', header=None, index=None)
    extract_locus = hl.read_table(extract_locus, no_header=True)
    extract_locus = extract_locus.annotate(
        locus=hl.parse_locus(extract_locus['f0'], reference_genome=geno_ref)
    )
    extract_locus = extract_locus.key_by("locus")


# def read_extract_locus(extract_files, grch37, log):
#     """
#     Extracting variants from multiple files
#     All files are confirmed to exist
#     Empty files are skipped without error/warning
#     Error out if no common variants exist

#     Parameters:
#     ------------
#     extract_files: a list of tab/white-delimited files
#     grch37: 

#     Returns:
#     ---------
#     keep_snp_: a hail.Table of common SNPs

#     """
#     geno_ref = "GRCh37" if grch37 else "GRCh38"
#     keep_snps_ = None
#     for i, extract_file in enumerate(extract_files):
#         if os.path.getsize(extract_file) == 0:
#             continue
#         ht = hl.import_table(extract_file, no_header=True, delimiter='\s+')
#         ht = ht.annotate(
#             locus=hl.parse_locus(ht['f0'], reference_genome=geno_ref)
#         )
#         ht = ht.key_by("locus")
#         if i == 0:
#             keep_snps_ = ht
#         else:
#             keep_snps_ = keep_snps_.filter(hl.is_defined(ht[keep_snps_.locus]))

#     keep_snps_ = keep_snps_.drop('f0')
#     if keep_snps_ is None or keep_snps_.count() == 0:
#         raise ValueError("no variants are common in --extract-locus(-cond)")

#     log.info(f"{keep_snps_.count()} variant(s) in --extract-locus(-cond) (logical 'and' for multiple files).")

#     return keep_snps_


def read_extract_locus(extract_files, grch37, log):
    """
    Extracting variants from multiple files
    All files are confirmed to exist
    Empty files are skipped without error/warning
    Error out if no common variants exist

    Parameters:
    ------------
    extract_files: a list of tab/white-delimited files
    grch37: if the reference genome is GRCh37

    Returns:
    ---------
    keep_locus: a hail.set of common loci

    """
    geno_ref = "GRCh37" if grch37 else "GRCh38"
    keep_locus = None
    unique_chrs = None
    for i, extract_file in enumerate(extract_files):
        if os.path.getsize(extract_file) == 0:
            continue
        locus = pd.read_csv(extract_file, header=None, sep='\s+')[0].to_list()
        if geno_ref == "GRCh38":
            locus = ["chr" + x for x in locus]
        chrs = set([contig for contig, _ in [v.split(":") for v in locus]])
        locus = hl.set(
            [hl.locus(contig, int(pos), reference_genome=geno_ref) 
             for contig, pos in [v.split(":") for v in locus]]
        )
        if i == 0:
            keep_locus = locus
            unique_chrs = chrs
        else:
            keep_locus = keep_locus.intersection(locus)
            unique_chrs = unique_chrs.intersection(chrs)

    if keep_locus is None or hl.eval(keep_locus.length()) == 0:
        raise ValueError("no variants are common in --extract-locus(-cond)")

    log.info(
        (f"{hl.eval(keep_locus.length())} variant(s) in --extract-locus(-cond) "
         "(logical 'and' for multiple files).")
    )

    return keep_locus, unique_chrs


# def read_exclude_locus(exclude_files, grch37, log):
#     """
#     Excluding SNPs from multiple files
#     All files are confirmed to exist
#     Empty files are skipped without error/warning
#     Error out if no SNPs exist

#     Parameters:
#     ------------
#     exclude_files: a list of tab/white-delimited files

#     Returns:
#     ---------
#     keep_snp_: pd.DataFrame of common SNPs

#     """
#     geno_ref = "GRCh37" if grch37 else "GRCh38"
#     exclude_snps_ = None
#     for i, exclude_file in enumerate(exclude_files):
#         if os.path.getsize(exclude_file) == 0:
#             continue
#         ht = hl.import_table(exclude_file, no_header=True, delimiter='\s+')
#         ht = ht.annotate(
#             locus=hl.parse_locus(ht['f0'], reference_genome=geno_ref)
#         )
#         ht = ht.key_by("locus")
#         if i == 0:
#             exclude_snps_ = ht
#         else:
#             exclude_snps_ = exclude_snps_.union(ht)

#     exclude_snps_ = exclude_snps_.drop('f0')
#     if exclude_snps_ is None or exclude_snps_.count() == 0:
#         raise ValueError("no variants in --extract-locus(-cond)")
    
#     log.info(f"{exclude_snps_.count()} variant(s) in --exclude-locus(-cond) (logical 'or' for multiple files).")

#     return exclude_snps_


def read_exclude_locus(exclude_files, grch37, log):
    """
    Excluding SNPs from multiple files
    All files are confirmed to exist
    Empty files are skipped without error/warning
    Error out if no SNPs exist

    Parameters:
    ------------
    exclude_files: a list of tab/white-delimited files

    Returns:
    ---------
    exclude_locus: a hail.set of locus 

    """
    geno_ref = "GRCh37" if grch37 else "GRCh38"
    exclude_locus = None
    for i, exclude_file in enumerate(exclude_files):
        if os.path.getsize(exclude_file) == 0:
            continue
        locus = pd.read_csv(exclude_file, header=None, sep='\s+')[0].to_list()
        if geno_ref == "GRCh38":
            locus = ["chr" + x for x in locus]
        locus = hl.set(
            [hl.locus(contig, int(pos), reference_genome=geno_ref) 
             for contig, pos in [v.split(":") for v in locus]]
        )
        if i == 0:
            exclude_locus = locus
        else:
            exclude_locus = exclude_locus.union(locus)

    if exclude_locus is None or hl.eval(exclude_locus.length()) == 0:
        raise ValueError("no variants in --extract-locus(-cond)")

    log.info(
        (f"{hl.eval(exclude_locus.length())} variant(s) in "
         "--exclude-locus(-cond) (logical 'or' for multiple files).")
    )

    return exclude_locus


def format_output(cate_pvalues, voxels, staar_only, sig_thresh):
    """
    organizing pvalues to a structured format

    Parameters:
    ------------
    cate_pvalues: a dict of pvalues of a variant category and n_variants
    voxels: zero-based voxel idxs of the image
    staar_only: if saving STAAR-O results only
    sig_thresh: significance threshold

    Returns:
    ---------
    output: a pd.DataFrame of pvalues with metadata

    """
    output = list()
    burden_output = list()

    for mask, cate_results in cate_pvalues.items():
        meta_data = pd.DataFrame(
            {
                "INDEX": voxels + 1, 
                "MASK": mask, 
                "N_VARIANTS": cate_results["n_variants"],
                "CMAC": cate_results["cMAC"]
            }
        )
        if staar_only:
            cate_results = cate_results["pvalues"][["STAAR-O"]]
        else:
            cate_results = cate_results["pvalues"]
        cate_results = pd.concat([meta_data, cate_results], axis=1)
        if sig_thresh is not None:
            to_keep = cate_results.iloc[:, 4:].min(axis=1) < sig_thresh
            cate_results = cate_results.loc[to_keep]
        if cate_results.shape[0] > 0:
            output.append(cate_results)

        if "burden_test" in cate_pvalues[mask] and cate_pvalues[mask]["burden_test"] is not None:
            burden_test = pd.concat([meta_data, cate_pvalues[mask]["burden_test"]], axis=1)
            if sig_thresh is not None:
                burden_test = burden_test.loc[to_keep]
            if burden_test.shape[0] > 0:
                burden_output.append(burden_test)
    
    if len(output) > 0:
        output = pd.concat(output)
    else:
        output = None

    if len(burden_output) > 0:
        burden_output = pd.concat(burden_output)
    else:
        burden_output = None

    return output, burden_output


class IndexFile:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        with open(self.out_dir, 'w') as file:
            file.write("VARIANT_SET\tCHR\tSTART\tEND\tRESULT_FILE\n")

    def write_index(self, gene, chr, start, end, result_file):
        with open(self.out_dir, 'a') as file:
            msg = f"{gene}\t{chr}\t{start}\t{end}\t{result_file}\n"
            file.write(msg)


class PermDistribution:
    def __init__(self, perm_file):
        self.bins = [(2,2), (3,3), (4,4), (5,5), (6,7), (8,9),
                     (10,11), (12,14), (15,20), (21,30), (31,60), (61,100)]
        self.sig_stats = {bin: dict() for bin in self.bins}
        self.count = {bin: 0 for bin in self.bins} 
        self.max_p = {bin: dict() for bin in self.bins}
        self.breaks = [bin[0] for bin in self.bins]

        h5file = h5py.File(f"{perm_file}", "r")
        all_bins = list_datasets(h5file)
        for bin_str in all_bins:
            bin1, bin2, voxel = tuple([int(x) for x in bin_str.split("_")])
            bin = tuple([bin1, bin2])
            data = h5file[bin_str]
            count = data.attrs["count"]
            self.sig_stats[bin][voxel] = data[:]
            self.count[bin] = count
            self.max_p[bin][voxel] = len(self.sig_stats[bin][voxel]) / count
        h5file.close()


def list_datasets(hdf5_file):
    dataset_names = []
    
    def visitor_func(name, node):
        if isinstance(node, h5py.Dataset):
            dataset_names.append(name)
    
    hdf5_file.visititems(visitor_func)
    return dataset_names


# class Table:
#     """
#     hail.Table processor for loci and annotations

#     """
#     def __init__(self, table, grch37):
#         """
#         table: a hail.Table

#         """
#         self.table = table
#         self.geno_ref = "GRCh37" if grch37 else "GRCh38"
#         self.table = self.table.add_index('idx')
#         self._create_keys()

#     @classmethod
#     def read_table(cls, dir, grch37):
#         """
#         Reading a hail.Table from a directory

#         """
#         table = hl.read_table(dir)
#         return cls(table, grch37)

#     def _create_keys(self):
#         self.table = self.table.key_by('locus')

#     def extract_locus(self, extract_locus, exclude_locus):
#         """
#         Extracting variants by locus

#         Parameters:
#         ------------
#         extract_locus: a pd.DataFrame of SNPs in `chr:pos` format
#         exclude_locus: a pd.DataFrame of SNPs in `chr:pos` format

#         """
#         if extract_locus is not None:
#             extract_locus = hl.Table.from_pandas(extract_locus[["locus"]])
#             extract_locus = extract_locus.annotate(locus=hl.parse_locus(extract_locus.locus))
#             # filtered_with_index = self.locus.semi_join(extract_locus)
#             filtered_with_index = self.locus.filter(extract_locus.contains(self.locus.locus))
#             indices = filtered_with_index.idx.collect()
#         if exclude_locus is not None:
#             exclude_locus = hl.Table.from_pandas(exclude_locus[["locus"]])
#             exclude_locus = exclude_locus.annotate(locus=hl.parse_locus(exclude_locus.locus))
#             filtered_with_index = self.locus.filter(~exclude_locus.contains(self.locus.locus))
#             # self.logger.info(f"{self.n_variants} variants remaining after --exclude.")

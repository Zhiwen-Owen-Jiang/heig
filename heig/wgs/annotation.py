import os
import logging
import hail as hl
from heig.wgs.utils import *


"""
Headers are required in annotation files
The first column must be variant string
Each variant is in `chr:pos:ref:alt` format

TODO: test general annotations

"""


class Annotation:
    """
    Processing and saving functional annotations

    """

    def __init__(self, annot, grch37):
        """
        Parameters:
        ------------
        annot: a hail.Table of functional annotation
        grch37: if the reference genome is GRCh37

        """
        self.annot = annot
        self.geno_ref = "GRCh37" if grch37 else "GRCh38"
        self._parse_variant()
        self._create_keys()
        self.logger = logging.getLogger(__name__)

    @classmethod
    def read_annot(cls, annot_dir, grch37, *args, **kwargs):
        """
        Reading annotation

        Parameters:
        ------------
        annot_dir: directory to unzipped annotation folder
            Hail will read all annotation files
        grch37: if the reference genome is GRCh37

        """
        annot = hl.import_table(annot_dir, *args, impute=True, **kwargs)
        return cls(annot, grch37)

    def _parse_variant(self):
        """
        Parsing variants

        """
        variant_column = list(self.annot.row)[0]
        parsed_variant = hl.parse_variant(
            self.annot[variant_column], reference_genome=self.geno_ref
        )
        self.annot = self.annot.annotate(parsed_variant=parsed_variant)

    def _create_keys(self):
        """
        Creating keys for merging

        """
        self.annot = self.annot.annotate(
            locus=self.annot.parsed_variant.locus,
            alleles=self.annot.parsed_variant.alleles,
        )
        self.annot = self.annot.key_by("locus", "alleles")
        self.annot = self.annot.drop("parsed_variant")

    def extract_exclude_locus(self, extract_locus=None, exclude_locus=None, extract_chrs=None):
        """
        Extracting and excluding variants by locus

        Parameters:
        ------------
        extract_locus: a hail.set of locus
        exclude_locus: a hail.set of locus
        extract_chrs: a set of chromosomes

        """
        if extract_locus is not None and extract_chrs is not None:
            # self.annot = self.annot.filter(hl.is_defined(extract_locus[self.annot.locus]))
            filter_chrs = hl.any(lambda c: self.annot.locus.contig == c, hl.set(extract_chrs))
            self.annot = self.annot.filter(filter_chrs)
            self.annot = self.annot.filter(extract_locus.contains(self.annot.locus))
        if exclude_locus is not None:
            # self.annot = self.annot.filter(~exclude_locus.contains(self.annot.locus))
            self.annot = self.annot.filter(~exclude_locus.contains(self.annot.locus))

    def extract_by_interval(self, chr_interval=None):
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
            self.annot = self.annot.filter(interval.contains(self.annot.locus))

    def extract_annots(self, annot_list=None):
        """
        annot_list: a list/array/series annotation names to extract

        """
        if annot_list is not None:
            annot_list = list(annot_list)
            self.annot = self.annot.select(*annot_list)
            self.logger.info(f"{annot_list} extracted from the annotation.")

    def save(self, output_dir):
        self.annot = self.annot.annotate_globals(
            reference_genome=self.annot.locus.dtype.reference_genome.name
        )
        self.annot.write(f"{output_dir}_annot.ht", overwrite=True)


class AnnotationFAVOR(Annotation):

    SELECTED_ANNOT = {
        "apc_conservation": hl.tfloat32,
        "apc_epigenetics": hl.tfloat32,
        "apc_epigenetics_active": hl.tfloat32,
        "apc_epigenetics_repressed": hl.tfloat32,
        "apc_epigenetics_transcription": hl.tfloat32,
        "apc_local_nucleotide_diversity": hl.tfloat32,
        "apc_mappability": hl.tfloat32,
        "apc_protein_function": hl.tfloat32,
        "apc_transcription_factor": hl.tfloat32,
        "cage_tc": hl.tstr,
        "metasvm_pred": hl.tstr,
        "rsid": hl.tstr,
        "fathmm_xf": hl.tfloat32,
        "genecode_comprehensive_category": hl.tstr,
        "genecode_comprehensive_info": hl.tstr,
        "genecode_comprehensive_exonic_category": hl.tstr,
        "genecode_comprehensive_exonic_info": hl.tstr,
        "genehancer": hl.tstr,
        "linsight": hl.tfloat32,
        "cadd_phred": hl.tfloat32,
        "rdhs": hl.tstr,
    }

    def __init__(self, annot, grch37):
        super().__init__(annot, grch37)
        self._drop_rename()
        self._add_more_annot()

    def _parse_variant(self):
        self.annot = self.annot.annotate(
            chromosome=hl.str("chr") + hl.str(self.annot.chromosome)
        )
        parsed_variant = hl.variant_str(
            hl.locus(
                self.annot.chromosome,
                self.annot.position,
                reference_genome=self.geno_ref,
            ),
            [self.annot.ref_vcf, self.annot.alt_vcf],
        )
        self.annot = self.annot.annotate(
            parsed_variant=hl.parse_variant(
                parsed_variant, reference_genome=self.geno_ref
            )
        )

    def _drop_rename(self):
        """
        Dropping fields and renaming annotation names

        """
        self.annot = self.annot.drop(
            "apc_conservation", "apc_local_nucleotide_diversity"
        )

        self.annot = self.annot.rename(
            {
                "apc_conservation_v2": "apc_conservation",
                "apc_local_nucleotide_diversity_v3": "apc_local_nucleotide_diversity",
                "apc_protein_function_v3": "apc_protein_function",
            }
        )

        annot_name = list(self.annot.row_value.keys())
        self.annot = self.annot.drop(
            *[field for field in annot_name if field not in self.SELECTED_ANNOT]
        )

    def _add_more_annot(self):
        """
        Filling NA for cadd_phred and creating a new annotation

        """
        annot_local_div = -10 * hl.log10(
            1 - 10 ** (-self.annot.apc_local_nucleotide_diversity / 10)
        )
        self.annot = self.annot.annotate(
            cadd_phred=hl.coalesce(self.annot.cadd_phred, 0),
            apc_local_nucleotide_diversity2=annot_local_div,
        )


def check_input(args, log):
    # required arguments
    if args.favor_annot is None and args.general_annot is None:
        raise ValueError("either --favor-annot or --general-annot is required")
    elif args.favor_annot is not None and args.general_annot is not None:
        log.info("WARNING: ignoring --general-annot as --favor-annot has been specified.")
    if args.annot_cols is not None:
        args.annot_cols = args.annot_cols.split(",")


def check_valid(annot, log):
    non_key_columns = len(annot.row) - len(annot.key)
    n_variants = annot.count()
    if non_key_columns == 0:
        raise ValueError("no annotation remaining after preprocessing")
    if n_variants == 0:
        raise ValueError("no variant remaining after preprocessing")
    log.info(
        (
            f"{n_variants} variants (including varying alleles) "
            f"with {non_key_columns} annotations processed."
        )
    )


def run(args, log):
    # check input and init
    check_input(args, log)
    try:
        init_hail(args.spark_conf, args.grch37, args.out, log)
        
        if args.extract_locus is not None:
            args.extract_locus, unique_chrs = read_extract_locus(args.extract_locus, args.grch37, log)
        else:
            unique_chrs = None
        if args.exclude_locus is not None:
            args.exclude_locus = read_exclude_locus(args.exclude_locus, args.grch37, log)

        # read annotation and preprocess
        if args.favor_annot is not None:
            log.info(f"Read FAVOR annotations from {args.favor_annot}")
            annot = AnnotationFAVOR.read_annot(
                args.favor_annot, args.grch37, delimiter=",", missing="", quote='"'
            )
        else:
            log.info(f"Read annotations from {args.general_annot}")
            annot = Annotation.read_annot(
                args.general_annot, args.grch37, delimiter="\s+", missing="", quote='"'
            )

        log.info(f"Processing annotations ...")
        annot.extract_annots(args.annot_cols)
        annot.extract_by_interval(args.chr_interval)
        annot.extract_exclude_locus(args.extract_locus, args.exclude_locus, unique_chrs)

        annot.save(args.out)
        annot = hl.read_table(f"{args.out}_annot.ht")
        try:
            check_valid(annot, log)
        except:
            os.remove(f"{args.out}_annot.ht")
            raise
        log.info(f"\nSaved the annotations to {args.out}_annot.ht")
    finally:
        clean(args.out)

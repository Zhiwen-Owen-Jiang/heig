import os
import numpy as np
from abc import ABC, abstractmethod
import hail as hl
from heig.wgs.wgs2 import RVsumstats
from heig.wgs.vsettest import VariantSetTest
from heig.wgs.utils import PermDistribution
from heig.wgs.utils import *


OFFICIAL_NAME = {
    "upstream": "upstream variants",
    "downstream": "downstream variants",
    "utr": "untranslated regions (UTR)",
    "promoter_cage": "promoter-CAGE variants",
    "promoter_dhs": "promoter-DHS variants",
    "enhancer_cage": "enhancer-CAGE variants",
    "enhancer_dhs": "enhancer-DHS variants",
    "ncrna": "long noncoding RNA (ncRNA)",
}


class Noncoding(ABC):
    def __init__(self, annot, variant_type, type=None):
        """
        Parameters:
        ------------
        annot: a hail.Table of annotations with key ('locus', 'alleles') and hail.struct of annotations
        type: subtype of variants

        """
        self.annot = annot
        self.type = type
        self.gencode_category = self.annot.annot[
            Annotation_name_catalog["GENCODE.Category"]
        ]
        self.genecode_info = self.annot.annot[
            Annotation_name_catalog["GENCODE.Info"]
        ]

        if variant_type != "indel":
            self.annot_cols = [
                Annotation_name_catalog[annot_name] for annot_name in Annotation_name
            ]
            self.annot_name = Annotation_name
        else:
            self.annot_cols, self.annot_name = None, None

    @abstractmethod
    def extract_variants(self, gene):
        """
        Extracting variants by boolean indices

        Parameters:
        ------------
        gene: gene_name

        """
        pass

    def parse_annot(self, variant_idx, use_annot_weights):
        """
        Parsing annotations and converting to np.array

        Parameters:
        ------------
        variant_idx: a hail.expr of boolean indices to extract variants
        use_annot_weights: boolean, using annotation weights

        Returns:
        ---------
        numeric_idx: a list of numeric indices for extracting sumstats
        phred_cate: a np.array of annotations

        """
        filtered_annot = self.annot.filter(variant_idx)
        numeric_idx = filtered_annot.idx.collect()
        if len(numeric_idx) <= 1:
            return numeric_idx, None, None, None, None
        
        all_locus = filtered_annot.locus.collect()
        chr = all_locus[0].contig
        start = all_locus[0].position
        end = all_locus[-1].position

        if self.annot_cols is not None and use_annot_weights:
            annot_phred = filtered_annot.annot.select(*self.annot_cols).collect()
            phred_cate = np.array(
                [[getattr(row, col) for col in self.annot_cols] for row in annot_phred]
            )
        else:
            phred_cate = None

        return numeric_idx, phred_cate, chr, start, end


class UpDown(Noncoding):
    def __init__(self, annot, variant_type, type):
        super().__init__(annot, variant_type, type)
        self.variant_idx1 = self.gencode_category == self.type
                                         
    def extract_variants(self, gene):
        """
        type is 'upstream' or 'downstream'

        """
        variant_idx2 = self.genecode_info.contains(gene)

        return self.variant_idx1 & variant_idx2


class UTR(Noncoding):
    def __init__(self, annot, variant_type, type):
        super().__init__(annot, variant_type, type)
        # self.annot = self.annot.annotate(utr=self.genecode_info.split(r'(')[0])
        # self.gencode_category = self.annot.annot[
        #     Annotation_name_catalog["GENCODE.Category"]
        # ]
        set1 = hl.literal({"UTR3", "UTR5", "UTR5;UTR3"})
        self.variant_idx1 = set1.contains(self.gencode_category)

    def extract_variants(self, gene):
        # variant_idx2 = self.annot.utr == gene
        variant_idx2 = self.genecode_info.startswith(gene)

        return self.variant_idx1 & variant_idx2


class Promoter(Noncoding):
    def __init__(self, annot, variant_type, type, promG_intervals):
        """
        promG_intervals: locus intervals of promG

        """
        super().__init__(annot, variant_type, type)
        self.annot = self.annot.annotate(split_genes=self.genecode_info.split(r'[(),;\\-]')[0])
        self.is_prom = hl.is_defined(promG_intervals.index(self.annot.locus))
        self.is_type = hl.is_defined(self.annot.annot[Annotation_name_catalog[self.type]])

    def extract_variants(self, gene):
        """
        type is 'CAGE' or 'DHS'

        """
        gene_idx = self.annot.split_genes == gene
        variant_idx = self.is_type & self.is_prom & gene_idx

        return variant_idx


class Enhancer(Noncoding):
    def __init__(self, annot, variant_type, type):
        super().__init__(annot, variant_type, type)
        genehancer = self.annot.annot[Annotation_name_catalog["GeneHancer"]]
        self.annot = self.annot.annotate(genehancer=genehancer.split('connected_gene=')[1].split(';')[0])
        self.is_type = hl.is_defined(self.annot.annot[Annotation_name_catalog[self.type]])
        
    def extract_variants(self, gene):
        """
        type is 'CAGE' or 'DHS'

        """
        gene_idx = self.annot.genehancer == gene
        variant_idx = self.is_type & gene_idx

        return variant_idx


class NcRNA(Noncoding):
    def __init__(self, annot, variant_type, type):
        super().__init__(annot, variant_type, type)
        self.annot = self.annot.annotate(ncrna=self.genecode_info.split(';')[0])
        self.gencode_category = self.annot.annot[
            Annotation_name_catalog["GENCODE.Category"]
        ]
        set1 = hl.literal({"ncRNA_exonic", "ncRNA_exonic;splicing", "ncRNA_splicing"})
        self.variant_idx1 = set1.contains(self.gencode_category)

    def extract_variants(self, gene):
        variant_idx2 = self.annot.ncrna.contains(gene)

        return self.variant_idx1 & variant_idx2


def read_promG(geno_ref):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    main_dir = os.path.dirname(os.path.dirname(base_dir))
    intervals = hl.read_table(os.path.join(main_dir, f"misc/wgs/promGdf_{geno_ref}.ht"))
    return intervals


def noncoding_vset_analysis(
    rv_sumstats, 
    annot, 
    variant_sets, 
    variant_type, 
    vset_test, 
    variant_category, 
    mac_thresh, 
    tests,
    cmac_min, 
    cmac_max,
    use_annot_weights,
    log
):
    """
    Single noncoding variant set analysis

    Parameters:
    ------------
    rv_sumstats: a RVsumstats instance
    annot: a hail.Table of locus containing annotations
    variant_sets: a pd.DataFrame of multiple variant sets to test
    variant_type: one of ('variant', 'snv', 'indel')
    vset_test: an instance of VariantSetTest
    variant_category: which category of variants to analyze,
        one of ('all', 'upstream', 'downstream', 'promoter_cage', 'promoter_dhs',
        'enhancer_cage', 'enhancer_dhs', 'ncrna')
    mac_thresh: a MAC threshold to denote ultrarare variants for ACAT-V
    tests: a list of rv tests
    cmac_min: the minimal cumulative MAC for a variant set
    cmac_max: the maximumcumulative MAC for a variant set
    use_annot_weights: boolean, using annotation weights
    log: a logger

    Returns:
    ---------
    cate_pvalues: a dict (keys: category, values: p-value)

    """
    # extracting specific variant category
    category_class_map = {
        "upstream": (UpDown, "upstream"),
        "downstream": (UpDown, "downstream"),
        "utr": (UTR, None),
        "promoter_cage": (Promoter, "CAGE"),
        "promoter_dhs": (Promoter, "DHS"),
        "enhancer_cage": (Enhancer, "CAGE"),
        "enhancer_dhs": (Enhancer, "DHS"),
        "ncrna": (NcRNA, None)
    }

    # extracting variants in sumstats and annot
    annot_locus = rv_sumstats.annotate(annot)

    # creating annotation class
    category_class_dict = dict()
    promGdf_intervals = read_promG(rv_sumstats.geno_ref)
    for category, (_category_class_, type) in category_class_map.items():
        if variant_category[0] == "all" or category in variant_category:
            if category == "promoter_cage" or category == "promoter_dhs":
                category_class = _category_class_(
                    annot_locus, variant_type, type, promGdf_intervals
                )
            else:
                category_class = _category_class_(
                    annot_locus, variant_type, type
                )

            category_class_dict[category] = category_class

    for _, gene in variant_sets.iterrows():
        # individual analysis
        log.info(f"\nGene: {gene[0]}")
        cate_pvalues = dict()

        for category, category_class in category_class_dict.items():
            variant_idx = category_class.extract_variants(gene[0])
            numeric_idx, phred_cate, chr, start, end = category_class.parse_annot(variant_idx, use_annot_weights)
            if len(numeric_idx) <= 1:
                log.info(f"Skipping {OFFICIAL_NAME[category]} (< 2 variants).")
                continue
            if phred_cate is not None and np.isnan(phred_cate).any():
                log.info(f"Skipping {OFFICIAL_NAME[category]} (NAs in annotation weights).")
                continue
            half_ldr_score, cov_mat, maf, mac = rv_sumstats.parse_data(numeric_idx)
            if half_ldr_score is None:
                continue
            cmac = int(np.sum(mac))
            if cmac < cmac_min or cmac > cmac_max:
                log.info(
                    f"Skipping {OFFICIAL_NAME[category]} (cMAC ({cmac}) out of range)."
                )
                continue
            is_rare = mac < mac_thresh
            vset_test.input_vset(half_ldr_score, cov_mat, maf, cmac, is_rare, phred_cate)
            log.info(
                (
                    f"Doing analysis for {OFFICIAL_NAME[category]} "
                    f"({vset_test.n_variants} variants, {cmac} alleles) ..."
                )
            )
            pvalues, burden_test = vset_test.do_inference_tests(tests, category_class.annot_name)
            cate_pvalues[category] = {
                "n_variants": vset_test.n_variants,
                "cMAC": cmac,
                "pvalues": pvalues,
                "burden_test": burden_test,
            }

        yield gene[0], chr, start, end, cate_pvalues


def check_input(args, log):
    # required arguments
    if args.rv_sumstats_part1 is None:
        raise ValueError("--rv-sumstats-part1 is required")
    if args.rv_sumstats_part2 is None:
        raise ValueError("--rv-sumstats-part2 is required")
    if args.spark_conf is None:
        raise ValueError("--spark-conf is required")
    if args.annot_ht is None:
        raise ValueError("--annot-ht (FAVOR annotation) is required")
    if args.variant_sets is None:
        raise ValueError("--variant-sets is required")
    log.info(f"{args.variant_sets.shape[0]} genes in --variant-sets.")
    if args.perm is None:
        raise ValueError("--perm is required")

    if args.rv_tests is None:
        args.rv_tests = ["staar"]

    if args.mac_thresh is None:
        args.mac_thresh = 10
        # log.info(f"Set --mac-thresh as default 10")
    elif args.mac_thresh < 0:
        raise ValueError("--mac-thresh must be greater than 0")
    
    if args.cmac_min is None:
        args.cmac_min = 2
        log.info(f"Set --cmac-min as default 2")
    if args.cmac_max is None:
        args.cmac_max = np.inf

    if args.cmac_min <= 1500 and ("staar" in args.rv_tests or "skat" in args.rv_tests):
        log.info(
            ("WARNING: SKAT/STAAR cannot be used for genes with cMAC <= 1500. "
             "Only burden test will be used.")
        )
    if args.cmac_min <= 1500 and args.use_annot_weights:
        log.info("WARNING: annotation weights cannot be used for genes with cMAC <= 1500.")

    if args.variant_category is None:
        variant_category = ["all"]
        log.info(f"Set --variant-category as default 'all'.")
    else:
        variant_category = list()
        args.variant_category = [x.lower() for x in args.variant_category.split(",")]
        for category in args.variant_category:
            if category == "all":
                variant_category = ["all"]
                break
            elif category not in {
                "upstream",
                "downstream",
                "utr",
                "promoter_cage",
                "promoter_dhs",
                "enhancer_cage",
                "enhancer_dhs",
                "ncrna"
            }:
                log.info(f"Ingoring invalid variant category {category}.")
            else:
                variant_category.append(category)
    if len(variant_category) == 0:
        raise ValueError("no valid variant category provided")

    return variant_category


def run(args, log):
    # checking if input is valid
    variant_category = check_input(args, log)
    try:
        init_hail(args.spark_conf, args.grch37, args.out, log)

        if args.extract_locus is not None:
            args.extract_locus, unique_chrs = read_extract_locus(args.extract_locus, args.grch37, log)
        else:
            unique_chrs = None
        if args.exclude_locus is not None:
            args.exclude_locus = read_exclude_locus(args.exclude_locus, args.grch37, log)

        # reading data and selecting voxels and LDRs
        log.info((f"Read rare variant summary statistics from "
                  f"{args.rv_sumstats_part1} and {args.rv_sumstats_part2}"))
        rv_sumstats = RVsumstats(args.rv_sumstats_part1, args.rv_sumstats_part2)
        rv_sumstats.extract_exclude_locus(args.extract_locus, args.exclude_locus, unique_chrs)
        rv_sumstats.extract_chr_interval(args.chr_interval)
        rv_sumstats.extract_maf(args.maf_min, args.maf_max)
        rv_sumstats.extract_mac(args.mac_min, args.mac_max)
        rv_sumstats.select_ldrs(args.n_ldrs)
        rv_sumstats.select_voxels(args.voxels)
        rv_sumstats.calculate_var()

        # reading annotation
        log.info(f"Read functional annotations from {args.annot_ht}")
        annot = hl.read_table(args.annot_ht)

        # reading permutation
        log.info(f"Read permutation from {args.perm}")
        perm = PermDistribution(args.perm)

        # single gene analysis
        if args.voxels is None:
            args.voxels = np.arange(rv_sumstats.bases.shape[0])
        vset_test = VariantSetTest(rv_sumstats.bases, rv_sumstats.var, perm, args.voxels)
        all_vset_test_pvalues = noncoding_vset_analysis(
            rv_sumstats,
            annot,
            args.variant_sets,
            rv_sumstats.variant_type,
            vset_test,
            variant_category,
            args.mac_thresh,
            args.rv_tests,
            args.cmac_min,
            args.cmac_max,
            args.use_annot_weights,
            log,
        )
        
        index_file = IndexFile(f"{args.out}_result_index.txt")
        log.info(f"Saved result index file to {args.out}_result_index.txt")

        for set_name, chr, start, end, cate_pvalues in all_vset_test_pvalues:
            cate_output, burden_output = format_output(
                cate_pvalues,
                rv_sumstats.voxel_idxs,
                False,
                args.sig_thresh
            )
            if cate_output is not None:
                out_path = f"{args.out}_{set_name}.txt"
                index_file.write_index(
                    set_name,
                    chr,
                    start,
                    end,
                    out_path
                )
                cate_output.to_csv(
                    out_path,
                    sep="\t",
                    header=True,
                    na_rep="NA",
                    index=None,
                    float_format="%.5e",
                )
                log.info(
                    f"Saved results for {set_name} to {args.out}_{set_name}.txt"
                )
            else:
                log.info(f"No significant results for {set_name}.")

            if burden_output is not None:
                out_path = f"{args.out}_{set_name}_burden.txt"
                burden_output.to_csv(
                    out_path,
                    sep="\t",
                    header=True,
                    na_rep="NA",
                    index=None,
                    float_format="%.5e",
                )
                log.info(
                    f"Saved burden results for {set_name} to {args.out}_{set_name}_burden.txt"
                )
    finally:
        clean(args.out)

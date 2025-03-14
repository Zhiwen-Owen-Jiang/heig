import hail as hl
import numpy as np
from heig.wgs.wgs2 import RVsumstats, get_interval, extract_chr_interval
from heig.wgs.vsettest import VariantSetTest
from heig.wgs.utils import *
from heig.wgs.utils import PermDistribution
from heig.utils import find_loc


class GeneralAnnotation:
    """
    Rare variant analysis using general annotations

    """

    def __init__(self, annot, annot_cols=None):
        """
        Parameters:
        ------------
        annot: a hail.Table of locus w/ or w/o annotations
        annot_cols: annotations used as weights

        """
        self.annot = annot
        self.annot_cols = annot_cols

    def parse_annot(self, idx=None, use_annot_weights=False):
        """
        Parsing annotations, maf, and is_rare from locus

        Parameters:
        ------------
        idx: a hail.expr of boolean indices to extract variants
        use_annot_weights: boolean, using annotation weights

        Returns:
        ---------
        numeric_idx: a list of numeric indices for extracting sumstats
        annot: a np.array of annotations

        """
        if idx is not None:
            filtered_annot = self.annot.filter(idx)
        else:
            filtered_annot = self.annot
        numeric_idx = filtered_annot.idx.collect()
        if len(numeric_idx) <= 1:
            return numeric_idx, None

        if "annot" in self.annot.row and self.annot_cols is not None and use_annot_weights:
            annot = filtered_annot.annot.select(*self.annot_cols).collect()
            annot = np.array(
                [[getattr(row, col) for col in self.annot_cols] for row in annot]
            )
            if annot.dtype == object:
                raise TypeError("annotations must be numerical data")
            if np.isnan(annot).any():
                raise ValueError("missing values are not allowed in annotations")
        else:
            annot = None

        return numeric_idx, annot


class SlidingWindow(GeneralAnnotation):
    """
    Rare variant analysis using fixed-length sliding window

    """

    def __init__(
        self,
        annot,
        window_length,
        geno_ref,
        sliding_length=None,
        annot_cols=None,
    ):
        """
        Parameters:
        ------------
        window_length: size of sliding window (bp)
        sliding_length: step size of moving forward (bp)

        """
        super().__init__(annot, annot_cols)
        self.geno_ref = geno_ref
        self.chr, self.start, self.end = get_interval(self.annot)
        self.window_length = window_length
        self.sliding_length = sliding_length

        # self.chr_intervals, self.windows = self._partition_windows()
        self.chr_intervals, self.windows = self._partition_windows_fast()

    def _partition_windows(self):
        """
        Partitioning variant set into overlapping windows
        sliding length is half of window length

        """
        windows = list()
        chr_intervals = list()
        cur_left = self.start
        cur_right = self.start + self.window_length

        while cur_right <= self.end:
            interval = hl.locus_interval(
                self.chr, cur_left, cur_right, reference_genome=self.geno_ref
            )
            window = interval.contains(self.annot.locus)
            windows.append(window)
            chr_intervals.append([cur_left, cur_right])
            cur_left += self.sliding_length
            cur_right += self.sliding_length

        return chr_intervals, windows
    
    def _partition_windows_fast(self):
        positions = np.array(self.annot.locus.position.collect())
        chr_intervals = list()
        windows = list()

        for start in range(self.start, self.end, self.sliding_length):
            end = start + self.window_length
            # chr_intervals.append((start, end))
            start_idx = find_loc(positions, start)
            end_idx = find_loc(positions, end) + 1
            if start_idx == -1 or positions[start_idx] != start:
                start_idx += 1
            if end_idx > start_idx + 1:
                windows.append(list(range(start_idx, end_idx)))
                chr_intervals.append((start, end-1))

        return chr_intervals, windows


def vset_analysis(
        rv_sumstats, 
        vset_test, 
        variant_sets, 
        annot, 
        annot_cols, 
        window_length, 
        sliding_length,
        mac_thresh,
        tests,
        cmac_min,
        cmac_max,
        use_annot_weights,
        log
    ):
    """
    Variant set analysis

    Parameters:
    ------------
    rv_sumstats: a RVsumstats instance
    annot: a hail.Table of locus containing annotations
    vset_test: an instance of VariantSetTest
    variant_sets: a pd.DataFrame of multiple variant sets to test 
    annot: general annotation data, can be None
    annot_cols: a list of columns of annotations to use
    window_length: window length
    sliding_length: silding length
    mac_thresh: a MAC threshold to denote ultrarare variants for ACAT-V
    tests: a list of rv tests
    cmac_min: the minimum cumulative MAC for a variant set
    cmac_min: the maximum cumulative MAC for a variant set
    use_annot_weights: boolean, using annotation weights
    log: a logger

    Returns:
    ---------
    pvalues: a dict (keys: vset/window_idx, values: p-value)

    """
    annot_locus = rv_sumstats.annotate(annot)
    # annot_locus = annot_locus.cache() # TODO: check this

    if window_length is None:
        for _, gene in variant_sets.iterrows():
            all_pvalues = dict()
            variant_set_locus = extract_chr_interval(
                annot_locus, gene[0], gene[1], rv_sumstats.geno_ref, log
            )
            if variant_set_locus is None:
                continue
            general_annot = GeneralAnnotation(variant_set_locus, annot_cols)
            chr, start, end = get_interval(variant_set_locus)

            # individual analysis
            numeric_idx, phred_cate = general_annot.parse_annot(use_annot_weights)
            half_ldr_score, cov_mat, maf, mac = rv_sumstats.parse_data(
                numeric_idx
            )
            if half_ldr_score is None:
                continue
            cmac = int(np.sum(mac))
            if cmac < cmac_min or cmac > cmac_max:
                log.info(
                    f"Skipping {gene[0]} (cMAC ({cmac}) out of range)."
                )
                continue
            is_rare = mac < mac_thresh
            vset_test.input_vset(half_ldr_score, cov_mat, maf, cmac, is_rare, phred_cate)
            log.info(
                (
                    f"Doing analysis for {gene[0]} "
                    f"({vset_test.n_variants} variants, {cmac} alleles) ..."
                )
            )
            pvalues = vset_test.do_inference(general_annot.annot_cols)
            pvalues, burden_test = vset_test.do_inference_tests(
                tests, general_annot.annot_cols
            )
            all_pvalues[gene[0]] = {
                "n_variants": vset_test.n_variants,
                "cMAC": cmac,
                "pvalues": pvalues,
                "burden_test": burden_test,
            }
            
            yield gene[0], chr, start, end, all_pvalues

    else:
        sliding_window = SlidingWindow(
            annot_locus, window_length, rv_sumstats.geno_ref, sliding_length, annot_cols
        )
        n_windows = len(sliding_window.windows)
        log.info(f"Partitioned the genotype data into {n_windows} windows.")
        
        window_i = 0
        for chr_interval, window in zip(sliding_window.chr_intervals, sliding_window.windows):
            all_pvalues = dict()
            # numeric_idx, phred_cate = sliding_window.parse_annot(window, use_annot_weights)
            numeric_idx, phred_cate = window, None
            if len(numeric_idx) <= 1:
                log.info(
                    (
                        f"Skipping window from {chr_interval[0]} to {chr_interval[1]} "
                        "(< 2 variants)."
                    )
                )
                continue
            half_ldr_score, cov_mat, maf, mac = rv_sumstats.parse_data(
                numeric_idx
            )
            if half_ldr_score is None:
                continue
            cmac = int(np.sum(mac))
            if cmac < cmac_min or cmac > cmac_max:
                log.info(f"Skipping window (cMAC ({cmac}) out of range).")
                continue
            window_i += 1
            is_rare = mac < mac_thresh
            vset_test.input_vset(half_ldr_score, cov_mat, maf, cmac, is_rare, phred_cate)
            log.info(
                (
                    f"Doing analysis for window{window_i} "
                    f"({vset_test.n_variants} variants, {cmac} alleles) ..."
                )
            )
            pvalues, burden_test = vset_test.do_inference_tests(
                tests, sliding_window.annot_cols
            )
            all_pvalues[f"window{window_i}"] = {
                "n_variants": vset_test.n_variants,
                "cMAC": cmac,
                "pvalues": pvalues,
                "burden_test": burden_test,
            }
            
            yield (
                f"window{window_i}", sliding_window.chr, 
                chr_interval[0], chr_interval[1], all_pvalues
            )


def check_input(args, log):
    # required arguments
    if args.rv_sumstats_part1 is None:
        raise ValueError("--rv-sumstats-part1 is required")
    if args.rv_sumstats_part2 is None:
        raise ValueError("--rv-sumstats-part2 is required")
    if args.spark_conf is None:
        raise ValueError("--spark-conf is required")
    if args.window_length is None and args.variant_sets is None:
        raise ValueError(("either --variant-sets (general annotation) " 
                          "or --window-length is required"))
    if args.annot_cols is not None:
        args.annot_cols = args.annot_cols.split(",")
    if args.window_length is not None and args.window_length < 2:
        raise ValueError("--window-length must be greater than 1")
    if args.sliding_length is not None and args.sliding_length < 1:
        raise ValueError("--sliding-length must be greater than 0")

    if args.window_length is not None and args.sliding_length is None:
        args.sliding_length = args.window_length // 2
        log.info(f"Set sliding length as {args.sliding_length}.")
    
    if args.rv_tests is None:
        args.rv_tests = ["staar"]

    if args.mac_thresh is None:
        args.mac_thresh = 10
        log.info(f"Set --mac-thresh as default 10")
    elif args.mac_thresh < 0:
        raise ValueError("--mac-thresh must be greater than 0")
    
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
        if args.annot_ht is not None:
            log.info(f"Read functional annotations from {args.annot_ht}")
            annot = hl.read_table(args.annot_ht)
        else:
            annot = None
            
        # reading permutation
        log.info(f"Read permutation from {args.perm}")
        perm = PermDistribution(args.perm)

        # single gene analysis
        if args.voxels is None:
            args.voxels = np.arange(rv_sumstats.bases.shape[0])
        vset_test = VariantSetTest(rv_sumstats.bases, rv_sumstats.var, perm, args.voxels)
        all_vset_test_pvalues = vset_analysis(
            rv_sumstats, 
            vset_test, 
            args.variant_sets, 
            annot,
            args.annot_cols, 
            args.window_length,
            args.sliding_length,
            args.mac_thresh,
            args.rv_tests,
            args.cmac_min,
            args.cmac_max,
            args.use_annot_weights,
            log
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
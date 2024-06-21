import h5py
import numpy as np
import pandas as pd
import hail as hl
from heig.wgs.staar import VariantSetTest
from heig.wgs.utils import *


class SlidingWindow:
    def __init__(self, snps, variant_type, window_length):
        """
        The specific type of variants have been extracted; snps data has aligned with covariates
        
        """
        self.snps = snps
        self.variant_type = variant_type
        self.window_length = window_length
        self.anno_pred = self.get_annotation()
        self.windows = self.get_windows()


    def get_annotation(self):
        """
        May use keys in `Annotation_name_catalog` as the column name
        return annotations for all coding variants in hail.Table

        """
        if self.variant_type != 'snv':
            anno_phred = self.snps.fa.annotate(null_weight=1)
        else:
            anno_cols = [Annotation_name_catalog[anno_name]
                        for anno_name in Annotation_name]

            # anno_phred = self.snps.fa[anno_cols].to_pandas()
            # anno_phred['cadd_phred'] = anno_phred['cadd_phred'].fillna(0)
            # anno_local_div = -10 * np.log10(1 - 10 ** (-anno_phred['apc_local_nucleotide_diversity']/10))
            # anno_phred['apc_local_nucleotide_diversity2'] = anno_local_div
            
            anno_phred = self.snps.fa.select(*anno_cols)
            anno_phred = anno_phred.annotate(cadd_phred=hl.coalesce(anno_phred.cadd_phred, 0))
            anno_local_div = -10 * np.log10(1 - 10 ** (-anno_phred.apc_local_nucleotide_diversity/10))
            anno_phred = anno_phred.annotate(apc_local_nucleotide_diversity2=anno_local_div)    
        return anno_phred
    
    def get_windows(self, start, end):
        windows = list()
        sliding_length = self.window_length // 2
        cur_left = start
        cur_right = start + sliding_length
        while cur_right <= end:
            windows.append(tuple([cur_left, cur_right]))
            cur_left, cur_right = cur_right, cur_right + sliding_length
        return windows
    

def single_gene_analysis(snps, start, end, variant_type, window_length, vset_test):
    """
    Single gene analysis

    Parameters:
    ------------
    snps: a MatrixTable of annotated vcf
    start: start position of gene
    end: end position of gene
    variant_type: one of ('variant', 'snv', 'indel')
    vset_test: an instance of VariantSetTest
    
    Returns:
    ---------
    cate_pvalues: a dict (keys: category, values: p-value)
    
    """
    # extracting specific variant type and the gene
    snps = extract_variant_type(snps, variant_type)
    snps = extract_gene(start, end, snps)
    vset = fillna_flip_snps(snps.GT.to_numpy())
    phred = slidingwindow.anno_phred.to_numpy()

    # individual analysis
    window_pvalues = dict()
    slidingwindow = SlidingWindow(snps, variant_type, window_length)
    for start_loc, end_loc in slidingwindow.windows:
        phred_ = phred[start_loc: end_loc]
        vset_ = vset[:, start_loc: end_loc]
        vset_test.input_vset(vset_, phred_)
        pvalues = vset_test.do_inference()
        window_pvalues[(start_loc, end_loc)] = pvalues


def check_input(args, log):
    pass


def run(args, log):
    # checking if input is valid
    start, end = check_input(args, log)

    # reading data
    with h5py.File(args.null_model, 'r') as file:
        covar = file['covar'][:]
        resid_ldr = file['resid_ldr'][:]
        var = file['var'][:]
        ids = file['ids'][:]

    bases = np.load(args.bases)
    inner_ldr = np.load(args.inner_ldr)

    vset_test = VariantSetTest(bases, inner_ldr, resid_ldr, covar, var)
    snps = hl.read_matrix_table(args.avcfmt)

    # extracting common ids
    snps_ids = set(snps.s.collect())
    common_ids = snps_ids.intersection(ids)
    snps = snps.filter_cols(hl.literal(common_ids).contains(snps.s))
    covar = covar[common_ids]
    resid_ldrs = resid_ldrs[common_ids]

    # single gene analysis (do parallel)
    res = single_gene_analysis(snps, start, end, args.window_length, vset_test)

   
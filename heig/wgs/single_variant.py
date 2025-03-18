import numpy as np
import pandas as pd
import hail as hl
from scipy.stats import chi2
from concurrent.futures import ThreadPoolExecutor
from heig.wgs.wgs2 import RVsumstats
from heig.wgs.coding import Coding
from heig.wgs.utils import *


"""
Computing variant-level associations

TODO:
1. support noncoding and w/o annotation

"""

class SingleVariant:
    def __init__(self, locus, half_score, cov_mat_diag, var, maf, mac, voxel_idxs):
        self.locus = locus
        self.half_score = half_score
        self.cov_mat_diag = cov_mat_diag
        self.var = var
        self.maf = maf
        self.mac = mac
        self.voxel_idxs = voxel_idxs

    def assoc(self, sig_thresh, threads):
        variant_var = self.var / self.cov_mat_diag
        variant_beta = self.half_score / self.cov_mat_diag
        variant_chisq = variant_beta ** 2 / variant_var
        sig_map = variant_chisq > sig_thresh
        sig_voxels = sig_map.any(axis=0)
        sig_variants_df = list()

        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [
                executor.submit(
                    self._assoc_voxel, 
                    i, 
                    voxel_idx, 
                    sig_voxels, 
                    sig_map, 
                    variant_beta, 
                    variant_var, 
                    variant_chisq
                )
                for i, voxel_idx in enumerate(self.voxel_idxs)
            ]

            for future in futures:
                result = future.result()
                if result is not None:
                    sig_variants_df.append(result)
        
        if len(sig_variants_df) > 0:
            sig_variants_df = pd.concat(sig_variants_df, axis=0)
        else:
            sig_variants_df = None
        
        return sig_variants_df
    
    def _assoc_voxel(self, i, voxel_idx, sig_voxels, sig_map, variant_beta, variant_var, variant_chisq):
        if sig_voxels[i]:
            sig_variants = self.locus[sig_map[:, i]].copy()
            sig_variants["MAF"] = self.maf[sig_map[:, i]]
            sig_variants["MAC"] = self.mac[sig_map[:, i]]
            sig_variants["BETA"] = variant_beta[sig_map[:, i], i]
            sig_variants["SE"] = np.sqrt(variant_var[sig_map[:, i], i])
            sig_variants["Z"] = sig_variants["BETA"] / sig_variants["SE"]
            sig_variants["P"] = chi2.sf(variant_chisq[sig_map[:, i], i], 1)
            sig_variants.insert(0, "INDEX", [voxel_idx + 1] * np.sum(sig_map[:, i]))
        else:
            sig_variants = None

        return sig_variants


def parse_sumstats_data(rv_sumstats):
    """
    Parsing sumstats for all variants
    
    """
    locus_df = rv_sumstats.locus.collect()
    locus_df = [(int(x.locus.contig.replace("chr", "")), x.locus.position, x.alleles[1], x.alleles[0], x.idx) for x in locus_df]
    locus_df = pd.DataFrame.from_records(locus_df, columns=['CHR', 'POS', 'A1', 'A2', 'idx']) # first is ref
    locus_df = locus_df.set_index('idx')

    numeric_idx = locus_df.index.values
    vset_half_covar_proj = np.array(rv_sumstats.vset_half_covar_proj[numeric_idx])
    vset_ld = rv_sumstats.vset_ld[numeric_idx][:, numeric_idx]
    half_ldr_score = np.array(rv_sumstats.half_ldr_score[numeric_idx])

    half_score = np.dot(half_ldr_score, rv_sumstats.bases.T)
    cov_mat_diag = np.array((vset_ld.diagonal() - np.sum(vset_half_covar_proj ** 2, axis=1))).reshape(-1, 1)
    maf = rv_sumstats.maf[numeric_idx]
    mac = rv_sumstats.mac[numeric_idx]

    return half_score, cov_mat_diag, maf, mac, locus_df


class DataParser:
    """
    Parsing sumstats for a variant category
    
    """
    def __init__(self, locus, rv_sumstats):
        variant_type = locus.variant_type.collect()[0]
        self.mask = Coding(locus, variant_type)
        self.rv_sumstats = rv_sumstats
        self.locus_df = self._get_locus_df()

    def _get_locus_df(self):
        locus_df = self.mask.annot.key_by().select('locus', 'alleles', 'idx').collect()
        locus_df = [(int(x.locus.contig.replace("chr", "")), x.locus.position, x.alleles[1], x.alleles[0], x.idx) for x in locus_df]
        locus_df = pd.DataFrame.from_records(locus_df, columns=['CHR', 'POS', 'A1', 'A2', 'idx']) # first is ref
        locus_df = locus_df.set_index('idx')
        return locus_df

    def parse(self, variant_category):
        mask_idx = self.mask.category_dict[variant_category]
        numeric_idx, _ = self.mask.parse_annot(mask_idx)
        vset_half_covar_proj = np.array(self.rv_sumstats.vset_half_covar_proj[numeric_idx])
        vset_ld = self.rv_sumstats.vset_ld[numeric_idx][:, numeric_idx]
        half_ldr_score = np.array(self.rv_sumstats.half_ldr_score[numeric_idx])

        half_score = np.dot(half_ldr_score, self.rv_sumstats.bases.T)
        cov_mat_diag = np.array((vset_ld.diagonal() - np.sum(vset_half_covar_proj ** 2, axis=1))).reshape(-1, 1)
        maf = self.rv_sumstats.maf[numeric_idx]
        mac = self.rv_sumstats.mac[numeric_idx]
        locus_df = self.locus_df.loc[numeric_idx]
    
        return half_score, cov_mat_diag, maf, mac, locus_df


def check_input(args, log):
    if args.rv_sumstats_part1 is None:
        raise ValueError("--rv-sumstats-part1 is required")
    if args.rv_sumstats_part2 is None:
        raise ValueError("--rv-sumstats-part2 is required")
    if args.spark_conf is None:
        raise ValueError("--spark-conf is required")
    if args.sig_thresh is None:
        args.sig_thresh = 5e-8
        log.info("Set significance threshold as 5e-8")
    if args.mac_min is None:
        args.mac_min = 100
        log.info(f"Set --mac-min as default 100")
    elif args.mac_min < 100:
        raise ValueError('--mac-min cannot be less than 100')
    if args.variant_category is not None:
        if args.annot_ht is None:
            raise ValueError("--annot-ht (FAVOR annotation) is required")
        variant_category = list()
        args.variant_category = [x.lower() for x in args.variant_category.split(",")]
        for category in args.variant_category:
            if category == "all":
                variant_category = ["all"]
                break
            elif category not in {
                "all",
                "plof",
                "plof_ds",
                "missense",
                "disruptive_missense",
                "synonymous",
                "ptv",
                "ptv_ds",
            }:
                log.info(f"Ingored invalid variant category {category}.")
            else:
                variant_category.append(category)
        if len(variant_category) == 0:
            raise ValueError("no valid variant category provided")
    else:
        variant_category = None
        
    return variant_category


def run(args, log):
    variant_category = check_input(args, log)

    try:
        init_hail(args.spark_conf, args.grch37, args.out, log)

        if args.extract_locus is not None:
            args.extract_locus = read_extract_locus(args.extract_locus, args.grch37, log)
        if args.exclude_locus is not None:
            args.exclude_locus = read_exclude_locus(args.exclude_locus, args.grch37, log)

        # reading data and selecting voxels and LDRs
        log.info((f"Read rare variant summary statistics from "
                  f"{args.rv_sumstats_part1} and {args.rv_sumstats_part2}"))
        rv_sumstats = RVsumstats(args.rv_sumstats_part1, args.rv_sumstats_part2)
        rv_sumstats.extract_exclude_locus(args.extract_locus, args.exclude_locus)
        rv_sumstats.extract_chr_interval(args.chr_interval)
        rv_sumstats.extract_maf(args.maf_min, args.maf_max)
        rv_sumstats.extract_mac(args.mac_min, args.mac_max)
        rv_sumstats.select_ldrs(args.n_ldrs)
        rv_sumstats.select_voxels(args.voxels)
        rv_sumstats.calculate_var()

        if variant_category is not None:
            # reading annotation
            log.info(f"Read functional annotations from {args.annot_ht}")
            annot = hl.read_table(args.annot_ht)
            locus = rv_sumstats.annotate(annot)
            data_parser = DataParser(locus, rv_sumstats)

            for category in variant_category:
                log.info(f"\nDoing analysis for {category} ...")
                half_score, cov_mat_diag, maf, mac, locus_df = data_parser.parse(category)
                single_variant = SingleVariant(
                    locus_df, half_score, cov_mat_diag, rv_sumstats.var, maf, mac, rv_sumstats.voxel_idxs
                )
                thresh_chisq = chi2.ppf(1 - args.sig_thresh, 1)
                sig_variants_df = single_variant.assoc(thresh_chisq, args.threads)
                if sig_variants_df is not None:
                    sig_variants_df.to_csv(f"{args.out}_{category}.txt", sep='\t', index=None, float_format="%.5e")
                    log.info(f"Saved rare variant association results to {args.out}_{category}.txt")
                else:
                    log.info(f"No significant rare variant associations for {category}.")
        else:
            log.info(f"\nDoing analysis for all variants ...")
            half_score, cov_mat_diag, maf, mac, locus_df = parse_sumstats_data(rv_sumstats)
            single_variant = SingleVariant(
                locus_df, half_score, cov_mat_diag, rv_sumstats.var, maf, mac, rv_sumstats.voxel_idxs
            )
            thresh_chisq = chi2.ppf(1 - args.sig_thresh, 1)
            sig_variants_df = single_variant.assoc(thresh_chisq, args.threads)
            if sig_variants_df is not None:
                sig_variants_df.to_csv(f"{args.out}.txt", sep='\t', index=None, float_format="%.5e")
                log.info(f"Saved rare variant association results to {args.out}.txt")
            else:
                log.info(f"No significant rare variant associations.")

    finally:
        clean(args.out)
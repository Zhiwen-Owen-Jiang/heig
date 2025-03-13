import os
import shutil
import numpy as np
import pandas as pd
import hail as hl
import heig.input.dataset as ds
from heig.wgs.relatedness import LOCOpreds
from heig.wgs.null import NullModel
from heig.wgs.coding import Coding
from heig.wgs.mt import SparseGenotype
from heig.wgs.vsettest import VariantSetTest
from heig.wgs.null import fit_null_model
from heig.wgs.wgs2 import get_interval, extract_chr_interval
from heig.wgs.utils import PermDistribution
from heig.wgs.utils import *


"""
Conditional analysis for a single variant set
user provides a list of SNPs to adjust for, prune by LD < 0.3

input:
1. sparse genotype data, geno_mt of nearby regions
2. rv sumstats
3. variant category
4. gene, chr_interval
5. null model
6. loco preds
7. a list of SNPs

"""


# def cond_adj(half_ldr_score, half_ldr_score_cond, cov_mat, cov_mat_cond, cov_cond):
#     """
#     half_ldr_score: (m, r) np.array, Z'(I-M)\Xi
#     half_ldr_score_cond: (m_c, r) np.array, Z_c'(I-M)\Xi
#     cov_mat: (m, m) np.array, Z'(I-M)Z
#     cov_mat_cond: (m_c, m_c) np.array, Z_c'(I-M)Z_c
#     cov_cond: (m, m_c) np.array, Z'(I-M)Z_c
    
#     """ 
#     cond_ld_inv = inv(cov_mat_cond)
#     proj_mat = np.dot(cov_cond, cond_ld_inv) # (m, m_c)
#     half_ldr_score_adj = half_ldr_score - np.dot(proj_mat, half_ldr_score_cond) # (m, r)
#     cov_mat_adj = cov_mat - np.dot(proj_mat, cov_cond.T) # (m, m)

#     return half_ldr_score_adj, cov_mat_adj


# def sumstats(resid_ldrs, covar, genotype, sparse_genotype):
#     """
#     resid_ldrs: (n, r) np.array, (I-M)\Xi
#     covar: (n, p) np.array, X
#     genotype: (m_c, n) np.array, Z_c'
    
#     """
#     half_ldr_score_cond = np.dot(genotype, resid_ldrs)
#     covar_U, _, covar_Vt = np.linalg.svd(covar, full_matrices=False)
#     half_covar_proj = np.dot(covar_U, covar_Vt).astype(np.float32)

#     # Z_c'X(X'X)^{-1/2} = Z_c'UV', where X = UDV'
#     geno_half = np.dot(genotype, half_covar_proj) # Z_c'UV'
#     sparse_geno_half = sparse_genotype @ half_covar_proj # Z'UV'

#     ld_cond = np.dot(genotype, genotype.T) # Z_c'Z_c
#     ld_cov = sparse_genotype @ genotype.T # Z'Z_c

#     # Z_c'(I-M)Z_c
#     cov_mat_cond = np.array(ld_cond - np.dot(geno_half, geno_half.T))
#     cov_cond = np.array(ld_cov - np.dot(sparse_geno_half, geno_half.T))

#     return half_ldr_score_cond, cov_mat_cond, cov_cond


def parse_gene(locus, gene, gene_interval, variant_category, vset, maf, mac, use_annot_weights, log):
    """
    Extracting genotype for the gene
    
    """
    locus = locus.add_index("idx")
    variant_type = locus.variant_type.collect()[0]
    geno_ref = locus.reference_genome.collect()[0]
    
    variant_set_locus = extract_chr_interval(
        locus, gene, gene_interval, geno_ref, log
    )
    if variant_set_locus is None:
        raise ValueError('less than two variants in the gene')
    coding = Coding(variant_set_locus, variant_type)
    chr, start, end = get_interval(variant_set_locus)
    mask_idx = coding.category_dict[variant_category]
    numeric_idx, phred_cate = coding.parse_annot(mask_idx, use_annot_weights)

    vset = vset[numeric_idx]
    maf = maf[numeric_idx]
    mac = mac[numeric_idx]

    log.info(f"{len(numeric_idx)} variants ({np.sum(mac)} alleles) for {variant_category}.")

    return vset, maf, mac, phred_cate, coding.annot_name, chr, start, end 


def adjust_cond(resid_ldr, covar, bases, vset, chr, loco_preds):
    """
    Adjusting for conditioned variants
    
    """
    # adjust for relatedness
    if loco_preds is not None:
        chr = int(chr.replace("chr", ""))
        resid_ldr = resid_ldr - loco_preds.data_reader(chr) # (I-M)\Xi, (n, r)
    resid_ldr = resid_ldr.astype(np.float32)

    # fit new null model
    resid_ldr = fit_null_model(covar, resid_ldr)

    # inner product of LDR residuals
    inner_ldr = np.dot(resid_ldr.T, resid_ldr).astype(np.float32) # \Xi'(I-M)\Xi, (r, r)
    var = np.sum(np.dot(bases, inner_ldr) * bases, axis=1)
    n_subs, n_covars = covar.shape
    var /= n_subs - n_covars  # (N, )

    # Z'(I-M)\Xi, (m, r)
    half_ldr_score = (vset @ resid_ldr).astype(np.float32)  

    # Z'(I-M)Z
    covar_U, _, covar_Vt = np.linalg.svd(covar, full_matrices=False)
    half_covar_proj = np.dot(covar_U, covar_Vt).astype(np.float32)
    vset_half_covar_proj = vset @ half_covar_proj # Z'UV'
    vset = vset.astype(np.uint16)
    vset_ld = vset @ vset.T # Z'Z
    cov_mat = np.array(vset_ld - vset_half_covar_proj @ vset_half_covar_proj.T)

    return half_ldr_score, cov_mat, var


def check_input(args, log):
    if args.spark_conf is None:
        raise ValueError("--spark-conf is required")
    if args.annot_ht is None:
        raise ValueError("--annot-ht (FAVOR annotation) is required")
    if args.null_model is None:
        raise ValueError("--null-model is required")
    if args.sparse_genotype is None:
        raise ValueError("--sparse-genotype is required")
    if args.geno_mt is None:
        raise ValueError("--geno-mt is required")
    if args.perm is None:
        raise ValueError("--perm is required")
    if args.variant_sets is None:
        raise ValueError("--variant-sets is required")
    log.info(f"{args.variant_sets.shape[0]} gene(s) in --variant-sets.")
    if args.variant_category is None:
        raise ValueError("--variant-category is required")
    
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
    if args.staar_only:
        log.info("Saving STAAR-O results only.")

    if args.mac_thresh is None:
        args.mac_thresh = 10
        log.info(f"Set --mac-thresh as default 10")
    elif args.mac_thresh < 0:
        raise ValueError("--mac-thresh must be greater than 0")
    
    # if args.maf_min is None:
    #     args.maf_min = 0
    #     log.info(f"Set --maf-min as default 0")

    if args.chr_interval_cond is not None:
        args.chr_interval_cond = args.chr_interval_cond.split(',')
    else:
        args.chr_interval_cond = []


def run(args, log):
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
        log.info((f"{sparse_genotype.vset.shape[1]} subjects and "
                  f"{sparse_genotype.vset.shape[0]} variants."))

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

        # read genotype data
        gprocessor = read_genotype_data(args, log)

        if args.extract_locus_cond is not None:
            args.extract_locus_cond = read_extract_locus(
                args.extract_locus_cond, args.grch37, log
            )
        if args.exclude_locus_cond is not None:
            args.exclude_locus_cond = read_exclude_locus(
                args.exclude_locus_cond, args.grch37, log
            )

        gprocessor.keep_remove_idvs(common_ids)
        gprocessor.extract_exclude_locus(args.extract_locus_cond, args.exclude_locus_cond)
        for chr_interval_cond in args.chr_interval_cond:
            gprocessor.extract_chr_interval(chr_interval_cond)
        # gprocessor.extract_chr_interval(args.chr_interval_cond)
        # gprocessor.do_processing('gwas')

        # extract common subjects and align data
        snp_mt_ids = np.array(gprocessor.subject_id(), dtype=str)
        common_ids = pd.MultiIndex.from_arrays([snp_mt_ids, snp_mt_ids], names=["FID", "IID"])
        sparse_genotype.keep(common_ids)
        
        if args.extract_locus is not None:
            args.extract_locus = read_extract_locus(args.extract_locus, args.grch37, log)
        if args.exclude_locus is not None:
            args.exclude_locus = read_exclude_locus(args.exclude_locus, args.grch37, log)

        sparse_genotype.extract_exclude_locus(args.extract_locus, args.exclude_locus)
        sparse_genotype.extract_chr_interval(args.chr_interval)
        sparse_genotype.extract_maf(args.maf_min, args.maf_max)
        sparse_genotype.extract_mac(args.mac_min, args.mac_max)

        # extract and align subjects with the genotype data
        null_model.keep(common_ids)
        null_model.remove_dependent_columns()
        log.info(f"{len(common_ids)} common subjects in the data.")
        log.info(
            (
                f"{null_model.covar.shape[1]} fixed effects in the covariates "
                "(including the intercept) after removing redundant effects."
            )
        )

        if args.loco_preds is not None:
            loco_preds.keep(common_ids)
        else:
            loco_preds = None
        
        # save temporary file
        temp_dir = get_temp_path(args.out)
        gprocessor.save_interim_data(temp_dir)
        gprocessor.check_valid()

        # LD pruning and get the genotype matrix
        log.info("Pruning conditional variants ...")
        gprocessor.ld_prune()
        cond_locus = gprocessor.get_locus()
        genotype = gprocessor.get_bm().to_numpy()
        # genotype = genotype[np.sum(genotype, axis=1) > 0]
        log.info(f"{genotype.shape[0]} variant(s) included in conditional analysis.")
        covar = np.concatenate([null_model.covar, genotype.T], axis=1, dtype=np.float32)

        # reading annotation
        log.info(f"Read functional annotations from {args.annot_ht}")
        annot = hl.read_table(args.annot_ht)
        log.info("Annotating sparse genotype data ...")
        sparse_genotype.annotate(annot, False)
        vset, locus, maf, mac = sparse_genotype.parse_data()

        # extracting gene
        gene_name, gene_interval = args.variant_sets.iloc[0]
        (
            vset, maf, mac, phred_cate, annot_name, chr, start, end 
        ) = parse_gene(
            locus, gene_name, gene_interval, args.variant_category, 
            vset, maf, mac, args.use_annot_weights, log
        )
        
        # adjusting for conditioned variants
        half_ldr_score, cov_mat, var = adjust_cond(
            null_model.resid_ldr, covar, null_model.bases, vset, chr, loco_preds
        )
        
        # index file
        index_file = IndexFile(f"{args.out}_result_index.txt")
        log.info(f"Saved result index file to {args.out}_result_index.txt")

        # reading permutation
        log.info(f"Read permutation from {args.perm}")
        perm = PermDistribution(args.perm)

        # analysis
        if args.voxels is None:
            args.voxels = np.arange(null_model.bases.shape[0])
        vset_test = VariantSetTest(null_model.bases, var, perm, args.voxels)
        is_rare = mac < args.mac_thresh
        cmac = int(np.sum(mac))
        vset_test.input_vset(half_ldr_score, cov_mat, maf, cmac, is_rare, phred_cate)
        pvalues, burden_test = vset_test.do_inference_tests(args.rv_tests, annot_name)
        cate_pvalues = {}
        cate_pvalues[args.variant_category] = {
            "n_variants": vset_test.n_variants,
            "cMAC": cmac,
            "pvalues": pvalues,
            "burden_test": burden_test,
        }

        cate_output, burden_output = format_output(
            cate_pvalues,
            args.voxels,
            False,
            args.sig_thresh
        )
        if cate_output is not None:
            out_path = f"{args.out}.txt"
            index_file.write_index(
                gene_name,
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
            cond_locus.to_csv(f"{args.out}_cond_variants.txt", header=None, index=None)
            log.info(f"\nSaved results for {gene_name} to {args.out}.txt")
            log.info(f"Saved conditioned variants to {args.out}_cond_variants.txt")
        else:
            log.info(f"\nNo significant results for {gene_name}.")

        if burden_output is not None:
            out_path = f"{args.out}_burden.txt"
            burden_output.to_csv(
                out_path,
                sep="\t",
                header=True,
                na_rep="NA",
                index=None,
                float_format="%.5e",
            )
            log.info(f"Saved burden results for {gene_name} to {args.out}_burden.txt")

    finally:
        clean(args.out)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
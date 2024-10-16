import os
import hail as hl
import shutil
import h5py
import numpy as np
import pandas as pd
from functools import reduce
from heig.wgs.staar import VariantSetTest, cauchy_combination, prepare_vset_test
import heig.input.dataset as ds
from heig.wgs.utils import *


OFFICIAL_NAME = {
    'plof': 'predicted loss of function (pLoF) variants',
    'synonymous': 'synonymous variants',
    'missense': 'missense variants',
    'disruptive_missense': 'disruptive missense variants',
    'plof_ds': 'pLoF variants with deleterious score',
    'ptv': 'protein truncating variants (PTV)',
    'ptv_ds': 'PTV with eleterious score'
}


class Coding:
    def __init__(self, snps_mt, variant_type, use_annotation_weights=True):
        """
        Extracting coding variants, generate annotation, and get index for each category

        Parameters:
        ------------
        snps_mt: a hail.MatrixTable of genotype data with annotation attached
            for a specific gene and variant type
        variant_type: one of ('variant', 'snv', 'indel')
        use_annotation_weights: if using annotation weights
        
        """
        self.snps_mt = snps_mt

        gencode_exonic_category = self.snps_mt.fa[Annotation_name_catalog['GENCODE.EXONIC.Category']]
        gencode_category = self.snps_mt.fa[Annotation_name_catalog['GENCODE.Category']]
        valid_exonic_categories = hl.literal({'stopgain', 'stoploss', 'nonsynonymous SNV', 'synonymous SNV'})
        valid_categories = hl.literal({'splicing', 'exonic;splicing', 'ncRNA_splicing', 'ncRNA_exonic;splicing'})
        lof_in_coding_snps_mt = (valid_exonic_categories.contains(gencode_exonic_category) | 
                                 valid_categories.contains(gencode_category))
        self.snps_mt = self.snps_mt.filter_rows(lof_in_coding_snps_mt)
        if self.snps_mt.rows().count() == 0:
            raise ValueError('no variants remaining')
        
        self.gencode_exonic_category = self.snps_mt.fa[Annotation_name_catalog['GENCODE.EXONIC.Category']]
        self.gencode_category = self.snps_mt.fa[Annotation_name_catalog['GENCODE.Category']]
        self.metasvm_pred = self.snps_mt.fa[Annotation_name_catalog['MetaSVM']]
        self.category_dict = self.get_category(variant_type)
        
        if variant_type == 'snv' and use_annotation_weights:
            self.annot_cols = [Annotation_name_catalog[annot_name] for annot_name in Annotation_name]
            self.annot_name = Annotation_name
        else:
            self.annot_cols, self.annot_name = None, None

    def get_category(self, variant_type):
        """
        Extracting different categories of variants

        Parameters:
        ------------
        variant_type: one of ('variant', 'snv', 'indel')

        Returns:
        ---------
        category_dict: a dict containing variant indices
        
        """
        category_dict = dict()
        set1 = hl.literal({'stopgain', 'stoploss'})
        set2 = hl.literal({'splicing', 'exonic;splicing', 'ncRNA_splicing', 'ncRNA_exonic;splicing'})
        set3 = hl.literal({'splicing', 'exonic;splicing'})
        set4 = hl.literal({'frameshift deletion', 'frameshift insertion'})

        category_dict['plof'] = (set1.contains(self.gencode_exonic_category) | 
                                 set2.contains(self.gencode_category))
        category_dict['synonymous'] = self.gencode_exonic_category == 'synonymous SNV'
        category_dict['missense'] = self.gencode_exonic_category == 'nonsynonymous SNV'
        category_dict['disruptive_missense'] = category_dict['missense'] & (self.metasvm_pred == 'D')
        category_dict['plof_ds'] = category_dict['plof'] | category_dict['disruptive_missense']

        ptv_snv = set1.contains(self.gencode_exonic_category) | set3.contains(self.gencode_category)
        ptv_indel = set4.contains(self.gencode_exonic_category)
        if variant_type == 'snv':
            category_dict['ptv'] = ptv_snv
            category_dict['ptv_ds'] = ptv_snv | category_dict['disruptive_missense']
        elif variant_type == 'indel':
            category_dict['ptv'] = ptv_indel
            category_dict['ptv_ds'] = ptv_indel | category_dict['disruptive_missense']
        else:
            category_dict['ptv'] = ptv_snv | ptv_indel 
            category_dict['ptv_ds'] = category_dict['ptv'] | category_dict['disruptive_missense']

        return category_dict


def single_gene_analysis(snps_mt, variant_type, vset_test,
                         variant_category, use_annotation_weights, 
                         log):
    """
    Single gene analysis

    Parameters:
    ------------
    snps_mt: a MatrixTable of annotated geno
    variant_type: one of ('variant', 'snv', 'indel')
    vset_test: an instance of VariantSetTest
    variant_category: which category of variants to analyze,
        one of ('all', 'plof', 'plof_ds', 'missense', 'disruptive_missense',
        'synonymous', 'ptv', 'ptv_ds')
    use_annotation_weights: if using annotation weights
    log: a logger
    
    Returns:
    ---------
    cate_pvalues: a dict (keys: category, values: p-value)
    
    """
    # getting annotations and specific categories of variants    
    coding = Coding(snps_mt, variant_type, use_annotation_weights)

    # individual analysis
    cate_pvalues = dict()
    for cate, idx in coding.category_dict.items():
        if variant_category[0] != 'all' and cate not in variant_category: 
            cate_pvalues[cate] = None
        else:
            snps_mt_cate = coding.snps_mt.filter_rows(idx)
            if snps_mt_cate.rows().count() <= 1:
                log.info(f'Less than 2 variants for {OFFICIAL_NAME[cate]}, skip.')
                continue
            if coding.annot_cols is not None:
                annot_phred = snps_mt_cate.fa.select(*coding.annot_cols).collect()
                phred_cate = np.array([[getattr(row, col) for col in coding.annot_cols] for row in annot_phred])
            else:
                phred_cate = None
            maf, is_rare, vset = prepare_vset_test(snps_mt_cate)
            vset_test.input_vset(vset, maf, is_rare, phred_cate)
            log.info(f'Doing analysis for {OFFICIAL_NAME[cate]} ({vset_test.n_variants} variants) ...')
            pvalues = vset_test.do_inference(coding.annot_name)
            cate_pvalues[cate] = {'n_variants': vset_test.n_variants, 'pvalues': pvalues}

    if 'missense' in cate_pvalues and 'disruptive_missense' in cate_pvalues:
        cate_pvalues['missense'] = process_missense(cate_pvalues['missense'], 
                                                    cate_pvalues['disruptive_missense'])
        
    return cate_pvalues


def process_missense(m_pvalues, dm_pvalues):
    """
    Incoporating disruptive missense results into missense

    Parameters:
    ------------
    m_pvalues: pvalues of missense variants
    dm_pvalues: pvalues of disruptive missense variants

    Returns:
    ---------
    m_pvalues: pvalues of missense variants incoporating disruptive missense results
    
    """

    dm_pvalues = dm_pvalues['pvalues']
    n_m_variants = m_pvalues['n_variants']
    m_pvalues = m_pvalues['pvalues']

    m_pvalues['SKAT(1,25)-Disruptive'] = dm_pvalues['SKAT(1,25)']
    m_pvalues['SKAT(1,1)-Disruptive'] = dm_pvalues['SKAT(1,1)']
    m_pvalues['Burden(1,25)-Disruptive'] = dm_pvalues['Burden(1,25)']
    m_pvalues['Burden(1,1)-Disruptive'] = dm_pvalues['Burden(1,1)']
    m_pvalues['ACAT-V(1,25)-Disruptive'] = dm_pvalues['ACAT-V(1,25)']
    m_pvalues['ACAT-V(1,1)-Disruptive'] = dm_pvalues['ACAT-V(1,1)']

    columns = m_pvalues.columns.values
    skat_1_25 = np.array([column.startswith('SKAT(1,25)') for column in columns])
    skat_1_1 = np.array([column.startswith('SKAT(1,1)') for column in columns])
    burden_1_25 = np.array([column.startswith('Burden(1,25)') for column in columns])
    burden_1_1 = np.array([column.startswith('Burden(1,1)') for column in columns])
    acatv_1_25 = np.array([column.startswith('ACAT-V(1,25)') for column in columns])
    acatv_1_1 = np.array([column.startswith('ACAT-V(1,1)') for column in columns])

    staar_s_1_25 = cauchy_combination(m_pvalues.loc[:, skat_1_25].values.T)
    staar_s_1_1 = cauchy_combination(m_pvalues.loc[:, skat_1_1].values.T)
    staar_b_1_25 = cauchy_combination(m_pvalues.loc[:, burden_1_25].values.T)
    staar_b_1_1 = cauchy_combination(m_pvalues.loc[:, burden_1_1].values.T)
    staar_a_1_25 = cauchy_combination(m_pvalues.loc[:, acatv_1_25].values.T)
    staar_a_1_1 = cauchy_combination(m_pvalues.loc[:, acatv_1_1].values.T)
    
    m_pvalues['STAAR-S(1,25)'] = staar_s_1_25
    m_pvalues['STAAR-S(1,1)'] = staar_s_1_1
    m_pvalues['STAAR-B(1,25)'] = staar_b_1_25
    m_pvalues['STAAR-B(1,1)'] = staar_b_1_1
    m_pvalues['STAAR-A(1,25)'] = staar_a_1_25
    m_pvalues['STAAR-A(1,1)'] = staar_a_1_1

    all_columns = [skat_1_25, skat_1_1, burden_1_25, burden_1_1, acatv_1_25, acatv_1_1]
    all_columns = reduce(np.logical_or, all_columns)
    all_columns = np.concatenate([all_columns, np.ones(6, dtype=bool)])
    m_pvalues['STAAR-O'] = cauchy_combination(m_pvalues.loc[:, all_columns].values.T)
    m_pvalues = {'n_variants': n_m_variants, 'pvalues': m_pvalues}

    return m_pvalues


def format_output(cate_pvalues, chr, start, end, n_variants, n_voxels, variant_category):
    """
    organizing pvalues to a structured format

    Parameters:
    ------------
    cate_pvalues: a pd.DataFrame of pvalues of the variant category
    chr: chromosome
    start: start position of the gene
    end: end position of the gene
    n_variants: #variants of the category
    n_voxels: #voxels of the image
    variant_category: which category of variants to analyze,
        one of ('all', 'plof', 'plof_ds', 'missense', 'disruptive_missense',
        'synonymous', 'ptv', 'ptv_ds')
    
    Returns:
    ---------
    output: a pd.DataFrame of pvalues with metadata

    """
    meta_data = pd.DataFrame({'INDEX': range(1, n_voxels+1), 
                              'VARIANT_CATEGORY': variant_category,
                              'CHR': chr,
                              'START': start, 'END': end,
                              'N_VARIANT': n_variants})
    output = pd.concat([meta_data, cate_pvalues], axis=1)
    return output


def check_input(args, log):
    # required arguments
    if args.geno_mt is None:
        raise ValueError('--geno-mt is required')
    if args.null_model is None:
        raise ValueError('--null-model is required')
    if args.range is None:
        raise ValueError('--range is required')
    
    # required files must exist
    if not os.path.exists(args.geno_mt):
        raise FileNotFoundError(f"{args.geno_mt} does not exist")
    if not os.path.exists(args.null_model):
        raise FileNotFoundError(f"{args.null_model} does not exist")

    # optional arguments
    if args.n_ldrs is not None and args.n_ldrs <= 0:
        raise ValueError('--n-ldrs should be greater than 0')
    
    if args.maf_min is not None:
        if args.maf_min > 0.5 or args.maf_min < 0:
            raise ValueError('--maf-min must be greater than 0 and less than 0.5')
    else:
        args.maf_min = 0

    if args.variant_type is None:
        args.variant_type = 'snv'
        log.info(f"Set --variant-type as default 'snv'")
    else:
        args.variant_type = args.variant_type.lower()
        if args.variant_type not in {'snv', 'variant', 'indel'}:
            raise ValueError("--variant-type must be one of ('variant', 'snv', 'indel')")
        
    if args.variant_category is None:
        variant_category = ['all']
        log.info(f"Set --variant-category as default 'all'")
    else:   
        variant_category = list()
        args.variant_category = [x.lower() for x in args.variant_category.split(',')]
        for category in args.variant_category:
            if category == 'all':
                variant_category = ['all']
                break
            if category not in {'all', 'plof', 'plof_ds', 'missense', 
                                'disruptive_missense','synonymous', 'ptv', 'ptv_ds'}:
                log.info(f'Ingore invalid variant category {category}.')
            else:
                variant_category.append(category)
        if len(variant_category) == 0:
            raise ValueError('no valid variant category provided')
        if 'missense' in variant_category and 'disruptive_missense' not in variant_category:
            variant_category.append('disruptive_missense')
    
    if args.maf_max is None:
        args.maf_max = 0.01
        log.info(f"Set --maf-max as default 0.01")
    elif args.maf_max > 0.5 or args.maf_max <= 0 or args.maf_max <= args.maf_min:
        raise ValueError(('--maf-max must be greater than 0, less than 0.5, '
                          'and greater than --maf-min'))
    
    if args.mac_thresh is None:
        args.mac_thresh = 10
        log.info(f"Set --mac-thresh as default 10")
    elif args.mac_thresh < 0:
        raise ValueError('--mac-thresh must be greater than 0')
    args.mac_thresh = int(args.mac_thresh)

    if args.use_annotation_weights is None:
        args.use_annotation_weights = False
        log.info(f"Set --use-annotation-weights as False")

    # process arguments
    try:
        start, end = args.range.split(',')
        start_chr, start_pos = [int(x) for x in start.split(':')]
        end_chr, end_pos = [int(x) for x in end.split(':')]
    except:
        raise ValueError('--range should be in this format: <CHR>:<POS1>,<CHR>:<POS2>')
    if start_chr != end_chr:
        raise ValueError((f'starting with chromosome {start_chr} '
                            f'while ending with chromosome {end_chr} '
                            'is not allowed'))
    if start_pos > end_pos:
        raise ValueError((f'starting with {start_pos} '
                            f'while ending with position is {end_pos} '
                            'is not allowed'))

    temp_path = os.path.join(os.path.dirname(args.out), 'temp')
    i = 0
    while os.path.exists(temp_path + str(i)):
        i += 1
    temp_path += str(i)

    if args.grch37 is None or not args.grch37:
        geno_ref = 'GRCh38'
    else:
        geno_ref = 'GRCh37'
    log.info(f'Set {geno_ref} as the reference genome.')

    return start_chr, start_pos, end_pos, variant_category, temp_path, geno_ref


def run(args, log):
    # checking if input is valid
    chr, start, end, variant_category, temp_path, geno_ref = check_input(args, log)

    # reading data for unrelated subjects
    log.info(f'Read null model from {args.null_model}')
    with h5py.File(args.null_model, 'r') as file:
        covar = file['covar'][:]
        resid_ldr = file['resid_ldr'][:]
        ids = file['id'][:].astype(str)
        bases = file['bases'][:]
    
    # subset voxels
    if args.voxel is not None:
        if np.max(args.voxel) + 1 <= bases.shape[0] and np.min(args.voxel) >= 0:
            log.info(f'{len(args.voxel)} voxels included.')
        else:
            raise ValueError('--voxel index (one-based) out of range')
    else:
        args.voxel = np.arange(bases.shape[0])
    bases = bases[args.voxel]

    # keep selected LDRs
    if args.n_ldrs is not None:
        resid_ldr, bases = keep_ldrs(args.n_ldrs, resid_ldr, bases)
        log.info(f'Keep the top {args.n_ldrs} LDRs and bases.')
        
    # keep subjects
    if args.keep is not None:
        keep_idvs = ds.read_keep(args.keep)
        log.info(f'{len(keep_idvs)} subjects in --keep.')
    else:
        keep_idvs = None
    common_ids = get_common_ids(ids, keep_idvs)

    # extract SNPs
    if args.extract is not None:
        keep_snps = ds.read_extract(args.extract)
        log.info(f"{len(keep_snps)} SNPs in --extract.")
    else:
        keep_snps = None
    
    # read genotype data
    hl.init(quiet=True)
    hl.default_reference = geno_ref

    log.info(f'Reading genotype data from {args.geno_mt}')
    gprocessor = GProcessor.read_matrix_table(args.geno_mt, geno_ref=geno_ref, 
                                              variant_type=args.variant_type,
                                              maf_min=args.maf_min, maf_max=args.maf_max)
    
    # do preprocessing
    log.info(f"Processing genotype data ...")
    gprocessor.extract_snps(keep_snps)
    gprocessor.extract_idvs(common_ids)
    gprocessor.do_processing(mode='wgs')
    gprocessor.extract_gene(chr=chr, start=start, end=end)
    
    # save processsed data for faster analysis
    if not args.not_save_genotype_data:
        log.info(f'Save preprocessed genotype data to {temp_path}')
        gprocessor.save_interim_data(temp_path)

    try:
        gprocessor.check_valid()
        # extract and align subjects with the genotype data
        snps_mt_ids = gprocessor.subject_id()
        idx_common_ids = extract_align_subjects(ids, snps_mt_ids)
        resid_ldr = resid_ldr[idx_common_ids]
        covar = covar[idx_common_ids]
        covar = remove_dependent_columns(covar)
        log.info(f'{len(idx_common_ids)} common subjects in the data.')
        log.info(f"{covar.shape[1]} fixed effects in the covariates after removing redundant effects.\n")

        # single gene analysis
        vset_test = VariantSetTest(bases, resid_ldr, covar)
        cate_pvalues = single_gene_analysis(gprocessor.snps_mt, args.variant_type, vset_test,
                                            variant_category, args.use_annotation_weights, 
                                            log)
        
        # format output
        n_voxels = bases.shape[0]
        log.info('')
        for cate, cate_results in cate_pvalues.items():
            cate_output = format_output(cate_results['pvalues'], chr, start, end,
                                        cate_results['n_variants'], n_voxels, cate)
            out_path = f'{args.out}_chr{chr}_start{start}_end{end}_{cate}.txt'
            cate_output.to_csv(out_path, sep='\t', header=True, na_rep='NA', 
                               index=None, float_format='%.5e')
            log.info(f'Save results for {OFFICIAL_NAME[cate]} to {out_path}')
    finally:
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
            log.info(f'Removed preprocessed genotype data at {temp_path}')
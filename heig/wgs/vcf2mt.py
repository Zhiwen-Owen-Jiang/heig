import os
import hail as hl
import heig.input.dataset as ds
from heig.wgs.utils import GProcessor 

"""
TODO: add an argument for inputing hail config with a JSON file

"""



SELECTED_ANNOT = {
    'apc_conservation': hl.tfloat32,
    'apc_epigenetics': hl.tfloat32,
    'apc_epigenetics_active': hl.tfloat32,
    'apc_epigenetics_repressed': hl.tfloat32,
    'apc_epigenetics_transcription': hl.tfloat32,
    'apc_local_nucleotide_diversity': hl.tfloat32,
    'apc_mappability': hl.tfloat32,
    'apc_protein_function': hl.tfloat32,
    'apc_transcription_factor': hl.tfloat32,
    'cage_tc': hl.tstr,
    'metasvm_pred': hl.tstr,
    'rsid': hl.tstr,
    'fathmm_xf': hl.tfloat32,
    'genecode_comprehensive_category': hl.tstr,
    'genecode_comprehensive_info': hl.tstr,
    'genecode_comprehensive_exonic_category': hl.tstr,
    'genecode_comprehensive_exonic_info': hl.tstr,
    'genehancer': hl.tstr,
    'linsight': hl.tfloat32,
    'cadd_phred': hl.tfloat32,
    'rdhs': hl.tstr
}


class Annotation:
    def __init__(self, annot, geno_ref):
        self.annot = annot
        self.geno_ref = geno_ref
        
        self._create_keys()
        self._drop_rename()
        self._convert_datatype()
        self._add_more_annot()

    @classmethod
    def read_annot(cls, favor_db, geno_ref, *args, **kwargs):
        """
        Reading FAVOR annotation
        
        """
        annot = hl.import_table(favor_db, *args, **kwargs)
        return cls(annot, geno_ref)
    
    def _create_keys(self):
        """
        Creating keys for merging
        
        """
        if self.geno_ref == 'GRCh38':
            self.annot = self.annot.annotate(chromosome=hl.str('chr') + self.annot.chromosome)
        chromosome = self.annot.chromosome
        position = hl.int(self.annot.position)
        ref_allele = self.annot.ref_vcf
        alt_allele = self.annot.alt_vcf
        locus = hl.locus(chromosome, position, reference_genome=self.geno_ref)

        self.annot = self.annot.annotate(locus=locus, alleles=[ref_allele, alt_allele])
        self.annot = self.annot.key_by('locus', 'alleles')

    def _drop_rename(self):
        """
        Dropping and renaming annotation names
        
        """
        for filed in ('apc_conservation', 'apc_local_nucleotide_diversity'):
            self.annot = self.annot.drop(filed)

        self.annot = self.annot.rename(
            {'apc_conservation_v2': 'apc_conservation',
             'apc_local_nucleotide_diversity_v3': 'apc_local_nucleotide_diversity',
             'apc_protein_function_v3': 'apc_protein_function'}
        )

        annot_name = list(self.annot.row_value.keys())

        for field in annot_name:
            if field not in SELECTED_ANNOT:
                self.annot = self.annot.drop(field)
    
    def _convert_datatype(self):
        """
        Converting numerical columns to float

        Parameters:
        ------------
        anno: a Table of functional annotation

        Returns:
        ---------
        anno: a Table of functional annotation

        """
        self.annot = self.annot.annotate(
            apc_conservation = hl.float32(self.annot.apc_conservation),
            apc_epigenetics = hl.float32(self.annot.apc_epigenetics),
            apc_epigenetics_active = hl.float32(self.annot.apc_epigenetics_active),
            apc_epigenetics_repressed = hl.float32(self.annot.apc_epigenetics_repressed),
            apc_epigenetics_transcription = hl.float32(self.annot.apc_epigenetics_transcription),
            apc_local_nucleotide_diversity = hl.float32(self.annot.apc_local_nucleotide_diversity),
            apc_mappability = hl.float32(self.annot.apc_mappability),
            apc_protein_function = hl.float32(self.annot.apc_protein_function), 
            apc_transcription_factor = hl.float32(self.annot.apc_transcription_factor),
            fathmm_xf = hl.float32(self.annot.fathmm_xf),
            linsight = hl.float32(self.annot.linsight),        
            cadd_phred = hl.float32(self.annot.cadd_phred)                                                    
        )

    def _add_more_annot(self):
        annot_local_div = -10 * hl.log10(1 - 10 ** (-self.annot.apc_local_nucleotide_diversity/10))
        self.annot = self.annot.annotate(
            cadd_phred = hl.coalesce(self.annot.cadd_phred, 0),
            apc_local_nucleotide_diversity2 = annot_local_div
        )


def read_vcf(dir, geno_ref, block_size=1024):
    if dir.endswith('vcf'):
        force_bgz = False
    elif dir.endswith('vcf.gz') or dir.endswith('vcf.bgz'):
        force_bgz = True
    else:
        raise ValueError('VCF suffix is incorrect')

    if geno_ref == 'GRCh38':
        recode = {f"{i}":f"chr{i}" for i in (list(range(1, 23)) + ['X', 'Y'])}
    else:
        recode = {f"chr{i}":f"{i}" for i in (list(range(1, 23)) + ['X', 'Y'])}

    vcf_mt = hl.import_vcf(dir, force_bgz=force_bgz, reference_genome=geno_ref,
                           contig_recoding=recode, block_size=block_size)
    return vcf_mt


def check_input(args, log):
    # required arguments
    if args.vcf is None:
        raise ValueError('--vcf is required')
    if args.favor_db is None:
        raise ValueError('--favor-db is required')
    
    # required files must exist
    if not os.path.exists(args.vcf):
        raise FileNotFoundError(f"{args.vcf} does not exist")
    if not os.path.exists(args.favor_db):
        raise FileNotFoundError(f"{args.favor_db} does not exist")
    args.favor_db = os.path.join(args.favor_db, 'chr*.csv')
    
    # process arguments
    if args.grch37 is None or not args.grch37:
        geno_ref = 'GRCh38'
    else:
        geno_ref = 'GRCh37'
    log.info(f'Set {geno_ref} as the reference.')

    if args.block_size is None:
        args.block_size = 1024
        log.info(f'Set --block-size as default 1024.')

    return geno_ref


def run(args, log):
    # check input and init
    geno_ref = check_input(args, log)
    hl.init(quiet=True)
    hl.default_reference = geno_ref

    # convert VCF to MatrixTable
    log.info(f'Read VCF from {args.vcf}')
    vcf_mt = read_vcf(args.vcf, geno_ref, block_size=args.block_size)

    # keep idvs
    gprocessor = GProcessor(vcf_mt)
    if args.keep is not None:
        keep_idvs = ds.read_keep(args.keep)
        log.info(f'{len(keep_idvs)} subjects in --keep.')
        gprocessor.extract_idvs(keep_idvs)
        
    # extract SNPs
    if args.extract is not None:
        keep_snps = ds.read_extract(args.extract)
        log.info(f"{len(keep_snps)} variants in --extract.")
        gprocessor.extract_snps(keep_snps)
    vcf_mt = gprocessor.snps_mt

    # read annotation and preprocess
    log.info(f'Read FAVOR annotation from {args.favor_db}')
    log.info(f'Processing annotation and annotating the VCF file ...')
    annot = Annotation.read_annot(args.favor_db, geno_ref, delimiter=',', 
                                  missing='', quote='"')
    vcf_mt = vcf_mt.annotate_rows(fa=annot.annot[vcf_mt.locus, vcf_mt.alleles])

    # save the MatrixTable
    out_dir = f'{args.out}_annotated_vcf.mt'
    vcf_mt.write(out_dir, overwrite=True)
    log.info(f'Write annotated VCF to MatrixTable {out_dir}')

import os
import time
import argparse
import traceback
import numexpr
import numpy as np
import heig.input.dataset as ds
from heig.utils import GetLogger, sec_to_str


# os.environ['NUMEXPR_MAX_THREADS'] = '8'
# numexpr.set_num_threads(int(os.environ['NUMEXPR_MAX_THREADS']))

VERSION = "1.5.0-alpha"
MASTHEAD = (
    "******************************************************************************\n"
)
MASTHEAD += "* Highly Efficient Imaging Genetics (HEIG)\n"
MASTHEAD += f"* Version {VERSION}\n"
MASTHEAD += f"* Zhiwen Jiang and Hongtu Zhu\n"
MASTHEAD += (
    f"* Department of Biostatistics, University of North Carolina at Chapel Hill\n"
)
MASTHEAD += f"* GNU General Public License v3\n"
MASTHEAD += f"* Correspondence: owenjf@live.unc.edu, zhiwenowenjiang@gmail.com\n"
MASTHEAD += (
    "******************************************************************************\n"
)


parser = argparse.ArgumentParser(
    description=f"\n Highly Efficient Imaging Genetics (HEIG) v{VERSION}"
)

common_parser = parser.add_argument_group(title="Common arguments")
herigc_parser = parser.add_argument_group(
    title="Arguments specific to heritability and (cross-trait) genetic correlation analysis"
)
image_parser = parser.add_argument_group(title="Arguments specific to reading images")
fpca_parser = parser.add_argument_group(title="Arguments specific to functional PCA")
ldr_parser = parser.add_argument_group(title="Arguments specific to constructing LDRs")
make_ld_parser = parser.add_argument_group(
    title="Arguments specific to making an LD matrix and its inverse"
)
sumstats_parser = parser.add_argument_group(
    title="Arguments specific to organizing and preprocessing GWAS summary statistics"
)
voxelgwas_parser = parser.add_argument_group(
    title="Arguments specific to recovering voxel-level GWAS results"
)
gwas_parser = parser.add_argument_group(
    title="Arguments specific to doing genome-wide association analysis"
)
partition_h2_parser = parser.add_argument_group(
    title="Arguments specific to partitioning heritability"
)
relatedness_parser = parser.add_argument_group(
    title="Arguments specific to removing genetic relatedness in LDRs"
)
make_mt_parser = parser.add_argument_group(
    title="Arguments specific to making a hail.MatrixTable of genotype data"
)
rv_null_parser = parser.add_argument_group(
    title="Arguments specific to the null model of rare variant analysis"
)
rv_sumstats_parser= parser.add_argument_group(
    title="Arguments specific to generating summary statistics for rare variant analysis"
)
rv_annotation_parser = parser.add_argument_group(
    title="Arguments specific to processing rare variant annotations"
)  
rv_coding_parser = parser.add_argument_group(
    title="Arguments specific to analyzing coding rare variants using FAVOR annotations"
)
rv_noncoding_parser = parser.add_argument_group(
    title="Arguments specific to analyzing non-coding rare variants using FAVOR annotations"
)
rv_parser = parser.add_argument_group(
    title="Arguments specific to analyzing rare variants w/ or w/o annotations"
)
cluster_parser = parser.add_argument_group(
    title="Arguments specific to cluster inference"
)
rv_cluster_parser = parser.add_argument_group(
    title="Arguments specific to cluster inference for rare variant associations"
)
rv_cond_parser = parser.add_argument_group(
    title="Arguments specific to conditional analysis for rare variant associations"
)
rv_single_parser = parser.add_argument_group(
    title="Arguments specific to single rare variant analysis"
)
tfce_parser = parser.add_argument_group(
    title="Arguments specific to threshold-free cluster enhancement (TFCE)"
)
permutation_parser = parser.add_argument_group(
    title="Arguments specific to permutation for burden test"
)


# module arguments
herigc_parser.add_argument(
    "--heri-gc",
    action="store_true",
    help="Heritability and (cross-trait) genetic correlation analysis.",
)
image_parser.add_argument("--read-image", action="store_true", help="Reading images.")
fpca_parser.add_argument("--fpca", action="store_true", help="Functional PCA.")
ldr_parser.add_argument("--make-ldr", action="store_true", help="Constructing LDRs.")
make_ld_parser.add_argument(
    "--ld-matrix", action="store_true", help="Making an LD matrix and its inverse."
)
sumstats_parser.add_argument(
    "--sumstats",
    action="store_true",
    help="Organizing and preprocessing GWAS summary statistics.",
)
voxelgwas_parser.add_argument(
    "--voxel-gwas", action="store_true", help="Recovering voxel-level GWAS results."
)
gwas_parser.add_argument(
    "--gwas", action="store_true", help="Genome-wide association analysis."
)
partition_h2_parser.add_argument(
    "--partition-h2", action="store_true", help="Partitioning heritability."
)
relatedness_parser.add_argument(
    "--relatedness", action="store_true", help="Removing genetic relatedness in LDRs."
)
make_mt_parser.add_argument(
    "--make-mt", action="store_true", help="Making a hail.MatrixTable of genotype data."
)
rv_null_parser.add_argument(
    "--rv-null",
    action="store_true",
    help="Fitting the null model for rare variant analysis.",
)
rv_sumstats_parser.add_argument(
    "--make-rv-sumstats",
    action="store_true",
    help="Generating summary statistics for rare variant analysis.",
)
rv_annotation_parser.add_argument(
    "--rv-annot",
    action="store_true",
    help="Preprocessing rare variant annotations.",
)
rv_coding_parser.add_argument(
    "--rv-coding",
    action="store_true",
    help="Analyzing rare coding variants using FAVOR annotations.",
)
rv_noncoding_parser.add_argument(
    "--rv-noncoding",
    action="store_true",
    help="Analyzing rare non-coding variants using FAVOR annotations.",
)
rv_parser.add_argument(
    "--rv",
    action="store_true",
    help="Analyzing rare variants w/ or w/o customized annotations.",
)
cluster_parser.add_argument(
    "--cluster",
    action="store_true",
    help="Generating null distribution of cluster size (number of associated voxels).",
)
rv_cluster_parser.add_argument(
    "--rv-cluster",
    action="store_true",
    help="Generating null distribution of cluster size for rare variant associations.",
)
rv_cond_parser.add_argument(
    "--rv-cond",
    action="store_true",
    help="Doing conditional analysis for rare variant associations.",
)
rv_single_parser.add_argument(
    "--rv-single",
    action="store_true",
    help="Doing single rare variant analysis.",
)
tfce_parser.add_argument(
    "--tfce",
    action="store_true",
    help="Evaluating significant rare variant associations by TFCE.",
)
permutation_parser.add_argument(
    "--permute",
    action="store_true",
    help="Generating null distribution of burden score statistics",
)



# common arguments
common_parser.add_argument("--out", help="Prefix of output.")
common_parser.add_argument(
    "--image",
    help=(
        "Directory to processed raw images in HDF5 format. "
        "Supported modules: --read-image, --fpca, --make-ldr."
    ),
)
common_parser.add_argument(
    "--n-ldrs",
    type=int,
    help=(
        "Number of LDRs. Supported modules: "
        "--make-ldr, --fpca, --heri-gc, --voxel-gwas, --gwas, --partition-h2, --cluster, "
        "--relatedness, --rv-null, --make-rv-sumstats, --rv-coding, "
        "--rv-noncoding, --rv, --rv-cluster, --rv-cond, --rv-single, --permute."
    ),
)
common_parser.add_argument(
    "--ldr-sumstats",
    help=(
        "Prefix of preprocessed LDR GWAS summary statistics. "
        "Supported modules: --heri-gc, --voxel-gwas, --partition-h2."
    ),
)
common_parser.add_argument(
    "--bases",
    help=(
        "Directory to functional bases. Supported modules: "
        "--make-ldr, --partition-h2, --heri-gc, --voxel-gwas, --cluster, --rv-null."
    ),
)
common_parser.add_argument(
    "--ldr-cov",
    help=(
        "Directory to variance-covariance marix of LDRs. "
        "Supported modules: --heri-gc, --voxel-gwas, --partition-h2, --cluster."
    ),
)
common_parser.add_argument(
    "--keep",
    help=(
        "Subject ID file(s). Multiple files are separated by comma. "
        "Only common subjects appearing in all files will be kept (logical and). "
        "Each file should be tab or space delimited, "
        "with the first column being FID and the second column being IID. "
        "Other columns will be ignored. "
        "Each row contains only one subject. "
        "Supported modules: --read-image, --fpca, --make-ldr, --ld-matrix, "
        "--gwas, --cluster, --make-mt, --relatedness, --rv-null, --make-rv-sumstats, "
        "--rv-cluster, --rv-cond, --permute."
    ),
)
common_parser.add_argument(
    "--remove",
    help=(
        "Subject ID file(s). Multiple files are separated by comma. "
        "Subjects appearing in any files will be removed (logical or). "
        "Each file should be tab or space delimited, "
        "with the first column being FID and the second column being IID. "
        "Other columns will be ignored. "
        "Each row contains only one subject. "
        "If a subject appears in both --keep and --remove, --remove takes precedence. "
        "Supported modules: --read-image, --fpca, --make-ldr, --gwas, --cluster, "
        "--make-mt, --relatedness, --rv-null, --make-rv-sumstats, --rv-cluster, "
        "--rv-cond, --permute."
    ),
)
common_parser.add_argument(
    "--extract",
    help=(
        "SNP file(s). Multiple files are separated by comma. "
        "Only common SNPs appearing in all files will be extracted (logical and). "
        "Each file should be tab or space delimited, "
        "with the first column being rsID. "
        "Other columns will be ignored. "
        "Each row contains only one SNP. "
        "Supported modules: --heri-gc, --ld-matrix, --voxel-gwas, --gwas, "
        "--partition-h2, --cluster, --make-mt, --relatedness."
    ),
)
common_parser.add_argument(
    "--extract-locus",
    help=(
        "Variant file(s). Multiple files are separated by comma. "
        "Only common Variants appearing in all files will be extracted (logical and). "
        "Each file should be tab or space delimited, "
        "with the first column being CHR:POS. "
        "Other columns will be ignored. "
        "Each row contains only one variant. "
        "Supported modules: --make-mt, --make-rv-sumstats, --rv-coding, --rv-noncoding, "
        "--rv-annot, --rv, --rv-cluster, --rv-cond, --rv-single, --permute."
    ),
)
common_parser.add_argument(
    "--exclude",
    help=(
        "SNP file(s). Multiple files are separated by comma. "
        "SNPs appearing in any files will be excluded (logical or). "
        "Each file should be tab or space delimited, "
        "with the first column being rsID. "
        "Other columns will be ignored. "
        "Each row contains only one SNP. "
        "Supported modules: --heri-gc, --ld-matrix, --voxel-gwas, --gwas, "
        "--partition-h2, --cluster, --make-mt, --relatedness."
    ),
)
common_parser.add_argument(
    "--exclude-locus",
    help=(
        "Variant file(s). Multiple files are separated by comma. "
        "Variants appearing in any files will be excluded (logical or). "
        "Each file should be tab or space delimited, "
        "with the first column being CHR:POS. "
        "Other columns will be ignored. "
        "Each row contains only one variant. "
        "Supported modules: --make-mt, --make-rv-sumstats, --rv-coding, --rv-noncoding, "
        "--rv-annot, --rv, --rv-cluster, --rv-cond, --rv-single, --permute."
    ),
)
common_parser.add_argument(
    "--maf-min",
    type=float,
    help=(
        "Minimum minor allele frequency for screening variants. "
        "Supported modules: --ld-matrix, --sumstats, --gwas, --cluster, --make-mt, "
        "--relatedness, --make-rv-sumstats, --rv-coding, --rv-noncoding, "
        "--rv, --rv-cluster, --rv-cond, --rv-single, --permute."
    ),
)
common_parser.add_argument(
    "--maf-max",
    type=float,
    help=(
        "Maximum minor allele frequency for screening variants. "
        "Supported modules: --ld-matrix, --sumstats, --gwas, --cluster, --make-mt, "
        "--relatedness, --make-rv-sumstats, --rv-coding, --rv-noncoding, "
        "--rv, --rv-cluster, --rv-cond, --rv-single, --permute."
    ),
)
common_parser.add_argument(
    "--mac-min",
    type=int,
    help=(
        "Minimum minor allele count for screening variants. "
        "Supported modules: --gwas, --cluster, --make-mt, "
        "--relatedness, --make-rv-sumstats, --rv-coding, --rv-noncoding, "
        "--rv, --rv-cluster, --rv-cond, --rv-single, --rv-permute."
    ),
)
common_parser.add_argument(
    "--mac-max",
    type=int,
    help=(
        "Maximum minor allele count for screening variants. "
        "Supported modules: --gwas, --cluster, --make-mt, "
        "--relatedness, --make-rv-sumstats, --rv-coding, --rv-noncoding, "
        "--rv, --rv-cluster, --rv-cond, --rv-single, --rv-permute."
    ),
)
common_parser.add_argument(
    "--hwe",
    type=float,
    help=(
        "A HWE p-value threshold. "
        "Variants with a HWE p-value less than the threshold "
        "will be removed."
        "Supported modules: --make-mt, --gwas, --cluster, --relatedness, "
        "--make-rv-sumstats."
    ),
)
common_parser.add_argument(
    "--call-rate",
    type=float,
    help=(
        "A genotype call rate threshold, equivalent to 1 - missing rate. "
        "Variants with a call rate less than the threshold "
        "will be removed."
        "Supported modules: --gwas, --relatedness, --make-mt, --cluster, "
        "--make-rv-sumstats."
    ),
)
common_parser.add_argument(
    "--covar",
    help=(
        "Directory to covariate file. "
        "The file should be tab or space delimited, with each row only one subject. "
        "Supported modules: --make-ldr, --gwas, --cluster, --relatedness, --rv-null."
    ),
)
common_parser.add_argument(
    "--cat-covar-list",
    help=(
        "List of categorical covariates to include in the analysis. "
        "Multiple covariates are separated by comma. "
        "Supported modules: --make-ldr, --gwas, --cluster, --relatedness, --rv-null."
    ),
)
common_parser.add_argument(
    "--bfile",
    help=(
        "Prefix of PLINK bfile triplets. "
        "When estimating LD matrix and its inverse, two prefices should be provided "
        "and seperated by a comma, e.g., `prefix1,prefix2`. "
        "When doing GWAS, only one prefix is allowed. "
        "Supported modules: --ld-matrix, --make-mt."
    ),
)
common_parser.add_argument(
    "--chr-interval", "--range",
    help=(
        "A segment of chromosome, e.g. `3:1000000-2000000`, "
        "from chromosome 3 bp 1000000 to chromosome 3 bp 2000000. "
        "Cross-chromosome is not allowed. And the end position must "
        "be greater than the start position. "
        "Supported modules: --voxel-gwas, --gwas, --cluster, --make-mt, --make-rv-sumstats, "
        "--rv-annot, --rv-coding, --rv-noncoding, --rv, --rv-cluster, --rv-cond, --rv-single, "
        "--permute."
    ),
)
common_parser.add_argument(
    "--voxels", "--voxel",
    help=(
        "One-based index of voxel or a file containing voxels. "
        "Supported modules: --read-image, --fpca, --make-ldr, "
        "--voxel-gwas, --partition-h2, --cluster, --rv-coding, --rv-noncoding, --rv, "
        "--rv-cluster, --rv-cond, --rv-single, --permute."
    ),
)
common_parser.add_argument(
    "--ldrs",
    help=(
        "Directory to LDR file. "
        "Supported modules: --gwas, --cluster, --relatedness, --rv-null."
    ),
)
common_parser.add_argument(
    "--geno-mt",
    help=(
        "Directory to genotype MatrixTable. "
        "Supported modules: --gwas, --make-mt, --cluster, --relatedness, "
        "--rv-cond."
    ),
)
common_parser.add_argument(
    "--grch37",
    action="store_true",
    help=(
        "Using reference genome GRCh37. Otherwise using GRCh38. "
        "Supported modules: --gwas, --make-mt, --cluster, --relatedness, --rv-annot, "
        "--make-rv-sumstats, --rv-coding, --rv-noncoding, --rv, --rv-cluster, "
        "--rv-cond, --rv-single, --permute"
    ),
)
common_parser.add_argument(
    "--variant-type",
    help=(
        "Variant type (case insensitive), "
        "must be one of ('variant', 'snv', 'indel'). "
        "Supported modules: --gwas, --make-mt, --cluster, --relatedness, "
        "--make-rv-sumstats."
    ),
)
common_parser.add_argument(
    "--not-save-genotype-data",
    action="store_true",
    help=(
        "Not saving preprocessed genotype data (Deprecated)."
    ),
)
common_parser.add_argument(
    "--partition",
    help=(
        "Genome partition file. "
        "The file should be tab or space delimited without header, "
        "with the first column being chromosome, "
        "the second column being the start position, "
        "and the third column being the end position."
        "Each row contains only one LD block. "
        "Supported modules: --ld-matrix, --relatedness."
    ),
)
common_parser.add_argument(
    "--threads",
    type=int,
    help=(
        "number of threads. "
        "Supported modules: --read-image, --sumstats, --voxel-gwas, "
        "--heri-gc, --make-ldr, --partition-h2, --cluster, --relatedness, "
        "--make-rv-sumstats, --rv-single, --tfce."
    ),
),
common_parser.add_argument(
    "--spark-conf",
    help=(
        "Spark configuration file. "
        "Supported modules: --relatedness, --gwas, --make-mt, --cluster, "
        "--make-rv-sumstats, --rv-annot, --rv-coding, --rv-noncoding, --rv, "
        "--rv-cluster, --rv-cond, --rv-single, --permute."
    ),
),
common_parser.add_argument(
    "--loco-preds",
    help=(
        "Leave-one-chromosome-out prediction file. "
        "Supported modules: --gwas, --cluster, --make-rv-sumstats, --rv-cluster, "
        "--rv-cond, --permute."
    ),
)
common_parser.add_argument(
    "--annot-ht",
    help=(
        "Directory to processed functional annotations "
        "for rare variant analysis in hail.Table format. "
        "Supported modules: --rv-coding, --rv-noncoding, --rv, --rv-cluster, "
        "--rv-cond, --rv-single, --permute."
    ),
)
common_parser.add_argument(
    "--rv-sumstats-part1",
    "--rv-sumstats",
    help=(
        "Prefix of rare variants summary statistics (part1) specific to images. "
        "Supported modules: --rv-coding, --rv-noncoding, --rv, --rv-single."
    )
)
common_parser.add_argument(
    "--rv-sumstats-part2",
    help=(
        "Prefix of rare variants summary statistics (part1) not specific to images. "
        "Supported modules: --rv-coding, --rv-noncoding, --rv, --rv-single."
    )
)
common_parser.add_argument(
    "--annot-cols",
    help=(
        "Annotation columns. Multiple columns are separated by comma. "
        "Supported modules: --rv-annot, --rv, --rv-cond."
    ),
)
common_parser.add_argument(
    "--variant-category",
    help=(
        "Variant category (case insensitive), "
        "must be one or some of ('all', 'plof', 'plof_ds', 'missense', "
        "'disruptive_missense', 'synonymous', 'ptv', 'ptv_ds') for coding variants, "
        "or one or some of ('all', 'upstream', 'downstream', 'promoter_cage', 'promoter_dhs', "
        "'enhancer_cage', 'enhancer_dhs') for noncoding variants, "
        "where 'all' means all categories; "
        "multiple categories should be separated by comma. "
        "Supported modules: --rv-coding, --rv-noncoding, --rv-cluster, --rv-cond, "
        "--rv-single, --tfce, --permute."
    ),
)
common_parser.add_argument(
    "--variant-sets",
    help=(
        "Variant sets file."
        "The file should be tab or space delimited without header. "
        "Each row contains only one variant set in format "
        "<gene name> <chr:start,chr:end>. "
        "Supported modules: --rv-coding, --rv-noncoding, --rv, --rv-cluster, "
        "--rv-cond, --permute."
    )
)
common_parser.add_argument(
    "--sig-thresh",
    type=float,
    help=(
        "p-Value threshold for significance, "
        "can be specified in a decimal 0.00000005 "
        "or in scientific notation 5e-08. "
        "Supported modules: --voxel-gwas, --cluster, --rv-coding, "
        "--rv-noncoding, --rv, --rv-cluster, --rv-cond, --rv-single, --tfce, "
        "--permute."
    ),
)
common_parser.add_argument(
    "--staar-only",
    action="store_true",
    help=(
        "Saving STAAR-O results only, the omnibus test aggregating all "
        "methods and functional annotations. "
        "Supported modules: --rv-coding, --rv-noncoding, --rv."
    )
)
common_parser.add_argument(
    "--mac-thresh",
    type=int,
    help=(
        "A minor allele count threshold. "
        "Variants with a MAC less than the threshold "
        "will be marked as a super rare variants in ACAT-V analysis. "
        "Default: 10. "
        "Supported modules: --rv-coding, --rv-noncoding, --rv, --rv-cluster."
    ),
)
common_parser.add_argument(
    "--null-model",
    help=(
        "Directory to null model. "
        "Supported modules: --make-rv-sumstats, --rv-cluster, --rv-cond, --permute."
    )
)
common_parser.add_argument(
    "--sparse-genotype",
    help=(
        "Prefix of sparse genotype data. "
        "Supported modules: --make-rv-sumstats, --rv-cluster, --rv-cond, --permute."
    )
)
common_parser.add_argument(
    "--n-bootstrap", "--n-points",
    type=float,
    help=(
        "Number of bootstrap samples or number of points in a null distribution. "
        "Supported modules: --cluster, --rv-cluster, --permute."
    )
)
common_parser.add_argument(
    "--cmac-min",
    type=int,
    help=(
        "The minimum of cumulative MAC in a variant set. "
        "Supported modules: --rv-coding, --rv-noncoding, --rv, --rv-cluster, --permute."
    )
)
common_parser.add_argument(
    "--cmac-max",
    type=int,
    help=(
        "The maximum of cumulative MAC in a variant set. "
        "Supported modules: --rv-coding, --rv-noncoding, --rv, --rv-cluster, --permute."
    )
)
common_parser.add_argument(
    "--coord-dir",
    help=(
        "Directory to mask or complementary image for coordinates. "
        "It should be a NIFTI file (nii.gz) for NIFTI images; "
        "a GIFTI file (gii) for CIFTI2 surface data; "
        "a FreeSurfer surface mesh file (.pial) for FreeSurfer morphometry data. "
        "Supported modules: --read-image, --tfce."
    ),
)
common_parser.add_argument(
    "--rv-tests",
    help=(
        "Gene based tests for rare variant analysis. "
        "Must be one or some of ('burden', 'skat', 'staar'). "
        "Multiple tests are separated by comma. "
        "STAAR will run all tests and combine individual p-values. "
        "Supported modules: --rv-coding, --rv-noncoding, --rv, --rv-cluster, --rv-cond."
    ),
)
common_parser.add_argument(
    "--use-annot-weights", 
    action="store_true", 
    help=(
        "Using annotation weights. "
        "Supported modules: --rv-coding, --rv-noncoding, --rv, --rv-cluster, --rv-cond."
    )
)
common_parser.add_argument(
    "--perm",
    help=(
        "Directory to permutation results. "
        "Supported modules: --rv-coding, --rv-noncoding, --rv, --rv-cluster, --rv-cond."
    )
)

# arguments for herigc.py
herigc_parser.add_argument(
    "--ld-inv",
    help=(
        "Prefix of inverse LD matrix. Multiple matrices can be specified using {:}, "
        "e.g., `ld_inv_chr{1:22}_unrel`."
    ),
)
herigc_parser.add_argument(
    "--ld",
    help=(
        "Prefix of LD matrix. Multiple matrices can be specified using {:}, "
        "e.g., `ld_chr{1:22}_unrel`."
    ),
)
herigc_parser.add_argument(
    "--y2-sumstats",
    help="Prefix of preprocessed GWAS summary statistics of non-imaging traits.",
)
herigc_parser.add_argument(
    "--overlap",
    action="store_true",
    help=(
        "Flag for indicating sample overlap between LDR summary statistics "
        "and non-imaging summary statistics. Only effective if --y2-sumstats is specified."
    ),
)
herigc_parser.add_argument(
    "--heri-only",
    action="store_true",
    help=(
        "Flag for only computing voxelwise heritability "
        "and skipping voxelwise genetic correlation within images."
    ),
)

# arguments for image.py
image_parser.add_argument(
    "--image-txt",
    help=(
        "Directory to images in txt format. "
        "The file should be tab or space delimited, with each row only one subject."
    ),
)
image_parser.add_argument(
    "--coord-txt",
    help=(
        "Directory to images in txt format. "
        "The file should be tab or space delimited, with each row only one voxel (vertex)."
    ),
)
image_parser.add_argument(
    "--image-dir",
    help=(
        "Directory to images. All images in the directory with matched suffix "
        "(see --image-suffix) will be loaded. "
        "Multiple directories can be provided and separated by comma. "
        "--keep can be used to load a subset of images (see --keep). "
        "The supported formats include NIFTI and CIFTI images "
        "and FreeSurfer morphometry data file."
    ),
)
image_parser.add_argument(
    "--image-suffix",
    help=(
        "Suffix of images. HEIG requires the name of each image in the format <ID><suffix>, "
        "e.g., `1000001_masked_FAskel.nii.gz`, where `1000001` is the ID "
        "and `_masked_FAskel.nii.gz` is the suffix. "
        "HEIG will collect ID for each image. "
        "Multiple suffixes can be specified and separated by comma "
        "and the number of directories must match the number of suffices."
    ),
)
image_parser.add_argument(
    "--image-list",
    help=(
        "Directory to multiple image HDF5 files, separated by comma."
    ),
)

# arguments for fpca.py
fpca_parser.add_argument(
    "--all-pc",
    action="store_true",
    help=(
        "Flag for generating all principal components which is min(n_subs, n_voxels), "
        "which may take longer time and very memory consuming."
    ),
)
fpca_parser.add_argument(
    "--bw-opt",
    type=float,
    help=(
        "The bandwidth you want to use in kernel smoothing. "
        "HEIG will skip searching the optimal bandwidth. "
        "For images of any dimension, just specify one number, e.g, 0.5 "
        "for 3D images."
    ),
)
fpca_parser.add_argument(
    "--skip-smoothing",
    action='store_true',
    help=(
        "Skipping kernel smoothing. "
    ),
)

# arguments for ldmatrix.py
make_ld_parser.add_argument(
    "--ld-regu",
    help=(
        "Regularization for LD matrix and its inverse. "
        "Two values should be separated by a comma and between 0 and 1, "
        "e.g., `0.85,0.80`."
    ),
)

# arguments for sumstats.py
sumstats_parser.add_argument(
    "--ldr-gwas",
    help=(
        "Directory to raw LDR GWAS summary statistics files. "
        "Multiple files can be provided using {:}, e.g., `ldr_gwas{1:10}.txt`."
    ),
)
sumstats_parser.add_argument(
    "--ldr-gwas-heig",
    help=(
        "Directory to raw LDR GWAS summary statistics files produced by --gwas. "
        "Multiple files can be provided using {:}, e.g., `ldr_gwas{1:10}.parquet`. "
        "One file may contain multiple LDRs. LDRs in these files must be in order."
    ),
)
sumstats_parser.add_argument(
    "--y2-gwas", help="Directory to raw non-imaging GWAS summary statistics file."
)
sumstats_parser.add_argument("--n", type=float, help="Sample size. A positive number.")
sumstats_parser.add_argument("--n-col", help="Sample size column.")
sumstats_parser.add_argument("--chr-col", help="Chromosome column.")
sumstats_parser.add_argument("--pos-col", help="Position column.")
sumstats_parser.add_argument("--snp-col", help="SNP column.")
sumstats_parser.add_argument("--a1-col", help="A1 column. The effective allele.")
sumstats_parser.add_argument("--a2-col", help="A2 column. The non-effective allele.")
sumstats_parser.add_argument(
    "--effect-col",
    help=(
        "Genetic effect column, usually refers to beta or odds ratio, "
        "should be specified in this format `BETA,0` where "
        "BETA is the column name and 0 is the null value. "
        "For odds ratio, the null value is 1."
    ),
)
sumstats_parser.add_argument(
    "--se-col",
    help=(
        "Standard error column. For odds ratio, the standard error must be in "
        "log(odds ratio) scale."
    ),
)
sumstats_parser.add_argument("--z-col", help="Z score column.")
sumstats_parser.add_argument("--p-col", help="p-Value column.")
sumstats_parser.add_argument("--maf-col", help="Minor allele frequency column.")
sumstats_parser.add_argument("--info-col", help="INFO score column.")
sumstats_parser.add_argument(
    "--info-min", type=float, help="Minimum INFO score for screening SNPs."
)

# arguments for voxelgwas.py

# arguments for partition_h2.py
partition_h2_parser.add_argument("--ref-ld-chr", help="Reference LD scores used as predictors.")
partition_h2_parser.add_argument("--w-ld-chr", help="LD scores used as weights.")
partition_h2_parser.add_argument("--frqfile-chr", help="MAF of reference SNPs.")

# arguments for relatedness.py
relatedness_parser.add_argument(
    "--bsize", type=int, help="Block size of genotype blocks. Default: 5000."
)

# arguments for gwas.py
gwas_parser.add_argument(
    "--ldr-col", help="One-based LDR indices. E.g., `3,4,5,6` and `3:6`, must be consecutive"
)

# arguments for mt.py
make_mt_parser.add_argument(
    "--qc-mode", help="Genotype data QC mode, either gwas or wgs. Default: gwas."
)
make_mt_parser.add_argument(
    "--save-sparse-genotype", action="store_true", 
    help="Saving sparse genotype for rare variant analysis."
)
make_mt_parser.add_argument(
    "--vcf",
    help="Direcotory to a VCF file."
)
make_mt_parser.add_argument(
    "--skip-qc",
    action="store_true",
    help="Skipping QC genotype data."
)
make_mt_parser.add_argument(
    "--lift-over",
    help="Target reference genome, either `GRCh37` or `GRCh38`."
)

# arguments for annotation.py
rv_annotation_parser.add_argument(
    "--favor-annot", 
    help=(
        "Directory to unzipped FAVOR annotation files. "
        "For multiple files, using * to match any string of characters. "
        "E.g., favor_db/chr*.csv."
    ),
)
rv_annotation_parser.add_argument(
    "--general-annot", 
    help=(
        "Directory to general annotation files. "
        "Each file should be tab or space delimited. "
        "Missing values are not allowed. "
        "Use double quote marks `\"`. "
        "For multiple files, using * to match any string of characters. "
        "E.g., chr*.csv."
    ),
)

# arguments for wgs.py
rv_sumstats_parser.add_argument(
    "--bandwidth",
    type=int,
    help="Bandwidth of banded LD matrix. Default: 3000000 (3 MB)."
)
rv_sumstats_parser.add_argument(
    "--make-part2",
    action="store_true",
    help="Generating rare variants summary statistics (part2) not specific to images."
)

# arguments for slidingwindow.py
rv_parser.add_argument(
    "--window-length",
    type=int,
    help="Length of sliding window. E.g., 10000 means 10kb."
)
rv_parser.add_argument(
    "--sliding-length",
    type=int,
    help="Sliding length. E.g., 10000 means 10kb. Default: --window-length // 2."
)

# arguments for condition.py
rv_cond_parser.add_argument(
    "--extract-locus-cond",
    help=(
        "Variant file(s). Multiple files are separated by comma. "
        "Only common Variants appearing in all files will be extracted (logical and). "
        "Each file should be tab or space delimited, "
        "with the first column being CHR:POS. "
        "Other columns will be ignored. "
        "Each row contains only one variant."
    ),
)
rv_cond_parser.add_argument(
    "--exclude-locus-cond",
    help=(
        "Variant file(s). Multiple files are separated by comma. "
        "Variants appearing in any files will be excluded (logical or). "
        "Each file should be tab or space delimited, "
        "with the first column being CHR:POS. "
        "Other columns will be ignored. "
        "Each row contains only one variant."
    ),
)
rv_cond_parser.add_argument(
    "--chr-interval-cond",
    help=(
        "Segment(s) of chromosome, e.g. `3:1000000-2000000,3:3000000-4000000`, "
        "from chromosome 3 bp 1000000 to chromosome 3 bp 2000000, "
        "and bp 3000000 to bp 4000000. "
        "Cross-chromosome is not allowed. And the end position must "
        "be greater than the start position."
    ),
)

# arguments for tfce.py
tfce_parser.add_argument(
    "--results-idx",
    help=(
        "Results index file returned by --rv-coding. Can be one file or multiple files. "
        "E.g., results_index_chr{1:22}.txt"
    )
)
tfce_parser.add_argument(
    "--null-assoc",
    help="Null associations returned by --rv-cluster."
)
tfce_parser.add_argument(
    "--tfce-quantile-level",
    type=float,
    help=(
        "TFCE quantile level to exclude associations, "
        "e.g., 0.95 means the 95% quantile. Default: 0."
    )
)
tfce_parser.add_argument(
    "--tfce-null",
    help="Directory to null distribution of TFCE."
)
tfce_parser.add_argument(
    "--sig-thresh2",
    type=float,
    help=(
        "Significant threshold for the most significant voxel in a cluster. "
        "Clusters do not meet this threshold will be excluded."
    )
)

# arguments for permutation.py
permutation_parser.add_argument(
    "--perm-list",
    help=(
        "a list of permutation files. "
        "Multiple files can be provided using {:}, e.g., `perm{1:10}.h5`."
    )
)


def check_accepted_args(module, args, log):
    """
    Checking if the provided arguments are accepted by the module

    """
    accepted_args = {
        "heri_gc": {
            "out",
            "heri_gc",
            "ld_inv",
            "ld",
            "y2_sumstats",
            "overlap",
            "heri_only",
            "n_ldrs",
            "ldr_sumstats",
            "bases",
            "ldr_cov",
            "extract",
            "exclude",
            "threads",
        },
        "read_image": {
            "out",
            "read_image",
            "keep",
            "remove",
            "voxels",
            "image_txt",
            "coord_txt",
            "image_dir",
            "image_suffix",
            "image",
            "image_list",
            "coord_dir",
            "threads",
        },
        "fpca": {
            "out",
            "fpca",
            "image",
            "voxels",
            "all_pc",
            "n_ldrs",
            "keep",
            "remove",
            "bw_opt",
            "skip_smoothing",
        },
        "make_ldr": {
            "out",
            "make_ldr",
            "image",
            "voxels",
            "bases",
            "n_ldrs",
            "covar",
            "cat_covar_list",
            "keep",
            "remove",
            "threads",
        },
        "ld_matrix": {
            "out",
            "ld_matrix",
            "partition",
            "ld_regu",
            "bfile",
            "keep",
            "extract",
            "maf_min",
        },
        "sumstats": {
            "out",
            "sumstats",
            "ldr_gwas",
            "y2_gwas",
            "ldr_gwas_heig",
            "n",
            "n_col",
            "chr_col",
            "pos_col",
            "snp_col",
            "a1_col",
            "a2_col",
            "effect_col",
            "se_col",
            "z_col",
            "p_col",
            "maf_col",
            "maf_min",
            "info_col",
            "info_min",
            "threads",
        },
        "voxel_gwas": {
            "out",
            "voxel_gwas",
            "sig_thresh",
            "voxels",
            "chr_interval",
            "extract",
            "exclude",
            "ldr_sumstats",
            "n_ldrs",
            "ldr_cov",
            "bases",
            "threads",
        },
        "gwas": {
            "out",
            "gwas",
            "keep",
            "remove",
            "extract",
            "exclude", 
            "maf_min",
            "maf_max",
            "mac_max",
            "mac_min",
            "variant_type",
            "hwe",
            "call_rate",
            "chr_interval",
            "ldr_col",
            "ldrs",
            "n_ldrs",
            "grch37",
            "geno_mt",
            "covar",
            "cat_covar_list",
            "loco_preds",
            "spark_conf",
        },
        "partition_h2": {
            "partition_h2",
            "out",
            "ref_ld_chr",
            "w_ld_chr",
            "frqfile_chr",
            "extract",
            "exclude",
            "bases",
            "n_ldrs",
            "voxels",
            "ldr_sumstats",
            "ldr_cov",
            "threads",
        },
        "relatedness": {
            "relatedness",
            "out",
            "keep",
            "remove",
            "extract",
            "exclude",
            "ldrs",
            "covar",
            "cat_covar_list",
            "partition",
            "maf_min",
            "maf_max",
            "mac_max",
            "mac_min",
            "variant_type",
            "hwe",
            "call_rate",
            "n_ldrs",
            "grch37",
            "geno_mt",
            "bsize",
            "spark_conf",
            "threads"
        }, 
        "make_mt": {
            "make_mt",
            "out",
            "keep",
            "remove",
            "extract",
            "exclude",
            "extract_locus",
            "exclude_locus",
            "bfile",
            "vcf",
            "geno_mt",
            "maf_min",
            "maf_max",
            "mac_max",
            "mac_min",
            "variant_type",
            "hwe",
            "call_rate",
            "chr_interval",
            "spark_conf",
            "qc_mode",
            "save_sparse_genotype",
            "grch37",
            "skip_qc",
            "lift_over",
        },
        "rv_null": {
            "rv_null",
            "out",
            "ldrs",
            "n_ldrs",
            "bases",
            "covar",
            "cat_covar_list",
            "keep",
            "remove",
        },
        "make_rv_sumstats":{
            "make_rv_sumstats",
            "out",
            "sparse_genotype",
            "null_model",
            "variant_type",
            "maf_max",
            "maf_min",
            "mac_max",
            "mac_min",
            "hwe",
            "call_rate",
            "chr_interval",
            "bandwidth",
            "n_ldrs",
            "keep",
            "remove",
            "extract_locus",
            "exclude_locus",
            "grch37",
            "loco_preds",
            "make_part2",
            "spark_conf",
            "threads"
        },
        "rv_annot":{
            "rv_annot",
            "out",
            "spark_conf",
            "grch37",
            "favor_annot",
            "general_annot",
            "annot_cols",
            "extract_locus",
            "exclude_locus",
            "chr_interval"
        },
        "rv_coding":{
            "rv_coding",
            "out",
            "rv_sumstats_part1",
            "rv_sumstats_part2",
            "variant_category",
            "variant_sets",
            "extract_locus",
            "exclude_locus",
            "chr_interval",
            "maf_max",
            "maf_min",
            "mac_max",
            "mac_min",
            "mac_thresh",
            "spark_conf",
            "grch37",
            "n_ldrs",
            "voxels",
            "annot_ht",
            "staar_only",
            "cmac_min",
            "cmac_max",
            "rv_tests",
            "use_annot_weights",
            "sig_thresh",
            "perm"
        },
        "rv_noncoding":{
            "rv_noncoding",
            "out",
            "rv_sumstats_part1",
            "rv_sumstats_part2",
            "variant_category",
            "variant_sets",
            "extract_locus",
            "exclude_locus",
            "chr_interval",
            "maf_max",
            "maf_min",
            "mac_max",
            "mac_min",
            "mac_thresh",
            "spark_conf",
            "grch37",
            "n_ldrs",
            "voxels",
            "annot_ht",
            "staar_only",
            "cmac_min",
            "cmac_max",
            "rv_tests",
            "use_annot_weights",
            "sig_thresh",
            "perm"
        },
        "rv":{
            "rv",
            "out",
            "rv_sumstats_part1",
            "rv_sumstats_part2",
            "extract_locus",
            "exclude_locus",
            "chr_interval",
            "maf_max",
            "maf_min",
            "mac_max",
            "mac_min",
            "mac_thresh",
            "spark_conf",
            "variant_sets",
            "grch37",
            "n_ldrs",
            "voxels",
            "annot_ht",
            "annot_cols",
            "window_length",
            "sliding_length",
            "staar_only",
            "cmac_min",
            "cmac_max",
            "rv_tests",
            "use_annot_weights",
            "sig_thresh",
            "perm",
        },
        "cluster":{
            "cluster",
            "out",
            "ldrs",
            "covar",
            "voxels",
            "cat_covar_list",
            "spark_conf",
            "grch37",
            "geno_mt",
            "bases",
            "ldr_cov",
            "sig_thresh",
            "n_bootstrap",
            "n_ldrs",
            "keep",
            "remove",
            "extract",
            "exclude",
            "maf_min",
            "maf_max",
            "mac_max",
            "mac_min",
            "hwe",
            "call_rate",
            "variant_type",
            "chr_interval",
            "loco_preds",
            "threads"
        },
        "rv_cluster":{
            "rv_cluster",
            "out",
            "null_model",
            "spark_conf",
            "grch37",
            "sparse_genotype",
            "sig_thresh",
            "n_bootstrap",
            "voxels",
            "n_ldrs",
            "keep",
            "remove",
            "extract_locus",
            "exclude_locus",
            "maf_min",
            "maf_max",
            "mac_max",
            "mac_min",
            "mac_thresh",
            "chr_interval",
            "loco_preds",
            "annot_ht",
            "variant_sets",
            "variant_category",
            "cmac_min",
            "cmac_max",
            "rv_tests",
            "use_annot_weights",
            "perm"
        },
        "rv_cond":{
            "rv_cond",
            "out",
            "null_model",
            "spark_conf",
            "grch37",
            "sparse_genotype",
            "sig_thresh",
            "voxels",
            "n_ldrs",
            "keep",
            "remove",
            "extract_locus",
            "exclude_locus",
            "extract_locus_cond",
            "exclude_locus_cond",
            "maf_min",
            "maf_max",
            "mac_max",
            "mac_min",
            "mac_thresh",
            "chr_interval",
            "chr_interval_cond",
            "loco_preds",
            "annot_ht",
            "annot_cols",
            "variant_sets",
            "variant_category",
            "geno_mt",
            "perm",
            "rv_tests",
            "use_annot_weights"
        },
        "rv_single":{
            "rv_single",
            "out",
            "rv_sumstats_part1",
            "rv_sumstats_part2",
            "extract_locus",
            "exclude_locus",
            "chr_interval",
            "maf_max",
            "maf_min",
            "mac_max",
            "mac_min",
            "variant_category",
            "annot_ht",
            "spark_conf",
            "grch37",
            "n_ldrs",
            "voxels",
            "sig_thresh",
            "threads"
        },
        "tfce":{
            "tfce",
            "out",
            "coord_dir",
            "results_idx",
            "null_assoc",
            "tfce_quantile_level",
            "sig_thresh",
            "variant_category",
            "tfce_null",
            "threads",
            "sig_thresh2"
        },
        "permute":{
            "permute",
            "out",
            "null_model",
            "spark_conf",
            "grch37",
            "sparse_genotype",
            "n_bootstrap",
            "voxels",
            "n_ldrs",
            "keep",
            "remove",
            "extract_locus",
            "exclude_locus",
            "maf_min",
            "maf_max",
            "mac_max",
            "mac_min",
            "chr_interval",
            "loco_preds",
            "annot_ht",
            "variant_sets",
            "variant_category",
            "cmac_min",
            "cmac_max",
            "perm_list",
            "sig_thresh",
        },
    }

    ignored_args = []
    for k, v in vars(args).items():
        if v is None or not v:
            continue
        elif k not in accepted_args[module]:
            ignored_args.append(k)
            setattr(args, k, None)

    if len(ignored_args) > 0:
        ignored_args = [f"--{arg.replace('_', '-')}" for arg in ignored_args]
        ignored_args_str = ", ".join(ignored_args)
        log.info(
            f"WARNING: {ignored_args_str} ignored by --{module.replace('_', '-')}."
        )


def split_files(arg):
    files = arg.split(",")
    for file in files:
        ds.check_existence(file)
    return files


def process_args(args, log):
    """
    Checking file existence and processing arguments

    """
    ds.check_existence(args.image)
    ds.check_existence(args.ldr_sumstats, ".snpinfo")
    ds.check_existence(args.ldr_sumstats, ".sumstats")
    ds.check_existence(args.bases)
    ds.check_existence(args.ldr_cov)
    ds.check_existence(args.covar)
    ds.check_existence(args.partition)
    ds.check_existence(args.ldrs)
    ds.check_existence(args.spark_conf)
    ds.check_existence(args.loco_preds)
    ds.check_existence(args.geno_mt)
    ds.check_existence(args.null_model)
    ds.check_existence(args.rv_sumstats_part1, "_rv_sumstats.h5")
    ds.check_existence(args.rv_sumstats_part2, "_data.h5")
    ds.check_existence(args.rv_sumstats_part2, "_locus_info.ht")
    ds.check_existence(args.annot_ht)
    ds.check_existence(args.variant_sets)
    ds.check_existence(args.perm)

    if args.n_ldrs is not None and args.n_ldrs <= 0:
        raise ValueError("--n-ldrs must be greater than 0")

    if args.threads is not None:
        if args.threads <= 0:
            raise ValueError("--threads must be greater than 0")
    else:
        args.threads = 1
    log.info(f"Using {args.threads} thread(s) in analysis.")

    if args.keep is not None:
        args.keep = split_files(args.keep)
        args.keep = ds.read_keep(args.keep)
        log.info(f"{len(args.keep)} subject(s) in --keep (logical 'and' for multiple files).")

    if args.remove is not None:
        args.remove = split_files(args.remove)
        args.remove = ds.read_remove(args.remove)
        log.info(f"{len(args.remove)} subject(s) in --remove (logical 'or' for multiple files).")

    if args.extract is not None:
        args.extract = split_files(args.extract)
        args.extract = ds.read_extract(args.extract)
        log.info(f"{len(args.extract)} SNP(s) in --extract (logical 'and' for multiple files).")
        
    if args.exclude is not None:
        args.exclude = split_files(args.exclude)
        args.exclude = ds.read_exclude(args.exclude)
        log.info(f"{len(args.exclude)} SNP(s) in --exclude (logical 'or' for multiple files).")

    if args.extract_locus is not None:
        args.extract_locus = split_files(args.extract_locus)
        # args.extract_locus = ds.read_extract(args.extract_locus, locus=True)
        # log.info(f"{len(args.extract_locus)} SNP(s) in --extract-locus (logical 'and' for multiple files).")
        
    if args.exclude_locus is not None:
        args.exclude_locus = split_files(args.exclude_locus)
        # args.exclude_locus = ds.read_exclude(args.exclude_locus, locus=True)
        # log.info(f"{len(args.exclude_locus)} SNP(s) in --exclude-locus (logical 'or' for multiple files).")
    
    if args.extract_locus_cond is not None:
        args.extract_locus_cond = split_files(args.extract_locus_cond)

    if args.exclude_locus_cond is not None:
        args.exclude_locus_cond = split_files(args.exclude_locus_cond)
    
    # if args.bfile is not None:
    #     for suffix in [".bed", ".fam", ".bim"]:
    #         ds.check_existence(args.bfile, suffix)

    if args.voxels is not None:
        try:
            args.voxels = np.array(
                [int(voxel) - 1 for voxel in ds.parse_input(args.voxels)]
            )
        except ValueError:
            ds.check_existence(args.voxels)
            args.voxels = ds.read_voxel(args.voxels)
        if np.min(args.voxels) <= -1:
            raise ValueError("voxel index must be one-based")
        log.info(f"{len(args.voxels)} voxel(s) in --voxels.")
    
    if args.sig_thresh is not None:
        if args.sig_thresh <= 0 or args.sig_thresh >= 1:
            raise ValueError("--sig-thresh should be greater than 0 and less than 1")
        # else:
        #     log.info(f"Saving results with a p-value less than {args.sig_thresh}")

    if args.maf_min is not None:
        if args.maf_min >= 0.5 or args.maf_min < 0:
            raise ValueError("--maf-min must be >= 0 and < 0.5")
        if args.maf_max is None:
            args.maf_max = 0.5
    if args.maf_max is not None:
        if args.maf_max > 0.5 or args.maf_max <= 0:
            raise ValueError("--maf-max must be > 0 and <= 0.5")
        if args.maf_min is None:
            args.maf_min = 0
    if args.mac_min is not None and args.mac_min < 0:
        raise ValueError("--mac-min must be >= 0")
    if args.mac_max is not None:
        if args.mac_max <= 0:
            raise ValueError("--mac-max must be > 0")
        if args.mac_min is None:
            args.maf_min = 0
    if args.hwe is not None and args.hwe <= 0:
        raise ValueError("--hwe must be greater than 0")
    if args.call_rate is not None and args.call_rate <= 0:
        raise ValueError("--call-rate must be greater than 0")
    
    if args.variant_type is not None:
        args.variant_type = args.variant_type.lower()
        if args.variant_type not in {"snv", "variant", "indel"}:
            raise ValueError(
                "--variant-type must be one of ('variant', 'snv', 'indel')"
            )
    
    if args.variant_sets is not None:
        args.variant_sets = ds.read_variant_sets(args.variant_sets)

    if args.cmac_min is not None and args.cmac_min <= 1:
        raise ValueError("--cmac-min must be greater than 1")
    if args.cmac_max is not None and args.cmac_min > args.cmac_max:
        raise ValueError("--cmac-min must be no greater than --cmac-max")
    
    if args.rv_tests is not None:
        args.rv_tests = [x.lower() for x in args.rv_tests.split(",")]
        for rv_test in args.rv_tests:
            if rv_test not in {"burden", "skat", "staar"}:
                raise ValueError(f"invalid rv test: {rv_test}")
        
    if args.n_bootstrap is not None:
        args.n_bootstrap = int(args.n_bootstrap)


def main(args, log):
    dirname = os.path.dirname(args.out)
    if dirname != "" and not os.path.exists(dirname):
        raise ValueError(f"{os.path.dirname(args.out)} does not exist")
    if (
        args.heri_gc
        + args.read_image
        + args.fpca
        + args.make_ldr
        + args.ld_matrix
        + args.sumstats
        + args.voxel_gwas
        + args.gwas
        + args.partition_h2
        + args.relatedness
        + args.make_mt
        + args.make_rv_sumstats
        + args.rv_null
        + args.rv_annot
        + args.rv_coding
        + args.rv_noncoding
        + args.rv
        + args.cluster
        + args.rv_cluster
        + args.rv_cond
        + args.rv_single
        + args.tfce
        + args.permute
        != 1
    ):
        raise ValueError(
            (
                "must raise one and only one of following module flags: "
                "--read-image, --fpca, --make-ldr, --heri-gc, --ld-matrix, --sumstats, "
                "--voxel-gwas, --gwas, --partition-h2, --relatedness, --make-mt, --rv-null, "
                "--make-rv-sumstats, --rv-annot, --rv-coding, --rv-noncoding, --rv, "
                "--cluster, --rv-cluster, --rv-cond, --rv-single, --tfce, --permute"
            )
        )

    if args.heri_gc:
        check_accepted_args("heri_gc", args, log)
        import heig.herigc as module
    elif args.read_image:
        check_accepted_args("read_image", args, log)
        import heig.image as module
    elif args.fpca:
        check_accepted_args("fpca", args, log)
        import heig.fpca as module
    elif args.make_ldr:
        check_accepted_args("make_ldr", args, log)
        import heig.ldr as module
    elif args.ld_matrix:
        check_accepted_args("ld_matrix", args, log)
        import heig.ldmatrix as module
    elif args.sumstats:
        check_accepted_args("sumstats", args, log)
        import heig.sumstats as module
    elif args.voxel_gwas:
        check_accepted_args("voxel_gwas", args, log)
        import heig.voxelgwas as module
    elif args.gwas:
        check_accepted_args('gwas', args, log)
        import heig.wgs.gwas as module
    elif args.partition_h2:
        check_accepted_args('partition_h2', args, log)
        import heig.partition_h2 as module
    elif args.relatedness:
        check_accepted_args('relatedness', args, log)
        import heig.wgs.relatedness as module
    elif args.make_mt:
        check_accepted_args('make_mt', args, log)
        import heig.wgs.mt as module
    elif args.make_rv_sumstats:
        check_accepted_args('make_rv_sumstats', args, log)
        import heig.wgs.wgs2 as module
    elif args.rv_null:
        check_accepted_args('rv_null', args, log)
        import heig.wgs.null as module
    elif args.rv_annot:
        check_accepted_args('rv_annot', args, log)
        import heig.wgs.annotation as module
    elif args.rv_coding:
        check_accepted_args('rv_coding', args, log)
        import heig.wgs.coding as module
    elif args.rv_noncoding:
        check_accepted_args('rv_noncoding', args, log)
        import heig.wgs.noncoding as module
    elif args.rv:
        check_accepted_args('rv', args, log)
        import heig.wgs.slidingwindow as module 
    elif args.cluster:
        check_accepted_args('cluster', args, log)
        import heig.wgs.cluster as module
    elif args.rv_cluster:
        check_accepted_args('rv_cluster', args, log)
        import heig.wgs.cluster_rare as module
    elif args.rv_cond:
        check_accepted_args('rv_cond', args, log)
        import heig.wgs.condition as module
    elif args.rv_single:
        check_accepted_args('rv_single', args, log)
        import heig.wgs.single_variant as module
    elif args.tfce:
        check_accepted_args('tfce', args, log)
        import heig.wgs.tfce as module
    elif args.permute:
        check_accepted_args('permute', args, log)
        import heig.wgs.permutation as module

    process_args(args, log)
    module.run(args, log)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.out is None:
        args.out = "heig"

    logpath = os.path.join(f"{args.out}.log")
    log = GetLogger(logpath)

    log.info(MASTHEAD)
    start_time = time.time()
    try:
        defaults = vars(parser.parse_args(""))
        opts = vars(args)
        non_defaults = [x for x in opts.keys() if opts[x] != defaults[x]]
        header = "heig.py \\\n"
        options = [
            "--" + x.replace("_", "-") + " " + str(opts[x]) + " \\"
            for x in non_defaults
        ]
        header += "\n".join(options).replace(" True", "").replace(" False", "")
        header = header + "\n"
        log.info(header)
        main(args, log)
    except Exception:
        log.info(traceback.format_exc())
        raise
    finally:
        log.info(f"\nAnalysis finished at {time.ctime()}")
        time_elapsed = round(time.time() - start_time, 2)
        log.info(f"Total time elapsed: {sec_to_str(time_elapsed)}")

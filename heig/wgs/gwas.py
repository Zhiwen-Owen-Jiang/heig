import os
import logging
import hail as hl
import heig.input.dataset as ds
from heig.wgs.relatedness import LOCOpreds
from heig.wgs.utils import read_genotype_data, init_hail, get_temp_path, clean


"""
TODO: 
1. consider providing more preprocessing options? such as --chr

"""


def parse_ldr_col(ldr_col):
    """
    Parsing string for LDR indices

    Parameters:
    ------------
    ldr_col: a string of one-based LDR column indices

    Returns:
    ---------
    res: a tuple of min and max (not included) zero-based indices

    """
    ldr_col = ldr_col.split(",")
    res = list()

    for col in ldr_col:
        if ":" in col:
            start, end = [int(x) for x in col.split(":")]
            if start > end:
                raise ValueError(f"{col} is invalid")
            res += list(range(start - 1, end))
        else:
            res.append(int(col) - 1)

    res = sorted(list(set(res)))
    if res[-1] - res[0] + 1 != len(res):
        raise ValueError(
            "it is very rare that columns in --ldr-col are not consective for LDR GWAS"
        )
    if res[0] < 0:
        raise ValueError("the min index less than 1")
    res = (res[0], res[-1] + 1)

    return res


def pandas_to_table(df, dir):
    """
    Converting a pd.DataFrame to hail.Table

    Parameters:
    ------------
    df: a pd.DataFrame to convert, it must have a single index 'IID'

    Returns:
    ---------
    table: a hail.Table

    """
    if not df.index.name == "IID":
        raise ValueError("the DataFrame must have a single index IID")
    df.to_csv(f"{dir}.txt", sep="\t", na_rep="NA")

    table = hl.import_table(
        f"{dir}.txt", key="IID", impute=True, types={"IID": hl.tstr}, missing="NA"
    )

    return table


def check_input(args, log):
    # required arguments
    if args.ldrs is None:
        raise ValueError("--ldrs is required")
    if args.covar is None:
        raise ValueError("--covar is required")
    if args.spark_conf is None:
        raise ValueError("--spark-conf is required")
    if args.geno_mt is None:
        raise ValueError(
            "--geno-mt is required. If you have bfile or vcf, convert it into a mt by --make-mt"
        )

    if args.ldr_col is not None:
        args.ldr_col = parse_ldr_col(args.ldr_col)
        if args.n_ldrs is not None:
            log.info(
                "WARNING: ignoring --n-ldrs as --ldr-col has been provided."
            )
    elif args.n_ldrs is not None:
        args.ldr_col = (0, args.n_ldrs)
    args.n_ldrs = None


class DoGWAS:
    """
    Conducting GWAS for all LDRs w/ or w/o relatedness

    """

    def __init__(self, gprocessor, ldrs, covar, temp_path, loco_preds=None, rand_v=1):
        """
        Parameters:
        ------------
        gprocessor: a GProcessor instance including hail.MatrixTable
        ldrs: a pd.DataFrame of LDRs with a single index 'IID'
        covar: a pd.DataFrame of covariates with a single index 'IID'
        temp_path: a temporary path for saving interim data
        loco_preds: a LOCOpreds instance of loco predictions
            loco_preds.data_reader(j) returns loco preds for chrj with matched subjects
        rand_v (n, 1): a np.array of random standard normal variable for wild bootstrap

        """
        self.gprocessor = gprocessor
        self.ldrs = ldrs
        self.covar = covar
        self.n_ldrs = self.ldrs.shape[1]
        self.n_covar = self.covar.shape[1]
        self.temp_path = temp_path
        self.loco_preds = loco_preds
        self.logger = logging.getLogger(__name__)
        self.n_variants = self.gprocessor.snps_mt.count_rows()

        covar_table = pandas_to_table(self.covar, f"{temp_path}_covar")
        self.gprocessor.annotate_cols(covar_table, "covar")

        if self.loco_preds is None:
            self.logger.info(
                (f"Doing GWAS for {self.n_variants} variants "
                 f"and {self.n_ldrs} LDRs without relatedness ...")
            )
            ldrs_table = pandas_to_table(self.ldrs * rand_v, f"{temp_path}_ldr")
            self.gprocessor.annotate_cols(ldrs_table, "ldrs")
            self.gwas = self.do_gwas(self.gprocessor.snps_mt)
        else:
            self.logger.info(
                (f"Doing GWAS for {self.n_variants} variants "
                 f"and {self.n_ldrs} LDRs considering relatedness ...")
            )
            unique_chrs = sorted(self.gprocessor.extract_unique_chrs())  # slow
            self.gwas = []
            for chr in unique_chrs:
                chr_mt = self._extract_chr(chr)
                resid_ldrs = (self.ldrs - self.loco_preds.data_reader(chr)) * rand_v
                ldrs_table = pandas_to_table(resid_ldrs, f"{temp_path}_ldr")
                chr_mt = self._annotate_cols(chr_mt, ldrs_table, "ldrs")
                self.gwas.append(self.do_gwas(chr_mt))
            self.gwas = hl.Table.union(*self.gwas, unify=False)

    def _extract_chr(self, chr):
        chr = str(chr)
        if hl.default_reference == "GRCh38":
            chr = "chr" + chr

        chr_mt = self.gprocessor.snps_mt.filter_rows(
            self.gprocessor.snps_mt.locus.contig == chr
        )

        return chr_mt

    @staticmethod
    def _annotate_cols(snps_mt, table, annot_name):
        """
        Annotating columns with values from a table
        the table is supposed to have the key 'IID'

        Parameters:
        ------------
        snps_mt: a hl.MatrixTable
        table: a hl.Table
        annot_name: annotation name

        """
        table = table.key_by("IID")
        annot_expr = {annot_name: table[snps_mt.s]}
        snps_mt = snps_mt.annotate_cols(**annot_expr)
        return snps_mt

    def do_gwas(self, snps_mt):
        """
        Conducting GWAS for all LDRs

        Parameters:
        ------------
        snps_mt: a hail.MatrixTable with LDRs and covariates annotated

        Returns:
        ---------
        gwas: gwas results in hail.Table

        """
        pheno_list = [snps_mt.ldrs[i] for i in range(self.n_ldrs)]
        covar_list = [snps_mt.covar[i] for i in range(self.n_covar)]

        gwas = hl.linear_regression_rows(
            y=pheno_list,
            x=snps_mt.GT.n_alt_alleles(),
            covariates=covar_list,
            pass_through=[snps_mt.rsid, snps_mt.info.n_called, snps_mt.info.AF],
        )

        gwas = gwas.annotate(
            chr=gwas.locus.contig,
            pos=gwas.locus.position,
            ref_allele=gwas.alleles[0],
            alt_allele=gwas.alleles[1],
            alt_allele_freq=gwas.AF[1],
        )
        gwas = gwas.key_by()
        gwas = gwas.drop(*["locus", "alleles", "y_transpose_x", "sum_x", "AF"])
        # gwas = gwas.drop(*['locus', 'alleles', 'n'])
        gwas = gwas.select(
            "chr",
            "pos",
            "rsid",
            "ref_allele",
            "alt_allele",
            "n_called",
            "alt_allele_freq",
            "beta",
            "standard_error",
            "t_stat",
            "p_value",
        )

        gwas = self._post_process(gwas)

        return gwas

    def _post_process(self, gwas):
        """
        Removing SNPs with any missing or infinity values.
        This step is originally done in sumstats.py.
        However, pandas is not convenient to handle nested arrays.

        """
        gwas = gwas.filter(
            ~(
                hl.any(lambda x: hl.is_missing(x) | hl.is_infinite(x), gwas.beta)
                | hl.any(
                    lambda x: hl.is_missing(x) | hl.is_infinite(x), gwas.standard_error
                )
                | hl.any(lambda x: hl.is_missing(x) | hl.is_infinite(x), gwas.t_stat)
                | hl.any(lambda x: hl.is_missing(x) | hl.is_infinite(x), gwas.p_value)
            )
        )

        return gwas

    def save(self, out_path):
        """
        Saving GWAS results as a parquet file

        """
        self.gwas = self.gwas.to_spark()
        self.gwas.write.mode("overwrite").parquet(f"{out_path}.parquet")


def run(args, log):
    # check input and configure hail
    check_input(args, log)
    try:
        init_hail(args.spark_conf, args.grch37, args.out, log)

        # read LDRs and covariates
        log.info(f"Read LDRs from {args.ldrs}")
        ldrs = ds.Dataset(args.ldrs)
        log.info(f"{ldrs.data.shape[1]} LDRs and {ldrs.data.shape[0]} subjects.")
        if args.ldr_col is not None:
            if ldrs.data.shape[1] < args.ldr_col[1]:
                raise ValueError(f"--ldr-col or --n-ldrs out of index")
            else:
                log.info(f"Keeping LDR{args.ldr_col[0]+1} to LDR{args.ldr_col[1]}.")
            ldrs.data = ldrs.data.iloc[:, args.ldr_col[0] : args.ldr_col[1]]

        log.info(f"Read covariates from {args.covar}")
        covar = ds.Covar(args.covar, args.cat_covar_list)

        # read loco preds
        if args.loco_preds is not None:
            log.info(f"Read LOCO predictions from {args.loco_preds}")
            loco_preds = LOCOpreds(args.loco_preds)
            loco_preds.select_ldrs(args.ldr_col)
            if loco_preds.ldr_col[1] - loco_preds.ldr_col[0] != ldrs.data.shape[1]:
                raise ValueError(
                    (
                        "inconsistent dimension in LDRs and LDR LOCO predictions. "
                        "Try to use --n-ldrs or --ldr-col"
                    )
                )
            common_ids = ds.get_common_idxs(
                ldrs.data.index,
                covar.data.index,
                loco_preds.ids,
                args.keep,
            )
        else:
            # keep subjects
            common_ids = ds.get_common_idxs(
                ldrs.data.index, covar.data.index, args.keep
            )
        common_ids = ds.remove_idxs(common_ids, args.remove, single_id=True)

        # read genotype data
        gprocessor = read_genotype_data(args, log)

        log.info(f"Processing genetic data ...")
        gprocessor.extract_exclude_snps(args.extract, args.exclude)
        gprocessor.extract_chr_interval(args.chr_interval)
        gprocessor.keep_remove_idvs(common_ids)
        gprocessor.do_processing(mode="gwas")

        # extract common subjects and align data
        snps_mt_ids = gprocessor.subject_id()
        ldrs.to_single_index()
        covar.to_single_index()
        ldrs.keep_and_remove(snps_mt_ids)
        covar.keep_and_remove(snps_mt_ids)
        covar.cat_covar_intercept()

        if args.loco_preds is not None:
            loco_preds.keep(snps_mt_ids)
        else:
            loco_preds = None
        log.info(f"{len(snps_mt_ids)} common subjects in the data.")
        log.info(
            f"{covar.data.shape[1]} fixed effects in the covariates (including the intercept)."
        )

        # gwas
        temp_path = get_temp_path(args.out)
        gprocessor.cache()
        gwas = DoGWAS(gprocessor, ldrs.data, covar.data, temp_path, loco_preds)

        # save gwas results
        gwas.save(args.out)
        log.info(f"\nSaved GWAS results to {args.out}.parquet")
    finally:
        if "temp_path" in locals():
            if os.path.exists(f"{temp_path}_covar.txt"):
                os.remove(f"{temp_path}_covar.txt")
                log.info(f"Removed temporary covariate data at {temp_path}_covar.txt")
            if os.path.exists(f"{temp_path}_ldr.txt"):
                os.remove(f"{temp_path}_ldr.txt")
                log.info(f"Removed temporary LDR data at {temp_path}_ldr.txt")
        if "loco_preds" in locals() and args.loco_preds is not None:
            loco_preds.close()

        clean(args.out)

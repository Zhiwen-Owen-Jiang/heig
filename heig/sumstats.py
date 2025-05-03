import os
import h5py
import ast
import logging
import concurrent.futures
import math
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from filelock import FileLock
from tqdm import tqdm
from scipy.stats import chi2
from heig import utils
from functools import partial
import heig.input.dataset as ds


def check_input(args, log):
    # required arguments
    if args.ldr_gwas is None and args.y2_gwas is None and args.ldr_gwas_heig is None:
        raise ValueError("--ldr-gwas, --ldr-gwas-heig, or --y2-gwas should be provided")

    # columns of LDR GWAS in HEIG are fixed
    if args.ldr_gwas_heig is not None:
        args.chr_col = "chr"
        args.pos_col = "pos"
        args.snp_col = "rsid"
        args.n_col = "n_called"
        args.a1_col = "alt_allele"
        args.a2_col = "ref_allele"
        args.effect_col = "beta,0"
        args.se_col = "standard_error"
        args.z_col = "t_stat"
        args.maf_col = "alt_allele_freq"
        args.info_col = None
        args.maf_min = None
        args.info_min = None
        args.ldr_gwas = None
        args.y2_gwas = None

    if args.snp_col is None:
        raise ValueError("--snp-col is required")
    if args.a1_col is None:
        raise ValueError("--a1-col is required")
    if args.a2_col is None:
        raise ValueError("--a2-col is required")

    # optional arguments
    if args.n_col is None and args.n is None:
        raise ValueError("either --n-col or --n is required")
    if args.ldr_gwas is not None and args.y2_gwas is not None:
        raise ValueError("can only specify --ldr-gwas or --y2-gwas")
    elif args.ldr_gwas is not None:
        if args.effect_col is None:
            raise ValueError("--effect-col is required for LDR summary statistics")
        if args.se_col is None:
            raise ValueError("--se-col is required for LDR summary statistics")
        if args.chr_col is None:
            raise ValueError("--chr-col is required for LDR summary statistics")
        if args.pos_col is None:
            raise ValueError("--pos-col is required for LDR summary statistics")
    elif args.y2_gwas is not None:
        if not (
            args.z_col is not None
            or args.effect_col is not None
            and args.se_col is not None
            or args.effect_col is not None
            and args.p_col is not None
        ):
            raise ValueError(
                (
                    "specify --z-col or --effect-col + --se-col or "
                    "--effect-col + --p-col for --y2-gwas"
                )
            )

    if args.maf_col is not None and args.maf_min is not None:
        if args.maf_min <= 0 or args.maf_min >= 0.5:
            raise ValueError("--maf-min must be greater than 0 and less than 0.5")
    elif args.maf_col is None and args.maf_min is not None:
        log.info("WARNING: ignoring --maf-min as --maf-col has not been provided.")
        args.maf_min = None

    if args.info_col is not None and args.info_min is not None:
        if args.info_min <= 0 or args.info_min >= 1:
            raise ValueError("--info-min must be between 0 and 1")
    elif args.info_col is None and args.info_min:
        log.info("WARNING: ignoring --info-min as --info-col has not been provided.")
        args.info_min = None

    if args.n is not None and args.n <= 0:
        raise ValueError("--n must be greater than 0")

    # processing some arguments
    if args.ldr_gwas is not None:
        ldr_gwas_files = ds.parse_input(args.ldr_gwas)
        for file in ldr_gwas_files:
            ds.check_existence(file)
        args.ldr_gwas = ldr_gwas_files
    elif args.ldr_gwas_heig is not None:
        ldr_gwas_heig_files = ds.parse_input(args.ldr_gwas_heig)
        for file in ldr_gwas_heig_files:
            ds.check_existence(file)
        args.ldr_gwas_heig = ldr_gwas_heig_files
    elif args.y2_gwas is not None:
        ds.check_existence(args.y2_gwas)
        args.y2_gwas = [args.y2_gwas]

    if args.effect_col is not None:
        try:
            args.effect, args.null_value = args.effect_col.split(",")
            args.null_value = int(args.null_value)
        except:
            raise ValueError("--effect-col must be specified as `BETA,0` or `OR,1`")
        if args.null_value not in (0, 1):
            raise ValueError("The null value must be 0 for BETA (log OR) or 1 for OR")
    else:
        args.effect, args.null_value = None, None


def map_cols(args):
    """
    Creating two dicts for mapping provided colnames and standard colnames

    Parameters:
    ------------
    args: instance of arguments

    Returns:
    ---------
    cols_map: keys are standard colnames, values are provided colnames
    cols_map2: keys are provided colnames, values are standard colnames

    """
    cols_map = dict()
    cols_map["N"] = args.n_col
    cols_map["n"] = args.n
    cols_map["CHR"] = args.chr_col
    cols_map["POS"] = args.pos_col
    cols_map["SNP"] = args.snp_col
    cols_map["EFFECT"] = args.effect
    cols_map["null_value"] = args.null_value
    cols_map["SE"] = args.se_col
    cols_map["A1"] = args.a1_col
    cols_map["A2"] = args.a2_col
    cols_map["Z"] = args.z_col
    cols_map["P"] = args.p_col
    cols_map["MAF"] = args.maf_col
    cols_map["maf_min"] = args.maf_min
    cols_map["INFO"] = args.info_col
    cols_map["info_min"] = args.info_min

    cols_map2 = dict()
    for k, v in cols_map.items():
        if v is not None and k not in ("n", "maf_min", "info_min", "null_value"):
            cols_map2[v] = k

    return cols_map, cols_map2


def read_sumstats(prefix):
    """
    Reading preprocessed summary statistics and creating a GWAS instance.

    Parameters:
    ------------
    prefix: the prefix of summary statistics file

    Returns:
    ---------
    a GWAS instance

    """
    snpinfo_dir = f"{prefix}.snpinfo"
    sumstats_dir = f"{prefix}.sumstats"

    if not os.path.exists(snpinfo_dir) or not os.path.exists(sumstats_dir):
        raise FileNotFoundError(f"either .sumstats or .snpinfo file does not exist")

    file = h5py.File(sumstats_dir, "r")
    snpinfo = pd.read_csv(snpinfo_dir, sep="\t", engine="pyarrow")

    if snpinfo.shape[0] != file.attrs["n_snps"]:
        raise ValueError(
            (
                "summary statistics and the meta data contain different number of SNPs, "
                "which means the files have been modified"
            )
        )

    return GWAS(file, snpinfo)


class GWAS:
    def __init__(self, file, snpinfo):
        """
        Parameters:
        ------------
        file: opened HDF5 file
        snpinfo: a pd.DataFrame of SNP info

        """
        self.n_snps = file.attrs["n_snps"]
        self.n_gwas = file.attrs["n_gwas"]
        self.n_blocks = file.attrs["n_blocks"]
        self.file = file
        self.snpinfo = snpinfo
        self.snp_idxs = None
        self.change_sign = None

    def close(self):
        self.file.close()

    def data_reader(self, data_type, gwas_idxs, snps_idxs, all_gwas=True):
        """
        Reading summary statistics in chunks, each chunk of 20 LDRs
        Two modes:
        1. Reading a batch of LDRs and a subset of sumstats as a generator
        2. Reading all LDRs and a small proportion of sumstats (e.g. hapmap3) into memory

        Parameters:
        ------------
        data_type: data type including `both`, `beta`, and `z`
        gwas_idxs (r, ): numerical indices of GWAS to extract
        snps_idxs (d, ): numerical/boolean indices of SNPs to extract
        all_gwas: if reading all GWAS sumstats

        Returns:
        ---------
        A np.array or a generator of sumstats

        """
        n_blocks = math.ceil(len(gwas_idxs) / 20)
        remaining = self.n_gwas

        if all_gwas:
            # in this case snps_idxs are numerical indices and only `z` is supported
            if data_type == "z":
                z_array = np.zeros((len(snps_idxs), self.n_gwas), dtype=np.float32)
                for block_idx in range(n_blocks):
                    if remaining >= 20:
                        z_array[:, block_idx * 20 : (block_idx + 1) * 20] = self.file[
                            f"z{block_idx}"
                        ][:][snps_idxs]
                        remaining -= 20
                    else:
                        z_array[:, block_idx * 20 : self.n_gwas] = self.file[
                            f"z{block_idx}"
                        ][:, :remaining][snps_idxs]
                return z_array
            else:
                raise ValueError("only z-score can be read for all LDRs")
        else:
            return self._data_reader_generator(data_type, snps_idxs, n_blocks)

    def _data_reader_generator(self, data_type, snps_idxs, n_blocks):
        """
        Reading data as a generator

        """
        remaining = self.n_gwas

        if data_type == "both":
            for block_idx in range(n_blocks):
                if remaining >= 20:
                    yield [
                        self.file[f"beta{block_idx}"][:][snps_idxs],
                        self.file[f"z{block_idx}"][:][snps_idxs],
                    ]
                    remaining -= 20
                else:
                    yield [
                        self.file[f"beta{block_idx}"][:, :remaining][snps_idxs],
                        self.file[f"z{block_idx}"][:, :remaining][snps_idxs],
                    ]
        elif data_type == "beta":
            for block_idx in range(n_blocks):
                if remaining >= 20:
                    yield self.file[f"beta{block_idx}"][:][snps_idxs]
                    remaining -= 20
                else:
                    yield self.file[f"beta{block_idx}"][:, :remaining][snps_idxs]
        else:
            raise ValueError("other data type is not supported")

    def extract_snps(self, keep_snps):
        """
        Extracting SNPs

        Parameters:
        ------------
        keep_snps: a pd.Series/DataFrame of SNPs

        """
        if isinstance(keep_snps, pd.Series):
            keep_snps = pd.DataFrame(keep_snps, columns=["SNP"])
        self.snpinfo["id"] = self.snpinfo.index  # keep the index in df
        self.snpinfo = keep_snps.merge(self.snpinfo, on="SNP")
        self.snp_idxs = self.snpinfo["id"].values
        del self.snpinfo["id"]

    def align_alleles(self, ref):
        """
        Aligning the summary statistics with the reference such that
        the Z scores are measured on the same allele.
        This function requires that the gwas and the reference have
        identical SNPs.

        Parameters:
        ------------
        ref: a pd.Dataframe of bim file

        """
        if not (np.array(ref["SNP"]) == np.array(self.snpinfo["SNP"])).all():
            raise ValueError("the GWAS and the reference have different SNPs")

        self.change_sign = ref["A1"].values != self.snpinfo["A1"].values
        self.snpinfo["A1"] = ref["A1"].values
        self.snpinfo["A2"] = ref["A2"].values


class ProcessGWAS(ABC):
    complement = {"A": "T", "T": "A", "C": "G", "G": "C"}
    required_cols = None

    def __init__(
        self, gwas_files, cols_map, cols_map2, out_dir, maf_min=None, info_min=None
    ):
        """
        Parameters:
        ------------
        gwas_files: a list of gwas files
        cols_map: a dict mapping standard colnames to provided colnames
        cols_map2: a dict mapping provided colnames to standard colnames
        out_dir: output directory
        maf_min: the minumum of MAF
        info_min: the minumum of INFO

        """
        self.gwas_files = gwas_files
        self.n_gwas_files = len(gwas_files)
        self.cols_map = cols_map
        self.cols_map2 = cols_map2
        self.out_dir = out_dir
        self.maf_min = maf_min
        self.info_min = info_min
        self.compression, self.delimiter = self._check_header(gwas_files[0])
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def _create_dataset(self, n_snps):
        pass

    @abstractmethod
    def _save_sumstats(self, index, **kwargs):
        pass

    def _save_snpinfo(self, snpinfo):
        snpinfo.to_csv(
            f"{self.out_dir}.snpinfo", sep="\t", index=None, na_rep="NA", float_format="%.3e"
        )

    @abstractmethod
    def process(self):
        pass

    def _read_gwas(self, gwas_file):
        """
        Reading a full GWAS file

        """
        gwas_data = pd.read_csv(
            gwas_file,
            sep=self.delimiter,
            compression=self.compression,
            usecols=list(self.cols_map2.keys()),
            na_values=["NONE", "."],
            dtype={"A1": "category", "A2": "category"},
            engine="pyarrow",
        )
        gwas_data = gwas_data.rename(self.cols_map2, axis=1)
        gwas_data["A1"] = gwas_data["A1"].str.upper().astype("category")
        gwas_data["A2"] = gwas_data["A2"].str.upper().astype("category")

        return gwas_data

    def _prune_snps(self, gwas, is_heig=False):
        """
        Pruning SNPs according to
        1) any missing values in required columns
        2) infinity in Z scores, less than 0 sample size
        3) any duplicates in rsID (indels)
        4) strand ambiguous
        5) an effective sample size less than 0.67 times the 90th percentage of sample size
        6) small MAF or small INFO score (optional)

        Parameters:
        ------------
        gwas: a pd.DataFrame of summary statistics with required columns
        is_heig: if it is HEIG GWAS

        Returns:
        ---------
        gwas: a pd.DataFrame of pruned summary statistics

        """
        n_snps = self._check_remaining_snps(gwas)
        self.logger.info(f"{n_snps} SNPs in the raw data.")

        if gwas["SNP"].isna().all():
            # if not gwas["CHR"].isna().all() and not gwas["POS"].isna().all():
            #     gwas.drop_duplicates(subset=["CHR", "POS"], keep=False, inplace=True)
            # else:
            #     raise ValueError("CHR, POS, and SNP are all missing values")
            raise ValueError("all SNPs missing an rsID")
        else:
            gwas.drop_duplicates(subset=["SNP"], keep=False, inplace=True)
        self.logger.info(f"Removed {n_snps - gwas.shape[0]} duplicated SNPs.")
        n_snps = self._check_remaining_snps(gwas)

        # increased a little memory
        if not is_heig:
            gwas = gwas.loc[~gwas.isin([np.inf, -np.inf, np.nan]).any(axis=1)]
            self.logger.info(
                f"Removed {n_snps - gwas.shape[0]} SNPs with any missing or infinite values."
            )
            n_snps = self._check_remaining_snps(gwas)

        not_strand_ambiguous = [
            (
                True
                if len(a2_) == 1
                and len(a1_) == 1
                and a2_ in self.complement
                and a1_ in self.complement
                and self.complement[a2_] != a1_
                else False
            )
            for a2_, a1_ in zip(gwas["A2"], gwas["A1"])
        ]
        gwas = gwas.loc[not_strand_ambiguous]
        self.logger.info(
            f"Removed {n_snps - gwas.shape[0]} non SNPs and strand-ambiguous SNPs."
        )
        n_snps = self._check_remaining_snps(gwas)

        n_thresh = int(gwas["N"].quantile(0.9) / 1.5)
        gwas = gwas.loc[gwas["N"] >= n_thresh]
        self.logger.info(f"Removed {n_snps - gwas.shape[0]} SNPs with N < {n_thresh}.")
        n_snps = self._check_remaining_snps(gwas)

        if self.maf_min is not None:
            gwas = gwas.loc[gwas["MAF"] >= self.maf_min]
            self.logger.info(
                f"Removed {n_snps - gwas.shape[0]} SNPs with MAF < {self.maf_min}."
            )
            n_snps = self._check_remaining_snps(gwas)

        if self.info_min is not None:
            gwas = gwas.loc[gwas["INFO"] >= self.info_min]
            self.logger.info(
                f"Removed {n_snps - gwas.shape[0]} SNPs with INFO < {self.info_min}."
            )
            n_snps = self._check_remaining_snps(gwas)

        self.logger.info(f"{n_snps} SNPs remaining after pruning.")

        return gwas

    @staticmethod
    def _check_remaining_snps(gwas):
        """
        Checking if gwas array is empty

        """
        n_snps = gwas.shape[0]
        if n_snps == 0:
            raise ValueError("no SNP remaining. Check if misspecified columns")
        return n_snps

    def _check_header(self, gwas_file):
        """
        Checking if all required columns exist;
        checking if all provided columns exist.

        Parameters:
        ------------
        gwas_file: directory to a gwas file

        Returns:
        ---------
        compression: compression mode
        delimiter: delimiter of gwas files

        """
        if gwas_file.endswith("parquet"):
            return None, None

        openfunc, compression = utils.check_compression(gwas_file)

        with openfunc(gwas_file, "r") as file:
            header = file.readline()
        if compression is not None:
            header = str(header, "UTF-8")

        # detecting tab or space
        if header.count("\t") > header.count(" "):
            delimiter = "\t"
        else:
            delimiter = " "
        header = header.split()

        for col in self.required_cols:
            if self.cols_map[col] not in header:
                raise ValueError(
                    f"{self.cols_map[col]} (case sensitive) cannot be found in {dir}"
                )

        for col, _ in self.cols_map2.items():
            if col not in header:
                raise ValueError(f"{col} (case sensitive) cannot be found in {dir}")

        return compression, delimiter

    def _check_median(self, data, effect, null_value):
        """
        Checking if the median value of effects (beta, or) is reasonable

        Parameters:
        ------------
        data: a pd.Series of effects
        effect: BETA or OR
        null_value: 1 or 0

        """
        median_beta = np.nanmedian(data)
        if np.abs(median_beta - null_value > 0.1):
            raise ValueError(
                (
                    f"median value of {effect} is {median_beta:.4f} "
                    f"(should be close to {null_value}). "
                    "This column may be mislabeled"
                )
            )
        else:
            self.logger.info(
                (
                    f"Median value of {effect} is {median_beta:.4f}, "
                    "which is reasonable."
                )
            )


class GWASLDR(ProcessGWAS):
    required_cols = ["CHR", "POS", "SNP", "A1", "A2"]

    def _create_dataset(self, n_snps):
        with h5py.File(f"{self.out_dir}.sumstats", "w") as file:
            file.attrs["n_snps"] = n_snps
            file.attrs["n_gwas"] = self.n_gwas_files
            file.attrs["n_blocks"] = math.ceil(self.n_gwas_files / 20)

    def _save_sumstats(self, block_idx, beta, z):
        """
        Saving sumstats and ensuring only one process is writing

        """
        chunk_size = np.min((beta.shape[0], 10000))
        lock_file = f"{self.out_dir}.sumstats.lock"
        with FileLock(lock_file):
            with h5py.File(f"{self.out_dir}.sumstats", "r+") as file:
                file.create_dataset(
                    f"beta{block_idx}",
                    data=beta,
                    dtype="float32",
                    chunks=(chunk_size, beta.shape[1]),
                )
                file.create_dataset(
                    f"z{block_idx}",
                    data=z,
                    dtype="float32",
                    chunks=(chunk_size, z.shape[1]),
                )

    def process(self, threads):
        """
        Processing LDR GWAS summary statistics. BETA and SE are required columns.

        """
        self.logger.info(
            (
                f"Reading and processing {self.n_gwas_files} LDR GWAS summary statistics file(s). "
                "Only the first GWAS file will be QCed ..."
            )
        )
        is_valid_snp, snpinfo = self._qc()
        self.logger.info("Reading and processing remaining GWAS files ...")
        self._read_in_parallel(is_valid_snp, threads)
        self._save_snpinfo(snpinfo)

    def _qc(self):
        """
        Quality control using the first GWAS file

        Returns:
        ---------
        is_valid_snp: boolean indices of valid SNPs
        snpinfo: SNP metadata

        """
        gwas_data = self._read_gwas(self.gwas_files[0])
        if self.cols_map["N"] is None:
            gwas_data["N"] = self.cols_map["n"]
        if self.cols_map["null_value"] == 1:
            raise ValueError("the null value of LDR GWAS effect size must be 0")
        self._check_median(gwas_data["EFFECT"], "EFFECT", self.cols_map["null_value"])
        if "MAF" in gwas_data.columns:
            orig_snps_list = gwas_data[["CHR", "POS", "SNP", "A1", "A2", "MAF", "N"]]
        else:
            orig_snps_list = gwas_data[["CHR", "POS", "SNP", "A1", "A2", "N"]]
        valid_snp_idxs = np.ones(gwas_data.shape[0], dtype=bool)

        self.logger.info(f"Pruning SNPs for the first GWAS file ...")
        gwas_data = self._prune_snps(gwas_data)
        final_snps_list = gwas_data["SNP"]
        valid_snp_idxs = (
            valid_snp_idxs & orig_snps_list["SNP"].isin(final_snps_list).values
        )

        is_valid_snp = valid_snp_idxs == 1
        snpinfo = orig_snps_list.loc[is_valid_snp].reset_index(drop=True)
        self._create_dataset(gwas_data.shape[0])

        return is_valid_snp, snpinfo

    def _read_gwas_effct(self, gwas_file):
        """
        Reading effect and se from a GWAS file

        """
        gwas_data = pd.read_csv(
            gwas_file,
            sep=self.delimiter,
            compression=self.compression,
            usecols=[self.cols_map["EFFECT"], self.cols_map["SE"]],
            na_values=["NONE", "."],
            engine="pyarrow",
        )
        gwas_data = gwas_data.rename(self.cols_map2, axis=1)

        return gwas_data

    def _read_save(self, is_valid_snp, block_idx, gwas_files):
        """
        Reading, processing, and saving a batch of LDR GWAS files

        Parameters:
        ------------
        is_valid_snp: boolean indices of valid SNPs
        block_idx: block index
        gwas_files: a list of GWAS files

        """
        beta_array = np.zeros((np.sum(is_valid_snp), len(gwas_files)), dtype=np.float32)
        z_array = np.zeros((np.sum(is_valid_snp), len(gwas_files)), dtype=np.float32)

        for i, gwas_file in enumerate(gwas_files):
            gwas_data = self._read_gwas_effct(gwas_file)
            gwas_data = gwas_data.loc[is_valid_snp]
            beta_array[:, i] = gwas_data["EFFECT"]
            try:
                z_array[:, i] = gwas_data["EFFECT"] / gwas_data["SE"]
            except:
                raise ValueError(
                    (
                        "did you input GWAS summary statistics generated by HEIG? "
                        "Using --ldr-gwas-heig to process"
                    )
                )

        self._save_sumstats(block_idx, beta=beta_array, z=z_array)

    def _read_in_parallel(self, is_valid_snp, threads):
        """
        Reading muliple LDR GWAS files in parallel

        Parameters:
        ------------
        is_valid_snp: boolean indices of valid SNPs
        threads: number of threads to use

        """
        partial_function = partial(self._read_save, is_valid_snp)
        n_blocks = math.ceil(self.n_gwas_files / 20)

        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            for block_idx in range(n_blocks):
                futures.append(
                    executor.submit(
                        partial_function,
                        block_idx,
                        self.gwas_files[block_idx * 20 : (block_idx + 1) * 20],
                    )
                )

            for _ in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc=f"{len(futures)} blocks",
            ):
                pass

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    executor.shutdown(wait=False)
                    raise RuntimeError(f"Computation terminated due to error: {exc}")

        if os.path.exists(f"{self.out_dir}.sumstats.lock"):
            os.remove(f"{self.out_dir}.sumstats.lock")


class GWASY2(ProcessGWAS):
    required_cols = ["SNP", "A1", "A2"]

    def _create_dataset(self, n_snps):
        with h5py.File(f"{self.out_dir}.sumstats", "w") as file:
            file.attrs["n_snps"] = n_snps
            file.attrs["n_gwas"] = self.n_gwas_files
            file.attrs["n_blocks"] = 1

    def _save_sumstats(self, block_idx, z):
        chunk_size = np.min((z.shape[0], 10000))
        with h5py.File(f"{self.out_dir}.sumstats", "r+") as file:
            file.create_dataset(
                f"z{block_idx}", data=z, dtype="float32", chunks=(chunk_size, 1)
            )

    def process(self, threads=None):
        """
        Preprocessing non-imaging GWAS summary statistics.

        """
        self.logger.info(
            f"Reading and processing the non-imaging GWAS summary statistics file ..."
        )

        gwas_file = self.gwas_files[0]
        gwas_data = self._read_gwas(gwas_file)

        if self.cols_map["N"] is None:
            gwas_data["N"] = self.cols_map["n"]

        if self.cols_map["EFFECT"] is not None and self.cols_map["SE"] is not None:
            self._check_median(
                gwas_data["EFFECT"], "EFFECT", self.cols_map["null_value"]
            )
            if self.cols_map["null_value"] == 1:
                gwas_data["EFFECT"] = np.log(gwas_data["EFFECT"])
            gwas_data["Z"] = gwas_data["EFFECT"] / gwas_data["SE"]
        elif self.cols_map["null_value"] is not None and self.cols_map["P"] is not None:
            abs_z_score = np.sqrt(chi2.ppf(1 - gwas_data["P"], 1))
            if self.cols_map["null_value"] == 0:
                gwas_data["Z"] = ((gwas_data["EFFECT"] > 0) * 2 - 1) * abs_z_score
            else:
                gwas_data["Z"] = ((gwas_data["EFFECT"] > 1) * 2 - 1) * abs_z_score
        else:
            self._check_median(gwas_data["Z"], "Z", 0)

        self.logger.info(f"Pruning SNPs for {gwas_file} ...")
        gwas_data = self._prune_snps(gwas_data)
        z = gwas_data["Z"].to_numpy().reshape(-1, 1)
        snpinfo = gwas_data[["SNP", "A1", "A2", "N"]].reset_index(drop=True)

        self._create_dataset(z.shape[0])
        self._save_sumstats(0, z=z)
        self._save_snpinfo(snpinfo)


class GWASHEIG(GWASLDR):
    def _create_dataset(self, n_snps):
        self.current_block_idx = -1
        self.current_empty_space = 0

        with h5py.File(f"{self.out_dir}.sumstats", "w") as file:
            file.attrs["n_snps"] = n_snps
            file.attrs["n_gwas"] = 0  # initialize as 0
            file.attrs["n_blocks"] = 0

    def _save_sumstats(self, beta, z, is_last_file):
        """
        Saving sumstats in blocks

        Parameters:
        ------------
        beta (n_snps, n_ldrs): a np.array of beta to save
        z (n_snps, n_ldrs): a np.array of z to save
        is_last_file: if it is the last gwas file
        idx: index of the current LDR to save
        self.current_empty_space: the number of empty columns in the current block
        self.current_block_idx: index of the corrent block

        """
        idx = 0
        n_ldrs = beta.shape[1]

        with h5py.File(f"{self.out_dir}.sumstats", "r+") as file:
            # checking if the current block is full, if not fill it first
            if self.current_empty_space != 0:
                file[f"beta{self.current_block_idx}"][
                    :, -self.current_empty_space :
                ] = beta[:, : self.current_empty_space]
                file[f"z{self.current_block_idx}"][:, -self.current_empty_space :] = z[
                    :, : self.current_empty_space
                ]

                if n_ldrs > self.current_empty_space:
                    file.attrs["n_gwas"] += self.current_empty_space
                    idx = self.current_empty_space
                    self.current_empty_space = 0
                else:
                    self.current_empty_space -= n_ldrs
                    file.attrs["n_gwas"] += n_ldrs
                    return

            # save remaining data to new blocks
            chunk_size = np.min((beta.shape[0], 10000))

            for i in range(idx, n_ldrs, 20):
                self.current_block_idx += 1
                file.attrs["n_blocks"] += 1
                file.create_dataset(
                    f"beta{self.current_block_idx}",
                    # data=beta[:, i: i+20],
                    shape=(file.attrs["n_snps"], 20),
                    dtype="float32",
                    chunks=(chunk_size, 20),
                )
                file.create_dataset(
                    f"z{self.current_block_idx}",
                    # data=z[:, i: i+20],
                    shape=(file.attrs["n_snps"], 20),
                    dtype="float32",
                    chunks=(chunk_size, 20),
                )

                end = np.min((i + 20, n_ldrs))
                file[f"beta{self.current_block_idx}"][:, : end - i] = beta[:, i:end]
                file[f"z{self.current_block_idx}"][:, : end - i] = z[:, i:end]
                file.attrs["n_gwas"] += end - i
                self.current_empty_space = 20 - end + i

            # remove the zero columns in the last block
            if is_last_file and self.current_empty_space > 0:
                last_beta = file[f"beta{self.current_block_idx}"][
                    :, : -self.current_empty_space
                ]
                last_z = file[f"z{self.current_block_idx}"][
                    :, : -self.current_empty_space
                ]
                del file[f"beta{self.current_block_idx}"]
                del file[f"z{self.current_block_idx}"]

                file.create_dataset(
                    f"beta{self.current_block_idx}",
                    data=last_beta,
                    dtype="float32",
                    chunks=(chunk_size, last_beta.shape[1]),
                )
                file.create_dataset(
                    f"z{self.current_block_idx}",
                    data=last_z,
                    dtype="float32",
                    chunks=(chunk_size, last_z.shape[1]),
                )

    def process(self, threads, is_valid_snp=None, snpinfo=None):
        """
        Processing LDR GWAS summary statistics.

        """
        if is_valid_snp is None and snpinfo is None:
            self.logger.info(
                (
                    f"Reading and processing {self.n_gwas_files} LDR GWAS summary statistics files. "
                    "Only the first GWAS file will be QCed ..."
                )
            )
            is_valid_snp, snpinfo = self._qc()
            self._save_snpinfo(snpinfo)
            self.logger.info("Reading and processing remaining GWAS files ...")

        self._create_dataset(snpinfo.shape[0])
        is_last_file = False
        for i, gwas_file in enumerate(self.gwas_files):
            if i == self.n_gwas_files - 1:
                is_last_file = True
            self._read_save(is_valid_snp, gwas_file, is_last_file, threads)

        return is_valid_snp, snpinfo

    def _qc(self):
        """
        Quality control using the first GWAS file

        Returns:
        ---------
        is_valid_snp: boolean indices of valid SNPs
        snpinfo: SNP metadata

        """
        gwas_data = self._read_gwas(self.gwas_files[0])
        orig_snps_list = gwas_data[["CHR", "POS", "SNP", "A1", "A2", "MAF", "N"]]
        valid_snp_idxs = np.ones(gwas_data.shape[0], dtype=bool)

        self.logger.info(f"Pruning SNPs for the first GWAS file ...")
        gwas_data = self._prune_snps(gwas_data, is_heig=True)
        if gwas_data["SNP"].isna().all():
            #     final_snps_list = gwas_data[["CHR", "POS"]]
            #     valid_snp_idxs = (
            #     valid_snp_idxs & orig_snps_list[["CHR", "POS"]].isin(final_snps_list).all(axis=1).values
            # )
            raise ValueError("all SNPs missing an rsID")
        else:
            final_snps_list = gwas_data["SNP"]
            valid_snp_idxs = (
                valid_snp_idxs & orig_snps_list["SNP"].isin(final_snps_list).values
            )

        is_valid_snp = valid_snp_idxs == 1
        snpinfo = orig_snps_list.loc[is_valid_snp].reset_index(drop=True)
        # self._create_dataset(gwas_data.shape[0])

        return is_valid_snp, snpinfo

    def _read_gwas(self, gwas_file):
        """
        Reading HEIG GWAS file from parquet, only SNP info

        """
        gwas_data = pd.read_parquet(
            gwas_file,
            columns=["chr", "pos", "rsid", "ref_allele", "alt_allele", "alt_allele_freq", "n_called"],
            engine="pyarrow",
        )
        gwas_data = gwas_data.rename(self.cols_map2, axis=1)
        gwas_data["A1"] = gwas_data["A1"].str.upper().astype("category")
        gwas_data["A2"] = gwas_data["A2"].str.upper().astype("category")

        return gwas_data

    def _read_gwas_effct(self, gwas_file):
        """
        Reading effects and z scores from HEIG GWAS results

        """
        gwas_data = pd.read_parquet(
            gwas_file,
            columns=["beta", "t_stat"],
            engine="pyarrow",
        )
        gwas_data = gwas_data.rename(self.cols_map2, axis=1)

        return gwas_data

    def _read_save(self, is_valid_snp, gwas_file, is_last_file, threads):
        """
        Reading, processing, and saving a batch of LDR GWAS files

        Parameters:
        ------------
        is_valid_snp: boolean indices of valid SNPs
        gwas_file: a GWAS file
        is_last_file: if it is the last gwas file
        threads: number of threads to use

        """
        gwas_data = self._read_gwas_effct(gwas_file)
        gwas_data = gwas_data.loc[is_valid_snp]
        beta_array = np.array(list(gwas_data["EFFECT"]))
        z_array = np.array(list(gwas_data["Z"]))
        self._save_sumstats(beta_array, z_array, is_last_file)


def run(args, log):
    check_input(args, log)
    cols_map, cols_map2 = map_cols(args)

    if args.ldr_gwas is not None:
        sumstats = GWASLDR(
            args.ldr_gwas, cols_map, cols_map2, args.out, args.maf_min, args.info_min
        )
    elif args.ldr_gwas_heig is not None:
        sumstats = GWASHEIG(
            args.ldr_gwas_heig,
            cols_map,
            cols_map2,
            args.out,
            args.maf_min,
            args.info_min,
        )
    elif args.y2_gwas is not None:
        sumstats = GWASY2(
            args.y2_gwas, cols_map, cols_map2, args.out, args.maf_min, args.info_min
        )
    sumstats.process(args.threads)

    log.info(f"\nSaved the processed summary statistics to {args.out}.sumstats")
    log.info(f"Saved the summary statistics information to {args.out}.snpinfo")

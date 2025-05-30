import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm
import heig.input.genotype as gt
import heig.input.dataset as ds
from heig.utils import find_loc


class LDmatrix:
    def __init__(self, ld_prefix):
        """
        Loading an existing LD matrix

        Parameters:
        ------------
        ld_prefix: prefix of LD matrix file

        """
        ld_prefix_list = ds.parse_input(ld_prefix)
        self.ldinfo = self._merge_ldinfo(ld_prefix_list)  # slow
        self.data = self._read_as_generator(ld_prefix_list)
        self.block_sizes, self.block_ranges = self._get_block_info(self.ldinfo)

    def _read_ldinfo(self, prefix):
        """
        Reading an individual ldinfo file

        Parameters:
        ------------
        prefix: prefix of LD file

        Returns:
        ---------
        ldinfo: a pd.DataFrame of ldinfo

        """
        ldinfo = pd.read_csv(
            f"{prefix}.ldinfo",
            sep="\t",
            header=None,
            engine="pyarrow",
            names=[
                "CHR",
                "SNP",
                "CM",
                "POS",
                "A1",
                "A2",
                "MAF",
                "block_idx",
                "block_idx2",
                "ldscore",
            ],
        )
        if (
            not ldinfo.groupby("CHR")["POS"]
            .apply(lambda x: x.is_monotonic_increasing)
            .all()
        ):
            raise ValueError(f"the SNPs in each chromosome are not sorted")
        if ldinfo.groupby("CHR")["POS"].apply(lambda x: x.duplicated()).any():
            raise ValueError(f"duplicated SNPs in LD matrix are not allowed")
        return ldinfo

    def _merge_ldinfo(self, prefix_list):
        """
        Merging multiple LD matrices with the current one

        Parameters:
        ------------
        prefix_list: a list of prefix of LD file

        Returns:
        ---------
        ldinfo: a pd.DataFrame of ldinfo

        """
        if len(prefix_list) == 0:
            raise ValueError("nothing in the LD list")
        ldinfo = self._read_ldinfo(prefix_list[0])
        for prefix in prefix_list[1:]:
            ldinfo_i = self._read_ldinfo(prefix)
            ldinfo_i["block_idx"] += ldinfo.loc[ldinfo.index[-1], "block_idx"] + 1
            ldinfo = pd.concat([ldinfo, ldinfo_i], axis=0, ignore_index=True)
        return ldinfo

    def _read_as_generator(self, prefix_list):
        """
        Reading multiple ldmatrix files as a generator

        Parameters:
        ------------
        prefix_list: a list of prefix of LD file

        Returns:
        ---------
        a generator of LD blocks

        """
        for prefix in prefix_list:
            file_path = f"{prefix}.ldmatrix"
            with h5py.File(file_path, "r") as file:
                for i in range(file.attrs["n_blocks"]):
                    yield file[f"block_{i}"][:]

    def _get_block_info(self, ldinfo):
        """
        Getting block sizes and block ranges from ldinfo

        Parameters:
        ------------
        ldinfo: a pd.DataFrame of ldinfo

        Returns:
        ---------
        block_sizes: a list of block sizes
        block_ranges: a list of tuples (begin, end)

        """
        block_sizes = ldinfo["block_idx"].value_counts().sort_index().to_list()
        block_ranges = []
        begin, end = 0, 0
        for block_size in block_sizes:
            begin = end
            end += block_size
            block_ranges.append((begin, end))
        return block_sizes, block_ranges

    def extract(self, snps):
        """
        Extracting SNPs from the LD matrix

        Parameters:
        ------------
        snps: a list/set of rdID

        """
        self.ldinfo = self.ldinfo.loc[self.ldinfo["SNP"].isin(snps)]
        block_dict = {
            k: g["block_idx2"].tolist() for k, g in self.ldinfo.groupby("block_idx")
        }
        self.block_sizes, self.block_ranges = self._get_block_info(self.ldinfo)
        self.data = (
            block[block_dict[i]] for i, block in enumerate(self.data) if i in block_dict
        )

    def merge_blocks(self):
        """
        Merging small blocks such that we have ~200 blocks with similar size

        Returns:
        ---------
        merged_blocks: a list of merged blocks, each element is a tuple of indices

        """
        n_blocks = len(self.block_sizes)
        if n_blocks <= 200:
            return [tuple([i]) for i in range(n_blocks)]
        mean_size = sum(self.block_sizes) / 200
        merged_blocks = []
        cur_size = 0
        cur_group = []
        for i, block_size in enumerate(self.block_sizes):
            if i < n_blocks - 1:
                if (
                    cur_size + block_size <= mean_size
                    or cur_size + block_size // 2 <= mean_size
                ):
                    cur_group.append(i)
                    cur_size += block_size
                else:
                    merged_blocks.append(tuple(cur_group))
                    cur_group = [i]
                    cur_size = block_size
            else:
                if (
                    cur_size + block_size <= mean_size
                    or cur_size + block_size // 2 <= mean_size
                ):
                    cur_group.append(i)
                    merged_blocks.append(tuple(cur_group))
                else:
                    merged_blocks.append(tuple([i]))

        return merged_blocks


class LDmatrixBED(LDmatrix):
    def __init__(self, num_snps_part, ldinfo, snp_getter, prop, inv=False):
        """
        Making an LD matrix from a bed file

        Parameters:
        ------------
        num_snps_part: a list of number of SNPs in each LD block
        ldinfo: SNP information
        snp_getter: a generator for getting SNPs
        prop: proportion of variance to keep for each LD block
        inv: if take inverse or not

        """
        self.data = []
        ldscore = []
        for _, num in enumerate(
            tqdm(num_snps_part, desc=f"Making {len(num_snps_part)} LD blocks")
        ):
            block = snp_getter(num)
            block = self._fill_na(block)
            corr = np.atleast_2d(np.corrcoef(block.T))
            ldscore_i = self._estimate_ldscore(corr, block.shape[0])
            ldscore.append(ldscore_i)
            values, bases = self._truncate(corr, prop)
            if inv:
                bases = bases * np.sqrt(values**-1)
            else:
                bases = bases * np.sqrt(values)
            self.data.append(bases)
        ldinfo["ldscore"] = np.concatenate(ldscore, axis=None)
        self.ldinfo = ldinfo

    def _truncate(self, block, prop):
        """
        Making an LD matrix from a bed file

        Parameters:
        ------------
        block: a squared correlation matrix
        prop: proportion of variance to keep for each LD block

        Returns:
        ---------
        values: an array of eigenvalues
        bases: an array of eigenvectors

        """
        values, bases = np.linalg.eigh(block)
        values = np.flip(values)
        bases = np.flip(bases, axis=1)
        prop_var = np.cumsum(values) / np.sum(values)
        idxs = (prop_var <= prop) & (values > 0)
        n_idxs = sum(idxs) + 1
        values = values[:n_idxs]
        bases = bases[:, :n_idxs]

        return values, bases

    def _fill_na(self, block):
        """
        Filling missing genotypes with sample mean

        Parameters:
        ------------
        block: a correlation matrix

        Returns:
        ---------
        block: an array of genotype data (n, p)

        """
        block_avg = np.nanmean(block, axis=0)
        nanidx = np.where(np.isnan(block))
        block[nanidx] = block_avg[nanidx[1]]

        return block

    def _estimate_ldscore(self, corr, n):
        """
        Estimating LD score from the LD matrix
        The Pearson correlation is adjusted by r2 - (1 - r2) / (n - 2)

        Parameters:
        ------------
        corr: a correlation matrix
        n: sample size of reference panel

        Returns:
        ---------
        adj_ld: an array of LD scores (p, )

        """
        raw_ld = np.sum(corr**2, axis=0)
        adj_ld = raw_ld - (corr.shape[0] - raw_ld) / (n - 2)

        return adj_ld

    def save(self, out, inv, regu):
        """
        Saving the LD matrix

        Parameters:
        ------------
        out: prefix of output
        inv: indicator of inverse
        regu: LD regularization level

        Returns:
        ---------
        prefix: prefix of LD matrix

        """
        regu = int(str(regu).split(".")[1])
        if not inv:
            prefix = f"{out}_ld_regu{regu}"
        else:
            prefix = f"{out}_ld_inv_regu{regu}"

        with h5py.File(f"{prefix}.ldmatrix", "w") as file:
            file.attrs["n_blocks"] = len(self.data)
            for i, block in enumerate(self.data):
                file.create_dataset(f"block_{i}", data=block, dtype="float32")
        self.ldinfo.to_csv(
            f"{prefix}.ldinfo", sep="\t", index=None, header=None, float_format="%.4f"
        )

        return prefix


def partition_genome(ld_bim, part, log):
    """
    Partitioning a chromosome to LD blocks
    All SNPs in ld_bim belong to one and only one block

    Parameters:
    ------------
    ld_bim: a pd.Dataframe of SNP information
    part: a pd.Dataframe of LD block annotation
    log: a logger

    Returns:
    ---------
    num_snps_part: a list of #SNPs in each block
    ld_bim: a pd.Dataframe of SNP information with block idxs

    """
    num_snps_part = []
    # end = -1
    cand = list(ld_bim.loc[ld_bim["CHR"] == part.iloc[0, 0], "POS"])
    end = find_loc(cand, part.iloc[0, 1])
    if end == 0:
        end = -1  # to include the 1st SNP into the 1st block
    ld_bim["block_idx"] = 0
    ld_bim["block_idx2"] = 0
    abs_begin = 0
    abs_end = 0
    n_skipped_blocks = 0
    block_idx = 0
    for i in range(part.shape[0]):
        cand = list(ld_bim.loc[ld_bim["CHR"] == part.iloc[i, 0], "POS"])
        begin = end
        end = find_loc(cand, part.iloc[i, 2])
        if end < begin:
            begin = -1
        if end > begin:
            block_size = end - begin
            if block_size < 2000:
                sub_blocks = [(begin, end)]
            else:
                log.info(
                    (
                        f"A large LD block with size {block_size}, "
                        "evenly partitioning it to small blocks with size ~1000."
                    )
                )
                sub_blocks = get_sub_blocks(begin, end)
            for sub_block in sub_blocks:
                sub_begin, sub_end = sub_block
                sub_block_size = sub_end - sub_begin
                num_snps_part.append(sub_block_size)
                if not abs_begin and not abs_end:
                    abs_begin = sub_begin + 1
                    abs_end = sub_end + 1
                else:
                    abs_begin = abs_end
                    abs_end += sub_block_size
                ld_bim.loc[ld_bim.index[abs_begin:abs_end], "block_idx"] = block_idx
                ld_bim.loc[ld_bim.index[abs_begin:abs_end], "block_idx2"] = range(
                    sub_block_size
                )
                block_idx += 1
        else:
            n_skipped_blocks += 1
    if len(num_snps_part) == 0:
        raise ValueError("no SNP overlapped with LD blocks")
    log.info(f"Skipping {n_skipped_blocks} blocks with no SNP.")

    return num_snps_part, ld_bim


def get_sub_blocks(begin, end):
    """
    Partitioning large LD blocks to smaller ones with size ~1000

    Parameters:
    ------------
    begin: begin index
    end: end index

    Returns:
    ---------
    sub_blocks: a list of tuples

    """
    block_size = end - begin
    n_sub_blocks = block_size // 1000
    sub_block_size = block_size // n_sub_blocks
    sub_blocks = []
    for _ in range(n_sub_blocks - 1):
        temp_end = begin + sub_block_size
        sub_blocks.append((begin, temp_end))
        begin = temp_end
    sub_blocks.append((begin, end))

    return sub_blocks


def check_input(args):
    # required arguments
    if args.bfile is None:
        raise ValueError("--bfile is required")
    if args.partition is None:
        raise ValueError("--partition is required")
    if args.ld_regu is None:
        raise ValueError("--ld-regu is required")
    if args.maf_min is None:
        args.maf_min = 0.01

    # processing some arguments
    try:
        ld_bfile, ld_inv_bfile = args.bfile.split(",")
    except:
        raise ValueError(
            "two bfiles must be provided with --bfile and separated with a comma"
        )
    for suffix in [".bed", ".fam", ".bim"]:
        ds.check_existence(ld_bfile, suffix)
        ds.check_existence(ld_inv_bfile, suffix)

    try:
        ld_regu, ld_inv_regu = [float(x) for x in args.ld_regu.split(",")]
    except:
        raise ValueError(
            (
                "two regularization levels must be provided with --prop "
                "and separated with a comma"
            )
        )
    if ld_regu >= 1 or ld_regu <= 0 or ld_inv_regu >= 1 or ld_inv_regu <= 0:
        raise ValueError(
            "both regularization levels must be greater than 0 and less than 1"
        )

    return ld_bfile, ld_inv_bfile, ld_regu, ld_inv_regu


def read_process_snps(bim_dir, log):
    log.info(f"Read SNP list from {bim_dir} and removed duplicated SNPs.")
    ld_bim = pd.read_csv(
        bim_dir,
        sep="\s+",
        header=None,
        names=["CHR", "SNP", "CM", "POS", "A1", "A2"],
        dtype={"A1": "category", "A2": "category"},
    )
    ld_bim.drop_duplicates(subset=["SNP"], keep=False, inplace=True)
    log.info(f"{ld_bim.shape[0]} SNPs remaining after removing duplicated SNPs.")

    return ld_bim


def read_process_idvs(fam_dir):
    ld_fam = pd.read_csv(
        fam_dir,
        sep="\s+",
        header=None,
        names=["FID", "IID", "FATHER", "MOTHER", "GENDER", "TRAIT"],
        dtype={"FID": str, "IID": str},
    )
    ld_fam = ld_fam.set_index(["FID", "IID"])

    return ld_fam


def filter_maf(
    ld_bfile,
    ld_keep_snp,
    ld_keep_idv,
    ld_inv_bfile,
    ld_inv_keep_snp,
    ld_inv_keep_idv,
    min_maf,
):
    ld_bim2, *_ = gt.read_plink(ld_bfile, ld_keep_snp, ld_keep_idv)
    ld_inv_bim2, *_ = gt.read_plink(ld_inv_bfile, ld_inv_keep_snp, ld_inv_keep_idv)
    if min_maf is not None:
        common_snps = ld_bim2.loc[
            (ld_bim2["MAF"] >= min_maf) & (ld_inv_bim2["MAF"] >= min_maf)
        ]

    return common_snps


def run(args, log):
    # checking if arguments are valid
    ld_bfile, ld_inv_bfile, ld_regu, ld_inv_regu = check_input(args)

    # reading and removing duplicated SNPs
    ld_bim = read_process_snps(ld_bfile + ".bim", log)
    ld_inv_bim = read_process_snps(ld_inv_bfile + ".bim", log)

    # merging two SNP lists
    ld_merged = ld_bim.merge(ld_inv_bim, on=["SNP", "A1", "A2"])
    log.info(
        f"{ld_merged.shape[0]} SNPs common in two bfiles with identical A1 and A2."
    )

    # extracting SNPs
    if args.extract is not None:
        ld_merged = ld_merged.loc[ld_merged["SNP"].isin(args.extract["SNP"])]
        log.info(f"{ld_merged.shape[0]} SNPs merged with --extract.")
    ld_keep_snp = ld_bim.merge(ld_merged, on="SNP")
    ld_inv_keep_snp = ld_inv_bim.merge(ld_merged, on="SNP")

    # keeping individuals
    if args.keep is not None:
        ld_fam = read_process_idvs(ld_bfile + ".fam")
        ld_keep_idv = ld_fam.loc[args.keep]
        log.info(f"Keeping {len(ld_keep_idv)} subjects in {ld_bfile}")
        ld_inv_fam = read_process_idvs(ld_inv_bfile + ".fam")
        ld_inv_keep_idv = ld_inv_fam.loc[args.keep]
        log.info(f"Keeping {len(ld_inv_keep_idv)} subjects in {ld_inv_bfile}")
    else:
        ld_keep_idv, ld_inv_keep_idv = None, None

    # filtering rare SNPs
    if args.maf_min is not None:
        log.info(f"Removing SNPs with MAF < {args.maf_min} ...")
        common_snps = filter_maf(
            ld_bfile,
            ld_keep_snp,
            ld_keep_idv,
            ld_inv_bfile,
            ld_inv_keep_snp,
            ld_inv_keep_idv,
            args.maf_min,
        )
        log.info(f"{len(common_snps)} SNPs remaining.")
    else:
        common_snps = None

    # reading bfiles
    log.info(f"Read bfile from {ld_bfile} with selected SNPs and individuals.")
    ld_bim, _, ld_snp_getter = gt.read_plink(ld_bfile, common_snps, ld_keep_idv)
    log.info(f"Read bfile from {ld_inv_bfile} with selected SNPs and individuals.")
    ld_inv_bim, _, ld_inv_snp_getter = gt.read_plink(
        ld_inv_bfile, common_snps, ld_inv_keep_idv
    )

    # reading and doing genome partition
    log.info(f"\nRead genome partition from {args.partition}")
    genome_part = ds.read_geno_part(args.partition)
    log.info(f"{genome_part.shape[0]} genome blocks to partition.")
    num_snps_part, ld_bim = partition_genome(ld_bim, genome_part, log)
    ld_inv_bim["block_idx"] = ld_bim["block_idx"].values
    ld_inv_bim["block_idx2"] = ld_bim["block_idx2"].values
    log.info(
        (
            f"{sum(num_snps_part)} SNPs partitioned into {len(num_snps_part)} blocks, "
            f"with the biggest one {np.max(num_snps_part)} SNPs."
        )
    )

    # making LD matrix and its inverse
    log.info(
        f"Regularization {ld_regu} for LD matrix, and {ld_inv_regu} for LD inverse matrix."
    )
    log.info(f"Making LD matrix and its inverse ...\n")
    ld = LDmatrixBED(num_snps_part, ld_bim, ld_snp_getter, ld_regu)
    ld_inv = LDmatrixBED(
        num_snps_part, ld_inv_bim, ld_inv_snp_getter, ld_inv_regu, inv=True
    )

    ld_prefix = ld.save(args.out, False, ld_regu)
    log.info(f"Saved LD matrix to {ld_prefix}.ldmatrix")
    log.info(f"Saved LD matrix info to {ld_prefix}.ldinfo")

    ld_inv_prefix = ld_inv.save(args.out, True, ld_inv_regu)
    log.info(f"Saved LD inverse matrix to {ld_inv_prefix}.ldmatrix")
    log.info(f"Saved LD inverse matrix info to {ld_inv_prefix}.ldinfo")

import os
import h5py
import logging
import unittest
import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal

from heig.sumstats import check_input, map_cols, read_sumstats, GWAS, GWASLDR, GWASY2

MAIN_DIR = os.path.join(os.getcwd(), "test", "test_sumstats")
log = logging.getLogger()
log.setLevel(logging.INFO)


class Args:
    def __init__(
        self,
        snp_col=None,
        a1_col=None,
        a2_col=None,
        n_col=None,
        effect_col=None,
        ldr_gwas=None,
        se_col=None,
        chr_col=None,
        pos_col=None,
        z_col=None,
        p_col=None,
        maf_col=None,
        maf_min=None,
        info_col=None,
        info_min=None,
        n=None,
        y2_gwas=None,
        ldr_gwas_heig=None,
        out=None,
    ):
        self.ldr_gwas = ldr_gwas
        self.y2_gwas = y2_gwas
        self.ldr_gwas_heig = ldr_gwas_heig
        self.snp_col = snp_col
        self.a1_col = a1_col
        self.a2_col = a2_col
        self.n_col = n_col
        self.effect_col = effect_col
        self.se_col = se_col
        self.chr_col = chr_col
        self.pos_col = pos_col
        self.z_col = z_col
        self.p_col = p_col
        self.maf_col = maf_col
        self.maf_min = maf_min
        self.info_col = info_col
        self.info_min = info_min
        self.n = n
        self.out = out
        self.threads = 1


class ProcessedArgs:
    def __init__(
        self,
        snp_col=None,
        a1_col=None,
        a2_col=None,
        n_col=None,
        effect=None,
        null_value=None,
        ldr_gwas=None,
        se_col=None,
        chr_col=None,
        pos_col=None,
        z_col=None,
        p_col=None,
        maf_col=None,
        maf_min=None,
        info_col=None,
        info_min=None,
        n=None,
        y2_gwas=None,
    ):
        self.ldr_gwas = ldr_gwas
        self.y2_gwas = y2_gwas
        self.snp_col = snp_col
        self.a1_col = a1_col
        self.a2_col = a2_col
        self.n_col = n_col
        self.effect = effect
        self.null_value = null_value
        self.se_col = se_col
        self.chr_col = chr_col
        self.pos_col = pos_col
        self.z_col = z_col
        self.p_col = p_col
        self.maf_col = maf_col
        self.maf_min = maf_min
        self.info_col = info_col
        self.info_min = info_min
        self.n = n

    def __eq__(self, other):
        for attr in vars(self).keys():
            if getattr(self, attr) != getattr(other, attr):
                print(
                    f"Attribute '{attr}' differs: '{getattr(self, attr)}' (self) vs '{getattr(other, attr)}' (other)"
                )
                return False
        return True


class Test_check_input(unittest.TestCase):
    def test_good_case1(self):
        """
        a standard case for LDR gwas without MAF and INFO filtering

        """
        args = Args(
            ldr_gwas=os.path.join(MAIN_DIR, "gwas{1:2}.txt"),
            snp_col="snp",
            a1_col="a1",
            a2_col="a2",
            n_col="n",
            effect_col="beta,0",
            se_col="se",
            chr_col="chr",
            pos_col="pos",
            z_col="z",
        )
        check_input(args, log)
        true_args = ProcessedArgs(
            ldr_gwas=[
                os.path.join(MAIN_DIR, "gwas1.txt"),
                os.path.join(MAIN_DIR, "gwas2.txt"),
            ],
            snp_col="snp",
            a1_col="a1",
            a2_col="a2",
            n_col="n",
            effect="beta",
            null_value=0,
            se_col="se",
            chr_col="chr",
            pos_col="pos",
            z_col="z",
        )
        self.assertEqual(args, true_args)

    def test_good_case2(self):
        """
        good cases with MAF and INFO filtering

        """
        args = Args(
            ldr_gwas=os.path.join(MAIN_DIR, "gwas{1:2}.txt"),
            snp_col="snp",
            a1_col="a1",
            a2_col="a2",
            n=10000,
            effect_col="beta,0",
            se_col="se",
            chr_col="chr",
            pos_col="pos",
            maf_col="maf",
            info_col="info",
        )
        check_input(args, log)
        true_args = ProcessedArgs(
            ldr_gwas=[
                os.path.join(MAIN_DIR, "gwas1.txt"),
                os.path.join(MAIN_DIR, "gwas2.txt"),
            ],
            snp_col="snp",
            a1_col="a1",
            a2_col="a2",
            n=10000,
            effect="beta",
            null_value=0,
            se_col="se",
            chr_col="chr",
            pos_col="pos",
            maf_col="maf",
            info_col="info",
            maf_min=None,
            info_min=None,
        )
        self.assertEqual(args, true_args)

        args = Args(
            ldr_gwas=os.path.join(MAIN_DIR, "gwas{1:2}.txt"),
            snp_col="snp",
            a1_col="a1",
            a2_col="a2",
            n=10000,
            effect_col="beta,0",
            se_col="se",
            chr_col="chr",
            pos_col="pos",
            maf_col="maf",
            info_col="info",
            maf_min=0.05,
            info_min=0.8,
        )
        check_input(args, log)
        true_args = ProcessedArgs(
            ldr_gwas=[
                os.path.join(MAIN_DIR, "gwas1.txt"),
                os.path.join(MAIN_DIR, "gwas2.txt"),
            ],
            snp_col="snp",
            a1_col="a1",
            a2_col="a2",
            n=10000,
            effect="beta",
            null_value=0,
            se_col="se",
            chr_col="chr",
            pos_col="pos",
            maf_col="maf",
            info_col="info",
            maf_min=0.05,
            info_min=0.8,
        )
        self.assertEqual(args, true_args)

        args = Args(
            ldr_gwas=os.path.join(MAIN_DIR, "gwas{1:2}.txt"),
            snp_col="snp",
            a1_col="a1",
            a2_col="a2",
            n=10000,
            effect_col="beta,0",
            se_col="se",
            chr_col="chr",
            pos_col="pos",
            maf_min=0.05,
            info_min=0.8,
        )
        check_input(args, log)
        true_args = ProcessedArgs(
            ldr_gwas=[
                os.path.join(MAIN_DIR, "gwas1.txt"),
                os.path.join(MAIN_DIR, "gwas2.txt"),
            ],
            snp_col="snp",
            a1_col="a1",
            a2_col="a2",
            n=10000,
            effect="beta",
            null_value=0,
            se_col="se",
            chr_col="chr",
            pos_col="pos",
        )
        self.assertEqual(args, true_args)

    def test_good_case3(self):
        """
        multiple ways of input for y2_gwas

        """
        # beta and se
        args = Args(
            y2_gwas=os.path.join(MAIN_DIR, "gwas1.txt"),
            snp_col="snp",
            a1_col="a1",
            a2_col="a2",
            n_col="n",
            effect_col="beta,0",
            se_col="se",
            chr_col="chr",
            pos_col="pos",
        )
        check_input(args, log)
        true_args = ProcessedArgs(
            y2_gwas=[os.path.join(MAIN_DIR, "gwas1.txt")],
            snp_col="snp",
            a1_col="a1",
            a2_col="a2",
            n_col="n",
            effect="beta",
            null_value=0,
            se_col="se",
            chr_col="chr",
            pos_col="pos",
        )
        self.assertEqual(args, true_args)

        # or and se
        args = Args(
            y2_gwas=os.path.join(MAIN_DIR, "gwas1.txt"),
            snp_col="snp",
            a1_col="a1",
            a2_col="a2",
            n_col="n",
            effect_col="or,1",
            se_col="se",
        )
        check_input(args, log)
        true_args = ProcessedArgs(
            y2_gwas=[os.path.join(MAIN_DIR, "gwas1.txt")],
            snp_col="snp",
            a1_col="a1",
            a2_col="a2",
            se_col="se",
            n_col="n",
            effect="or",
            null_value=1,
        )
        self.assertEqual(args, true_args)

        # z
        args = Args(
            y2_gwas=os.path.join(MAIN_DIR, "gwas1.txt"),
            snp_col="snp",
            a1_col="a1",
            a2_col="a2",
            n_col="n",
            z_col="z",
        )
        check_input(args, log)
        true_args = ProcessedArgs(
            y2_gwas=[os.path.join(MAIN_DIR, "gwas1.txt")],
            snp_col="snp",
            a1_col="a1",
            a2_col="a2",
            n_col="n",
            z_col="z",
        )
        self.assertEqual(args, true_args)

        # p-value and effect
        args = Args(
            y2_gwas=os.path.join(MAIN_DIR, "gwas1.txt"),
            snp_col="snp",
            a1_col="a1",
            a2_col="a2",
            n_col="n",
            p_col="pv",
            effect_col="beta,0",
        )
        check_input(args, log)
        true_args = ProcessedArgs(
            y2_gwas=[os.path.join(MAIN_DIR, "gwas1.txt")],
            snp_col="snp",
            a1_col="a1",
            a2_col="a2",
            n_col="n",
            p_col="pv",
            effect="beta",
            null_value=0,
        )
        self.assertEqual(args, true_args)

    def test_bad_cases(self):
        # missing any required component for LDR gwas
        def exclude_key(d, key_to_exclude):
            return {key: value for key, value in d.items() if key != key_to_exclude}

        args_dict = {
            "ldr_gwas": os.path.join(MAIN_DIR, "gwas{1:2}.txt"),
            "snp_col": "snp",
            "a1_col": "a1",
            "a2_col": "a2",
            "n_col": "n",
            "effect_col": "beta,0",
            "se_col": "se",
            "chr_col": "chr",
            "pos_col": "pos",
        }
        for key in args_dict.keys():
            args_dict = exclude_key(args_dict, key)
            args = Args(**args_dict)
            with self.assertRaises(ValueError):
                check_input(args, log)

        # missing required components for y2 gwas
        args = Args(
            y2_gwas=os.path.join(MAIN_DIR, "gwas1.txt"),
            snp_col="snp",
            a1_col="a1",
            a2_col="a2",
            n_col="n",
            p_col="pv",
        )
        with self.assertRaises(ValueError):
            check_input(args, log)

        args = Args(
            y2_gwas=os.path.join(MAIN_DIR, "gwas1.txt"),
            snp_col="snp",
            a1_col="a1",
            a2_col="a2",
            n_col="n",
        )
        with self.assertRaises(ValueError):
            check_input(args, log)

        # non-existing files
        args = Args(
            y2_gwas=os.path.join(MAIN_DIR, "gwas3.txt"),
            snp_col="snp",
            a1_col="a1",
            a2_col="a2",
            n_col="n",
            z_col="z",
        )
        with self.assertRaises(FileNotFoundError):
            check_input(args, log)

        # invalid MAF/INFO threshold
        args = Args(
            y2_gwas=os.path.join(MAIN_DIR, "gwas1.txt"),
            snp_col="snp",
            a1_col="a1",
            a2_col="a2",
            n_col="n",
            p_col="pv",
            effect_col="beta,0",
            maf_col="maf",
            maf_min=-1,
        )
        with self.assertRaises(ValueError):
            check_input(args, log)

        args = Args(
            y2_gwas=os.path.join(MAIN_DIR, "gwas1.txt"),
            snp_col="snp",
            a1_col="a1",
            a2_col="a2",
            n_col="n",
            p_col="pv",
            effect_col="beta,0",
            info_col="maf",
            info_min=-1,
        )
        with self.assertRaises(ValueError):
            check_input(args, log)


class Test_map_cols(unittest.TestCase):
    def test_map_cols(self):
        args = Args(
            ldr_gwas=os.path.join(MAIN_DIR, "gwas{1:2}.txt"),
            snp_col="snp",
            a1_col="a1",
            a2_col="a2",
            n=10000,
            effect_col="beta,0",
            se_col="se",
            chr_col="chr",
            pos_col="pos",
            maf_col="maf",
            info_col="info",
        )
        check_input(args, log)
        true_col_map1 = {
            "SNP": "snp",
            "A1": "a1",
            "A2": "a2",
            "EFFECT": "beta",
            "SE": "se",
            "CHR": "chr",
            "POS": "pos",
            "MAF": "maf",
            "INFO": "info",
            "N": None,
            "n": 10000,
            "P": None,
            "null_value": 0,
            "Z": None,
            "maf_min": None,
            "info_min": None,
        }
        true_col_map2 = {
            "snp": "SNP",
            "a1": "A1",
            "a2": "A2",
            "beta": "EFFECT",
            "se": "SE",
            "chr": "CHR",
            "pos": "POS",
            "maf": "MAF",
            "info": "INFO",
        }
        col_map1, col_map2 = map_cols(args)
        self.assertEqual(true_col_map1, col_map1)
        self.assertEqual(true_col_map2, col_map2)


class Test_GWAS(unittest.TestCase):
    def get_ldr_sumstats(self, args, log):
        check_input(args, log)
        cols_map1, cols_map2 = map_cols(args)
        sumstats = GWASLDR(
            args.ldr_gwas, cols_map1, cols_map2, args.out, args.maf_min, args.info_min
        )
        sumstats.process(args.threads)

        with h5py.File(os.path.join(MAIN_DIR, "gwas.sumstats"), "r") as file:
            beta = file["beta0"][:]
            z = file["z0"][:]
        snpinfo = pd.read_csv(os.path.join(MAIN_DIR, "gwas.snpinfo"), sep="\t")

        return beta, z, snpinfo

    def get_y2_sumstats(self, args, log):
        check_input(args, log)
        cols_map1, cols_map2 = map_cols(args)
        sumstats = GWASY2(
            args.y2_gwas, cols_map1, cols_map2, args.out, args.maf_min, args.info_min
        )
        sumstats.process(args.threads)

        with h5py.File(os.path.join(MAIN_DIR, "gwasy2.sumstats"), "r") as file:
            z = file["z0"][:]
        snpinfo = pd.read_csv(os.path.join(MAIN_DIR, "gwasy2.snpinfo"), sep="\t")

        return z, snpinfo

    def test_ldr_gwas(self):
        true_beta = np.array([0.0, 0.0, 0.0, 0.0]).reshape((2, 2))
        true_se = np.array([0.5, 0.5, 0.5, 0.5]).reshape((2, 2))
        true_z = true_beta / true_se
        true_snpinfo = pd.DataFrame(
            {
                "CHR": [2, 2],
                "POS": [10, 4],
                "SNP": ["rs2", "rs4"],
                "A1": ["C", "C"],
                "A2": ["A", "A"],
                "MAF": [0.05, 0.1],
                "N": [100, 100],
            }
        )
        # regular case
        args = Args(
            ldr_gwas=os.path.join(MAIN_DIR, "gwas{1:2}.txt"),
            snp_col="snp",
            a1_col="a1",
            a2_col="a2",
            n_col="n",
            effect_col="beta,0",
            se_col="se",
            chr_col="chr",
            pos_col="pos",
            maf_col="maf",
            out=os.path.join(MAIN_DIR, "gwas"),
        )
        beta, z, snpinfo = self.get_ldr_sumstats(args, log)
        assert_array_equal(beta, true_beta)
        assert_array_equal(z, true_z)
        assert_array_equal(snpinfo.values, true_snpinfo.values)

        # nonexisting columns
        args = Args(
            ldr_gwas=os.path.join(MAIN_DIR, "gwas{1:2}.txt"),
            snp_col="snp",
            a1_col="a1",
            a2_col="a2",
            n_col="n",
            effect_col="beta,0",
            se_col="se",
            chr_col="chr",
            pos_col="pos",
            maf_col="maf11",
            out=os.path.join(MAIN_DIR, "gwas"),
        )
        with self.assertRaises(ValueError):
            self.get_ldr_sumstats(args, log)

        args = Args(
            ldr_gwas=os.path.join(MAIN_DIR, "gwas{1:2}.txt"),
            snp_col="snp",
            a1_col="a1",
            a2_col="a222",
            n_col="n",
            effect_col="beta,0",
            se_col="se",
            chr_col="chr",
            pos_col="pos",
            maf_col="maf11",
            out=os.path.join(MAIN_DIR, "gwas"),
        )
        with self.assertRaises(ValueError):
            self.get_ldr_sumstats(args, log)

    def test_y2_gwas(self):
        true_z = np.array([0.0, 0.0]).reshape((2, 1))
        true_snpinfo = pd.DataFrame(
            {"SNP": ["rs2", "rs4"], "A1": ["C", "C"], "A2": ["A", "A"], "N": [100, 100]}
        )
        # beta and se, maf
        args = Args(
            y2_gwas=os.path.join(MAIN_DIR, "gwas1.txt"),
            snp_col="snp",
            a1_col="a1",
            a2_col="a2",
            n_col="n",
            effect_col="beta,0",
            se_col="se",
            chr_col="chr",
            pos_col="pos",
            maf_col="maf",
            out=os.path.join(MAIN_DIR, "gwasy2"),
        )
        z, snpinfo = self.get_y2_sumstats(args, log)
        assert_array_equal(z, true_z)
        assert_array_equal(snpinfo, true_snpinfo.values)

        # or and se
        args = Args(
            y2_gwas=os.path.join(MAIN_DIR, "gwas1.txt"),
            snp_col="snp",
            a1_col="a1",
            a2_col="a2",
            n_col="n",
            effect_col="or,1",
            se_col="se",
            chr_col="chr",
            pos_col="pos",
            out=os.path.join(MAIN_DIR, "gwasy2"),
        )
        z, snpinfo = self.get_y2_sumstats(args, log)
        assert_array_equal(z, true_z)

        # z
        args = Args(
            y2_gwas=os.path.join(MAIN_DIR, "gwas1.txt"),
            snp_col="snp",
            a1_col="a1",
            a2_col="a2",
            n_col="n",
            z_col="z",
            out=os.path.join(MAIN_DIR, "gwasy2"),
        )
        z, snpinfo = self.get_y2_sumstats(args, log)
        assert_array_equal(z, true_z)

        # effect and p, info
        true_snpinfo = pd.DataFrame(
            {"SNP": ["rs11", "rs2", "rs4"], 
             "A1": ["T", "C", "C"], 
             "A2": ["G", "A", "A"], 
             "N": [100, 100, 100]}
        )
        true_z = np.array([-0.67448974, 0.0, 0.0], dtype=np.float32).reshape((3, 1))
        args = Args(
            y2_gwas=os.path.join(MAIN_DIR, "gwas1.txt"),
            snp_col="snp",
            a1_col="a1",
            a2_col="a2",
            n_col="n",
            effect_col="or,1",
            p_col="p",
            info_col="info",
            out=os.path.join(MAIN_DIR, "gwasy2"),
        )
        z, snpinfo = self.get_y2_sumstats(args, log)
        assert_array_equal(z, true_z)
        assert_array_equal(snpinfo.values, true_snpinfo.values)

        # nonexisting columns
        args = Args(
            y2_gwas=os.path.join(MAIN_DIR, "gwas1.txt"),
            snp_col="snp221",
            a1_col="a1",
            a2_col="a2",
            n_col="n",
            z_col="z",
            out=os.path.join(MAIN_DIR, "gwasy2"),
        )
        with self.assertRaises(ValueError):
            self.get_y2_sumstats(args, log)


class Test_read_sumstats(unittest.TestCase):
    def test_read_sumstats(self):
        true_beta = np.array([0, 0, 0, 0]).reshape((2, 2))
        true_se = np.array([1, 1, 1, 1]).reshape((2, 2))
        true_z = true_beta / true_se
        true_snpinfo = pd.DataFrame(
            {
                "CHR": [2, 2],
                "POS": [10, 4],
                "SNP": ["rs2", "rs4"],
                "A1": ["C", "C"],
                "A2": ["A", "A"],
                "MAF": [0.05, 0.1],
                "N": [100, 100],
            }
        )
        sumstats = read_sumstats("test/test_sumstats/gwas")
        assert_array_equal(sumstats.file["beta0"][:], true_beta)
        assert_array_equal(sumstats.file["z0"][:], true_z)
        assert_array_equal(sumstats.snpinfo.values, true_snpinfo.values)


class Test_extract_snps(unittest.TestCase):
    def test_extract_snps(self):
        true_z = np.array([0.0, 0.0]).reshape((1, 2))
        true_snpinfo = pd.DataFrame(
            {
                "SNP": ["rs2"],
                "CHR": [2],
                "POS": [10],
                "A1": ["C"],
                "A2": ["A"],
                "MAF": [0.05],
                "N": [100],
            }
        )
        sumstats = read_sumstats("test/test_sumstats/gwas")
        sumstats.extract_snps(pd.Series(["rs2"], name="SNP"))

        assert_array_equal(sumstats.file["z0"][sumstats.snp_idxs], true_z)
        assert_array_equal(sumstats.snpinfo.values, true_snpinfo.values)


class toy_h5file:
    def __init__(self):
        self.attrs = {"n_snps": None, "n_gwas": None, "n_blocks": None}


class Test_align_alleles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        snpinfo = pd.DataFrame(
            {"SNP": ["rs1", "rs2"], "A1": ["A", "C"], "A2": ["G", "T"]}
        )
        cls.file = toy_h5file()
        cls.gwas = GWAS(cls.file, snpinfo)

        cls.bim = pd.DataFrame(
            {"SNP": ["rs1", "rs2"], "A1": ["A", "C"], "A2": ["G", "T"]}
        )

    def test_matched_alleles(self):
        change_sign = np.ones([0, 0], dtype=bool)
        assert_array_equal(change_sign, self.gwas.align_alleles(self.bim))

    def test_unmatched_alleles(self):
        snpinfo = pd.DataFrame(
            {"SNP": ["rs1", "rs2"], "A1": ["G", "C"], "A2": ["A", "T"]}
        )
        gwas = GWAS(self.file, snpinfo)
        change_sign = np.ones([1, 0], dtype=bool)
        assert_array_equal(change_sign, gwas.align_alleles(self.bim))

    def test_diff_snps(self):
        bim = pd.DataFrame({"SNP": ["rs1", "rs3"], "A1": ["A", "C"], "A2": ["G", "T"]})
        with self.assertRaises(ValueError):
            self.gwas.align_alleles(bim)

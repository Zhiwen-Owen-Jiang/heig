import os
import unittest
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal, assert_index_equal

from heig.input.dataset import (
    read_keep,
    read_geno_part,
    read_extract,
    Dataset,
    Covar,
    get_common_idxs,
    parse_input,
)
from heig.input.genotype import read_plink


MAIN_DIR = os.getcwd()


class Test_read_keep(unittest.TestCase):
    def setUp(self):
        folder = os.path.join(MAIN_DIR, "test", "test_input", "keep_files")

        self.keep_files = ["keep1.txt", "keep2.txt", "keep3.txt", "keep_empty.txt"]
        self.keep_files = [os.path.join(folder, x) for x in self.keep_files]

        self.bad_keep_files = ["keep_bad1.txt"]
        self.bad_keep_files = [os.path.join(folder, x) for x in self.bad_keep_files]

    def test_read_keep(self):
        true_value = pd.MultiIndex.from_arrays(
            [("s2", "s3"), ("s2", "s3")], names=["FID", "IID"]
        )
        self.assertTrue(read_keep(self.keep_files).equals(true_value))

        with self.assertRaises(ValueError):
            read_keep(self.bad_keep_files)


class Test_read_extract(unittest.TestCase):
    def setUp(self):
        folder = os.path.join(MAIN_DIR, "test", "test_input", "extract_files")
        self.good_files = ["extract1.txt", "extract2.txt", "extract3.txt"]
        self.good_files = [os.path.join(folder, x) for x in self.good_files]

        self.bad_files = ["extract1.txt", "extract4.txt"]
        self.bad_files = [os.path.join(folder, x) for x in self.bad_files]

    def test_read_extract(self):
        true_value = pd.DataFrame({"SNP": ["rs2"]})
        self.assertTrue(read_extract(self.good_files).equals(true_value))

        with self.assertRaises(ValueError):
            read_extract(self.bad_files)


class Test_read_plink(unittest.TestCase):
    def setUp(self):
        self.folder = os.path.join(MAIN_DIR, "test", "test_input", "plink")
        self.bed_mat = np.load(os.path.join(self.folder, "plink_bed.npy"))
        self.bim_df = pd.read_csv(
            os.path.join(self.folder, "plink.bim"),
            sep="\t",
            header=None,
            names=["CHR", "SNP", "CM", "POS", "A1", "A2"],
        )
        self.fam_df = pd.read_csv(
            os.path.join(self.folder, "plink.fam"),
            sep="\t",
            header=None,
            usecols=[0, 1, 4],
            names=["FID", "IID", "SEX"],
        ).set_index(["FID", "IID"])

    def screen_maf(self, min_maf=None):
        maf = np.nanmean(self.bed_mat, axis=0) / 2
        maf[maf > 0.5] = 1 - maf[maf > 0.5]
        self.bim_df["MAF"] = maf
        self.bim_df["MAF"] = self.bim_df["MAF"].astype(np.float64)
        if min_maf is not None:
            self.bim_df = self.bim_df.loc[self.bim_df["MAF"] > min_maf]
            self.bed_mat = self.bed_mat[:, maf > min_maf]

    def test_basic(self):
        bim, fam, snp_getter = read_plink(os.path.join(self.folder, "plink"))
        snp_mat = snp_getter(len(bim))

        self.screen_maf()

        assert_array_equal(self.bed_mat, snp_mat)
        assert_frame_equal(self.bim_df, bim)
        assert_frame_equal(self.fam_df, fam)

    def test_keep_snps(self):
        keep_snps = pd.DataFrame({"SNP": ["rs6151412", "rs5770964", "rs78222150"]})
        bim, fam, snp_getter = read_plink(
            os.path.join(self.folder, "plink"), keep_snps=keep_snps
        )
        snp_mat = snp_getter(len(bim))

        self.bim_df = self.bim_df.iloc[:3, :]
        self.bed_mat = self.bed_mat[:, :3]
        self.screen_maf()

        assert_array_equal(self.bed_mat[:, :3], snp_mat)
        assert_frame_equal(self.bim_df, bim)
        assert_frame_equal(self.fam_df, fam)

    def test_keep_invids(self):
        invids_list = ["s" + str(i) for i in range(1, 1 + 15)]
        keep_invids = pd.MultiIndex.from_arrays(
            [tuple(invids_list), tuple(invids_list)], names=["FID", "IID"]
        )
        bim, fam, snp_getter = read_plink(
            os.path.join(self.folder, "plink"), keep_indivs=keep_invids
        )
        snp_mat = snp_getter(len(bim))

        self.fam_df = self.fam_df.iloc[:15, :]
        self.bed_mat = self.bed_mat[:15]
        self.screen_maf()

        assert_array_equal(self.bed_mat[:15], snp_mat)
        assert_frame_equal(self.bim_df, bim)
        assert_frame_equal(self.fam_df, fam)

    def test_maf(self):
        bim, fam, snp_getter = read_plink(os.path.join(self.folder, "plink"), maf=0.2)
        snp_mat = snp_getter(len(bim))

        self.screen_maf(min_maf=0.2)

        assert_array_equal(snp_mat, self.bed_mat)
        assert_frame_equal(self.bim_df, bim)
        assert_frame_equal(self.fam_df, fam)


class Test_read_geno_part(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = os.path.join(MAIN_DIR, "test", "test_input", "geno_part_files")
        cls.geno_part = pd.read_csv(
            os.path.join(cls.folder, "geno_part"), sep="\s+", header=None
        )

    def test_good_cases(self):
        geno_part = read_geno_part(os.path.join(self.folder, "geno_part"))
        self.assertTrue(geno_part.equals(self.geno_part))
        geno_part = read_geno_part(os.path.join(self.folder, "geno_part.gz"))
        self.assertTrue(geno_part.equals(self.geno_part))
        geno_part = read_geno_part(os.path.join(self.folder, "geno_part.bz2"))
        self.assertTrue(geno_part.equals(self.geno_part))

    def test_bad_cases(self):
        with self.assertRaises(ValueError):
            read_geno_part(os.path.join(self.folder, "geno_part.tar"))
            read_geno_part(os.path.join(self.folder, "geno_part.zip"))


class Test_DataSet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = os.path.join(MAIN_DIR, "test", "test_input", "dataset")

    def test_dup_rows_remove_na(self):
        # there are missing values in data, so it is float
        true_data = pd.DataFrame(
            {
                "FID": ["s1", "s2", "s6"],
                "IID": ["s1", "s2", "s6"],
                "data": [1.0, 2.0, 6.0],
            }
        ).set_index(["FID", "IID"])
        data = Dataset(os.path.join(self.folder, "data_dup_rows_na.txt"))
        assert_frame_equal(true_data, data.data)

    def test_dup_cols(self):
        with self.assertRaises(ValueError):
            Dataset(os.path.join(self.folder, "data_dup_cols.txt"))

    def test_keep_rows(self):
        true_data = pd.DataFrame(
            {"FID": ["s6", "s1"], "IID": ["s6", "s1"], "data": [6.0, 1.0]}
        ).set_index(["FID", "IID"])
        data = Dataset(os.path.join(self.folder, "data_dup_rows_na.txt"))
        keep_rows = pd.MultiIndex.from_arrays(
            [("s6", "s1"), ("s6", "s1")], names=["FID", "IID"]
        )
        data.keep_and_remove(keep_idx=keep_rows)
        assert_frame_equal(true_data, data.data)


class Test_Covar(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = os.path.join(MAIN_DIR, "test", "test_input", "dataset")

    def test_cont_covar(self):
        # there is no missing value in covar1, so it is int
        true_data = pd.DataFrame(
            {
                "FID": ["s1", "s2", "s3"],
                "IID": ["s1", "s2", "s3"],
                "intercept": [1.0, 1.0, 1.0],
                "covar1": [1, 2, 3],
            }
        ).set_index(["FID", "IID"])
        data = Covar(os.path.join(self.folder, "covar_cont.txt"))
        data.cat_covar_intercept()
        assert_frame_equal(true_data, data.data)

    def test_cate_covar(self):
        # forgot specifying any categorical variables
        with self.assertRaises(ValueError):
            data = Covar(os.path.join(self.folder, "covar_cate.txt"))
            data.cat_covar_intercept()

        # forgot specifying a categorical variable
        with self.assertRaises(ValueError):
            data = Covar(os.path.join(self.folder, "covar_cate.txt"), "covar2")
            data.cat_covar_intercept()

        # wrong syntax to specify a categorical variable
        with self.assertRaises(ValueError):
            data = Covar(os.path.join(self.folder, "covar_cate.txt"), "covar2.covar3")
            data.cat_covar_intercept()

        # nonexisting columns
        with self.assertRaises(ValueError):
            data = Covar(os.path.join(self.folder, "covar_cate.txt"), "covar10")
            data.cat_covar_intercept()

        # singularity
        with self.assertRaises(ValueError):
            data = Covar(
                os.path.join(self.folder, "covar_cate_sing.txt"), "covar2,covar3"
            )
            data.cat_covar_intercept()

        # correct case
        true_data = pd.DataFrame(
            {
                "FID": ["s1", "s2", "s3", "s4", "s5", "s6"],
                "IID": ["s1", "s2", "s3", "s4", "s5", "s6"],
                "intercept": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "covar2_b": [0, 1, 1, 0, 0, 0],
                "covar2_c": [0, 0, 0, 0, 1, 1],
                "covar3_b": [0, 1, 1, 1, 0, 0],
                "covar1": [1, 2, 3, 4, 5, 6],
            }
        ).set_index(["FID", "IID"])
        data = Covar(os.path.join(self.folder, "covar_cate.txt"), "covar2,covar3")
        data.cat_covar_intercept()
        assert_frame_equal(true_data, data.data)


class Test_get_common_idxs(unittest.TestCase):
    def test_get_common_idxs(self):
        idxs1 = pd.MultiIndex.from_arrays(
            [("s1", "s2", "s3"), ("s1", "s2", "s3")], names=["FID", "IID"]
        )
        idxs2 = pd.MultiIndex.from_arrays(
            [("s3", "s4", "s2"), ("s3", "s4", "s2")], names=["FID", "IID"]
        )
        idxs3 = pd.MultiIndex.from_arrays(
            [("s5", "s6", "s7"), ("s5", "s6", "s7")], names=["FID", "IID"]
        )

        # correct cases
        true1 = pd.MultiIndex.from_arrays(
            [("s2", "s3"), ("s2", "s3")], names=["FID", "IID"]
        )
        assert_index_equal(true1, get_common_idxs(idxs1, idxs2))

        true2 = pd.MultiIndex.from_arrays(
            [("s3", "s2"), ("s3", "s2")], names=["FID", "IID"]
        )
        assert_index_equal(true2, get_common_idxs(idxs2, idxs1, None))

        # specified other data types
        with self.assertRaises(TypeError):
            get_common_idxs(idxs2, idxs1, [1, 2, 3])

        # no overlap
        with self.assertRaises(ValueError):
            get_common_idxs(idxs1, idxs3)


class Test_parse_input(unittest.TestCase):
    def test_parse_input_good(self):
        true_value = ["file1", "file2"]
        self.assertEqual(true_value, parse_input("file{1:2}"))
        true_value = ["file1_a", "file2_a"]
        self.assertEqual(true_value, parse_input("file{1:2}_a"))
        true_value = ["file1-a", "file2-a"]
        self.assertEqual(true_value, parse_input("file{1:2}-a"))
        true_value = ["file1~a", "file2~a"]
        self.assertEqual(true_value, parse_input("file{1:2}~a"))
        true_value = ["file1.a", "file2.a"]
        self.assertEqual(true_value, parse_input("file{1:2}.a"))
        true_value = ["file1.a", "file2.a"]
        self.assertEqual(true_value, parse_input("file{2:1}.a"))
        true_value = ["file1.a"]
        self.assertEqual(true_value, parse_input("file{1:1}.a"))
        true_value = ["file1"]
        self.assertEqual(true_value, parse_input("file1"))
        true_value = ["file1:a"]
        self.assertEqual(true_value, parse_input("file1:a"))
        true_value = ["file{}"]
        self.assertEqual(true_value, parse_input("file{}"))
        true_value = ["file1", "file2"]
        self.assertEqual(true_value, parse_input("file1,file2"))
        true_value = ["file1", ":file2"]
        self.assertEqual(true_value, parse_input("file1,:file2"))

    def test_parse_input_bad(self):
        with self.assertRaises(ValueError):
            parse_input("file{1:}.a")
        with self.assertRaises(ValueError):
            parse_input("file{1:a}.a")
        with self.assertRaises(ValueError):
            parse_input("file{:}.a")
        with self.assertRaises(ValueError):
            parse_input("file{:},a")
            
    def test_split_voxels(self):
        true_value = ["1"]
        self.assertEqual(true_value, parse_input("1"))
        true_value = ["1", "2"]
        self.assertEqual(true_value, parse_input("1,2"))
        true_value = ["1", "2", "3"]
        self.assertEqual(true_value, parse_input("{1:3}"))
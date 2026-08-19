import os
import tempfile
import unittest
import h5py
import pandas as pd
import numpy as np
from pandas.testing import assert_series_equal
from numpy.testing import assert_array_equal, assert_array_almost_equal


from heig.herigc import CommonSNPs, OneSample


class toy_GWAS:
    def __init__(self, snpinfo):
        self.snpinfo = snpinfo


class toy_LDmatrix:
    def __init__(self, ldinfo):
        self.ldinfo = ldinfo


class Test_get_common_snps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        snpinfo = pd.DataFrame(
            {"SNP": ["rs1", "rs3"], "A1": ["G", "C"], "A2": ["A", "A"]}
        )
        cls.gwas = toy_GWAS(snpinfo)

        ldinfo1 = pd.DataFrame(
            {"SNP": ["rs1", "rs2", "rs3"], "A1": ["A", "C", "C"], "A2": ["G", "T", "A"]}
        )
        cls.ld1 = toy_LDmatrix(ldinfo1)

        ldinfo2 = pd.DataFrame(
            {"SNP": ["rs1", "rs2", "rs3"], "A1": ["A", "C", "G"], "A2": ["G", "T", "A"]}
        )
        cls.ld2 = toy_LDmatrix(ldinfo2)

        ldinfo3 = pd.DataFrame(
            {"SNP": ["rs1", "rs3", "rs3"], "A1": ["A", "C", "G"], "A2": ["G", "T", "A"]}
        )
        cls.ld3 = toy_LDmatrix(ldinfo3)

        cls.keep_snps1 = pd.DataFrame({"SNP": ["rs2", "rs3"]})
        cls.keep_snps2 = pd.DataFrame({"SNP": ["rs4"]})

    def test_null(self):
        """
        no snp list provided

        """
        with self.assertRaises(ValueError):
            CommonSNPs(exclude_snps=None, threads=1)

    def test_case1(self):
        """
        test if snp list order matters

        """
        true_res = pd.Series({0: "rs1", 1: "rs3"}, name="SNP")
        assert_series_equal(
            true_res,
            CommonSNPs(self.gwas, self.ld1, exclude_snps=None, threads=1).common_snps,
        )
        assert_series_equal(
            true_res,
            CommonSNPs(self.ld1, self.gwas, exclude_snps=None, threads=1).common_snps,
        )

    def test_case2(self):
        """
        test different combinations of snp lists

        """
        true_res1 = pd.Series({0: "rs3"}, name="SNP")
        true_res2 = pd.Series({0: "rs2", 1: "rs3"}, name="SNP")
        assert_series_equal(
            true_res1,
            CommonSNPs(
                self.gwas, self.ld1, self.keep_snps1, exclude_snps=None, threads=1
            ).common_snps,
        )
        assert_series_equal(
            true_res1,
            CommonSNPs(
                self.gwas, self.keep_snps1, exclude_snps=None, threads=1
            ).common_snps,
        )
        assert_series_equal(
            true_res2,
            CommonSNPs(
                self.ld1, self.keep_snps1, exclude_snps=None, threads=1
            ).common_snps,
        )

    def test_case3(self):
        """
        no common SNPs exist

        """
        with self.assertRaises(ValueError):
            CommonSNPs(
                self.gwas, self.ld1, self.keep_snps2, exclude_snps=None, threads=1
            ).common_snps

    def test_case4(self):
        """
        inconsistent alleles

        """
        true_res = pd.Series({0: "rs1"}, name="SNP")
        assert_series_equal(
            true_res,
            CommonSNPs(self.gwas, self.ld2, exclude_snps=None, threads=1).common_snps,
        )

    def test_case5(self):
        """
        duplicated SNPs

        """
        true_res = pd.Series({0: "rs1"}, name="SNP")
        assert_series_equal(
            true_res,
            CommonSNPs(self.gwas, self.ld3, exclude_snps=None, threads=1).common_snps,
        )


class Test_check_alleles(unittest.TestCase):
    def test_check_alleles(self):
        ldinfo1 = pd.DataFrame(
            {"SNP": ["rs1", "rs2", "rs3"], "A1": ["A", "C", "G"], "A2": ["G", "T", "A"]}
        )
        ldinfo2 = pd.DataFrame(
            {"SNP": ["rs1", "rs2", "rs3"], "A1": ["A", "C", "G"], "A2": ["G", "T", "A"]}
        )
        ldinfo3 = pd.DataFrame(
            {
                "SNP": ["rs1", "rs2", "rs3"],
                "A1": ["A", "C", "G"],
                "A2": ["G", "T", "A"],
            },
            index=[10, 20, 30],
        )
        assert_array_equal(ldinfo1[["A1", "A2"]].values, ldinfo2[["A1", "A2"]].values)
        assert_array_equal(ldinfo1[["A1", "A2"]].values, ldinfo3[["A1", "A2"]].values)
        self.assertTrue(
            np.equal(ldinfo1[["A1", "A2"]].values, ldinfo3[["A1", "A2"]].values).all()
        )


class Test_get_gene_cor_se_block(unittest.TestCase):
    """
    Each block covers rows [start, start+100) of the (N, N) genetic correlation
    matrix, so its diagonal sits at global columns [start, start+100), not at
    columns [0, 100).

    """

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        N, r = 250, 8  # blocks starting at 0, 100, 200
        cls.N = N

        obj = OneSample.__new__(OneSample)  # skip __init__, no GWAS/LD needed
        obj.N = N
        obj.bases = rng.normal(size=(N, r))
        a = rng.normal(size=(r, r))
        obj.ldr_gene_cov = a @ a.T
        b = rng.normal(size=(r, r))
        obj.ldr_cov = b @ b.T
        obj.gene_var = np.einsum("ij,jk,ik->i", obj.bases, obj.ldr_gene_cov, obj.bases)
        obj.heri = rng.uniform(0.1, 0.5, size=N)
        obj.nbar = 30000.0
        obj.ld_rank = 1.5e6
        cls.obj = obj

        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = os.path.join(temp_dir, "test")
            with h5py.File(f"{out_dir}_gc.h5", "w") as file:
                gc = file.create_dataset("gc", shape=(N, N), dtype="float32")
                se = file.create_dataset("se", shape=(N, N), dtype="float32")
                for start in range(0, N, 100):
                    obj._get_gene_cor_se_block(start, gc, se, out_dir)
                cls.gc = gc[:]
                cls.se = se[:]

    def test_diagonal(self):
        assert_array_equal(np.diag(self.gc), np.ones(self.N, dtype=np.float32))
        assert_array_equal(np.diag(self.se), np.zeros(self.N, dtype=np.float32))

    def test_no_off_diagonal_overwritten(self):
        off_diag = ~np.eye(self.N, dtype=bool)
        overwritten = off_diag & (self.gc == 1) & (self.se == 0)
        self.assertEqual(np.sum(overwritten), 0)

    def test_gene_cor(self):
        obj = self.obj
        gene_cov = obj.bases @ obj.ldr_gene_cov @ obj.bases.T
        gene_cov[gene_cov == 0] = 0.01
        gene_cor = gene_cov / np.sqrt(np.outer(obj.gene_var, obj.gene_var))
        gene_cor = np.clip(gene_cor, -1, 1)
        np.fill_diagonal(gene_cor, 1)
        assert_array_almost_equal(self.gc, gene_cor, decimal=5)

    def test_symmetry(self):
        assert_array_almost_equal(self.gc, self.gc.T, decimal=5)
        assert_array_almost_equal(self.se, self.se.T, decimal=5)

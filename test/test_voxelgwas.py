import os
import logging
import unittest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

from heig.voxelgwas import (
    check_input,
)


MAIN_DIR = os.path.join(os.getcwd(), "test", "test_secondary")
log = logging.getLogger()
log.setLevel(logging.INFO)


class Args:
    def __init__(
        self,
        ldr_sumstats=None,
        bases=None,
        ldr_cov=None,
        extract=None,
        n_ldrs=None,
        voxels=None,
        chr_interval=None,
        sig_thresh=None,
    ):
        self.ldr_sumstats = ldr_sumstats
        self.bases = bases
        self.ldr_cov = ldr_cov
        self.extract = extract
        self.n_ldrs = n_ldrs
        self.voxels = voxels
        self.chr_interval = chr_interval
        self.sig_thresh = sig_thresh


class Test_check_input(unittest.TestCase):
    def test_good_cases(self):
        # voxels
        args = Args(
            ldr_sumstats=os.path.join(MAIN_DIR, "gwas"),
            bases=os.path.join(MAIN_DIR, "bases.npy"),
            ldr_cov=os.path.join(MAIN_DIR, "ldr_cov.npy"),
            voxels=1,
        )
        start_chr, start_pos, end_pos = check_input(args, log)
        self.assertEqual(start_chr, None)
        self.assertEqual(start_pos, None)
        self.assertEqual(end_pos, None)

        # chr_interval
        args = Args(
            ldr_sumstats=os.path.join(MAIN_DIR, "gwas"),
            bases=os.path.join(MAIN_DIR, "bases.npy"),
            ldr_cov=os.path.join(MAIN_DIR, "ldr_cov.npy"),
            chr_interval="3:1-3",
        )
        start_chr, start_pos, end_pos = check_input(args, log)
        self.assertEqual(start_chr, 3)
        self.assertEqual(start_pos, 1)
        self.assertEqual(end_pos, 3)

        # extract
        # args = Args(ldr_sumstats=os.path.join(MAIN_DIR, 'gwas'),
        #             bases=os.path.join(MAIN_DIR, 'bases.npy'),
        #             ldr_cov=os.path.join(MAIN_DIR, 'ldr_cov.npy'),
        #             extract=[os.path.join(MAIN_DIR, 'extract.txt')])
        # *_, keep_snps = check_input(args, log)
        # true_keep_snps = pd.DataFrame({'SNP': ['rs2']})
        # assert_frame_equal(keep_snps, true_keep_snps)

    def test_bad_cases(self):
        # missing required arguments
        def exclude_key(d, key_to_exclude):
            return {key: value for key, value in d.items() if key != key_to_exclude}

        args_dict = {
            "ldr_sumstats": os.path.join(MAIN_DIR, "gwas"),
            "bases": os.path.join(MAIN_DIR, "bases.npy"),
            "ldr_cov": os.path.join(MAIN_DIR, "ldr_cov.npy"),
        }
        for key in args_dict.keys():
            args_dict = exclude_key(args_dict, key)
            args = Args(**args_dict)
            with self.assertRaises(ValueError):
                check_input(args, log)

        # wrong specifications of chr_interval
        args = Args(
            ldr_sumstats=os.path.join(MAIN_DIR, "gwas"),
            bases=os.path.join(MAIN_DIR, "bases.npy"),
            ldr_cov=os.path.join(MAIN_DIR, "ldr_cov.npy"),
            chr_interval="3:1,4:3",
        )
        with self.assertRaises(ValueError):
            check_input(args, log)

        args = Args(
            ldr_sumstats=os.path.join(MAIN_DIR, "gwas"),
            bases=os.path.join(MAIN_DIR, "bases.npy"),
            ldr_cov=os.path.join(MAIN_DIR, "ldr_cov.npy"),
            chr_interval="3:5,3:3",
        )
        with self.assertRaises(ValueError):
            check_input(args, log)

        args = Args(
            ldr_sumstats=os.path.join(MAIN_DIR, "gwas"),
            bases=os.path.join(MAIN_DIR, "bases.npy"),
            ldr_cov=os.path.join(MAIN_DIR, "ldr_cov.npy"),
            chr_interval="3:5;3:3",
        )
        with self.assertRaises(ValueError):
            check_input(args, log)

        # invalid arguments
        # args = Args(ldr_sumstats=os.path.join(MAIN_DIR, 'gwas'),
        #             bases=os.path.join(MAIN_DIR, 'bases.npy'),
        #             ldr_cov=os.path.join(MAIN_DIR, 'ldr_cov.npy'),
        #             voxels=0)
        # with self.assertRaises(ValueError):
        #     check_input(args, log)

        # args = Args(
        #     ldr_sumstats=os.path.join(MAIN_DIR, "gwas"),
        #     bases=os.path.join(MAIN_DIR, "bases.npy"),
        #     ldr_cov=os.path.join(MAIN_DIR, "ldr_cov.npy"),
        #     sig_thresh=0,
        # )
        # with self.assertRaises(ValueError):
        #     check_input(args, log)

        # args = Args(ldr_sumstats=os.path.join(MAIN_DIR, 'gwas'),
        #             bases=os.path.join(MAIN_DIR, 'bases.npy'),
        #             ldr_cov=os.path.join(MAIN_DIR, 'ldr_cov.npy'),
        #             n_ldrs=0)
        # with self.assertRaises(ValueError):
        #     check_input(args, log)


# class Test_recover_se(unittest.TestCase):
#     def test_recover_se(self):
#         bases = np.load(os.path.join(MAIN_DIR, 'bases.npy'))[:, :3]
#         ldr_cov = np.load(os.path.join(MAIN_DIR, 'ldr_cov.npy'))[:3, :3]
#         sumstats = read_sumstats(os.path.join(MAIN_DIR, 'gwas'))
#         n = np.array(sumstats.snpinfo['N']).reshape(-1, 1)[:4, ]
#         ldr_beta = sumstats.beta[:4, ]
#         ztz_inv = np.ones(4).reshape(4, 1)

#         se = recover_se(bases[0], ldr_cov, n, ldr_beta, ztz_inv).reshape(-1)
#         voxel_beta = np.dot(ldr_beta, bases[0].T)
#         self.assertEqual(se.shape, voxel_beta.shape)

#         se = recover_se(bases, ldr_cov, n, ldr_beta, ztz_inv)
#         voxel_beta = np.dot(ldr_beta, bases.T)
#         self.assertEqual(se.shape, voxel_beta.shape)

import os
import h5py
import unittest
import logging
import pandas as pd
import numpy as np
from pandas.testing import assert_index_equal
from numpy.testing import assert_array_equal
from heig.image import get_image_list, NIFTIReader, ImageManager, merge_images

MAIN_DIR = os.getcwd()
log = logging.getLogger()
log.setLevel(logging.INFO)


class Test_get_image_list(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.folder = os.path.join(MAIN_DIR, "test", "test_ksm")

    def test_single_dir(self):
        true_ids = pd.MultiIndex.from_arrays(
            [("s1000", "s1002", "s1004"), ("s1000", "s1002", "s1004")],
            names=["FID", "IID"],
        )
        true_img_list = [
            "s1000_example_image.nii.gz",
            "s1002_example_image.nii.gz",
            "s1004_example_image.nii.gz",
        ]
        true_img_list = [
            os.path.join(self.folder, "image_dir1", x) for x in true_img_list
        ]

        ids, img_list = get_image_list(
            [os.path.join(self.folder, "image_dir1")], ["_example_image.nii.gz"], log
        )
        assert_index_equal(true_ids, ids)
        self.assertEqual(true_img_list, img_list)

        # test duplicated subjects
        ids, img_list = get_image_list(
            [
                os.path.join(self.folder, "image_dir1"),
                os.path.join(self.folder, "image_dir3"),
            ],
            ["_example_image.nii.gz", "_example_image.nii.gz"],
            log,
        )
        assert_index_equal(true_ids, ids)
        self.assertEqual(true_img_list, img_list)

    def test_multiple_dirs(self):
        true_ids = pd.MultiIndex.from_arrays(
            [
                ("s1000", "s1001", "s1002", "s1003", "s1004", "s1005"),
                ("s1000", "s1001", "s1002", "s1003", "s1004", "s1005"),
            ],
            names=["FID", "IID"],
        )
        true_img_list = [
            "image_dir1/s1000_example_image.nii.gz",
            "image_dir2/s1001_example_image.nii.gz",
            "image_dir1/s1002_example_image.nii.gz",
            "image_dir2/s1003_example_image.nii.gz",
            "image_dir1/s1004_example_image.nii.gz",
            "image_dir2/s1005_example_image.nii.gz",
        ]
        true_img_list = [os.path.join(self.folder, x) for x in true_img_list]

        ids, img_list = get_image_list(
            [
                os.path.join(self.folder, "image_dir1"),
                os.path.join(self.folder, "image_dir2"),
            ],
            ["_example_image.nii.gz", "_example_image.nii.gz"],
            log,
        )
        assert_index_equal(true_ids, ids)
        self.assertEqual(true_img_list, img_list)

    def test_keep_and_remove(self):
        true_ids = pd.MultiIndex.from_arrays(
            [("s1000", "s1001"), ("s1000", "s1001")], names=["FID", "IID"]
        )
        true_img_list = [
            "image_dir1/s1000_example_image.nii.gz",
            "image_dir2/s1001_example_image.nii.gz",
        ]
        true_img_list = [os.path.join(self.folder, x) for x in true_img_list]

        keep_idvs = pd.MultiIndex.from_arrays(
            [("s1000", "s1001"), ("s1000", "s1001")], names=["FID", "IID"]
        )
        ids, img_list = get_image_list(
            [
                os.path.join(self.folder, "image_dir1"),
                os.path.join(self.folder, "image_dir2"),
            ],
            ["_example_image.nii.gz", "_example_image.nii.gz"],
            log,
            keep_idvs,
        )
        assert_index_equal(true_ids, ids)
        self.assertEqual(true_img_list, img_list)

        remove_idvs = pd.MultiIndex.from_arrays(
            [
                ("s1002", "s1003", "s1004", "s1005"),
                ("s1002", "s1003", "s1004", "s1005"),
            ],
            names=["FID", "IID"],
        )
        ids, img_list = get_image_list(
            [
                os.path.join(self.folder, "image_dir1"),
                os.path.join(self.folder, "image_dir2"),
            ],
            ["_example_image.nii.gz", "_example_image.nii.gz"],
            log,
            remove_idvs=remove_idvs,
        )
        assert_index_equal(true_ids, ids)
        self.assertEqual(true_img_list, img_list)

        keep_idvs = pd.MultiIndex.from_arrays(
            [("s1000", "s1001", "s1002"), ("s1000", "s1001", "s1002")],
            names=["FID", "IID"],
        )
        ids, img_list = get_image_list(
            [
                os.path.join(self.folder, "image_dir1"),
                os.path.join(self.folder, "image_dir2"),
            ],
            ["_example_image.nii.gz", "_example_image.nii.gz"],
            log,
            keep_idvs=keep_idvs,
            remove_idvs=remove_idvs,
        )
        assert_index_equal(true_ids, ids)
        self.assertEqual(true_img_list, img_list)

    def test_one_image(self):
        true_ids = pd.MultiIndex.from_arrays(
            [("s1000",), ("s1000",)], names=["FID", "IID"]
        )
        true_img_list = [
            "image_dir3/s1000_example_image.nii.gz",
        ]
        true_img_list = [os.path.join(self.folder, x) for x in true_img_list]

        ids, img_list = get_image_list(
            [
                os.path.join(self.folder, "image_dir3"),
            ],
            [
                "_example_image.nii.gz",
            ],
            log,
        )
        assert_index_equal(true_ids, ids)
        self.assertEqual(true_img_list, img_list)


class Test_load_nifti(unittest.TestCase):
    def setUp(self):
        self.true_images = np.array(
            [[1, 1000], [1, 1001], [1, 1002], [1, 1003], [1, 1004], [1, 1005]],
            dtype=np.float32,
        ).reshape(6, 2)
        self.true_coord = np.array([[5, 5, 5], [5, 5, 6]])
        true_ids = [f"s{1000 + i}" for i in range(6)]
        self.true_ids = np.array([[val, val] for val in true_ids], dtype="|S10")

        self.folder = os.path.join(MAIN_DIR, "test", "test_ksm")

    def test_multiple_dir(self):
        ids, img_files = get_image_list(
            [
                os.path.join(self.folder, "image_dir1"),
                os.path.join(self.folder, "image_dir2"),
                os.path.join(self.folder, "image_dir3"),
            ],
            ["_example_image.nii.gz", "_example_image.nii.gz", "_example_image.nii.gz"],
            log,
        )
        img_reader = NIFTIReader(
            img_files, ids, None, os.path.join(self.folder, "all_images.h5")
        )
        img_reader.create_dataset(img_files[0])
        img_reader.read_save_image(1)

        with h5py.File(os.path.join(self.folder, "all_images.h5"), "r") as file:
            images = file["images"][:]
            coord = file["coord"][:]
            ids = file["id"][:]

        assert_array_equal(self.true_images, images)
        assert_array_equal(self.true_coord, coord)
        assert_array_equal(self.true_ids, ids)

    def test_single(self):
        for i in range(1, 4):
            ids, img_files = get_image_list(
                [os.path.join(self.folder, f"image_dir{i}")],
                ["_example_image.nii.gz"],
                log,
            )
            img_reader = NIFTIReader(
                img_files, ids, None, os.path.join(self.folder, f"dir{i}_images.h5")
            )
            img_reader.create_dataset(img_files[0])
            img_reader.read_save_image(1)

            with h5py.File(os.path.join(self.folder, f"dir{i}_images.h5"), "r") as file:
                images = file["images"][:]
                coord = file["coord"][:]
                ids = file["id"][:]

            if i == 1:
                assert_array_equal(self.true_images[[0, 2, 4]], images)
                assert_array_equal(self.true_coord, coord)
                assert_array_equal(self.true_ids[[0, 2, 4]], ids)
            elif i == 2:
                assert_array_equal(self.true_images[[1, 3, 5]], images)
                assert_array_equal(self.true_coord, coord)
                assert_array_equal(self.true_ids[[1, 3, 5]], ids)
            else:
                assert_array_equal(self.true_images[[0]], images)
                assert_array_equal(self.true_coord, coord)
                assert_array_equal(self.true_ids[[0]], ids)


class Test_image_manager(unittest.TestCase):
    def setUp(self):
        self.true_images = np.array(
            [[1, 1000], [1, 1001], [1, 1002], [1, 1003], [1, 1004], [1, 1005]],
            dtype=np.float32,
        ).reshape(6, 2)
        self.true_coord = np.array([[5, 5, 5], [5, 5, 6]])
        true_ids = [f"s{1000 + i}" for i in range(6)]
        self.true_ids = np.array([[val, val] for val in true_ids], dtype="|S10")

        self.folder = os.path.join(MAIN_DIR, "test", "test_ksm")

    def test_keep(self):
        image_manager = ImageManager(os.path.join(self.folder, "dir1_images.h5"), None)
        to_keep_id = pd.MultiIndex.from_tuples(
            [["s1000", "s1000"]], names=["FID", "IID"]
        )
        image_manager.keep_and_remove(keep_idvs=to_keep_id, remove_idvs=None)
        image_manager.save(os.path.join(self.folder, "dir1_keep_images.h5"))

        with h5py.File(os.path.join(self.folder, "dir1_keep_images.h5"), "r") as file:
            images = file["images"][:]
            coord = file["coord"][:]
            ids = file["id"][:]

        assert_array_equal(self.true_images[[0]], images)
        assert_array_equal(self.true_coord, coord)
        assert_array_equal(self.true_ids[[0]], ids)

    def test_remove(self):
        image_manager = ImageManager(os.path.join(self.folder, "dir1_images.h5"), None)
        to_remove_id = pd.MultiIndex.from_tuples(
            [["s1000", "s1000"]], names=["FID", "IID"]
        )
        image_manager.keep_and_remove(keep_idvs=None, remove_idvs=to_remove_id)
        image_manager.save(os.path.join(self.folder, "dir1_remove_images.h5"))

        with h5py.File(os.path.join(self.folder, "dir1_remove_images.h5"), "r") as file:
            images = file["images"][:]
            coord = file["coord"][:]
            ids = file["id"][:]

        assert_array_equal(self.true_images[[2, 4]], images)
        assert_array_equal(self.true_coord, coord)
        assert_array_equal(self.true_ids[[2, 4]], ids)

    def test_remove_nonexist(self):
        image_manager = ImageManager(os.path.join(self.folder, "dir1_images.h5"), None)
        to_remove_id = pd.MultiIndex.from_tuples(
            [["s1000", "s1000"], ["s1010", "s1010"]], names=["FID", "IID"]
        )
        image_manager.keep_and_remove(keep_idvs=None, remove_idvs=to_remove_id)
        image_manager.save(os.path.join(self.folder, "dir1_remove_images.h5"))

        with h5py.File(os.path.join(self.folder, "dir1_remove_images.h5"), "r") as file:
            images = file["images"][:]
            coord = file["coord"][:]
            ids = file["id"][:]

        assert_array_equal(self.true_images[[2, 4]], images)
        assert_array_equal(self.true_coord, coord)
        assert_array_equal(self.true_ids[[2, 4]], ids)

    def test_doing_nothing(self):
        image_manager = ImageManager(os.path.join(self.folder, "dir3_images.h5"), None)
        image_manager.keep_and_remove()
        image_manager.save(os.path.join(self.folder, "dir3_doing_nothing_images.h5"))

        with h5py.File(
            os.path.join(self.folder, "dir3_doing_nothing_images.h5"), "r"
        ) as file:
            images = file["images"][:]
            coord = file["coord"][:]
            ids = file["id"][:]

        assert_array_equal(self.true_images[[0]], images)
        assert_array_equal(self.true_coord, coord)
        assert_array_equal(self.true_ids[[0]], ids)


class Test_merge_images(unittest.TestCase):
    def setUp(self):
        self.true_images = np.array(
            [[1, 1000], [1, 1001], [1, 1002], [1, 1003], [1, 1004], [1, 1005]],
            dtype=np.float32,
        ).reshape(6, 2)
        self.true_coord = np.array([[5, 5, 5], [5, 5, 6]])
        true_ids = [f"s{1000 + i}" for i in range(6)]
        self.true_ids = np.array([[val, val] for val in true_ids], dtype="|S10")

        self.folder = os.path.join(MAIN_DIR, "test", "test_ksm")

    def test_keep(self):
        image_files = [
            os.path.join(self.folder, "dir1_images.h5"),
            os.path.join(self.folder, "dir2_images.h5"),
        ]
        to_keep_ids = pd.MultiIndex.from_tuples(
            [["s1000", "s1000"], ["s1001", "s1001"], ["s1004", "s1004"]],
            names=["FID", "IID"],
        )
        merge_images(
            image_files,
            None,
            os.path.join(self.folder, "dir12_keep_images.h5"),
            log,
            to_keep_ids,
        )

        with h5py.File(os.path.join(self.folder, "dir12_keep_images.h5"), "r") as file:
            images = file["images"][:]
            coord = file["coord"][:]
            ids = file["id"][:]

        assert_array_equal(self.true_images[[0, 4, 1]], images)
        assert_array_equal(self.true_coord, coord)
        assert_array_equal(self.true_ids[[0, 4, 1]], ids)

    def test_merge(self):
        # w/o duplicated ID
        image_files = [
            os.path.join(self.folder, "dir1_images.h5"),
            os.path.join(self.folder, "dir2_images.h5"),
        ]
        merge_images(image_files, None, os.path.join(self.folder, "dir12_images.h5"), log)

        with h5py.File(os.path.join(self.folder, "dir12_images.h5"), "r") as file:
            images = file["images"][:]
            coord = file["coord"][:]
            ids = file["id"][:]

        assert_array_equal(self.true_images[[0, 2, 4, 1, 3, 5]], images)
        assert_array_equal(self.true_coord, coord)
        assert_array_equal(self.true_ids[[0, 2, 4, 1, 3, 5]], ids)

        # w/ duplicated ID
        image_files = [
            os.path.join(self.folder, "dir1_images.h5"),
            os.path.join(self.folder, "dir3_images.h5"),
        ]
        merge_images(image_files, None, os.path.join(self.folder, "dir13_images.h5"), log)

        with h5py.File(os.path.join(self.folder, "dir13_images.h5"), "r") as file:
            images = file["images"][:]
            coord = file["coord"][:]
            ids = file["id"][:]

        assert_array_equal(self.true_images[[0, 2, 4]], images)
        assert_array_equal(self.true_coord, coord)
        assert_array_equal(self.true_ids[[0, 2, 4]], ids)

    def test_keep_and_remove(self):
        image_files = [
            os.path.join(self.folder, "dir1_images.h5"),
            os.path.join(self.folder, "dir2_images.h5"),
        ]
        to_keep_ids = pd.MultiIndex.from_tuples(
            [["s1000", "s1000"], ["s1001", "s1001"], ["s1004", "s1004"]],
            names=["FID", "IID"],
        )
        to_remove_id = pd.MultiIndex.from_tuples(
            [["s1000", "s1000"]], names=["FID", "IID"]
        )
        merge_images(
            image_files,
            None,
            os.path.join(self.folder, "dir12_keep_remove_images.h5"),
            log,
            to_keep_ids,
            to_remove_id,
        )

        with h5py.File(
            os.path.join(self.folder, "dir12_keep_remove_images.h5"), "r"
        ) as file:
            images = file["images"][:]
            coord = file["coord"][:]
            ids = file["id"][:]

        assert_array_equal(self.true_images[[4, 1]], images)
        assert_array_equal(self.true_coord, coord)
        assert_array_equal(self.true_ids[[4, 1]], ids)

    def test_keep_nonexist(self):
        image_files = [
            os.path.join(self.folder, "dir1_images.h5"),
            os.path.join(self.folder, "dir2_images.h5"),
        ]
        to_keep_ids = pd.MultiIndex.from_tuples(
            [["s1010", "s1010"]], names=["FID", "IID"]
        )

        with self.assertRaises(ValueError):
            merge_images(
                image_files,
                None,
                os.path.join(self.folder, "dir12_keep_nonexist_images.h5"),
                log,
                to_keep_ids,
            )

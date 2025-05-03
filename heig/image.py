import os
import logging
import h5py
import concurrent.futures
from abc import ABC, abstractmethod
from filelock import FileLock
from tqdm import tqdm
import numpy as np
import pandas as pd
import nibabel as nib
import heig.input.dataset as ds


class ImageReader(ABC):
    """
    An abstract class for reading images

    """

    def __init__(self, img_files, ids, voxels, out_dir):
        self.img_files = img_files
        self.n_images = len(self.img_files)
        self.ids = ids
        self.voxels = voxels
        self.out_dir = out_dir
        self.logger = logging.getLogger(__name__)

    def create_dataset(self, coord_img_file):
        """
        Creating a HDF5 file saving images, coordinates, and ids

        """
        self.coord = self._get_coord(coord_img_file)
        if self.voxels is not None:
            self.coord = self.coord[self.voxels]
        self.n_voxels = self.coord.shape[0]

        with h5py.File(self.out_dir, "w") as h5f:
            images = h5f.create_dataset(
                "images", shape=(self.n_images, self.n_voxels), dtype="float32"
            )
            h5f.create_dataset("id", data=np.array(self.ids.tolist(), dtype="S10"))
            h5f.create_dataset("coord", data=self.coord)
        self.logger.info(
            (
                f"{self.n_images} subjects and {self.n_voxels} voxels (vertices) "
                "in the imaging data."
            )
        )

    def read_save_image(self, threads):
        """
        Reading and saving images in parallel

        """
        self.logger.info("Reading images ...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [
                executor.submit(self._read_save_image, idx, img_file)
                for idx, img_file in enumerate(self.img_files)
            ]
            for _ in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc=f"{len(futures)} images",
            ):
                pass

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    executor.shutdown(wait=False)
                    if os.path.exists(f"{self.out_dir}.lock"):
                        os.remove(f"{self.out_dir}.lock")
                    if os.path.exists(self.out_dir):
                        os.remove(self.out_dir)
                    raise RuntimeError(f"Computation terminated due to error: {exc}")

        self.logger.info("Done.")
        if os.path.exists(f"{self.out_dir}.lock"):
            os.remove(f"{self.out_dir}.lock")

    def _read_save_image(self, idx, img_file):
        """
        Reading and writing a single image

        """
        image = self._read_image(img_file)
        if len(image) != self.n_voxels:
            raise ValueError(
                f"{img_file} is of resolution {len(image)} but the coordinate is of resolution {self.n_voxels}"
            )
        lock_file = f"{self.out_dir}.lock"
        with FileLock(lock_file):
            with h5py.File(self.out_dir, "r+") as h5f:
                h5f["images"][idx] = image

    @abstractmethod
    def _get_coord(self, coord_img_file):
        pass

    @abstractmethod
    def _read_image(self, img_file):
        pass


class NIFTIReader(ImageReader):
    """
    Reading NIFTI images and coordinates.

    """

    def _get_coord(self, coord_img_file):
        img = nib.load(coord_img_file)
        data = img.get_fdata()
        coord = np.stack(np.nonzero(data)).T
        return coord

    def _read_image(self, img_file):
        try:
            img = nib.load(img_file)
            data = img.get_fdata()
            data = data[tuple(self.coord.T)]
            if np.std(data) == 0:
                raise ValueError(f"{img_file} is an invalid image with variance 0")
            return data
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise RuntimeError(
                f"cannot read {img_file}, did you provide a wrong NIFTI image?"
            )


class CIFTIReader(ImageReader):
    """
    Reading CIFTI images and coordinates.

    """

    def _get_coord(self, coord_img_file):
        """
        Reading coordinates from a GIFTI image.

        """
        coord = nib.load(coord_img_file).darrays[0].data
        return coord

    def _read_image(self, img_file):
        try:
            img = nib.load(img_file)
            data = img.get_fdata()[0]
            if self.voxels is not None:
                data = data[self.voxels]
            if np.std(data) == 0:
                raise ValueError(f"{img_file} is an invalid image with variance 0")
            return data
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise RuntimeError(
                f"cannot read {img_file}, did you provide a wrong CIFTI image?"
            )


class FreeSurferReader(ImageReader):
    """
    Reading FreeSurfer outputs and coordinates.

    """

    def _get_coord(self, coord_img_file):
        """
        Reading coordinates from a Freesurfer surface mesh file

        """
        coord = nib.freesurfer.read_geometry(coord_img_file)[0]
        return coord

    def _read_image(self, img_file):
        try:
            data = nib.freesurfer.read_morph_data(img_file)
            if self.voxels is not None:
                data = data[self.voxels]
            if np.std(data) == 0:
                raise ValueError(f"{img_file} is an invalid image with variance 0")
            return data
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise RuntimeError(
                (
                    f"cannot read {img_file}, did you provide a wrong "
                    "FreeSurfer morphometry data file?"
                )
            )


def get_image_list(img_dirs, suffixes, log, keep_idvs=None, remove_idvs=None):
    """
    Getting file path of images from multiple directories.

    Parameters:
    ------------
    img_dirs: a list of directories
    suffixes: a list of suffixes of images
    log: a logger
    keep_idvs: a pd.MultiIndex instance of IDs (FID, IID)
    remove_idvs: a pd.MultiIndex instance of IDs (FID, IID)

    Returns:
    ---------
    ids: a pd.MultiIndex instance of IDs
    img_files_list: a list of image files to read

    """
    img_files = {}
    n_dup = 0

    for img_dir, suffix in zip(img_dirs, suffixes):
        for img_file in os.listdir(img_dir):
            img_id = img_file.replace(suffix, "")
            if img_file.endswith(suffix) and (
                (keep_idvs is not None and img_id in keep_idvs) or (keep_idvs is None)
            ):
                if (remove_idvs is not None and img_id not in remove_idvs) or (
                    remove_idvs is None
                ):
                    if img_id in img_files:
                        n_dup += 1
                    else:
                        img_files[img_id] = os.path.join(img_dir, img_file)
    img_files = dict(sorted(img_files.items()))
    ids = pd.MultiIndex.from_arrays(
        [img_files.keys(), img_files.keys()], names=["FID", "IID"]
    )
    img_files_list = list(img_files.values())
    if n_dup > 0:
        log.info(f"WARNING: {n_dup} duplicated subject(s). Keeping the first one.")

    return ids, img_files_list


def save_images(out_dir, images, coord, id):
    """
    Save imaging data to a HDF5 file.

    Parameters:
    ------------
    out_dir: directory of output
    images (n, N): a np.array of imaging data
    coord (N, dim): a np.array of coordinate
    id: a pd.MultiIndex instance of IDs (FID, IID)

    """
    with h5py.File(out_dir, "w") as file:
        dset = file.create_dataset("images", data=images, dtype="float32")
        file.create_dataset("id", data=np.array(id.tolist(), dtype="S10"))
        file.create_dataset("coord", data=coord)


class ImageManager:
    """
    Image management class
    which can keep, remove, and read in batch for a single image HDF5 file

    """

    def __init__(self, image_file, voxels=None):
        """
        Parameters:
        ------------
        image_file: a image HDF5 file path
        voxels: a np.array of voxel indices to keep (0 based)

        """
        self.file = h5py.File(image_file, "r")
        self.images = self.file["images"]
        self.coord = self.file["coord"][:]
        ids = self.file["id"][:]
        self.ids = pd.MultiIndex.from_arrays(ids.astype(str).T, names=["FID", "IID"])
        self.n_sub, self.n_voxels = self.images.shape
        self.dim = self.coord.shape[1]
        self.id_idxs = np.arange(len(self.ids))
        self.extracted_ids = self.ids
        self.logger = logging.getLogger(__name__)
        
        if voxels is not None:
            self.voxels = voxels
            self.n_voxels = len(self.voxels)
            self.coord = self.coord[self.voxels]
        else:
            self.voxels = np.arange(self.n_voxels)

        self.logger.info(
            f"{self.n_sub} subjects and {self.n_voxels} voxels (vertices) in {image_file}"
        )

    def keep_and_remove(self, keep_idvs=None, remove_idvs=None, check_empty=True):
        """
        Keeping and removing subjects

        Parameters:
        ------------
        keep_idvs: subject indices in pd.MultiIndex to keep
        remove_idvs: subject indices in pd.MultiIndex to remove
        check_empty: if check the current image set is empty

        """
        if keep_idvs is not None:
            self.extracted_ids = ds.get_common_idxs(self.extracted_ids, keep_idvs)
        if remove_idvs is not None:
            self.extracted_ids = ds.remove_idxs(self.extracted_ids, remove_idvs)
        if check_empty and len(self.extracted_ids) == 0:
            raise ValueError("no subject remaining after --keep and/or --remove")

        self.n_sub = len(self.extracted_ids)
        self.id_idxs = np.arange(len(self.ids))[self.ids.isin(self.extracted_ids)]

    def image_reader(self, batch_size=None):
        """
        Reading imaging data in chunks as a generator

        Parameters:
        ------------
        batch_size: an int of batch size

        """
        if batch_size is None:
            memory_use = (
                self.n_sub * self.n_voxels * np.dtype(np.float32).itemsize / (1024**3)
            )
            if memory_use <= 5:
                batch_size = self.n_sub
            else:
                batch_size = int(self.n_sub / memory_use * 5)

        for i in range(0, self.n_sub, batch_size):
            id_idx_chuck = self.id_idxs[i : i + batch_size]
            yield self.images[id_idx_chuck][:, self.voxels], self.ids[id_idx_chuck]

    def save(self, out_dir):
        """
        Saving the processed images to a new HDF5 file

        Parameters:
        ------------
        out_dir: directory of output

        """
        self.logger.info(f"{self.n_sub} subjects in the output image file.")
        try:
            with h5py.File(out_dir, "w") as output:
                dset = output.create_dataset(
                    "images", shape=(self.n_sub, self.n_voxels), dtype="float32"
                )
                output.create_dataset(
                    "id", data=self.extracted_ids.tolist(), dtype="S10"
                )
                output.create_dataset("coord", data=self.coord)

                start, end = 0, 0
                for images_, _ in self.image_reader():
                    start = end
                    end += images_.shape[0]
                    dset[start:end] = images_

        except Exception:
            if os.path.exists(out_dir):
                os.remove(out_dir)
            raise

    def close(self):
        self.file.close()


def merge_images(image_files, voxels, out_dir, log, keep_idvs=None, remove_idvs=None):
    """
    Merging multiple image files

    Parameters:
    ------------
    image_files: a list of image files
    voxels: a np.array of voxel indices to keep (0 based)
    out_dir: a directory of output
    log: a logger
    keep_idvs: a pd.MultiIndex instance of IDs (FID, IID)
    remove_idvs: a pd.MultiIndex instance of IDs (FID, IID)

    """
    try:
        image_managers = list()
        for image_file in image_files:
            image_managers.append(ImageManager(image_file, voxels))
            if (
                len(image_managers) > 1
                and not np.equal(
                    image_managers[-1].coord, image_managers[-2].coord
                ).all()
            ):
                raise ValueError(
                    f"{image_file} has different coordinates than other files"
                )

        all_ids = ds.get_union_idxs(
            *[image_manager.ids for image_manager in image_managers]
        )
        n_unique_sub = len(all_ids)
        n_all_sub = sum((image_manager.n_sub for image_manager in image_managers))
        log.info(
            (
                f"{n_unique_sub} unique subjects in these image files. "
                f"{n_all_sub - n_unique_sub} duplicated subject(s). Keeping the first one."
            )
        )
        all_ids = ds.get_common_idxs(all_ids, keep_idvs)
        all_ids = ds.remove_idxs(all_ids, remove_idvs)

        for image_manager in image_managers:
            image_manager.keep_and_remove(
                keep_idvs=all_ids, remove_idvs=None, check_empty=False
            )

        with h5py.File(out_dir, "w") as output:
            dset = output.create_dataset(
                "images",
                shape=(len(all_ids), image_managers[0].n_voxels),
                dtype="float32",
            )
            output.create_dataset("coord", data=image_managers[0].coord)

            ids_read = None
            start, end = 0, 0
            for image_manager in image_managers:
                if len(image_manager.id_idxs) > 0:
                    for images_, image_ids_ in image_manager.image_reader():
                        if ids_read is not None:
                            images_ = images_[~(image_ids_.isin(ids_read))]
                            ids_read = ids_read.union(image_ids_, sort=False)
                        else:
                            ids_read = image_ids_
                        start = end
                        end += images_.shape[0]
                        dset[start:end] = images_

            output.create_dataset("id", data=np.array(ids_read.tolist(), dtype="S10"))
    except:
        if os.path.exists(out_dir):
            os.remove(out_dir)
        raise
    finally:
        if "image_managers" in locals():
            for image_manager in image_managers:
                image_manager.close()


def check_input(args):
    if (
        args.image_dir is None
        and args.image_suffix is None
        and args.image_txt is None
        and args.coord_txt is None
        and args.image_list is None
        and args.image is None
    ):
        raise ValueError(
            (
                "--image-txt + --coord-txt or --image-dir + --image-suffix or "
                "--image-list or --image is required"
            )
        )

    if args.image_txt is not None and args.coord_txt is None:
        raise ValueError("--coord-txt is required")
    elif args.image_txt is None and args.coord_txt is not None:
        raise ValueError("--image-txt is required")
    elif args.image_dir is not None and args.image_suffix is None:
        raise ValueError("--image-suffix is required")
    elif args.image_dir is None and args.image_suffix is not None:
        raise ValueError("--image-dir is required")

    ds.check_existence(args.image_txt)
    ds.check_existence(args.coord_txt)
    ds.check_existence(args.image)

    ## process arguments
    if args.image_dir is not None and args.image_suffix is not None:
        if args.coord_dir is None:
            raise ValueError("--coord-dir is required")
        args.image_dir = args.image_dir.split(",")
        args.image_suffix = args.image_suffix.split(",")
        if len(args.image_dir) != len(args.image_suffix):
            raise ValueError("--image-dir and --image-suffix do not match")
        for image_dir in args.image_dir:
            ds.check_existence(image_dir)
        ds.check_existence(args.coord_dir)
    elif args.image_list is not None:
        args.image_list = args.image_list.split(",")
        for image_file in args.image_list:
            ds.check_existence(image_file)


def run(args, log):
    # check input
    check_input(args)

    # read images
    out_dir = f"{args.out}_images.h5"
    if args.image_txt is not None:
        images = ds.Dataset(args.image_txt, all_num_cols=True)
        log.info(
            (
                f"{images.data.shape[0]} subjects and {images.data.shape[1]} "
                f"voxels (vertices) read from {args.image_txt}"
            )
        )

        images.keep_and_remove(args.keep, args.remove, merge=True)
        ids = images.get_ids()
        images = np.array(images.data, dtype=np.float32)
        if args.voxels is not None:
            images = images[:, args.voxels]
        log.info(f"Keeping {images.shape[0]} subjects and {images.shape[1]} voxels.")

        coord = pd.read_csv(args.coord_txt, sep="\s+", header=None)
        log.info(f"Read coordinates from {args.coord_txt}")
        if args.voxels is not None:
            coord = coord.iloc[args.voxels]
        if coord.isnull().sum().sum() > 0:
            raise ValueError("no missing values allowed in coordinates")
        if coord.shape[0] != images.shape[1]:
            raise ValueError("images and coordinates have different resolution")
        save_images(out_dir, images, coord, ids)

    elif args.image is not None:
        try:
            log.info(f"Processing {args.image}")
            image_manager = ImageManager(args.image, args.voxels)
            image_manager.keep_and_remove(args.keep, args.remove)
            image_manager.save(out_dir)
        finally:
            if "image_manager" in locals():
                image_manager.close()

    elif args.image_list is not None:
        log.info(f"Merging image files {args.image_list}")
        merge_images(args.image_list, args.voxels, out_dir, log, args.keep, args.remove)

    else:
        ids, img_files = get_image_list(
            args.image_dir, args.image_suffix, log, args.keep, args.remove
        )
        if len(img_files) == 0:
            raise ValueError(
                f"no image in {args.image_dir} with suffix {args.image_suffix}"
            )
        if args.coord_dir.endswith("nii.gz") or args.coord_dir.endswith("nii"):
            log.info("Reading NIFTI images.")
            img_reader = NIFTIReader(img_files, ids, args.voxels, out_dir)
        elif args.coord_dir.endswith("gii.gz") or args.coord_dir.endswith("gii"):
            log.info("Reading CIFTI images.")
            img_reader = CIFTIReader(img_files, ids, args.voxels, out_dir)
        else:
            log.info("Reading FreeSurfer morphometry data.")
            img_reader = FreeSurferReader(img_files, ids, args.voxels, out_dir)
        img_reader.create_dataset(args.coord_dir)
        img_reader.read_save_image(args.threads)

    log.info(f"\nSaved the images to {out_dir}")

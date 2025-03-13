import h5py
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import label
from concurrent.futures import ThreadPoolExecutor
import heig.input.dataset as ds
from heig.wgs.utils import list_datasets
from heig.utils import find_loc


"""
Threshold-free cluster enhancement (TFCE) analysis for 
significant/null pvalues

1. summarize significant results
2. summarize null results

TODO:
1. support more image format

"""

class TFCE:
    def __init__(self, coord, roi_mask, slices, h=2, E=0.5, dh=0.1):
        self.coord = coord
        self.roi_mask = roi_mask
        self.slices = slices
        self.h = h
        self.E = E
        self.dh = dh

    def _tfce(self, stat_map):
        """
        Compute Threshold-Free Cluster Enhancement (TFCE) on a statistical map.

        Parameters:
        ------------
        stat_map (np.ndarray): Input statistical map (2D or 3D).
        h (float): Height exponent.
        E (float): Extent exponent.
        dh (float): Step size for integration.

        Returns:
        ------------
        tfce_map (np.ndarray): Enhanced TFCE statistical map.
        
        """
        tfce_map = np.zeros_like(stat_map)
        tfce_map[stat_map == 0.001] = 0.001
        non_zeros = stat_map[stat_map > 0.001]
        thresholds = np.arange(np.min(non_zeros), np.max(non_zeros), self.dh)

        for threshold in thresholds:
            # Binary mask of voxels exceeding the current threshold
            binarized_map = stat_map >= threshold

            # Label connected components
            labeled_clusters, num_clusters = label(binarized_map)

            for cluster_id in range(1, num_clusters + 1):
                cluster_mask = labeled_clusters == cluster_id
                cluster_size = np.sum(cluster_mask)
                increment = (cluster_size ** self.E) * (threshold ** self.h) * self.dh
                tfce_map[cluster_mask] += increment

        return tfce_map
    
    def tfce(self, index, results):
        """
        Generating a cropped TFCE map

        Parameters:
        ------------
        index: a np.array of zero-based idxs
        results: a np.array of -log10 pvalues
        
        """
        all_res = np.ones(len(self.coord[0])) * 0.001
        all_res[index] = results
        stat_map = np.zeros(self.roi_mask.shape)
        tfce_map = np.zeros(self.roi_mask.shape)
        stat_map[self.roi_mask] = all_res

        stat_map_crop = stat_map[self.slices]
        tfce_map_crop = self._tfce(stat_map_crop)
        tfce_map[self.slices] = tfce_map_crop

        return tfce_map, stat_map


class TFCEnull:
    def __init__(self, tfce_null_file):
        self.bins = [(2,2), (3,3), (4,4), (5,5), (6,7), (8,9), 
                     (10,11), (12,14), (15,20), (21,30), (31,60), 
                     (61,100), (101,500), (501,)]
        self.sig_stats = dict()
        self.count = dict()
        self.min_quantile = dict() 
        self.breaks = list() 

        h5file = h5py.File(f"{tfce_null_file}", "r")
        all_bins = list_datasets(h5file)
        for bin_str in all_bins:
            bin = tuple([int(x) for x in bin_str.split("_")])
            data = h5file[bin_str]
            count = data.attrs["count"]
            self.sig_stats[bin] = data[:]
            self.count[bin] = count
            self.min_quantile[bin] = 1 - len(self.sig_stats[bin]) / count
            self.breaks.append(bin[0])
        self.breaks.sort()
        h5file.close()

    def quantile(self, cmac, cluster_thresh):
        bin_idx = find_loc(self.breaks, cmac)
        bin = self.bins[bin_idx]
        if cmac < bin[0] or (len(bin) > 1 and cmac > bin[1]):
            raise ValueError(f"CMAC {cmac} not included in the null distribution")
        if cluster_thresh <= self.min_quantile[bin]:
            return 0
        else:
            idx = int(self.count[bin] * (cluster_thresh - self.min_quantile[bin]))
            return self.sig_stats[bin][idx]
        

def crop_image_with_margin(image, margin=1):
    """
    Crops a 2D or 3D image to the smallest bounding box containing nonzero values,
    while keeping a margin of at least 'margin' voxels.

    Parameters:
    ------------
    image (np.ndarray): Input 2D or 3D image.
    margin (int): Number of voxels to leave as a margin.

    Returns:
    ------------
    slices (np.ndarray): voxel idxs of cropped image.

    """
    # Find nonzero voxel coordinates
    nonzero_coords = np.where(image != 0)
    
    # Get bounding box (min and max indices along each axis)
    min_indices = [max(np.min(axis) - margin, 0) for axis in nonzero_coords]
    max_indices = [
        min(np.max(axis) + margin + 1, image.shape[i]) 
        for i, axis in enumerate(nonzero_coords)
    ]
    
    # Crop image
    slices = tuple(
        slice(min_idx, max_idx) for min_idx, max_idx in zip(min_indices, max_indices)
    )
    # cropped_image = image[slices]

    return slices


def nifti_coord_mask(coord_img_file):
    img = nib.load(coord_img_file)
    data = img.get_fdata()
    roi_mask = data > 0
    coord = np.stack(np.nonzero(data)).T
    coord = tuple(zip(*coord))
    slices = crop_image_with_margin(data)

    return coord, roi_mask, slices


def summarize_results(
        tfce, 
        results_idx, 
        tfce_null, 
        variant_category,
        sig_thresh, 
        tfce_quantile_level, 
        sig_thresh2
    ):
    """
    Computing TFCE for significant associations
    
    """
    gene = list()
    chr = list()
    start = list()
    end = list()
    n_variants = list()
    cmac = list()
    most_sig_pv = list()
    n_clusters = list()
    cluster_info = list()
    max_tfce = list()
    sig_thresh2 = -np.log10(sig_thresh2) 

    for _, result_info in results_idx.iterrows():
        results = pd.read_csv(result_info['RESULT_FILE'], sep='\t')
        test = "STAAR-O" if "STAAR-O" in results.columns else "Burden(1,1)"
        results = results[
            (results["MASK"] == variant_category) & (results[test] < sig_thresh)
        ].copy()

        if len(results) == 0:
            continue
        results["INDEX"] -= 1
        results.loc[results[test] == 0, test] = results.loc[results[test] > 0, test].min()
        log_pvalues = -np.log10(results[test])
        tfce_res, stat_map = tfce.tfce(results["INDEX"], log_pvalues)

        if tfce_null is not None:
            tfce_thresh = tfce_null.quantile(results['CMAC'].to_list()[0], tfce_quantile_level)
        else:
            tfce_thresh = 0
        labeled_clusters, num_clusters = label(tfce_res > tfce_thresh + 0.001)
        
        if num_clusters == 0:
            continue
        
        voxels_in_cluster_list = list()
        cluster_tfce_list = list()
        n_valid_clusters = 0
        for cluster in range(1, num_clusters + 1):
            if (
                np.max(stat_map[labeled_clusters == cluster]) > sig_thresh2 and 
                np.sum(labeled_clusters == cluster) > 1
            ):
                voxels_in_cluster = np.where(labeled_clusters[tfce.roi_mask] == cluster)[0] + 1
                voxels_in_cluster_list.append(voxels_in_cluster)
                cluster_tfce_list.append(np.max(tfce_res[labeled_clusters == cluster]))
                n_valid_clusters += 1

        if n_valid_clusters == 0:
            continue

        n_clusters.append(n_valid_clusters)
        global_max_tfce = round(np.max(cluster_tfce_list), 3)
        cluster_max_tfce = ';'.join([str(round(x, 3)) for x in cluster_tfce_list])
        max_tfce.append(cluster_max_tfce)
        voxels_in_cluster = ';'.join([','.join(x.astype(str)) for x in voxels_in_cluster_list])
        cluster_info.append(voxels_in_cluster)
        
        gene.append(result_info['VARIANT_SET'])
        chr.append(result_info['CHR'])
        start.append(result_info['START'])
        end.append(result_info['END'])
        n_variants.append(results['N_VARIANTS'].to_list()[0])
        cmac.append(results['CMAC'].to_list()[0])
        most_sig_pv.append(results[test].min())

    if n_clusters:
        results_summary = pd.DataFrame(
            {
                'GENE': gene,
                'CHR': chr,
                'START': start,
                'END': end,
                'CATEGORY': variant_category,
                'N_VARIANTS': n_variants,
                'CMAC': cmac,
                'MOST_SIG_PV': most_sig_pv,
                'GLOBAL_MAX_TFCE': global_max_tfce,
                # 'N_SIG_VOXELS': n_sig_voxels,
                'N_CLUSTERS': n_clusters,
                'MAX_TFCE_OF_EACH_CLUSTER': max_tfce,
                'VOXELS_IN_EACH_CLUSTER': cluster_info,
            }
        )
        return results_summary
    else:
        return None


def summarize_null_results(tfce, null_assoc, sig_thresh, threads):
    """
    Computing max TFCE for null clusters
    
    """
    null_assoc = null_assoc[null_assoc["P"] < sig_thresh].copy()
    null_assoc.loc[null_assoc["P"] == 0, "P"] = null_assoc.loc[null_assoc["P"] > 0, "P"].min()
    null_assoc["LOG10P"] = -np.log10(null_assoc["P"])
    null_assoc["INDEX"] -= 1
    null_assoc_group = null_assoc.groupby(["SAMPLE_ID", "GENE_ID"])
    null_assoc_tfce = list()

    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [
            executor.submit(tfce.tfce, null_assoc_["INDEX"], null_assoc_["LOG10P"])
            for _, null_assoc_ in null_assoc_group
        ]
        
        for future in futures:
            result = future.result()
            if result is not None and np.max(result) > 0.001:
                null_assoc_tfce.append(np.max(result))

    return np.sort(null_assoc_tfce)
        

def check_input(args, log):
    if args.coord_dir is None:
        raise ValueError("--coord-dir is required")
    else:
        ds.check_existence(args.coord_dir)

    if args.results_idx is None and args.null_assoc is None:
        raise ValueError("--result-idx or --null-assoc is required")
    if args.results_idx is not None:
        args.null_assoc = None
        args.results_idx = ds.parse_input(args.results_idx)
        for file in args.results_idx:
            ds.check_existence(file)
        if args.tfce_quantile_level is None:
            args.tfce_quantile_level = 0
            log.info("Set TFCE quantile level as 0")
        if args.variant_category is None:
            raise ValueError("--variant-category is required")
        else:
            args.variant_category = args.variant_category.lower()
            if args.variant_category not in {
                    "plof",
                    "plof_ds",
                    "missense",
                    "disruptive_missense",
                    "synonymous",
                    "ptv",
                    "ptv_ds",
                }:
                raise ValueError(f"invalid variant category: {args.variant_category}")
    if args.null_assoc is not None:
        ds.check_existence(args.null_assoc)
    if args.sig_thresh is None:
        args.sig_thresh = 2.5e-6
        log.info("Set significance threshold as 2.5e-6")
    if args.sig_thresh2 is None:
        args.sig_thresh2 = args.sig_thresh
        log.info(f"Set significance threshold as {args.sig_thresh}")


def run(args, log):
    check_input(args, log)

    log.info(f"Read mask image from {args.coord_dir}")
    coord, roi_mask, slices = nifti_coord_mask(args.coord_dir)
    tfce = TFCE(coord, roi_mask, slices)

    if args.results_idx is not None:
        if args.tfce_null is not None:
            tfce_null = TFCEnull(args.tfce_null)
        else:
            tfce_null = None
        results_summary_list = list()
        for results_idx_file in args.results_idx:
            log.info(f"Read result index file from {results_idx_file}")
            results_idx = pd.read_csv(results_idx_file, sep='\t')

            results_summary = summarize_results(
                tfce,
                results_idx, 
                tfce_null,
                args.variant_category, 
                args.sig_thresh, 
                args.tfce_quantile_level,
                args.sig_thresh2
            )
            if results_summary is not None:
                results_summary_list.append(results_summary)
        if results_summary_list:
            results_summary = pd.concat(results_summary_list, axis=0)
            results_summary.to_csv(f"{args.out}_tfce.txt", sep="\t", index=None)
            log.info(f"\nSaved TFCE of significant associations to {args.out}_tfce.txt")
        else:
            log.info(f"\nNo significant results.")

    else:
        log.info(f"Read null associations from {args.null_assoc}")
        null_assoc = pd.read_csv(args.null_assoc, sep="\t")
        cmac_breaks = [0, 2, 3, 4, 5, 7, 9, 11, 14, 20, 30, 60, 100, 500, 10000000]
        cmac_bins = [(2,2), (3,3), (4,4), (5,5), (6,7), (8,9), (10,11), 
                     (12,14), (15,20), (21,30), (31,60), (61,100), (101,500), (501,)]
        null_assoc["cmac_bin"] = pd.cut(null_assoc["CMAC"], bins = cmac_breaks, labels=cmac_bins)
        null_assoc_by_cmac_bin = null_assoc.groupby("cmac_bin", observed=True)

        log.info("Computing TFCE ...")
        all_null_assoc_results = dict()
        all_cmac_bin_count = dict()
        for cmac_bin, null_assoc_bin in null_assoc_by_cmac_bin:
            null_assoc_results = summarize_null_results(
                tfce,
                null_assoc_bin,
                args.sig_thresh,
                args.threads
            )
            all_null_assoc_results[cmac_bin] = null_assoc_results
            all_cmac_bin_count[cmac_bin] = null_assoc_bin['CMAC_BIN_COUNT'].iloc[0]
        
        with h5py.File(f"{args.out}_tfce.h5", "w") as file:
            for cmac_bin, null_assoc_results in all_null_assoc_results.items():
                bin_str = "_".join(str(x) for x in cmac_bin)
                dataset = file.create_dataset(
                    bin_str, data=null_assoc_results, dtype=np.float32
                )
                dataset.attrs["count"] = all_cmac_bin_count[cmac_bin]
                
        log.info(f"\nSaved TFCE of null associations to {args.out}_tfce.h5")
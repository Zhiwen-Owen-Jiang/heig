import time
import logging
import h5py
import numpy as np
import hail as hl
from tqdm import tqdm
from scipy.sparse import csr_matrix, vstack
from scipy.stats import chi2
import heig.input.dataset as ds
from heig.wgs.null import NullModel
from heig.wgs.relatedness import LOCOpreds
from heig.wgs.mt import SparseGenotype
from heig.wgs.utils import *
from heig.wgs.utils import list_datasets


"""
Using permutation to test variant sets with a cMAC < 300

"""

CMAC_BINS = [(2,2), (3,3), (4,4), (5,5), (6,7), (8,9),
            (10,11), (12,14), (15,20), (21,30), (31,60), 
            (61,100), (101,200), (201,300)]


class Permutation:
    """
    Computing null distribution of burden/skat test
    for variant sets with small cMAC

    """

    def __init__(
            self, 
            mask,
            cmac_bins,
            n_samples=5*10**8,
            sig_thresh=2.5e-6,
            threads=1,
        ):
        """
        Parameters:
        ------------
        mask: an instance of CreatingMask
        cmac_bins: CMAC bins to permute
        n_samples: total number of permutation points
        sig_thresh: significant threshold
        threads: number of threads

        """
        self.cov_mat_dict = mask.cov_mat_dict
        self.resid_voxels = mask.resid_voxels
        self.gene_numeric_idxs = mask.gene_numeric_idxs
        self.vset = mask.vset
        self.n_masks = mask.n_masks
        self.voxels = mask.voxels
        self.total_points = n_samples
        self.n_batch = self.total_points // self.n_masks[list(self.n_masks.keys())[0]]
        self.cmac_bins = cmac_bins 
        self.threads = threads
        self.n_subs = self.resid_voxels.shape[0]
        self.sig_thresh = chi2.ppf(1 - sig_thresh, 1)
        self.logger = logging.getLogger(__name__)

    def _permute(self):
        resid_voxels_rand = self.resid_voxels[np.random.permutation(self.n_subs)]
        half_score = self.vset @ resid_voxels_rand
        return half_score

    def _variant_set_test(self, half_score, bin):
        """
        A wrapper function of variant set test for multiple sets
        """
        burden_stats_list = []

        for idx in range(self.n_masks[bin]):
            burden_stat = self._variant_set_test_(
                self.cov_mat_dict[bin][idx], 
                half_score[self.gene_numeric_idxs[bin][idx]]
            )
            if burden_stat is not None:
                burden_stats_list.append(burden_stat)

        return vstack(burden_stats_list) if burden_stats_list else None

    def _variant_set_test_(self, cov_mat, half_score):
        """
        Testing a single variant set
        
        """
        burden_stats = np.einsum('ij->j', half_score) ** 2 / cov_mat
        col = np.nonzero(burden_stats > self.sig_thresh)[0]
        if col.size > 0:
            row = np.zeros_like(col)
            values = burden_stats[col]
            burden_stats = csr_matrix((values, (row, col)),shape=(1, burden_stats.size))
            return burden_stats
        return None
        
    def run(self):
        """
        The main function for permutation

        """
        burden_sig_stats_dict = dict()
        burden_sig_stats_temp_dict = {bin: list() for bin in self.cmac_bins}
        burden_count_dict = {bin: 0 for bin in self.cmac_bins}
        total_permute_time = 0
        total_data_time = 0
        
        for _ in tqdm(range(self.n_batch), desc=f"{self.n_batch} batches"):
            start_time = time.perf_counter()
            half_score = self._permute()
            elapsed_time = (time.perf_counter() - start_time)
            total_data_time += elapsed_time

            for bin in self.cmac_bins:
                start_time = time.perf_counter()
                burden_stats = self._variant_set_test(half_score, bin)
                elapsed_time = (time.perf_counter() - start_time)
                total_permute_time += elapsed_time

                if burden_stats is not None:
                    burden_sig_stats_temp_dict[bin].append(burden_stats)

                burden_count_dict[bin] += self.n_masks[bin] 

        self.logger.info(f"total data time {total_data_time}s")
        self.logger.info(f"total test time {total_permute_time}s")
        
        for bin, voxel_sig_stats in burden_sig_stats_temp_dict.items():
            if len(voxel_sig_stats) == 0:
                voxel_sig_stats_dict = {voxel: list() for voxel in self.voxels}
            else:
                voxel_sig_stats = vstack(voxel_sig_stats).tocsc()
                voxel_sig_stats_dict = {
                    self.voxels[i]: voxel_sig_stats[:, i].data
                    for i in range(voxel_sig_stats.shape[1])
                }
                burden_sig_stats_dict[bin] = voxel_sig_stats_dict

        return burden_sig_stats_dict, burden_count_dict
    

class CreatingMask:
    """
    Generating summary statistics for each variant set
    
    """
    def __init__(
            self,
            null_model,
            voxels,
            locus, 
            vset,
            mac, 
            cmac_bins,
            loco_preds,
            n_points,
    ):
        self.locus = locus
        self.vset = vset
        self.resid_ldr = null_model.resid_ldr
        self.covar = null_model.covar
        self.n_subs, self.n_covars = self.covar.shape
        self.bases = null_model.bases
        self.mac = mac
        self.cmac_bins = cmac_bins
        self.n_points = n_points
        
        if voxels is None:
            self.voxels = np.arange(self.bases.shape[0])
        else:
            self.voxels = voxels
        
        self.chr, self.n_variants = self._extract_variants()
        self.resid_voxels = self._misc(loco_preds)
        self.gene_numeric_idxs = self._parse_genes_sliding_window()
        self.n_masks = {k:len(v) for k, v in self.gene_numeric_idxs.items()}
        self.vset_ld = self._get_ld_matrix()
        self.cov_mat_dict = self._sumstats()

    def _get_ld_matrix(self):
        vset = self.vset.astype(np.uint16)
        return vset @ vset.T

    def _extract_variants(self):
        geno_ref = self.locus.reference_genome.collect()[0]
        chr = self.locus.aggregate(hl.agg.take(self.locus.locus.contig, 1)[0])
        if geno_ref == "GRCh38":
            chr = int(chr.replace("chr", ""))
        else:
            chr = int(chr)
        
        if self.vset.shape[0] > 100000:
            self.vset = self.vset[:100000]
            self.mac = self.mac[:100000]
            
        return chr, len(self.mac)
    
    def _misc(self, loco_preds):
        """
        Correcting sample relatedness and calculating var (irrelavant to permutation)
        
        """
        if loco_preds is not None:
            self.resid_ldr = self.resid_ldr - loco_preds.data_reader(self.chr)
        resid_voxels = np.dot(self.resid_ldr, self.bases.T)
        inner_ldr = np.dot(self.resid_ldr.T, self.resid_ldr).astype(np.float32)
        var = np.sum(np.dot(self.bases, inner_ldr) * self.bases, axis=1)
        var /= self.n_subs - self.n_covars  # (N, )
        resid_voxels = resid_voxels / np.sqrt(var) # normalized residuals
        
        return resid_voxels
    
    def _parse_genes_sliding_window(self):
        """
        Get independent genes for each cMAC bin
        
        """
        n_replicates = 100000 if self.n_points > 1e7 else self.n_points // 10
        gene_numeric_idxs = dict()
        variant_idxs = np.arange(self.n_variants)
        for bin in self.cmac_bins:
            output = list()
            window_range = (max(2, int(bin[0]*0.1)), bin[1] + 1)
            while True:
                permuted_variant_idxs = variant_idxs[np.random.permutation(self.n_variants)]
                start = 0
                window_size = np.random.choice(list(range(*window_range)), 1)[0]
                skip_size = int(window_size * 0.8) + 1
                while start + window_size < self.n_variants:
                    end = start + window_size
                    selected_variants = permuted_variant_idxs[start: end]
                    if bin[0] <= np.sum(self.mac[selected_variants]) <= bin[1]:
                        output.append(selected_variants.tolist())
                        if len(output) >= n_replicates:
                            break
                    start += skip_size
                if len(output) >= n_replicates:
                    break
            gene_numeric_idxs[bin] = output

        return gene_numeric_idxs
    
    def _sumstats(self):
        """
        Calculating summary statistics for burden test
        
        """
        covar_U, _, covar_Vt = np.linalg.svd(self.covar, full_matrices=False)
        half_covar_proj = np.dot(covar_U, covar_Vt).astype(np.float32)
        vset_half_covar_proj = np.array(self.vset @ half_covar_proj)

        cov_mat_dict = {bin: list() for bin in self.cmac_bins}
        for bin, numeric_idx_list in self.gene_numeric_idxs.items():
            for numeric_idx in numeric_idx_list:
                x = vset_half_covar_proj[numeric_idx]
                vset_ld = self.vset_ld[numeric_idx][:, numeric_idx]
                cov_mat_dict[bin].append(np.array((vset_ld - np.dot(x, x.T))).sum())
            
        return cov_mat_dict
            

def merge_perm_files(perm_files, out, sig_thresh, cmac_bins):
    """
    Merge a list of permutation files for selected CMAC bins
    
    """
    burden_sig_stats_dict = {bin: dict() for bin in cmac_bins}
    burden_count_dict = {bin: 0 for bin in cmac_bins}
    all_bins = None

    for perm_file in perm_files:
        h5file = h5py.File(f"{perm_file}", "r")
        if all_bins is None:
            all_bins = list_datasets(h5file)
        for bin_str in all_bins:
            bin1, bin2, voxel = tuple([int(x) for x in bin_str.split("_")])
            bin = tuple([bin1, bin2])
            if bin not in burden_sig_stats_dict:
                continue
            data = h5file[bin_str]
            count = data.attrs["count"]
            if voxel in burden_sig_stats_dict[bin]:
                burden_sig_stats_dict[bin][voxel].append(data[:])
            else:
                burden_sig_stats_dict[bin][voxel] = [data[:]]
        for bin in cmac_bins:
            burden_count_dict[bin] += count
        h5file.close()

    save(out, burden_sig_stats_dict, burden_count_dict, sig_thresh)


def save(out, burden_sig_stats_dict, burden_count_dict, sig_thresh):
     """
     Save permutation results in a HDF5 file
     
     """
     with h5py.File(f"{out}_burden_perm.h5", 'w') as file:
         for bin, voxel_sig_stats in burden_sig_stats_dict.items():
             for voxel, sig_stats in voxel_sig_stats.items():
                 if isinstance(sig_stats, list):
                     sig_stats = np.concatenate(sig_stats)
                 sig_stats = np.sort(sig_stats).astype(np.float32)
                 if sig_thresh is not None:
                     sig_stats = sig_stats[-int(sig_thresh * burden_count_dict[bin]):]
                 bin_str = "_".join(str(x) for x in bin) + "_" + str(voxel)
                 dataset = file.create_dataset(bin_str, data=sig_stats)
                 dataset.attrs["count"] = burden_count_dict[bin]


def check_input(args, log):
    if args.perm_list is None:
        if args.sparse_genotype is None:
            raise ValueError("--sparse-genotype is required")
        if args.spark_conf is None:
            raise ValueError("--spark-conf is required")
        if args.null_model is None:
            raise ValueError("--null-model is required")
        if args.n_bootstrap is None:
            args.n_bootstrap = 5e7
            log.info("Set total number of permutation as 5e7")
        if args.variant_type is None:
            args.variant_type = "variant"
            log.info(f"Set --variant-type as default 'variant'.")
    else:
        args.perm_list = ds.parse_input(args.perm_list)
        for x in args.perm_list:
            ds.check_existence(x)
    if args.cmac_bins is None:
            args.cmac_bins = CMAC_BINS
    else:
        cmac_bins = args.cmac_bins.split(',')
        args.cmac_bins = set()
        for x in cmac_bins:
            cmac_bin = (int(x.split('_')[0]), int(x.split('_')[1]))
            if cmac_bin not in CMAC_BINS:
                raise ValueError(f"invalid CMAC bin {x}")
            args.cmac_bins.add(cmac_bin)
    if args.sig_thresh is None:
        args.sig_thresh = 2.5e-6
        log.info("Set significance threshold as 2.5e-6")
    

def run(args, log):
    # checking if input is valid
    check_input(args, log)

    if args.perm_list is not None:
        log.info(f"Merging permutation results from {len(args.perm_list)} files ...")
        merge_perm_files(args.perm_list, args.out, args.sig_thresh, args.cmac_bins)
        log.info(f"\nSaved merged permutation results to {args.out}_burden_perm.h5")
    else:
        try:
            init_hail(args.spark_conf, args.grch37, args.out, log)
            # reading data and selecting LDRs
            log.info(f"Read null model from {args.null_model}")
            null_model = NullModel(args.null_model)
            null_model.select_ldrs(args.n_ldrs)
            null_model.select_voxels(args.voxels)

            # reading sparse genotype data
            sparse_genotype = SparseGenotype(args.sparse_genotype)
            log.info(f"Read sparse genotype data from {args.sparse_genotype}")
            log.info((f"{sparse_genotype.vset.shape[1]} subjects and "
                    f"{sparse_genotype.vset.shape[0]} variants."))

            # reading loco preds
            if args.loco_preds is not None:
                log.info(f"Read LOCO predictions from {args.loco_preds}")
                loco_preds = LOCOpreds(args.loco_preds)
                if args.n_ldrs is not None:
                    loco_preds.select_ldrs((0, args.n_ldrs))
                if loco_preds.ldr_col[1] - loco_preds.ldr_col[0] != null_model.n_ldrs:
                    raise ValueError(
                        (
                            "inconsistent dimension in LDRs and LDR LOCO predictions. "
                            "Try to use --n-ldrs"
                        )
                    )
                common_ids = ds.get_common_idxs(
                    sparse_genotype.ids.index,
                    null_model.ids,
                    loco_preds.ids,
                    args.keep,
                )
            else:
                common_ids = ds.get_common_idxs(
                    sparse_genotype.ids.index, null_model.ids, args.keep
                )
            common_ids = ds.remove_idxs(common_ids, args.remove)

            # extract and align subjects with the genotype data
            null_model.keep(common_ids)
            null_model.remove_dependent_columns()
            log.info(f"{len(common_ids)} common subjects in the data.")
            log.info(
                (
                    f"{null_model.covar.shape[1]} fixed effects in the covariates "
                    "(including the intercept) after removing redundant effects.\n"
                )
            )

            if args.loco_preds is not None:
                loco_preds.keep(common_ids)
            else:
                loco_preds = None

            # log.info(f"Processing sparse genetic data ...")
            if args.extract_locus is not None:
                args.extract_locus, unique_chrs = read_extract_locus(args.extract_locus, args.grch37, log)
            else:
                unique_chrs = None
            if args.exclude_locus is not None:
                args.exclude_locus = read_exclude_locus(args.exclude_locus, args.grch37, log)
            
            sparse_genotype.keep(common_ids)
            sparse_genotype.extract_variant_type(args.variant_type)
            sparse_genotype.extract_exclude_locus(args.extract_locus, args.exclude_locus, unique_chrs)
            sparse_genotype.extract_chr_interval(args.chr_interval)
            sparse_genotype.extract_maf(args.maf_min, args.maf_max)
            sparse_genotype.extract_mac(args.mac_min, args.mac_max)
            vset, locus, maf, mac = sparse_genotype.parse_data()

            # creating mask
            mask = CreatingMask(
                null_model,
                args.voxels,
                locus, 
                vset,
                mac, 
                args.cmac_bins,
                loco_preds,
                args.n_bootstrap,
            )

            log.info(f"{mask.n_variants} variants used in permutation.")
            max_key_len = max(len(str(key)) for key in mask.n_masks.keys())
            max_val_len = max(len(str(value)) for value in mask.n_masks.values())
            max_len = max([max_key_len, max_val_len])
            keys_str = "  ".join(f"{str(key):<{max_len}}" for key in mask.n_masks.keys())
            values_str = "  ".join(f"{str(value):<{max_len}}" for value in mask.n_masks.values())
            log.info("Number of genes in each cMAC bin:")
            log.info(keys_str)
            log.info(values_str)

            # permutation
            log.info("Doing permutation ...")
            permutation = Permutation(mask, args.cmac_bins, args.n_bootstrap, args.sig_thresh, args.threads)
            burden_sig_stats_dict, burden_count_dict = permutation.run()

            # save results
            save(args.out, burden_sig_stats_dict, burden_count_dict, args.sig_thresh)
            log.info(f"\nSaved permutation results to {args.out}_burden_perm.h5")

        finally:
            if "loco_preds" in locals() and args.loco_preds is not None:
                loco_preds.close()

            clean(args.out)
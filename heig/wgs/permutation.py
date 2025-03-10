import time
import logging
import h5py
import numpy as np
import hail as hl
from tqdm import tqdm
# from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.sparse import csr_matrix, vstack
from scipy.stats import chi2
import heig.input.dataset as ds
from heig.wgs.null import NullModel
from heig.wgs.relatedness import LOCOpreds
from heig.wgs.mt import SparseGenotype
# from heig.wgs.coding import Coding
# from heig.utils import find_loc
from heig.wgs.utils import *

"""
Using permutation to test variant sets with a cMAC < 100

Input
1. images
2. annot-ht
3. variant-sets
4. sparse-genotype
5. n-permutes

Output
1. null distribution of test statistics

Steps
1. determine a list of independent voxels

"""

class Permutation:
    """
    Computing null distribution of burden/skat test
    for variant sets with small cMAC

    """

    def __init__(
            self, 
            mask,
            n_samples=5*10**8,
            threads=1,
        ):
        """
        Parameters:
        ------------
        mask: an instance of CreatingMask
        threads: number of threads

        """
        self.cov_mat_dict = mask.cov_mat_dict
        self.resid_voxels = mask.resid_voxels
        self.gene_numeric_idxs = mask.gene_numeric_idxs
        self.vset = mask.vset
        self.n_masks = mask.n_masks
        self.voxels = mask.voxels
        self.total_points = n_samples
        self.n_batch = self.total_points // 10000
        self.cmac_bins = [(2,2), (3,3), (4,4), (5,5), (6,7), (8,9),
                          (10,11), (12,14), (15,20), (21,30), (31,60), (61,100)]
        self.threads = threads
        self.n_subs = self.resid_voxels.shape[0]
        self.sig_thresh = chi2.ppf(1 - 2.5e-6, 1)
        self.logger = logging.getLogger(__name__)

        # self.burden_stats_denom_dict = self._get_burden_stats_denom()

    # def _get_burden_stats_denom(self):
    #     """
    #     Calculating denominator of burden stats (irrelavant to permutation)

    #     """
    #     burden_stats_denom_dict = {bin: list() for bin in self.cmac_bins}
    #     for bin, cov_mat_list in self.cov_mat_dict.items():
    #         for cov_mat in cov_mat_list:
    #             burden_stats_denom_dict[bin].append(self.var * np.sum(cov_mat))
    #     return burden_stats_denom_dict

    def _permute(self):
        resid_voxels_rand = self.resid_voxels[np.random.permutation(self.n_subs)]
        half_score = self.vset @ resid_voxels_rand
        return half_score

    # def _variant_set_test(self, half_score, bin):
    #     """
    #     A wrapper function of variant set test for multiple sets

    #     """
    #     burden_stats_list = []
    
    #     with ThreadPoolExecutor(max_workers=self.threads) as executor:
    #         futures = [
    #             executor.submit(
    #                 self._variant_set_test_, 
    #                 self.cov_mat_dict[bin][idx], 
    #                 half_score[self.gene_numeric_idxs[bin][idx]]
    #             )
    #             for idx in range(self.n_masks[bin])
    #         ]
            
    #         burden_stats_list = [
    #             future.result() for future in as_completed(futures) if future.result() is not None
    #         ]
        
    #     return vstack(burden_stats_list) if len(burden_stats_list) > 0 else None
    
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
        # finished_bins = set()
        total_permute_time = 0
        total_data_time = 0
        
        # while len(finished_bins) < len(burden_count_dict):
        for _ in tqdm(range(self.n_batch), desc=f"{self.n_batch} batches"):
            start_time = time.perf_counter()
            half_score = self._permute()
            elapsed_time = (time.perf_counter() - start_time)
            total_data_time += elapsed_time

            for bin in self.cmac_bins:
                # if bin not in finished_bins:
                start_time = time.perf_counter()
                burden_stats = self._variant_set_test(half_score, bin)
                elapsed_time = (time.perf_counter() - start_time)
                total_permute_time += elapsed_time

                if burden_stats is not None:
                    burden_sig_stats_temp_dict[bin].append(burden_stats)

                burden_count_dict[bin] += self.n_masks[bin] 
                # if burden_count_dict[bin] >= self.total_points or self.n_masks[bin] == 0:
                #     finished_bins.add(bin)
                #     self.logger.info(f"cMAC bin {bin} finished.")

        self.logger.info(f"total data time {total_data_time}s")
        self.logger.info(f"total test time {total_permute_time}s")
        
        for bin, voxel_sig_stats in burden_sig_stats_temp_dict.items():
            if len(voxel_sig_stats) == 0:
                voxel_sig_stats_dict = {voxel: list() for voxel in self.voxels}
            else:
                voxel_sig_stats = vstack(voxel_sig_stats).tocsc()
                voxel_sig_stats_dict = {
                    self.voxels[i]: voxel_sig_stats[:, i].data.tolist() 
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
            loco_preds,
    ):
        self.locus = locus
        # self.variant_sets = variant_sets
        # self.variant_category = variant_category
        self.vset = vset
        self.resid_ldr = null_model.resid_ldr
        self.covar = null_model.covar
        self.n_subs, self.n_covars = self.covar.shape
        self.bases = null_model.bases
        self.mac = mac
        self.cmac_bins = [(2,2), (3,3), (4,4), (5,5), (6,7), (8,9),
                          (10,11), (12,14), (15,20), (21,30), (31,60), (61,100)]
        
        if voxels is None:
            self.voxels = np.arange(self.bases.shape[0])
        else:
            self.voxels = voxels
        
        # self.geno_ref, self.chr, self.positions = self._extract_variant_category()
        self.chr, self.n_variants = self._extract_variants()
        self.resid_voxels = self._misc(loco_preds)
        # self.gene_numeric_idxs = self._parse_genes(cmac_min, cmac_max)
        # self.gene_numeric_idxs = self._parse_genes_backtrack()
        self.gene_numeric_idxs = self._parse_genes_sliding_window()
        self.n_masks = {k:len(v) for k, v in self.gene_numeric_idxs.items()}
        # self.vset_ld = self._get_band_ld_matrix()
        self.vset_ld = self._get_ld_matrix()
        self.cov_mat_dict = self._sumstats()

    def _get_ld_matrix(self):
        vset = self.vset.astype(np.uint16)
        return vset @ vset.T

    # def _extract_variant_category(self):
    #     """
    #     Extracting a variant category from annotation
        
    #     """
    #     self.locus = self.locus.add_index("idx")
    #     variant_type = self.locus.variant_type.collect()[0]
    #     geno_ref = self.locus.reference_genome.collect()[0]
    #     chr = self.locus.aggregate(hl.agg.take(self.locus.locus.contig, 1)[0])
    #     if geno_ref == "GRCh38":
    #         chr = int(chr.replace("chr", ""))
    #     else:
    #         chr = int(chr)
        
    #     mask = Coding(self.locus, variant_type)
    #     mask_idx = mask.category_dict[self.variant_category]
    #     numeric_idx, _ = mask.parse_annot(mask_idx, False)

    #     self.vset = self.vset[numeric_idx]
    #     self.maf = self.maf[numeric_idx]
    #     self.mac = self.mac[numeric_idx]
    #     positions = np.array(self.locus.locus.position.collect())[numeric_idx]

    #     return geno_ref, chr, positions
    
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
    
    # def _parse_genes(self, cmac_min, cmac_max):
    #     """
    #     Parsing genes and put into cMAC bins
        
    #     """
    #     gene_numeric_idxs = {bin: list() for bin in self.cmac_bins}
    #     for _, gene in self.variant_sets.iterrows():
    #         _, start, end = parse_interval(gene[1], self.geno_ref)
    #         start_idx = find_loc(self.positions, start)
    #         end_idx = find_loc(self.positions, end) + 1
    #         if start_idx == -1 or self.positions[start_idx] != start:
    #             start_idx += 1
    #         if end_idx > start_idx + 1:
    #             numeric_idxs = list(range(start_idx, end_idx))
    #             cmac = np.sum(self.mac[numeric_idxs])
    #             if cmac >= cmac_min and cmac <= cmac_max:
    #                 bin_idx = find_loc([2,3,4,5,6,8,10,12,15,21,31,61], cmac)
    #                 gene_numeric_idxs[self.cmac_bins[bin_idx]].append(numeric_idxs)
                         
    #     for bin, numeric_idx_list in gene_numeric_idxs.items():
    #         if len(numeric_idx_list) == 0:
    #             raise ValueError(f"no genes for cMAC bin {bin}")

    #     return gene_numeric_idxs
    
    # def _parse_genes_backtrack(self):
    #     """
    #     Find all genes within cMAC bins using backtrack
        
    #     """
    #     variant_idxs = list(range(len(self.mac)))
    #     def backtrack(output, target_cmac, n, temp, cur_cmac, i):
    #         if cur_cmac == target_cmac and len(temp) >= 2:
    #             output.append(temp[:])
    #             return
    #         if len(output) >= n:
    #             return
    #         for ii, mac in enumerate(self.mac[i:]):
    #             if cur_cmac + mac <= target_cmac:
    #                 if mac > 1 or np.random.rand() < 0.5:
    #                     temp.append(variant_idxs[i:][ii])
    #                     backtrack(output, target_cmac, n, temp, cur_cmac + mac, i + ii + 1)
    #                     temp.pop()

    #     gene_numeric_idxs = dict()
    #     for bin in self.cmac_bins:
    #         output = list()
    #         for target_cmac in range(bin[0], bin[1] + 1):
    #             output_ = list()
    #             n = 10000 // (bin[1] - bin[0] + 1)
    #             backtrack(output_, target_cmac, n, [], 0, 0)
    #             output.extend(output_[:n])
    #         gene_numeric_idxs[bin] = output
        
    #     return gene_numeric_idxs

    def _parse_genes_sliding_window(self):
        """
        Get independent genes for each cMAC bin
        
        """
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
                        if len(output) >= 10000:
                            break
                    start += skip_size
                if len(output) >= 10000:
                    break
            gene_numeric_idxs[bin] = output

        return gene_numeric_idxs
    
    # def _get_band_ld_matrix(self):
    #     """
    #     Creating a banded sparse LD matrix with the bandwidth being
    #     the length of the largest gene
        
    #     """
    #     bandwidth = 100
    #     n_variants = len(self.maf)
    #     diagonal_data = list()
    #     banded_data = list()
    #     banded_row = list()
    #     banded_col = list()

    #     for start in range(0, n_variants, bandwidth):
    #         end1 = start + bandwidth
    #         end2 = end1 + bandwidth
    #         vset_block1 = self.vset[start:end1].astype(np.uint16)
    #         vset_block2 = self.vset[start:end2].astype(np.uint16)
    #         ld_rec = vset_block1 @ vset_block2.T
    #         ld_rec_row, ld_rec_col = ld_rec.nonzero()
    #         ld_rec_row += start
    #         ld_rec_col += start
    #         ld_rec_data = ld_rec.data

    #         diagonal_data.append(ld_rec_data[ld_rec_row == ld_rec_col])
    #         mask = (np.abs(ld_rec_row - ld_rec_col) <= bandwidth) & (
    #             ld_rec_col > ld_rec_row
    #         )
    #         banded_row.append(ld_rec_row[mask])
    #         banded_col.append(ld_rec_col[mask])
    #         banded_data.append(ld_rec_data[mask])

    #     diagonal_data = np.concatenate(diagonal_data)
    #     banded_row = np.concatenate(banded_row)
    #     banded_col = np.concatenate(banded_col)
    #     banded_data = np.concatenate(banded_data)
    #     shape = np.array([n_variants, n_variants])

    #     lower_row = banded_col
    #     lower_col = banded_row
    #     diag_row_col = np.arange(shape[0])

    #     full_row = np.concatenate([banded_row, lower_row, diag_row_col])
    #     full_col = np.concatenate([banded_col, lower_col, diag_row_col])
    #     full_data = np.concatenate([banded_data, banded_data, diagonal_data])

    #     vset_ld = csr_matrix((full_data, (full_row, full_col)), shape=shape)
    #     return vset_ld
    
    def _sumstats(self):
        """
        Calculating summary statistics for burden test
        
        """
        covar_U, _, covar_Vt = np.linalg.svd(self.covar, full_matrices=False)
        half_covar_proj = np.dot(covar_U, covar_Vt).astype(np.float32)
        vset_half_covar_proj = self.vset @ half_covar_proj

        cov_mat_dict = {bin: list() for bin in self.cmac_bins}
        for bin, numeric_idx_list in self.gene_numeric_idxs.items():
            for numeric_idx in numeric_idx_list:
                x = np.array(vset_half_covar_proj[numeric_idx])
                vset_ld = self.vset_ld[numeric_idx][:, numeric_idx]
                cov_mat_dict[bin].append(np.array((vset_ld - x @ x.T)).sum())
            
        return cov_mat_dict
            

def merge_perm_files(perm_files, out, sig_thresh):
    """
    Merge a list of permutation files
    
    """
    cmac_bins = [(2,2), (3,3), (4,4), (5,5), (6,7), (8,9),
                 (10,11), (12,14), (15,20), (21,30), (31,60), (61,100)]
    burden_sig_stats_dict = {bin: dict() for bin in cmac_bins}
    burden_count_dict = {bin: 0 for bin in cmac_bins}
    all_bins = None

    for perm_file in perm_files:
        h5file = h5py.File(f"{perm_file}", "r")
        if all_bins is None:
            all_bins = _list_datasets(h5file)
        for bin_str in all_bins:
            bin1, bin2, voxel = tuple([int(x) for x in bin_str.split("_")])
            bin = tuple([bin1, bin2])
            data = h5file[bin_str]
            count = data.attrs["count"]
            if voxel in burden_sig_stats_dict[bin]:
                burden_sig_stats_dict[bin][voxel].append(data[:])
            else:
                burden_sig_stats_dict[bin][voxel] = [data[:]]
        for bin in cmac_bins:
            burden_count_dict[bin] += count
        h5file.close()

    with h5py.File(f"{out}_burden_perm.h5", 'w') as file:
        for bin, voxel_sig_stats in burden_sig_stats_dict.items():
            for voxel, sig_stats in voxel_sig_stats.items():
                sig_stats = np.sort(np.concatenate(sig_stats)).astype(np.float32)
                if sig_thresh is not None:
                    sig_stats = sig_stats[-int(sig_thresh * burden_count_dict[bin]):]
                bin_str = "_".join(str(x) for x in bin) + "_" + str(voxel)
                dataset = file.create_dataset(bin_str, data=sig_stats)
                dataset.attrs["count"] = burden_count_dict[bin]


def _list_datasets(hdf5_file):
    dataset_names = []
    
    def visitor_func(name, node):
        if isinstance(node, h5py.Dataset):
            dataset_names.append(name)
    
    hdf5_file.visititems(visitor_func)
    return dataset_names


def check_input(args, log):
    if args.perm_list is None:
        if args.sparse_genotype is None:
            raise ValueError("--sparse-genotype is required")
        if args.spark_conf is None:
            raise ValueError("--spark-conf is required")
        # if args.annot_ht is None:
        #     raise ValueError("--annot-ht (FAVOR annotation) is required")
        if args.null_model is None:
            raise ValueError("--null-model is required")
        # if args.variant_category is None:
        #     raise ValueError("--variant-category is required")
        # if args.cmac_min is None:
        #     args.cmac_min = 2
        #     log.info(f"Set --cmac-min as default 2")
        # if args.cmac_max is None:
        #     args.cmac_max = 100
        #     log.info(f"Set --cmac-max as default 100")
        if args.n_bootstrap is None:
            args.n_bootstrap = 5e7
            log.info("Set total number of permutation as 5e7")
    else:
        args.perm_list = ds.parse_input(args.perm_list)
        for x in args.perm_list:
            ds.check_existence(x)

    # args.variant_category = args.variant_category.lower()
    # if args.variant_category not in {
    #     "plof",
    #     "plof_ds",
    #     "missense",
    #     "disruptive_missense",
    #     "synonymous",
    #     "ptv",
    #     "ptv_ds",
    #     }:
    #     raise ValueError(f"invalid variant category: {args.variant_category}")
    

def run(args, log):
    # checking if input is valid
    check_input(args, log)

    if args.perm_list is not None:
        # if args.sig_thresh is not None:
        #     log.info(f"Permutation results will be trimmed for {args.sig_thresh}")
        log.info(f"Merging permutation results from {len(args.perm_list)} files ...")
        merge_perm_files(args.perm_list, args.out, args.sig_thresh)
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
                args.extract_locus = read_extract_locus(args.extract_locus, args.grch37, log)
            if args.exclude_locus is not None:
                args.exclude_locus = read_exclude_locus(args.exclude_locus, args.grch37, log)
            
            sparse_genotype.keep(common_ids)
            sparse_genotype.extract_exclude_locus(args.extract_locus, args.exclude_locus)
            sparse_genotype.extract_chr_interval(args.chr_interval)
            sparse_genotype.extract_maf(args.maf_min, args.maf_max)
            sparse_genotype.extract_mac(args.mac_min, args.mac_max)

            # # reading annotation
            # log.info(f"Read functional annotations from {args.annot_ht}")
            # annot = hl.read_table(args.annot_ht)
            # sparse_genotype.annotate(annot)
            vset, locus, maf, mac = sparse_genotype.parse_data()

            # creating mask
            mask = CreatingMask(
                null_model,
                args.voxels,
                locus, 
                vset,
                mac, 
                loco_preds,
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
            permutation = Permutation(mask, args.n_bootstrap, args.threads)
            burden_sig_stats_dict, burden_count_dict = permutation.run()
            with h5py.File(f"{args.out}_burden_perm.h5", 'w') as file:
                for bin, voxel_sig_stats in burden_sig_stats_dict.items():
                    for voxel, sig_stats in voxel_sig_stats.items():
                        sig_stats = np.sort(sig_stats).astype(np.float32)
                        bin_str = "_".join(str(x) for x in bin) + "_" + str(voxel)
                        dataset = file.create_dataset(bin_str, data=sig_stats)
                        dataset.attrs["count"] =  burden_count_dict[bin]

            # save results
            log.info(f"\nSaved permutation results to {args.out}_burden_perm.h5")

        finally:
            if "loco_preds" in locals() and args.loco_preds is not None:
                loco_preds.close()

            clean(args.out)
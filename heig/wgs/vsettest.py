import numpy as np
import pandas as pd
from functools import reduce
from scipy.stats import chi2, cauchy, beta
from heig.wgs.pvalue import saddle2
from heig.utils import find_loc



class VariantSetTest:
    def __init__(self, bases, var, perm, voxels):
        """
        Variant set test for rare variants

        Parameters:
        ------------
        N: number of voxels
        bases: (N, r) np.array, functional bases
        var: (N, ) np.array, voxel variance
        perm: an instance of PermDistribution
        voxels: a list of voxel idxs

        """
        self.N = bases.shape[0]
        self.bases = bases
        self.var = var
        self.perm = perm
        self.voxels = voxels

    def input_vset(
        self,
        half_ldr_score,
        cov_mat,
        maf,
        cmac,
        is_rare,
        annotation_pred=None,
        annot_transform=True,
    ):
        """
        Inputing variant set and computing half scores and covariance matrix

        Parameters:
        ------------
        half_ldr_score: (m, r) np.array, Z'(I-M)\Xi
        cov_mat: (m, m) np.array, Z'(I-M)Z
        maf: (m, ) np.array of MAF
        is_rare: (m, ) np.array boolean index indicating MAC < mac_threshold
        annotation_pred: (m, q) np.array of functional annotation or None
        annot_transform: if transforming FAVOR annotations to rank

        """
        self.maf = maf
        self.cmac = cmac
        self.annot_weights = annotation_pred if cmac > 1000 else None
        self.is_rare = is_rare
        self.half_ldr_score = half_ldr_score  # Z'(I-M)\Xi, (m, r)
        self.half_score = np.dot(self.half_ldr_score, self.bases.T)  # Z'(I-M)Y, (m, N)
        self.cov_mat = cov_mat
        self.weights = self._get_weights(self.annot_weights, annot_transform)
        self.n_variants = self.half_ldr_score.shape[0]

    def _get_weights(self, annot=None, annot_transform=True):
        """
        Vertically stacking weights, i.e., each row is a (m, ) vector

        Parameters:
        ------------
        annot: (m, q) array, m is #variants, q is #functional weights
        annot_transform: if transforming FAVOR annotations to rank

        Returns:
        ---------
        weights_dict: a dict of weights

        """
        w1 = beta.pdf(self.maf, 1, 25).reshape(1, -1)
        w2 = beta.pdf(self.maf, 1, 1).reshape(1, -1)
        # w3 = beta.pdf(self.maf, 0.5, 0.5).reshape(1, -1)
        weights_dict = dict()

        if annot is None:
            weights_dict["skat(1,25)"] = w1
            # weights_dict["skat(1,1)"] = w2
            # weights_dict["burden(1,25)"] = w1
            weights_dict["burden(1,1)"] = w2
            # weights_dict["acatv(1,25)"] = (w1 / w3) ** 2
            # weights_dict["acatv(1,1)"] = (w2 / w3) ** 2
        else:
            if (annot <= 0).any():
                raise ValueError("annotation weights must be greater than 0")
            if annot_transform:
                annot = 1 - 10 ** (-annot / 10)
            annot = annot.T
            weights_dict["skat(1,25)"] = self._combine_weights(w1, np.sqrt(annot))
            # weights_dict["skat(1,1)"] = self._combine_weights(w2, np.sqrt(annot))
            # weights_dict["burden(1,25)"] = self._combine_weights(w1, annot)
            weights_dict["burden(1,1)"] = self._combine_weights(w2, annot)
            # weights_dict["acatv(1,25)"] = self._combine_weights((w1 / w3) ** 2, annot)
            # weights_dict["acatv(1,1)"] = self._combine_weights((w2 / w3) ** 2, annot)

        return weights_dict

    def _combine_weights(self, w, w_annot):
        """
        Combining MAF weights with annotation weights

        Parameters:
        ------------
        w: (m, ) array of MAF weights
        w_annot (q, m) array of annotation weights

        Returns:
        ---------
        weights: a dict of weights

        """
        w_annot = w_annot * w
        weights = np.vstack([w, w_annot])
        return weights

    def _skat_test(self, weights):
        """
        Computing SKAT pvalues for one weight and all voxels. (tested)

        For a single trait Y, the score test statistic of SKAT:
        Y'(I-M)ZWWZ'(I-M)Y/\sigma^2.
        W is a m by m diagonal matrix, Z is a n by m matrix,
        M = X(X'X)^{-1}X'.
        Under the null, it follows a mixture of chisq(1) distribution,
        where the weights are eigenvalues of WZ'(I-M)ZW.
        All voxels share the same eigenvalues.
        Use Saddle-point approximation to compute pvalues.

        Parameters:
        ------------
        weights: (m, ) array

        Returns:
        ---------
        pvalues: (N, ) array

        """
        weighted_half_ldr_score = weights.reshape(-1, 1) * self.half_ldr_score  # (m, r)
        ldr_score = np.dot(weighted_half_ldr_score.T, weighted_half_ldr_score)  # (r, r)
        score_stat = (
            np.sum(np.dot(self.bases, ldr_score) * self.bases, axis=1) / self.var
        )  # (N, )

        wcov_mat = weights.reshape(-1, 1) * self.cov_mat * weights  # (m, m)
        egvalues, _ = np.linalg.eigh(
            wcov_mat
        )  # (m, ) all voxels share the same eigenvalues

        egvalues = np.flip(egvalues)
        egvalues[egvalues < 10**-8] = 0

        pvalues = saddle2(score_stat, egvalues)
        return pvalues

    def _burden_test(self, weights):
        """
        Computing Burden pvalues for one weight and all voxels. (tested)

        For a single trait Y, the burden test statistic
        (w'Z'(I-M)Y)^2 / (\hat{\sigma}^2 w'Z'(I-M)Zw)
        where w is a m by 1 vector, Z is a n by m matrix,
        M = X(X'X)^{-1}X'.
        Under the null, it follows a chisq(1) distribution.

        Parameters:
        ------------
        weights: (m, ) array

        Returns:
        ---------
        pvalues: (N, ) array

        """
        # burden_score_num = np.dot(weights, self.half_score) ** 2  # (N, )
        # burden_score_denom = self.var * np.dot(np.dot(weights, self.cov_mat), weights)
        # burden_score = burden_score_num / burden_score_denom  # (N, )
        # pvalues = chi2.sf(burden_score, 1)
        w2cov_mat = np.dot(np.dot(weights, self.cov_mat), weights)
        burden_effect = np.dot(weights, self.half_score) / w2cov_mat
        burden_se = np.sqrt(self.var / w2cov_mat)
        burden_chisq = (burden_effect / burden_se) ** 2

        if self.cmac > 100:
            pvalues = chi2.sf(burden_chisq, 1)
        else:
            bin_idx = find_loc(self.perm.breaks, self.cmac)
            bin = self.perm.bins[bin_idx]
            
            pvalues = [
                self._search_pvalue(
                    self.perm.sig_stats[bin][voxel], 
                    burden_chisq[i], 
                    self.perm.count[bin], 
                    self.perm.max_p[bin][voxel]
                ) for i, voxel in enumerate(self.voxels)
            ]

        return burden_effect, burden_se, pvalues
    
    def _search_pvalue(self, sig_stats, burden_chisq, count, max_p):
        idx = np.searchsorted(sig_stats, burden_chisq, side='left')
        pvalue = (len(sig_stats) - idx + 1) / (count + 1)
        return pvalue if pvalue < max_p else np.nan

    def _acatv_test(self, weights_A, weights_B):
        """
        Computing ACATV pvalues for one weight and all voxels. (tested)

        First split the variant set to very rare (MAC <= mac_thresh) and common sets.
        For very rare sets, do Burden test; for common set, do individual test,
        then combine the pvalues by cauchy combination.
        Individual score test:
        numerator: Y'(I-M)ZZ'(I-M)Y
        denominator: Y'(I-M)Y * [Z'(I-M)Z]/(n-p)

        Parameters:
        ------------
        weights_A: (m, ) array, ACAT weights
        weights_B: (m, ) array, Burden weights

        Returns:
        ---------
        pvalues: (N, ) array

        """
        ## score test for individual common variants
        if (~self.is_rare).any():
            denom = np.diag(self.cov_mat[~self.is_rare][:, ~self.is_rare]).reshape(
                -1, 1
            ) * self.var.reshape(
                1, -1
            )  # (m, N)
            common_variant_pv = chi2.sf(
                (self.half_score[~self.is_rare] ** 2 / denom), 1
            )  # (m1, N)
            common_weights = weights_A[~self.is_rare]  # (m1, )
        else:
            common_variant_pv = None
            common_weights = None

        ## Burden test for rare variants
        if (self.is_rare).any():
            rare_burden_score_num = (
                np.dot(weights_B[self.is_rare], self.half_score[self.is_rare]) ** 2
            )  # (N, )
            rare_burden_score_denom = self.var * np.dot(
                np.dot(
                    weights_B[self.is_rare], self.cov_mat[self.is_rare][:, self.is_rare]
                ),
                weights_B[self.is_rare],
            )  # (N, )
            rare_burden_score = rare_burden_score_num / rare_burden_score_denom
            rare_burden_pv = chi2.sf(rare_burden_score, 1).reshape(1, -1)  # (1, N)
            rare_weights = np.atleast_1d(np.mean(weights_A[self.is_rare]))  # (1, )
        else:
            rare_burden_pv = None
            rare_weights = None

        if common_variant_pv is not None and rare_burden_pv is not None:
            pvalues = cauchy_combination(
                np.vstack([common_variant_pv, rare_burden_pv]),
                np.concatenate([common_weights, rare_weights]),
            )
        elif common_variant_pv is not None:
            pvalues = cauchy_combination(common_variant_pv, common_weights)
        else:
            pvalues = rare_burden_pv

        return pvalues

    def do_inference(self, annot_name=None):
        """
        Doing inference for the variant set using multiple weights and methods.
        Using cauchy combination to get final pvalues.

        Parameters:
        ------------
        annot_name: a list of functional annotation names

        Returns:
        ---------
        all_results_df: a pd.DataFrame of results, each columns is a method

        """
        n_weights = self.weights["skat(1,25)"].shape[0]
        skat_1_25_pvalues = np.zeros((n_weights, self.N))
        skat_1_1_pvalues = np.zeros((n_weights, self.N))
        burden_1_25_pvalues = np.zeros((n_weights, self.N))
        burden_1_1_pvalues = np.zeros((n_weights, self.N))
        # acatv_1_25_pvalues = np.zeros((n_weights, self.N))
        # acatv_1_1_pvalues = np.zeros((n_weights, self.N))

        for i in range(n_weights):
            skat_1_25_pvalues[i] = self._skat_test(self.weights["skat(1,25)"][i])
            skat_1_1_pvalues[i] = self._skat_test(self.weights["skat(1,1)"][i])
            burden_1_25_pvalues[i] = self._burden_test(self.weights["burden(1,25)"][i])
            burden_1_1_pvalues[i] = self._burden_test(self.weights["burden(1,1)"][i])
            # acatv_1_25_pvalues[i] = self._acatv_test(
            #     self.weights["acatv(1,25)"][i], self.weights["burden(1,25)"][i]
            # )
            # acatv_1_1_pvalues[i] = self._acatv_test(
            #     self.weights["acatv(1,1)"][i], self.weights["burden(1,1)"][i]
            # )

        all_pvalues = np.vstack(
            [
                skat_1_25_pvalues,
                skat_1_1_pvalues,
                burden_1_25_pvalues,
                burden_1_1_pvalues,
                # acatv_1_25_pvalues,
                # acatv_1_1_pvalues,
            ]
        )
        results_STAAR_O = pd.DataFrame(
            cauchy_combination(all_pvalues), columns=["STAAR-O"]
        )
        # results_ACAT_O = cauchy_combination(
        #     np.vstack(
        #         [
        #             skat_1_25_pvalues[0],
        #             skat_1_1_pvalues[0],
        #             burden_1_25_pvalues[0],
        #             burden_1_1_pvalues[0],
        #             acatv_1_25_pvalues[0],
        #             acatv_1_1_pvalues[0],
        #         ]
        #     )
        # )
        # results_ACAT_O = pd.DataFrame(results_ACAT_O, columns=["ACAT-O"])
        # all_results = [results_STAAR_O, results_ACAT_O]
        all_results = [results_STAAR_O]

        for pvalues, test_method in (
            (skat_1_25_pvalues, "SKAT(1,25)"),
            (skat_1_1_pvalues, "SKAT(1,1)"),
            (burden_1_25_pvalues, "Burden(1,25)"),
            (burden_1_1_pvalues, "Burden(1,1)"),
            # (acatv_1_25_pvalues, "ACAT-V(1,25)"),
            # (acatv_1_1_pvalues, "ACAT-V(1,1)"),
        ):
            if n_weights > 1:
                comb_pvalues = cauchy_combination(pvalues).reshape(-1, 1)
            else:
                comb_pvalues = None
            all_pvalues = format_results(
                pvalues.T, comb_pvalues, test_method, annot_name
            )
            all_results.append(all_pvalues)
        all_results_df = pd.concat(all_results, axis=1)

        return all_results_df
    
    def do_inference_tests(self, tests, annot_name=None, compute_burden_effect=True):
        """
        Doing inference for the variant set using multiple weights and methods.
        if tests == burden or skat, do maf(1,1), maf(1,25), and weights
        if burden, output burden stats
        if tests == staar, do burden, skat, staar-burden, staar-skat, and staaro

        Parameters:
        ------------
        tests: a list of tests
        annot_name: a list of functional annotation names
        compute_burden_effect: if computing burden effect

        Returns:
        ---------
        all_results_df: a pd.DataFrame of results, each columns is a method
        
        """
        n_weights = self.weights["burden(1,1)"].shape[0]
        all_results = list()
        # burden_1_25_pvalues = np.zeros((n_weights, self.N))
        burden_1_1_pvalues = np.zeros((n_weights, self.N))
        burden_effect, burden_se, burden_pvalue = None, None, None
        skat_1_25_pvalues = np.zeros((n_weights, self.N))
        # skat_1_1_pvalues = np.zeros((n_weights, self.N))

        if "burden" in tests or "staar" in tests or self.cmac <= 500:
            if compute_burden_effect:
                burden_effect, burden_se, burden_pvalue = self._burden_test(self.weights["burden(1,1)"][0])
            for i in range(n_weights):
                # _, _, burden_1_25_pvalues[i] = self._burden_test(self.weights["burden(1,25)"][i])
                _, _, burden_1_1_pvalues[i] = self._burden_test(self.weights["burden(1,1)"][i])

        if ("skat" in tests or "staar" in tests) and self.cmac > 500:
            for i in range(n_weights):
                skat_1_25_pvalues[i] = self._skat_test(self.weights["skat(1,25)"][i])
                # skat_1_1_pvalues[i] = self._skat_test(self.weights["skat(1,1)"][i])
        
        if "staar" in tests and self.cmac > 500:
            all_pvalues = np.vstack(
                [
                    skat_1_25_pvalues,
                    # skat_1_1_pvalues,
                    # burden_1_25_pvalues,
                    burden_1_1_pvalues,
                ]
            )
            results_STAAR_O = pd.DataFrame(
                cauchy_combination(all_pvalues), columns=["STAAR-O"]
            )
            all_results.append(results_STAAR_O)

        for pvalues, test_method in (
            (skat_1_25_pvalues, "SKAT(1,25)"),
            # (skat_1_1_pvalues, "SKAT(1,1)"),
            # (burden_1_25_pvalues, "Burden(1,25)"),
            (burden_1_1_pvalues, "Burden(1,1)"),
        ):
            if pvalues.sum() == 0: # nan == 0 is False
                continue
            if n_weights > 1 and self.cmac > 1000:
                comb_pvalues = cauchy_combination(pvalues).reshape(-1, 1)
            else:
                comb_pvalues = None
            all_pvalues = format_results(
                pvalues.T, comb_pvalues, test_method, annot_name
            )
            all_results.append(all_pvalues)
        all_results_df = pd.concat(all_results, axis=1)

        if burden_effect is not None:
            burden_test = pd.DataFrame(
                {'EFFECT': burden_effect, 
                 'SE': burden_se,
                 'P': burden_pvalue}
            )
        else:
            burden_test = None

        return all_results_df, burden_test

        
def cauchy_combination(pvalues, weights=None, axis=0):
    """
    Cauchy combination for an array of pvalues and weights.

    Parameters:
    ------------
    pvalues: (m1, N) array
    weights: (m1, ) array
    axis: which axis to take average

    Returns:
    ---------
    cct_pvalues: (N, ) array

    """
    n_weights, n_voxels = pvalues.shape

    if weights is None:
        weights = np.ones(n_weights)
    elif (weights <= 0).any():
        raise ValueError("weights must be positive")
    elif n_weights != weights.shape[0]:
        raise ValueError(
            "the length of weights should be the same as that of the p-values"
        )
    elif np.isnan(weights).any():
        return np.full(n_voxels, np.nan)

    output0_voxels = (pvalues == 0).any(axis=0)
    output1_voxels = (pvalues == 1).any(axis=0)

    nan_voxels = reduce(np.logical_and, [output0_voxels, output1_voxels])
    nan_voxels = reduce(np.logical_or, [nan_voxels, np.isnan(pvalues).any(axis=0)])

    good_voxels = ~reduce(np.logical_or, [nan_voxels, output0_voxels, output1_voxels])
    pvalues = pvalues[:, good_voxels]

    normed_weights = weights / np.sum(weights)
    normed_weights = np.tile(normed_weights, (pvalues.shape[1], 1)).T

    is_small = pvalues < 10**-16
    if not is_small.any():
        stats = np.sum(np.tan((0.5 - pvalues) * np.pi) * normed_weights, axis=axis)
    else:
        stats1 = np.sum(
            np.where(~is_small, np.tan((0.5 - pvalues) * np.pi) * normed_weights, 0),
            axis=axis,
        )
        stats2 = (
            np.sum(np.where(is_small, normed_weights / pvalues, 0), axis=axis) / np.pi
        )
        stats = stats1 + stats2

    cct_pvalues = np.zeros(stats.shape)
    is_large = stats > 10**15
    cct_pvalues[~is_large] = cauchy.sf(stats[~is_large], loc=0, scale=1)
    cct_pvalues[is_large] = 1 / stats[is_large] / np.pi

    output = np.zeros(n_voxels)
    output[good_voxels] = cct_pvalues
    output[nan_voxels] = np.nan
    output[output0_voxels] = 0
    output[output1_voxels] = 1

    return output


def format_results(indiv_pvalues, comb_pvalues, method_name, indiv_annot_name=None):
    """
    Formating test results. Individual p-values go first followed by combined p-values.

    Parameters:
    ------------
    indiv_pvalues: (N, 1) array
    comb_pvalues: (N, ) array
    indiv_annot_name: name of each functional annotation
    method_name: test method with beta distribution parameters, such as SKAT(1,25)

    Returns:
    ---------
    cct_pvalues: (N, ) array

    """
    if comb_pvalues is not None:
        col_names = [method_name]
        if indiv_annot_name is not None:
            for annot in indiv_annot_name:
                col_names.append(f"{method_name}-{annot}")
        col_names.append(f"STAAR-{method_name}")
        res = pd.DataFrame(
            np.column_stack([indiv_pvalues, comb_pvalues]), columns=col_names
        )
    else:
        res = pd.DataFrame(indiv_pvalues, columns=[method_name])

    return res

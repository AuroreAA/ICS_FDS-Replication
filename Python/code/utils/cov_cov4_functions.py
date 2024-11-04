import numpy as np
from .moments_functions import *
from .helpers import sort_eigenvalues_eigenvectors


def compute_moment(vect_ij, mu_y_groups, eps, k):
    """
    Computes a moment of the components of Y. The expression of the moment depends on the values of vect_ij.
    :param vect_ij: ndarray (k-1, ) containing power of the components of Y
    :param mu_y_groups: ndarray (k, p) containing means of the groups of Y:=X-mu_x
    :param eps: ndarray (k, ) containing group proportions
    :param k: (int) number of groups
    :return: (float) moment E[y_1^(vect_ij[0])*y_2^(vect_ij[1])*...*y_k-1^(vect_ij[-1])]
    """
    if max(vect_ij) == 4:
        # Compute E(y_a^4)
        ind4 = vect_ij.tolist().index(4)
        exp = expectation_y_4(ind4, mu_y_groups, eps, k)

    elif max(vect_ij) == 3:
        # Compute E(y_a^3*y_b)
        ind3 = vect_ij.tolist().index(3)
        ind1 = vect_ij.tolist().index(1)
        exp = expectation_y_3_1(ind3, ind1, mu_y_groups, eps, k)

    elif max(vect_ij) == 2:
        ind2 = [i for i, x in enumerate(vect_ij) if x == 2]
        if len(ind2) == 2:
            # Compute E(y_a^2*y_b^2)
            ind2_a, ind2_b = ind2
            exp = expectation_y_2_2(ind2_a, ind2_b, mu_y_groups, eps, k)
        else:
            # Compute E(y_c^2*y_a*y_b)
            ind2 = ind2[0]
            ind1_a, ind1_b = [i for i, x in enumerate(vect_ij) if x == 1]
            exp = expectation_y_2_1_1(ind2, ind1_a, ind1_b, mu_y_groups, eps, k)
    else:
        # Compute E(y_a*y_b*y_c*y_d)
        ind1_a, ind1_b, ind1_c, ind1_d = [i for i, x in enumerate(vect_ij) if x == 1]
        exp = expectation_y_1_1_1_1(ind1_a, ind1_b, ind1_c, ind1_d, mu_y_groups, eps, k)

    return exp


def compute_cov(mu_y_groups, eps, k, p):
    """Computes COV = sigma_b + sigma_w"""
    sigma_w = np.identity(p)
    sigma_b = np.zeros((p, p))
    for i in range(k):
        sigma_b += eps[i] * np.outer(mu_y_groups[i], mu_y_groups[i])
    cov = sigma_b + sigma_w
    return cov


def compute_cov4(cov, dict_beta, mu_y_groups, eps, k, p, q):
    """Computes COV_4 = (1/p+2) * E[D^2YY^T]"""

    cov4 = cov.copy()

    for alph_i in range(q):
        for alph_j in range(q):
            if alph_i <= alph_j:
                res = 0
                for i in range(q):
                    for j in range(q):
                        beta_ij = "beta_" + str(i + 1) + str(j + 1)
                        vect_ij = dict_beta[beta_ij][1].copy()
                        vect_ij[alph_i] += 1
                        vect_ij[alph_j] += 1
                        res += (
                                dict_beta[beta_ij][0] *
                                compute_moment(vect_ij, mu_y_groups, eps, k)
                        )
                if alph_i == alph_j:
                    res += (p - q) * expectation_y_2(alph_i, mu_y_groups, eps, k)
                else:
                    res += (p - q) * expectation_y_1_1(alph_i, alph_j, mu_y_groups, eps, k)
                res = (1 / (p + 2)) * res
                cov4[alph_i][alph_j] = res
            else:
                cov4[alph_i][alph_j] = cov4[alph_j][alph_i]

    return cov4


def eig_cc4(eps, mu_x_groups, reduced=False, detailed=False):
    """
    Given group proportions and group means, computes the eigenvalues of COV-1COV4 with the assumption that X follows
    a mixture of Gaussian distributions.
    :param eps: (ndarray (k,)) group proportions
    :param mu_x_groups: (ndarray (k,p)) means of the groups in rows
    :param reduced: (bool, default=False) if reduced=True, mu_x_grouped is trimmed and becomes ndarray (k,q), q being
     the dimension of the space spanned by mu_x_groups
    :param detailed: (bool, default=False) if detailed=True, return more objects
    :return: (ndarray (k+p,)) concatenation of eps and eigenvalues of COV-1COV4 if detailed=False (default)
    :return: cov, cov4, cov_inv_cov4, cov_inv_cov4_eigenvalues, cov_inv_cov4_eigenvectors if detailed=True
    """

    # Check sum of eps = 1
    assert eps.sum().round(2) == 1.00, "Sum of epsilons must be 1"

    # q is the dimension of the space spanned by mu_x_groups
    q = mu_x_groups[:, np.any(mu_x_groups != 0, axis=0)].shape[1]

    if reduced:
        # mu_x_grouped is trimmed and becomes ndarray (k,q)
        mu_x_groups = mu_x_groups[:, np.any(mu_x_groups != 0, axis=0)]

    # Name k the number of group and p the dimension
    k = np.shape(eps)[0]
    p = mu_x_groups.shape[1]

    # if q < (k-1) and not reduced:
    #     raise ValueError("Condition failed: q must be == (k-1) when reduced is False")

    # Defining a centered distribution Y:=X-mu_x
    mu_x = np.dot(eps, mu_x_groups)
    mu_y_groups = mu_x_groups - mu_x

    # Compute cov and its inverse
    cov = compute_cov(mu_y_groups, eps, k, p)
    cov_inv = np.linalg.inv(cov)

    # Name gammas the terms of cov and betas the terms of its inverse
    dict_beta = {}
    for i in range(q):
        for j in range(q):
            # vect_ij, associated with beta_ij, stored the values of i and j
            # eg: [1, 1, 0, ...] is associated with beta_12, [0, 2, 0, ...] is associated with beta_22
            # it is useful to compute the power of the components of Y when multiplied by YY^T
            vect_ij = np.zeros(q)
            vect_ij[i] += 1
            vect_ij[j] += 1
            beta_ij = "beta_" + str(i+1) + str(j+1)
            dict_beta[beta_ij] = (cov_inv[i][j], vect_ij)

    # Compute cov4
    cov4 = compute_cov4(cov, dict_beta, mu_y_groups, eps, k, p, q)

    # Eigenvalues of COV_inverse * COV_4
    cov_inv_cov4 = np.dot(cov_inv, cov4)
    cov_inv_cov4_eigenvalues, cov_inv_cov4_eigenvectors = np.linalg.eig(cov_inv_cov4)
    # Sort in decreasing order (eigenvectors are row vectors)
    cov_inv_cov4_eigenvalues, cov_inv_cov4_eigenvectors = sort_eigenvalues_eigenvectors(cov_inv_cov4_eigenvalues,
                                                                                        cov_inv_cov4_eigenvectors)

    if detailed:
        return cov, cov4, cov_inv_cov4, cov_inv_cov4_eigenvalues, cov_inv_cov4_eigenvectors
    else:
        return np.concatenate((eps, cov_inv_cov4_eigenvalues), axis=None)

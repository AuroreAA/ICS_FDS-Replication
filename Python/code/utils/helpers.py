import numpy as np
from numpy.linalg import multi_dot


def pad_to_length(arr, length):
    """
    Function to pad each array to the desired length
    """
    return np.pad(arr, (0, length - arr.size), 'constant', constant_values=0)


def sort_eigenvalues_eigenvectors(eigenvalues, eigenvectors):
    """
    Sort eigenvalues and associated eigenvectors in descending order
    :param eigenvalues: ndarray(p,) containing the eigenvalues
    :param eigenvectors: ndarray(p,p) containing the corresponding eigenvectors
    :return: updated eigenvalues and eigenvectors
    """
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvalues,  eigenvectors


def sqrt_symmetric_matrix(A, inverse=False):
    """
    Compute the matrix square root (or the inverse thereof) of a symmetric matrix.
    :param A: (N,N) array_like: Symmetric matrix whose square root to evaluate
    :param inverse: bool, optional: If true, compute the square root of the inverse of A (Default: False)
    """
    # Check symmetry
    # assert np.allclose(A, A.T), "A must be symmetric"
    # Eigen decomposition
    A_eigenval, A_eigenvect = np.linalg.eig(A)
    A_eigenval, A_eigenvect = sort_eigenvalues_eigenvectors(A_eigenval, A_eigenvect)
    # Compute matrix square root or its inverse
    power = -0.5 if inverse else 0.5
    A_sqrt = multi_dot([A_eigenvect, np.diag(A_eigenval**power), A_eigenvect.T])
    return A_sqrt

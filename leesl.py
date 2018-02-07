"""
Implementation of leesL bivariate spatial association
statistic.

The paper is titled:
    Developing a Bivariate Spatial Association Measure: An Integration
    of Pearson's r and Moran's I

Here's a link:
    https://www.researchgate.net/publication/220449126
"""
import numpy as np


def stats_matrix(X, spatial_weight_matrix, Y=None):
    """
    Produces a lees l statistic matrix. If only X is given lees L
    matrix will represent spatial association of the attributes in X.
    The rows of the matrices X and Y should represent the nodes in the
    spatial weight matrix. The row order of the X Y must correspond
    to the row/col order in the spatial weight matrix.
    :param X: numpy matrix or pandas dataframe.
    :param spatial_weight_matrix: nXn numpy matrix or pandas dataframe.
    :param Y: numpy matrix or pandas dataframe.
    :return: numpy matrix lees l statistic matrix.
    """

    # Using similar notation as paper, see equation (18).
    V = spatial_weight_matrix

    # V^T
    VT = spatial_weight_matrix.transpose()

    # (V^T V)
    VTV = np.dot(VT, V)

    # 1^T (V^T V) 1
    denominator = VTV.sum().sum()

    # Z^T
    Z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)
    ZT = Z.transpose()

    # Two sets of attributes case:
    # Replace the right side of the equation with second matrix,
    # then Z score.
    Y_is_a_matrix = Y != None
    if Y_is_a_matrix:
        Z = (Y - Y.mean(axis=0)) / Y.std(axis=0, dd0f=0)

    # Z^T (V^T V) Z : 'Z' is Y if two sets of data are given.
    ZTVTVZ = np.dot(np.dot(ZT,  VTV), Z)

    return ZTVTVZ / denominator


def L(Z, V):
    """"Equation (18) from the paper."""
    VTV = np.dot(V.transpose(), V)
    ZTVTVZ = np.dot(np.dot(Z.transpose(),  VTV), Z)
    return ZTVTVZ / VTV.sum().sum()


def spatial_lag(x, w):
    """Produces a spatial lag matrix. See equation (4) in paper."""
    x_squiggle = np.dot(w, x)
    return x_squiggle



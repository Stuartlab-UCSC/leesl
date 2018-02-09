"""
Implementation of leesL bivariate spatial association statistic.

In an attempt to ease understanding of the Lee's L publication
and the code, the verbiage between the code and the paper is aligned
and functions in the code site their respective equation number in the
paper.

The paper is titled:
    Developing a Bivariate te Spatial Association Measure: An
    Integration of Pearson's r and Moran's I

Here's a link:
    https://www.researchgate.net/publication/220449126
"""
import numpy as np
import multiprocessing


def statistic_matrix(X, spatial_weight_matrix, Y=None):
    """
    Produce a lees l statistic matrix.

    The rows of the matrices X and Y must be parallel to the rows/cols
    in the spatial weight matrix.
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
    Y_is_a_matrix = Y is not None
    if Y_is_a_matrix:
        Z = (Y - Y.mean(axis=0)) / Y.std(axis=0, ddof=0)

    # Z^T (V^T V) Z : 'Z' is Y if two sets of data are given.
    ZTVTVZ = np.dot(np.dot(ZT, VTV), Z)

    return ZTVTVZ / denominator


def permutation_test(
        x,
        y,
        spatial_weight_matrix,
        n=1000,
        n_jobs=1,
        test="two-sided"
):
    """
    Statistical evaluation of the spatial relationship between x and y.

    See section 4.4 of paper.
    :param x: numeric vector in space represented by spatial weights
    :param y: numeric vector in space represented by spatial weights
    :param spatial_weight_matrix: arbitrary nxn matrix
    :param n: number of permutations to generate.
    :param n_jobs: number of parallel process to initiate.
    :param test: one of "two-sided" "greater-than" or "less-than"
    :return: (tuple) numpy array of lees L stats from permutation,
     actual lees L stat from x and y, pvalue of the permutation test.
    """
    actual = statistic_matrix(x, spatial_weight_matrix, y)

    # Make an array of arguments to feed into a function via map.
    argument_array = \
        [
            (
                x[index_shuffle],
                spatial_weight_matrix,
                y[index_shuffle]
            ) for index_shuffle in random_indices_gen(n, len(x))
        ]

    if n_jobs == 1:
        randomized_leesls = \
            map(stat_wrapper, argument_array)

    if n_jobs > 1:
        pool = multiprocessing.Pool(processes=n_jobs)

        randomized_leesls = \
            np.array(pool.map(stat_wrapper, argument_array))
        pool.close()
        pool.join()

    pvalue = calculate_pvalue(actual, randomized_leesls, test)

    return randomized_leesls, actual, pvalue


def L(Z, V):
    """See equation (18) from the paper."""
    VTV = np.dot(V.transpose(), V)
    ZTVTVZ = np.dot(np.dot(Z.transpose(), VTV), Z)
    return ZTVTVZ / VTV.sum().sum()


def spatial_smoothing_scalar(X, spatial_weight_matrix):
    """Produce the SSS. See equation (9) of paper."""
    # Left side of the product in (9).
    numerator_left = float(len(X))  # n
    denomonator_left = (spatial_weight_matrix.sum(axis=1)**2).sum()

    left_side = numerator_left / denomonator_left

    # Right side of the product in (9).
    # Inner parentheses for numerator is a mean centered X.
    mean_center = X - X.mean(axis=0)
    outer_parentheses = np.dot(spatial_weight_matrix, mean_center)
    numerator_right = (outer_parentheses ** 2).sum(axis=0)
    denomonator_right = (mean_center ** 2).sum(axis=0)

    right_side = numerator_right / denomonator_right

    return left_side * right_side


def random_indices_gen(n, length):
    """Generate n number of randomly shuffled numbers [0:length)."""
    number_of_times = 0
    while number_of_times < n:
        yield np.random.choice(
            range(length),
            size=length,
            replace=False
        )
        number_of_times += 1


def calculate_pvalue(actual, randomized, test="two-sided"):
    """Calculate pvalue for a permutation test."""
    n = len(randomized)

    # Functions calculating counts for a type of test.
    def greater(randomized, actual):
        return (actual > randomized).sum()

    def lesser(randomized, actual):
        return (actual < randomized).sum()

    def both(randomized, actual):
        return lesser(randomized, actual) + greater(randomized, actual)

    # Dictionary is working as a switch statement for type of test.
    tests = {
        "greater-than": greater,
        "less-than": lesser,
        "two-sided": both
    }

    sum_passing = tests[test](actual, randomized)
    pvalue = float(sum_passing) / n
    return pvalue


def stat_wrapper(argument_triple):
    """Unpacks a triple and calls lees L statistic function."""
    leesL_statistic = statistic_matrix(
        X=argument_triple[0],
        spatial_weight_matrix=argument_triple[1],
        Y=argument_triple[2]
    )
    return leesL_statistic

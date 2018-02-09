"""
Tests for lees L.

These tests assure that using pandas or numpys results in the
correct set of answers.

Tests are attributes on a 6x6 grid with neighbors being squares that
share an edge.
"""
import pandas as pd
import numpy as np
import leesl
from itertools import product

def test_calculate_pvalue_greater_then():
    data = np.concatenate([np.repeat(1, 10), np.repeat(5, 10)])
    answer = leesl.calculate_pvalue(2, data, "greater-than")
    assert answer == .5


def test_calculate_pvalue_less_then():
    data = np.concatenate([np.repeat(1, 10), np.repeat(5, 10)])
    answer = leesl.calculate_pvalue(2, data, "less-than")
    assert answer == .5


def test_calculate_pvalue_two_sided():
    data = np.concatenate([np.repeat(1, 10), np.repeat(5, 10)])
    answer = leesl.calculate_pvalue(3, data, "two-sided")
    assert answer == 1



def test_permutation_test_sample_size():
    spatial_weight_matrix = checkerboard_similarity()
    known_arrangements = get_spatial_vectors_df()[["A","B"]]
    x = known_arrangements["A"].values
    y = known_arrangements["B"].values
    randomized, a, p = leesl.permutation_test(
        x,
        y,
        spatial_weight_matrix,
        n=30
    )
    assert len(randomized) == 30


def test_permutation_test_sample_size_n_jobs():
    spatial_weight_matrix = checkerboard_similarity()
    known_arrangements = get_spatial_vectors_df()[["A","B"]]
    x = known_arrangements["A"].values
    y = known_arrangements["B"].values
    randomized, a, p = leesl.permutation_test(
        x,
        y,
        spatial_weight_matrix,
        n=30,
        n_jobs=2
    )
    assert len(randomized) == 30


def test_L_many_pandas():
    spatial_weight_matrix = checkerboard_similarity()
    known_arrangements = get_spatial_vectors_df()
    known_answers = get_known_answers()
    mean_centered = known_arrangements - known_arrangements.mean(axis=0)
    stdev = known_arrangements.std(axis=0, ddof=0)
    Z =  mean_centered / stdev
    passes = np.allclose(
        known_answers,
        leesl.L(Z, spatial_weight_matrix)
    )

    assert passes


def test_L_many_numpy():
    spatial_weight_matrix = checkerboard_similarity()
    known_arrangements = get_spatial_vectors_df().values
    known_answers = get_known_answers()
    mean_centered = known_arrangements - known_arrangements.mean(axis=0)
    stdev = known_arrangements.std(axis=0, ddof=0)
    Z =  mean_centered / stdev
    passes = np.allclose(
        known_answers,
        leesl.L(Z, spatial_weight_matrix)
    )

    assert passes


def test_L_single_numpy():
    spatial_weight_matrix = checkerboard_similarity()
    known_arrangement = get_spatial_vectors_df()["B"].values
    known_answers = get_known_answers()[1,1]

    mean_centered = known_arrangement - known_arrangement.mean(axis=0)
    stdev = known_arrangement.std(axis=0, ddof=0)
    Z =  mean_centered / stdev

    passes = np.isclose(
        known_answers,
        leesl.L(Z, spatial_weight_matrix)
    )

    assert passes


def test_L_single_pandas():
    spatial_weight_matrix = checkerboard_similarity()
    known_arrangement = get_spatial_vectors_df()["B"]
    known_answers = get_known_answers()[1,1]

    mean_centered = known_arrangement - known_arrangement.mean(axis=0)
    stdev = known_arrangement.std(axis=0, ddof=0)
    Z =  mean_centered / stdev

    passes = np.isclose(
        known_answers,
        leesl.L(Z, spatial_weight_matrix)
    )

    assert passes


def test_statistic_matrix_both_sides_pandas():
    spatial_weight_matrix = checkerboard_similarity()
    known_arrangements = get_spatial_vectors_df()
    known_answers = get_known_answers()

    passes = np.allclose(
        known_answers,
        leesl.statistic_matrix(
            known_arrangements,
            spatial_weight_matrix,
            known_arrangements
        )
    )
    assert passes


def test_statistic_matrix_both_sides_np():
    spatial_weight_matrix = checkerboard_similarity()
    known_arrangements_np = get_spatial_vectors_df().values
    known_answers = get_known_answers()

    passes = np.allclose(
        known_answers,
        leesl.statistic_matrix(
            known_arrangements_np,
            spatial_weight_matrix,
            known_arrangements_np
        )
    )
    assert passes


def test_statistic_matrix_np_and_pandas():
    spatial_weight_matrix = checkerboard_similarity()
    known_arrangements_np = get_spatial_vectors_df().values
    known_arrangements_pd = get_spatial_vectors_df()
    known_answers = get_known_answers()

    passes = np.allclose(
        known_answers,
        leesl.statistic_matrix(
            known_arrangements_np,
            spatial_weight_matrix,
            known_arrangements_pd
        )
    )
    assert passes


def test_statistic_matrix_pandas_and_np():
    spatial_weight_matrix = checkerboard_similarity()
    known_arrangements_pd = get_spatial_vectors_df()
    known_arrangements_np = get_spatial_vectors_df().values
    known_answers = get_known_answers()

    passes = np.allclose(
        known_answers,
        leesl.statistic_matrix(
            known_arrangements_pd,
            spatial_weight_matrix,
            known_arrangements_np
        )
    )
    assert passes


def test_statistic_matrix_many_pandas():
    spatial_weight_matrix = checkerboard_similarity()
    known_arrangements = get_spatial_vectors_df()
    known_answers = get_known_answers()

    passes = np.allclose(
        known_answers,
        leesl.statistic_matrix(known_arrangements, spatial_weight_matrix)
    )
    assert passes


def test_statistic_matrix_many_numpy():
    # Switch from pandas to numpy.
    spatial_weight_matrix = checkerboard_similarity().values
    known_attrs = get_spatial_vectors_df().values
    known_answers = get_known_answers()

    passes = np.allclose(
        known_answers,
        leesl.statistic_matrix(known_attrs, spatial_weight_matrix)
    )
    assert passes


def test_statistic_matrix_single_pandas():
    spatial_weight_matrix = checkerboard_similarity()
    known_arrangements = get_spatial_vectors_df()[["A", "B"]]
    known_answers = get_known_answers()[0:2, 0:2]

    passes = np.allclose(
        known_answers,
        leesl.statistic_matrix(known_arrangements, spatial_weight_matrix)
    )
    assert passes


def test_statistic_matrix_single_numpy():
    # Switch from pandas to numpy.
    spatial_weight_matrix = checkerboard_similarity().values
    known_arrangements_np = get_spatial_vectors_df()[["A", "B"]].values
    known_answers = get_known_answers()[0:2, 0:2]

    passes = np.allclose(
        known_answers,
        leesl.statistic_matrix(known_arrangements_np, spatial_weight_matrix)
    )
    assert passes


def test_SSS_many_pandas():
    spatial_weight_matrix = checkerboard_similarity()
    spatial_vectors_df = get_spatial_vectors_df()
    sss = leesl.spatial_smoothing_scalar(
        spatial_vectors_df,
        spatial_weight_matrix
    )
    known_sss = get_known_answers().diagonal()

    assert np.allclose(sss, known_sss)


def test_SSS_many_numpy():
    spatial_weight_matrix = checkerboard_similarity()
    spatial_vectors_np = get_spatial_vectors_df().values
    sss = leesl.spatial_smoothing_scalar(
        spatial_vectors_np,
        spatial_weight_matrix
    )
    known_sss = get_known_answers().diagonal()

    assert np.allclose(sss, known_sss)


def test_SSS_single_pandas():
    spatial_weight_matrix = checkerboard_similarity()
    spatial_vector_pd = get_spatial_vectors_df()["A"]
    sss = leesl.spatial_smoothing_scalar(
        spatial_vector_pd,
        spatial_weight_matrix
    )
    known_sss = get_known_answers()[0,0]

    assert np.allclose(sss, known_sss)


def test_SSS_single_numpy():

    spatial_weight_matrix = checkerboard_similarity()
    spatial_vector_np = get_spatial_vectors_df()["A"].values
    sss = leesl.spatial_smoothing_scalar(
        spatial_vector_np,
        spatial_weight_matrix
    )
    known_sss = get_known_answers()[0, 0]

    assert np.allclose(sss, known_sss)


def checkerboard_similarity():
    """Make a 6x6 grid where any square sharing an edge is
    connected."""
    # Making a 6x6 Square grid. (0, 0) in top left corner.
    board_length = 6

    # Dictionary to hold (row, col) -> similarity
    square_sim_dict = {}

    # Functions to run through at each square producing
    # neighbors.
    def neighbors(row_n, col_n):
         return [
            (row_n + 1, col_n), #down
            (row_n - 1, col_n), #up
            (row_n, col_n + 1), #right
            (row_n, col_n - 1), #left
        ]

    def on_board(row_col, length=6):
        return \
            not (
                (row_col[0] > length-1 or row_col[1] > length-1)
                or (row_col[0] < 0 or row_col[1] < 0)
            )

    board_indecies = [range(board_length), range(board_length)]
    for row_n, col_n in product(*board_indecies):
        neighbors_on_board = filter(
            on_board,
            neighbors(row_n, col_n)
        )
        square_sim_dict[(row_n, col_n)] = neighbors_on_board

    # Throw it in a dataframe.
    df = pd.DataFrame(
        np.reshape(np.repeat(0, 36*36), (36, 36)),
        index=square_sim_dict.keys(),
        columns=square_sim_dict.keys()
    )
    for node in square_sim_dict.keys():
        edges = square_sim_dict[node]
        for edge in edges:
            df.loc[node, edge] = 1

    # Make sure the ordering is properly defined.
    simdf = df.loc[sorted(df.columns), sorted(df.columns)]

    return simdf


def get_known_answers():
    # Correct values came from the R implementation of
    # leesl in the package spdep Version 0.6-15.
    known_answers = np.array(
        [
            [.358653846, 0, 0, -0.044989875,  0.184495192],
            [0, 1,  -1, .0678401,  -.0193506],
            [0,  -1,   1, -.0678401, .0193506],
            [-.0449899, .0678401,  -.0678401, .1385315, -.135423],
            [.184495,  -.01935059,   .01935059, -.1354234, .409495192]
        ]
    )
    return known_answers


def get_spatial_vectors_df():
    # Attributes with known lees l values. 2d array represents
    # squares on a 6x6 grid.
    A = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
    ])
    B = np.array([
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
    ])
    C = np.array([
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
    ])
    D = np.array([
        [0, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 1, 0],
        [1, 0, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0],
    ])
    E = np.array([
        [0, 0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
    ])

    known_attrs = pd.DataFrame(
        {"A": A.flatten(),
         "B": B.flatten(),
         "C": C.flatten(),
         "D": D.flatten(),
         "E": E.flatten()}
    )
    return known_attrs
import pandas as pd
import numpy as np
from ast import literal_eval
from utils.helpers import pad_to_length
from utils.cov_cov4_functions_simplified import eig_cc4_simplified
import plotly.graph_objects as go


config_path = 'config/groups_means_grid.csv'
path_results_output = '../results/thresholds/'
path_figures_output = '../figures/thresholds/'


k_list = [k for k in range(2, 11)]


def str_to_np_array(s):
    """
    Converts a string representation of a list or array into a NumPy array.
    :param s: (str)  string representing a list, tuple, or other object that can be converted into a NumPy array.
    Example: "[1, 2, 3]" or "(4, 5, 6)".
    :return: (np.ndarray) NumPy array constructed from the literal evaluation of the input string.
    """
    return np.array(literal_eval(s))


def create_alpha_grid_alpha1_decrease(k, precision=3):
    """
    For a given number of groups k, create a grid with group proportions (alphas) in rows: sum of each row = 1.
    Proportions are ordered in ascending order such that:
    - alpha 1: 1%, 2%, ..., 1/k; if precision=2
    - alpha i: 1/k; i=2, ..., k-1
    - alpha k = 1 - sum from 1 to k of epsilons
    :param k: (int) number of groups
    :param precision: (int) number of decimals of the group proportions
    :return: (ndarray) group proportions between 0 and 1
    """
    one_over_k = round(1/k, precision)
    step = 10 ** -precision
    e = np.arange(step, one_over_k + step, step)
    alpha_grid = np.vstack((e, np.ones([k-2, e.shape[0]])*one_over_k)).T
    alpha_grid = np.c_[alpha_grid, 1-np.sum(alpha_grid, axis=1)]
    alpha_grid = np.around(alpha_grid, precision)
    return alpha_grid


def create_alpha_grid_near_threshold(k, df_threshold, precision=3):
    """
    For a given number of groups k, create a grid with group proportions (alphas) in rows: sum of each row = 1. This
    method requires at least 3 groups. Proportions are generated such that:
    - alpha 1: 1%, 2%, ..., 1/k; if precision=2
    - alpha 2: threshold found with Scenario 1 (create_alpha_grid_alpha1_decrease) + 0.02
    - alpha i: 1/k; i=3, ..., k-1
    - alpha k = 1 - sum from 1 to k of epsilons
    :param k: (int) number of groups
    :param precision: (int) number of decimals of the group proportions
    :return: (ndarray) group proportions between 0 and 1
    """
    assert k > 2, "k must be > 2"
    threshold_init = df_threshold[df_threshold['k'] == k].iloc[0, 1] + 0.02
    alpha_grid = create_alpha_grid_alpha1_decrease(k, precision=3)
    alpha_grid[:, 1] = threshold_init
    alpha_grid = alpha_grid[:, :-1]
    alpha_grid = np.c_[alpha_grid, 1 - np.sum(alpha_grid, axis=1)]
    alpha_grid = np.around(alpha_grid, precision)
    return alpha_grid


def create_alpha_grid_alpha2_5pct(k, precision=3):
    """
    For a given number of groups k, create a grid with group proportions (alphas) in rows: sum of each row = 1. This
    method requires at least 3 groups. Proportions are generated such that:
    - alpha 1: 1%, 2%, ..., 1/k; if precision=2
    - alpha 2: 0.05
    - alpha i: 1/k; i=3, ..., k-1
    - alpha k = 1 - sum from 1 to k of epsilons
    :param k: (int) number of groups
    :param precision: (int) number of decimals of the group proportions
    :return: (ndarray) group proportions between 0 and 1
    """
    assert k > 2, "k must be > 2"
    alpha_grid = create_alpha_grid_alpha1_decrease(k, precision=3)
    alpha_grid[:, 1] = 0.05
    alpha_grid = alpha_grid[:, :-1]
    alpha_grid = np.c_[alpha_grid, 1 - np.sum(alpha_grid, axis=1)]
    alpha_grid = np.around(alpha_grid, precision)
    return alpha_grid


def compute_thresholds(k_list, df_means):
    """
    Compute the threshold at which the number of eigenvalues > 1 goes from 0 to 1, for a list of numbers of groups. The
    group proportion grid used is the one resulting from create_grid_alpha1_decrease. This is referred to as Setup 1.
    :param k_list: (list) numbers of groups to compute the thresholds for
    :param df_means: (pd.DataFrame) one row is one configuration of group means for up to 10 groups
    :return: (pd.DataFrame) containing one column for the number of group k and one column for the associated threshold
    """
    threshold = []
    for k in k_list:
        p = k-1
        # Take any row from df_means as the means have no impact on the eigenvalues when q = k-1
        means_reduced = [np.array(df_means.iloc[0, col]) for col in range(k-1)]
        means_reduced.append(np.array([0]))
        means_groups = np.vstack(
            [pad_to_length(arr, p) for arr in means_reduced]  # add 0 to group means to obtain required shape
        )
        # Create grid and compute eigenvalues for each configuration (each row of the grid)
        alpha_grid = create_alpha_grid_alpha1_decrease(k, precision=3)
        cc4_eigenvalues = np.apply_along_axis(eig_cc4_simplified, 1, alpha_grid, means_groups, reduced=True,
                                              use_cov_inv_sqrt=True, detailed=False)
        col_names_lambda = ["lambda_" + str(i+1) for i in range(p)]
        col_names = ["alpha_" + str(i+1) for i in range(k)] + col_names_lambda
        df_eigenvalues = pd.DataFrame(cc4_eigenvalues, columns=col_names)

        # The threshold is the first value of 'alpha_1' for which there is 1 eigenvalue > 1
        df_eigenvalues["nb_lambda_gt_1"] = df_eigenvalues[col_names_lambda].gt(1).sum(axis=1)
        df_eigenvalues = df_eigenvalues.sort_values(by='alpha_1', ascending=False).copy()
        threshold_i = np.array([k, df_eigenvalues[df_eigenvalues["nb_lambda_gt_1"] == 1].iloc[0, 0]])
        threshold.append(threshold_i)
    df_threshold = pd.DataFrame(threshold, columns=['k', 'threshold'])
    return df_threshold


def compute_thresholds_near_threshold(k_list, df_means, df_threshold):
    """
    Compute the threshold at which the number of eigenvalues > 1 goes from 0 to 1, for a list of numbers of groups. The
    group proportion grid used is the one resulting from create_alpha_grid_near_threshold. This is referred to as
    Setup 2.
    :param k_list: (list) numbers of groups to compute the thresholds for
    :param df_means: (pd.DataFrame) one row is one configuration of group means for up to 10 groups
    :param df_threshold: (pd.DataFrame) one row per number of group k and its associated threshold computed with
    Scenario 1.
    :return: (pd.DataFrame) containing one column for the number of group k and one column for the associated threshold
    """
    threshold = []
    for k in k_list:
        p = k - 1
        # Take any row from df_means as the means have no impact on the eigenvalues when q = k-1
        means_reduced = [np.array(df_means.iloc[0, col]) for col in range(k - 1)]
        means_reduced.append(np.array([0]))
        means_groups = np.vstack(
            [pad_to_length(arr, p) for arr in means_reduced]  # add 0 to group means to obtain required shape
        )
        # Create grid and compute eigenvalues for each configuration (each row of the grid)
        alpha_grid = create_alpha_grid_near_threshold(k, df_threshold, precision=3)
        cc4_eigenvalues = np.apply_along_axis(eig_cc4_simplified, 1, alpha_grid, means_groups, reduced=True,
                                              use_cov_inv_sqrt=True, detailed=False)
        col_names_lambda = ["lambda_" + str(i + 1) for i in range(p)]
        col_names = ["alpha_" + str(i + 1) for i in range(k)] + col_names_lambda
        df_eigenvalues = pd.DataFrame(cc4_eigenvalues, columns=col_names)

        # The threshold is the first value of 'alpha_1' for which there is 1 eigenvalue > 1
        df_eigenvalues["nb_lambda_gt_1"] = df_eigenvalues[col_names_lambda].gt(1).sum(axis=1)
        df_eigenvalues = df_eigenvalues.sort_values(by='alpha_1', ascending=False).copy()
        threshold_i = np.array([k, df_eigenvalues[df_eigenvalues["nb_lambda_gt_1"] == 1].iloc[0, 0]])
        threshold.append(threshold_i)
    df_threshold = pd.DataFrame(threshold, columns=['k', 'threshold'])
    return df_threshold


def compute_thresholds_alpha2_5pct(k_list, df_means):
    """
    Compute the threshold at which the number of eigenvalues > 1 goes from 0 to 1, for a list of numbers of groups. The
    group proportion grid used is the one resulting from create_alpha_grid_alpha2_5pct. This is referred to as
    Setup 3.
    :param k_list: (list) numbers of groups to compute the thresholds for
    :param df_means: (pd.DataFrame) one row is one configuration of group means for up to 10 groups
    :return: (pd.DataFrame) containing one column for the number of group k and one column for the associated threshold
    """
    threshold = []
    for k in k_list:
        p = k - 1
        # Take any row from df_means as the means have no impact on the eigenvalues when q = k-1
        means_reduced = [np.array(df_means.iloc[0, col]) for col in range(k - 1)]
        means_reduced.append(np.array([0]))
        means_groups = np.vstack(
            [pad_to_length(arr, p) for arr in means_reduced]  # add 0 to group means to obtain required shape
        )
        # Create grid and compute eigenvalues for each configuration (each row of the grid)
        alpha_grid = create_alpha_grid_alpha2_5pct(k, precision=3)
        cc4_eigenvalues = np.apply_along_axis(eig_cc4_simplified, 1, alpha_grid, means_groups, reduced=True,
                                              use_cov_inv_sqrt=True, detailed=False)
        col_names_lambda = ["lambda_" + str(i + 1) for i in range(p)]
        col_names = ["alpha_" + str(i + 1) for i in range(k)] + col_names_lambda
        df_eigenvalues = pd.DataFrame(cc4_eigenvalues, columns=col_names)

        # The threshold is the first value of 'alpha_1' for which there are 2 eigenvalue > 1
        df_eigenvalues["nb_lambda_gt_1"] = df_eigenvalues[col_names_lambda].gt(1).sum(axis=1)
        df_eigenvalues = df_eigenvalues.sort_values(by='alpha_1', ascending=False).copy()
        threshold_i = np.array([k, df_eigenvalues[df_eigenvalues["nb_lambda_gt_1"] == 2].iloc[0, 0]])
        threshold.append(threshold_i)
    df_threshold = pd.DataFrame(threshold, columns=['k', 'threshold'])
    return df_threshold


def scatterplot_threshold_per_group(df_threshold, df_threshold_near, df_threshold_1alpha_5pct):
    """
    Generates a scatter plot comparing threshold values across different setups.
    :param df_threshold: (pd.DataFrame) threshold valued obtained with compute_thresholds(), it is referred to as
    Setup 1
    :param df_threshold_near: (pd.DataFrame) threshold valued obtained with compute_thresholds_near_threshold(), it is
    referred to as Setup 2
    :param df_threshold_1alpha_5pct: (pd.DataFrame) threshold valued obtained with compute_thresholds_alpha2_5pct(),
    it is referred to as Setup 3
    :return: plotly figure
    """

    fig = go.Figure(
        data=go.Scatter(x=[str(int(val)) for val in df_threshold['k']],
                        y=df_threshold['threshold'],
                        mode='lines+markers', marker=dict(symbol='circle'), name='Setup 1'))
    fig.add_trace(go.Scatter(x=[str(int(val)) for val in df_threshold_near['k']],
                             y=df_threshold_near['threshold'],
                             mode='lines+markers', marker=dict(symbol='square'), name='Setup 2'))
    fig.add_trace(go.Scatter(x=[str(int(val)) for val in df_threshold_1alpha_5pct['k']],
                             y=df_threshold_1alpha_5pct['threshold'],
                             mode='lines+markers', marker=dict(symbol='diamond'), name='Setup 3'))

    fig.update_xaxes(
        title_text='Number of groups',
        title_font=dict(size=13),
        tickfont=dict(size=12)
    )
    fig.update_yaxes(
        title_text='Threshold',
        title_font=dict(size=13),
        tickfont=dict(size=12)
    )
    fig.update_layout(plot_bgcolor='rgba(229,236,246,0.5)')

    return fig


def table_threshold_per_group(df_threshold, df_threshold_near, df_threshold_1alpha_5pct):
    """
    Merges threshold data from three different setups into a single DataFrame for comparison.
    :param df_threshold: (pd.DataFrame) threshold valued obtained with compute_thresholds(), it is referred to as
    Setup 1
    :param df_threshold_near: (pd.DataFrame) threshold valued obtained with compute_thresholds_near_threshold(), it is
    referred to as Setup 2
    :param df_threshold_1alpha_5pct: (pd.DataFrame) threshold valued obtained with compute_thresholds_alpha2_5pct(),
    it is referred to as Setup 3
    :return: (pd.DataFrame) rows are the number of groups (k) and columns are the setups
    """
    # Rename 'threshold' column to a unique name
    df_threshold = df_threshold.rename(columns={'threshold': ' Setup 1'})
    df_threshold_near = df_threshold_near.rename(columns={'threshold': 'Setup 2'})
    df_threshold_1alpha_5pct = df_threshold_1alpha_5pct.rename(columns={'threshold': 'Setup 3'})

    # Merge DataFrames on 'k'
    df_merged_thresholds = (df_threshold.merge(df_threshold_near, on='k', how='outer')
                            .merge(df_threshold_1alpha_5pct, on='k', how='outer'))

    # Format
    df_merged_thresholds['k'] = df_merged_thresholds['k'].astype(int)
    df_merged_thresholds.iloc[:, 1:] = df_merged_thresholds.iloc[:, 1:].round(3)

    return df_merged_thresholds


if __name__ == "__main__":
    df_means = pd.read_csv(config_path, sep=",")
    df_means = df_means.map(str_to_np_array)
    # Scenario 1
    df_threshold = compute_thresholds(k_list, df_means)
    if 2 in k_list:
        k_list.remove(2)
    # Scenario 2 and 3 (k >= 3)
    df_threshold_near = compute_thresholds_near_threshold(k_list, df_means, df_threshold)
    df_threshold_alpha2_5pct = compute_thresholds_alpha2_5pct(k_list, df_means)
    # Figure 6
    fig = scatterplot_threshold_per_group(
        df_threshold, df_threshold_near, df_threshold_alpha2_5pct)
    fig.write_image(path_figures_output + "scatterplot_3setups.jpg", scale=2)
    # Table 1
    df_output = table_threshold_per_group(df_threshold, df_threshold_near, df_threshold_alpha2_5pct)
    df_output.to_csv(path_results_output + 'results_3setups.csv', index=False)
